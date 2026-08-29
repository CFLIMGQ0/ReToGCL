"""按 A→B→C 继承最优值，运行单个 idea/数据集的顺序五折调参。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.evaluation import METRIC_KEYS, linear_probe_five_fold_metrics, summarize
from src.baselines.gcl_baselines import set_seed
from src.models.idea_parameter_registry import parameter_space, value_token
from src.models.research_ideas import IDEA_IDS, build_research_idea_model, get_idea_spec
from src.scripts.run_research_ideas import DATASETS, _augment_pair, extract_embeddings
from src.scripts.run_sota_five_fold import choose_device, load_graphs, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", choices=IDEA_IDS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/main_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("IDEA_PARA"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolved_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    return data_root.resolve(), output_dir.resolve()


def configuration_id(parameters: dict[str, float]) -> str:
    payload = json.dumps(parameters, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def selection_key(result: dict) -> tuple[float, float, float, float, float]:
    """四指标等权；平分时依次使用 Accuracy、F1、NMI、ARI。"""

    values = [float(result[f"{metric}_percent"]["mean"]) for metric in METRIC_KEYS]
    score = sum(values) / len(values)
    return score, values[0], values[3], values[1], values[2]


def build_model(args: argparse.Namespace, in_dim: int, classes: int, parameters: dict[str, float]):
    return build_research_idea_model(
        args.idea,
        in_dim,
        args.hidden_dim,
        args.layers,
        classes,
        auxiliary_weight=0.25,
        idea_temperature=0.2,
        operator_mix=1.0,
        idea_parameters=parameters,
    )


def smoke_test(args: argparse.Namespace, graphs, in_dim: int, classes: int, device: torch.device) -> None:
    """逐一改变 15 个参数点，验证前向、反向、更新和参数影响。"""

    space = parameter_space(args.idea)
    batch = next(iter(DataLoader(graphs[:8], batch_size=min(8, len(graphs))))).to(device)
    baseline_loss = None
    tested = 0
    for parameter in space.parameters:
        for value in tqdm(parameter.values, desc=f"{args.idea}/{parameter.name}", unit="值", leave=False):
            values = space.defaults
            values[parameter.name] = value
            set_seed(args.seed)
            model = build_model(args, in_dim, classes, values).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            first, second = _augment_pair(batch)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.idea_loss(first, second)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{args.idea}/{parameter.name}={value} 损失非有限")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gradient):
                raise RuntimeError(f"{args.idea}/{parameter.name}={value} 梯度非有限")
            optimizer.step()
            if baseline_loss is None:
                baseline_loss = float(loss.detach())
            tested += 1
            del model, optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"{args.idea}/{args.dataset} 冒烟通过：{tested} 个参数点，参考损失={baseline_loss:.6f}")


def run_trial(
    args: argparse.Namespace,
    graphs,
    in_dim: int,
    classes: int,
    parameters: dict[str, float],
    data_output: Path,
    device: torch.device,
) -> dict:
    config_id = configuration_id(parameters)
    trial_path = data_output / "trials" / args.idea / args.dataset / f"{config_id}.json"
    if trial_path.exists() and not args.force:
        return json.loads(trial_path.read_text(encoding="utf-8"))

    set_seed(args.seed)
    model = build_model(args, in_dim, classes, parameters).to(device)
    checkpoint = data_output / "checkpoints" / args.idea / args.dataset / f"{config_id}.pt"
    log_path = data_output / "logs" / args.idea / args.dataset / f"{config_id}.jsonl"
    batch_size = min(args.batch_size, 64 if args.dataset == "COLLAB" else args.batch_size)

    if checkpoint.exists() and not args.force:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            progress = tqdm(
                range(1, args.epochs + 1),
                desc=f"{args.idea}/{args.dataset}/{config_id}",
                unit="轮",
                leave=False,
            )
            for epoch in progress:
                model.train()
                total_loss = 0.0
                total_graphs = 0
                for batch in loader:
                    batch = batch.to(device)
                    first, second = _augment_pair(batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss, diagnostics = model.idea_loss(first, second)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"{args.idea}/{args.dataset}/{config_id}/epoch={epoch} 损失非有限"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += float(loss.detach()) * batch.num_graphs
                    total_graphs += batch.num_graphs
                record = {
                    "epoch": epoch,
                    "loss": total_loss / max(total_graphs, 1),
                    **diagnostics,
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                progress.set_postfix(loss=f"{record['loss']:.4f}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "idea_id": args.idea,
                "dataset": args.dataset,
                "config_id": config_id,
                "parameters": parameters,
                "epochs": args.epochs,
            },
            checkpoint,
        )

    embeddings, labels = extract_embeddings(model, graphs, batch_size, device)
    metrics = linear_probe_five_fold_metrics(
        embeddings,
        labels,
        device=device,
        seed=args.seed,
        folds=args.folds,
        epochs=args.probe_epochs,
    )
    result = {
        "idea_id": args.idea,
        "dataset": args.dataset,
        "config_id": config_id,
        "parameters": parameters,
        "protocol": "sequential_greedy_A_then_B_then_C",
        "selection_metric": "mean(accuracy,nmi,ari,macro_f1)",
        "fixed_non_idea_parameters": {
            "seed": args.seed,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "probe_epochs": args.probe_epochs,
            "folds": args.folds,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "base_contrastive_temperature": 0.2,
        },
        **{f"{key}_percent": summarize(values) for key, values in metrics.items()},
    }
    result["selection_score"] = round(selection_key(result)[0], 6)
    save_json(trial_path, result)
    return result


def run_sequence(args: argparse.Namespace, data_root: Path, output_dir: Path, device: torch.device) -> Path:
    final_path = output_dir / "jobs" / f"{args.idea}_{args.dataset}.json"
    if final_path.exists() and not args.force:
        print(f"已存在，跳过：{final_path}")
        return final_path

    graphs, in_dim, classes = load_graphs(data_root, args.dataset)
    space = parameter_space(args.idea)
    selected = space.defaults
    stages = []
    for stage_index, parameter in enumerate(space.parameters):
        candidates = []
        for value in parameter.values:
            trial_parameters = dict(selected)
            trial_parameters[parameter.name] = value
            result = run_trial(
                args, graphs, in_dim, classes, trial_parameters, output_dir, device
            )
            candidates.append({
                "value": value,
                "config_id": result["config_id"],
                "selection_score": result["selection_score"],
                **{f"{metric}_percent": result[f"{metric}_percent"] for metric in METRIC_KEYS},
            })
        best_result = max(
            (run_trial(
                args,
                graphs,
                in_dim,
                classes,
                {**selected, parameter.name: candidate["value"]},
                output_dir,
                device,
            ) for candidate in candidates),
            key=selection_key,
        )
        selected[parameter.name] = float(best_result["parameters"][parameter.name])
        stages.append({
            "stage": "ABC"[stage_index],
            "parameter": parameter.name,
            "inherited_parameters": {
                name: value for name, value in selected.items() if name != parameter.name
            },
            "candidates": candidates,
            "selected_value": selected[parameter.name],
            "selected_config_id": best_result["config_id"],
            "selected_score": best_result["selection_score"],
        })
        partial = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "idea_id": args.idea,
            "dataset": args.dataset,
            "status": f"stage_{'ABC'[stage_index]}_completed",
            "selected_parameters": selected,
            "stages": stages,
        }
        save_json(output_dir / "sequences" / f"{args.idea}_{args.dataset}.json", partial)

    final_trial = run_trial(args, graphs, in_dim, classes, selected, output_dir, device)
    spec = get_idea_spec(args.idea)
    final = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "idea_id": args.idea,
        "family": spec.family_name,
        "revision": spec.revision,
        "dataset": args.dataset,
        "status": "completed",
        "protocol": "sequential_greedy_A_then_B_then_C",
        "selection_rule": {
            "primary": "四指标五折均值等权平均",
            "tie_break": ["accuracy", "macro_f1", "nmi", "ari"],
        },
        "selected_parameters": selected,
        "selected_config_id": final_trial["config_id"],
        "selection_score": final_trial["selection_score"],
        "stages": stages,
        "final_metrics": {
            metric: final_trial[f"{metric}_percent"] for metric in METRIC_KEYS
        },
        "fixed_non_idea_parameters": final_trial["fixed_non_idea_parameters"],
        "fidelity": "differentiable_prototype",
    }
    save_json(final_path, final)
    return final_path


def main() -> None:
    args = parse_args()
    data_root, output_dir = resolved_paths(args)
    device = choose_device(args.device)
    print(f"使用设备：{device}")
    if args.smoke_test:
        graphs, in_dim, classes = load_graphs(data_root, args.dataset)
        smoke_test(args, graphs, in_dim, classes, device)
        return
    path = run_sequence(args, data_root, output_dir, device)
    print(f"顺序调参完成：{path}")


if __name__ == "__main__":
    main()
