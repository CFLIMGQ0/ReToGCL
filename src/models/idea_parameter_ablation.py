"""110 个研究 idea 的单因素参数消融网格。"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PARAMETERS = {
    "idea_loss_weight": 0.25,
    "idea_temperature": 0.2,
    "operator_mix": 1.0,
}

PARAMETER_VALUES = {
    "idea_loss_weight": (0.0, 0.1, 0.25, 0.5, 1.0),
    # 默认温度 0.2 由 loss-weight 网格中的共同默认配置提供，避免重复任务。
    "idea_temperature": (0.05, 0.1, 0.15, 0.3, 0.5),
    # 默认混合强度 1.0 同理由共同默认配置提供。
    "operator_mix": (0.0, 0.25, 0.5, 0.75, 1.25),
}


@dataclass(frozen=True)
class ParameterConfig:
    config_id: str
    varied_parameter: str
    varied_value: float
    parameters: dict[str, float]


def _value_token(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def build_parameter_configs() -> tuple[ParameterConfig, ...]:
    configs = []
    seen = set()
    for parameter, values in PARAMETER_VALUES.items():
        for value in values:
            parameters = dict(DEFAULT_PARAMETERS)
            parameters[parameter] = float(value)
            signature = tuple(sorted(parameters.items()))
            if signature in seen:
                raise RuntimeError(f"参数网格产生重复配置：{signature}")
            seen.add(signature)
            configs.append(ParameterConfig(
                config_id=f"{parameter}__{_value_token(float(value))}",
                varied_parameter=parameter,
                varied_value=float(value),
                parameters=parameters,
            ))
    if len(configs) != 15:
        raise RuntimeError(f"期望 15 个唯一配置，实际得到 {len(configs)} 个")
    return tuple(configs)


PARAMETER_CONFIGS = build_parameter_configs()
CONFIG_BY_ID = {config.config_id: config for config in PARAMETER_CONFIGS}
