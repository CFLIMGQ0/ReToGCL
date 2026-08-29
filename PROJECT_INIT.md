# PROJECT_INIT.md（项目初始化与目录结构规范）

## 根目录文档职责（必须遵守）

项目根目录默认维护以下说明文档：

- `AGENTS.md`：开发代理协作规范文档，用于说明与用户沟通、文档编写、过程总结、执行日志等语言规范和协作约束。
- `PROJECT_INIT.md`：项目初始化与目录结构规范文档，用于说明项目标准目录、各目录职责、根目录文档职责和后续开发约定。
- `README.md`：项目运行逻辑文档，用于说明项目目标、整体流程、训练/验证/测试运行逻辑、主要入口脚本、配置文件和实验输出位置。
- `DATASETS.md`：数据说明文档，用于记录数据来源、数据目录结构、标签定义、数据统计、数据清洗、划分策略、数据相关脚本和数据操作注意事项。
- `EXPERIMENT_PLAN.md`：实验方案文档，用于记录研究定位、数据划分、预处理、Baseline、主模型、评价指标和论文实验设计。

后续维护时应默认遵守以下约定：

- 修改语言规范、沟通规范或代理协作规则时，优先更新 `AGENTS.md`。
- 修改项目目录结构、初始化模板或根目录文档职责时，优先更新 `PROJECT_INIT.md`。
- 修改项目运行方式、训练入口、实验流程或部署/复现实验说明时，优先更新 `README.md`。
- 修改数据统计、数据清洗规则、数据划分方式或数据文件说明时，优先更新 `DATASETS.md`。
- 修改研究定位、实验设计、模型对照方案或论文指标说明时，优先更新 `EXPERIMENT_PLAN.md`。

## 项目目录结构规范（必须遵守）

新项目默认采用统一的根目录结构。项目根目录通常形如 `Projectx/`，其下包含 `AGENTS.md`、`PROJECT_INIT.md`、`README.md`、`DATASETS.md`、`EXPERIMENT_PLAN.md`、`datasets/`、`outputs/`、`pre_weights/` 和 `src/`：

```text
Projectx/
├── AGENTS.md
├── PROJECT_INIT.md
├── README.md
├── DATASETS.md
├── EXPERIMENT_PLAN.md
├── datasets/
│   └── main_data/
├── outputs/
├── pre_weights/
└── src/
    ├── models/
    ├── baselines/
    ├── configs/
    │   └── train.yaml
    └── scripts/
```

各文档和目录职责如下：

- `AGENTS.md`：开发代理协作规范文档，用于约束语言规范和协作方式。
- `PROJECT_INIT.md`：项目初始化与目录结构规范文档，用于记录项目标准结构和维护约定。
- `README.md`：项目运行逻辑文档，用于记录项目如何运行、训练和评估。
- `DATASETS.md`：数据说明文档，用于记录数据相关操作、统计和文件说明。
- `EXPERIMENT_PLAN.md`：实验方案文档，用于记录研究定位、实验对照和论文实验设计。
- `datasets/`：数据集根目录，用于存放项目使用的数据，不放项目源码。
- `datasets/main_data/`：主数据集目录，默认作为训练、验证、测试数据的主要读取路径。
- `outputs/`：输出目录，用于存放训练结果、测试结果、日志、checkpoint、图表、分析文件和其他实验产物。
- `pre_weights/`：预训练模型权重目录，用于存放外部下载或预先训练好的模型参数文件。
- `src/`：项目代码目录，用于存放训练、测试、模型、数据处理、可视化和工具脚本等源码。
- `src/models/`：模型目录，用于存放本项目正式使用的模型结构、模块组件和模型封装代码。
- `src/baselines/`：基线模型目录，用于存放 baseline 实验代码和基础对照模型实现。
- `src/configs/`：配置文件目录，用于存放训练、验证、测试等配置文件。
- `src/configs/train.yaml`：训练参数配置文件，用于统一管理数据路径、模型参数、优化器、学习率、batch size、epoch、输出路径等训练相关参数。
- `src/scripts/`：脚本目录，用于存放数据统计、数据清洗、训练启动、评估、可视化、结果汇总等脚本文件。

后续开发时应默认遵守以下约定：

- 读取数据时，默认从 `datasets/main_data/` 开始组织路径。
- 保存任何实验产物时，默认写入 `outputs/`，不要写入 `src/`。
- 读取预训练权重时，默认从 `pre_weights/` 中查找。
- 新增代码文件、配置文件和脚本时，默认放在 `src/` 下，除非用户明确指定其他位置。
- 新增正式模型实现时，默认放入 `src/models/`。
- 新增基线模型或 baseline 实验代码时，默认放入 `src/baselines/`。
- 新增训练配置或调整训练参数时，默认维护 `src/configs/train.yaml`。
- 新增一次性处理、批量运行或结果分析脚本时，默认放入 `src/scripts/`。
- 脚本中的路径参数可以保持英文键名，例如 `data_root`、`output_dir`、`pretrained_path`，但默认值应符合上述目录结构。
