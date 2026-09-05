"""从 IDEA-PARA.md 读取 110 个 idea 的专属顺序调参空间。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENT = PROJECT_ROOT / "IDEA-PARA.md"


@dataclass(frozen=True)
class IdeaParameter:
    """一个 idea 专属参数及其五个候选值。"""

    name: str
    values: tuple[float, ...]
    default: float
    description: str


@dataclass(frozen=True)
class IdeaParameterSpace:
    """单个 idea 的 A、B、C 三个有序参数。"""

    idea_id: str
    title: str
    parameters: tuple[IdeaParameter, ...]

    @property
    def defaults(self) -> dict[str, float]:
        return {parameter.name: parameter.default for parameter in self.parameters}


_HEADING = re.compile(r"^### (D\d+-I\d{2})　(.+)$", re.MULTILINE)
_PARAMETER = re.compile(
    r"^([123])\. `([^`]+)`(?:（[^\n]*?）)?：`\[([^\]]+)\]`。(.+)$",
    re.MULTILINE,
)


def _parse_value(token: str) -> tuple[float, bool]:
    token = token.strip()
    is_default = token.startswith("**") and token.endswith("**")
    if is_default:
        token = token[2:-2]
    return float(token), is_default


def load_parameter_spaces(path: Path = DEFAULT_DOCUMENT) -> dict[str, IdeaParameterSpace]:
    """解析文档并严格验证 110×3×5 的参数空间。"""

    text = path.read_text(encoding="utf-8")
    headings = list(_HEADING.finditer(text))
    spaces: dict[str, IdeaParameterSpace] = {}
    field_names: set[str] = set()
    for index, heading in enumerate(headings):
        idea_id, title = heading.group(1), heading.group(2).strip()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = text[heading.end():end]
        matches = _PARAMETER.findall(block)
        if [match[0] for match in matches] != ["1", "2", "3"]:
            raise ValueError(f"{idea_id} 必须按 A、B、C 顺序定义三个参数")
        parameters = []
        for _, name, value_text, description in matches:
            parsed = [_parse_value(token) for token in value_text.split(",")]
            values = tuple(value for value, _ in parsed)
            defaults = [value for value, is_default in parsed if is_default]
            if len(values) != 5 or len(defaults) != 1:
                raise ValueError(f"{idea_id}/{name} 必须有五个值和一个默认值")
            if name in field_names:
                raise ValueError(f"参数字段跨 idea 重复：{name}")
            field_names.add(name)
            parameters.append(IdeaParameter(name, values, defaults[0], description.strip()))
        spaces[idea_id] = IdeaParameterSpace(idea_id, title, tuple(parameters))

    expected = [f"D{family}-I{variant:02d}" for family in range(1, 12) for variant in range(1, 11)]
    if list(spaces) != expected:
        raise ValueError(f"参数文档 idea 顺序或数量异常：实际 {len(spaces)}，预期 110")
    if len(field_names) != 330:
        raise ValueError(f"专属字段数量异常：实际 {len(field_names)}，预期 330")
    return spaces


PARAMETER_SPACES = load_parameter_spaces()


def parameter_space(idea_id: str) -> IdeaParameterSpace:
    try:
        return PARAMETER_SPACES[idea_id.upper()]
    except KeyError as error:
        raise ValueError(f"没有 {idea_id} 的专属参数空间") from error


def value_token(value: float) -> str:
    """生成可用于结果文件名的稳定数值标记。"""

    return format(float(value), ".10g").replace("-", "m").replace(".", "p").replace("+", "")
