"""配置加载器 — 从 YAML 文件加载和发现模型配置。"""

from __future__ import annotations

from pathlib import Path

import yaml

from docmill.config.schema import ModelSpec
from docmill.utils.errors import InvalidSpecError, ModelNotFoundError
from docmill.utils.logging import get_logger

logger = get_logger("config.loader")


def load_model_spec(config_path: str | Path) -> ModelSpec:
    """从 YAML 文件加载模型配置。

    Args:
        config_path: 配置文件路径 (config.yaml)。

    Returns:
        经过验证的 ModelSpec 实例。

    Raises:
        ModelNotFoundError: 配置文件不存在。
        InvalidSpecError: 配置内容校验失败。
    """
    path = Path(config_path)
    if not path.exists():
        raise ModelNotFoundError(str(path), "配置文件不存在")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise InvalidSpecError(f"YAML 解析失败: {e}") from e

    if not isinstance(raw, dict):
        raise InvalidSpecError("配置文件顶层必须是字典")

    try:
        spec = ModelSpec(**raw)
    except Exception as e:
        raise InvalidSpecError(str(e)) from e

    logger.info("已加载模型配置: %s (pipeline=%s, execution=%s)", spec.name, spec.pipeline.value, spec.execution.value)
    return spec


def discover_models(models_dir: str | Path) -> dict[str, ModelSpec]:
    """扫描目录发现所有模型配置。

    遍历 models_dir 下的每个子目录，查找 config.yaml。

    Args:
        models_dir: 模型目录根路径。

    Returns:
        {模型名: ModelSpec} 字典。
    """
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        logger.warning("模型目录不存在: %s", models_dir)
        return {}

    specs: dict[str, ModelSpec] = {}

    # 查找直接放在 models_dir 下的 config.yaml
    for config_file in sorted(models_dir.rglob("config.yaml")):
        try:
            spec = load_model_spec(config_file)
            if spec.name in specs:
                logger.warning("模型名称冲突: %s (已跳过 %s)", spec.name, config_file)
                continue
            specs[spec.name] = spec
            logger.debug("发现模型: %s -> %s", spec.name, config_file)
        except (InvalidSpecError, ModelNotFoundError) as e:
            logger.warning("跳过无效配置 %s: %s", config_file, e)
            continue

    logger.info("共发现 %d 个模型配置", len(specs))
    return specs
