"""DocMill 配置加载器 — 支持 YAML + ENV + CLI 三层覆盖。

加载优先级：
YAML 文件 (docmill.yaml) → ENV 环境变量覆盖 → CLI 参数覆盖 → 最终 DocMillConfig
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from docmill.config.schema import DocMillConfig, EngineConfig, ServerConfig, VLLMConfig
from docmill.utils.logging import get_logger

logger = get_logger("config.loader")


def load_config(config_path: str | Path | None = None) -> DocMillConfig:
    """加载配置。
    
    优先级：YAML 文件 → ENV 环境变量 → 默认值
    
    Args:
        config_path: 配置文件路径。如果为 None，尝试默认路径。
    
    Returns:
        加载完成的 DocMillConfig 对象。
    """
    config_dict: dict[str, Any] = {}
    
    # 1. 加载 YAML 文件
    if config_path is None:
        # 尝试默认路径
        default_paths = [
            Path.cwd() / "docmill.yaml",
            Path.cwd() / "config" / "docmill.yaml",
            Path.home() / ".docmill" / "docmill.yaml",
        ]
        for path in default_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            logger.info("加载配置文件: %s", config_path)
            with open(config_path) as f:
                config_dict = yaml.safe_load(f) or {}
        else:
            logger.warning("配置文件不存在: %s", config_path)
    
    # 2. ENV 环境变量覆盖
    config_dict = _apply_env_overrides(config_dict)
    
    # 3. 构建配置对象
    config = DocMillConfig.model_validate(config_dict)
    
    logger.info("配置加载完成: %d 个 Engine", len(config.engines))
    return config


def _apply_env_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
    """应用环境变量覆盖。
    
    环境变量命名规则：
    - DOCMILL_SERVER_* -> server.*
    - DOCMILL_ENGINE_<name>_VLLM_* -> engines.<name>.vllm.*
    
    例如：
    - DOCMILL_SERVER_API_PORT=8080 -> server.api_port = 8080
    - DOCMILL_ENGINE_PADDLE_VLLM_GPU_ID=1 -> engines.paddle.vllm.gpu_id = 1
    """
    prefix = "DOCMILL_"
    
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        
        # 解析键名
        parts = key[len(prefix):].split("_")
        
        if parts[0] == "SERVER":
            # DOCMILL_SERVER_API_PORT -> server.api_port
            if len(parts) >= 2:
                field_path = parts[1].lower()
                config_dict.setdefault("server", {})
                config_dict["server"][field_path] = _parse_value(value)
                logger.debug("ENV 覆盖: server.%s = %s", field_path, value)
        
        elif parts[0] == "ENGINE" and len(parts) >= 4:
            # DOCMILL_ENGINE_PADDLE_OCR_VL_VLLM_GPU_ID -> engines.paddle_ocr_vl.vllm.gpu_id
            engine_name = parts[1].lower()
            if len(parts) >= 5 and parts[2].upper() == "VLLM":
                # vLLM 配置
                field_path = "_".join(parts[3:]).lower()
                config_dict.setdefault("engines", {}).setdefault(engine_name, {}).setdefault("vllm", {})
                config_dict["engines"][engine_name]["vllm"][field_path] = _parse_value(value)
                logger.debug("ENV 覆盖: engines.%s.vllm.%s = %s", engine_name, field_path, value)
            else:
                # Engine 配置
                field_path = "_".join(parts[2:]).lower()
                config_dict.setdefault("engines", {}).setdefault(engine_name, {})
                config_dict["engines"][engine_name][field_path] = _parse_value(value)
                logger.debug("ENV 覆盖: engines.%s.%s = %s", engine_name, field_path, value)
    
    return config_dict


def _parse_value(value: str) -> Any:
    """解析环境变量值为适当的 Python 类型。"""
    # 布尔值
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    
    # 整数
    try:
        return int(value)
    except ValueError:
        pass
    
    # 浮点数
    try:
        return float(value)
    except ValueError:
        pass
    
    # 字符串列表（逗号分隔）
    if "," in value:
        return [item.strip() for item in value.split(",")]
    
    return value


def validate_config(config: DocMillConfig) -> list[str]:
    """验证配置的有效性。
    
    Args:
        config: DocMillConfig 对象。
    
    Returns:
        验证错误列表。如果为空则验证通过。
    """
    errors: list[str] = []
    
    for name, engine_cfg in config.engines.items():
        # 检查 Engine 是否存在
        try:
            from docmill.engines.registry import EngineRegistry
            engine_class = EngineRegistry.get(engine_cfg.engine)
            if engine_class is None:
                errors.append(f"Engine '{engine_cfg.engine}' 不存在")
                continue
            
            # 检查 vLLM 配置
            if engine_class.requires_vllm_sidecar():
                if engine_cfg.vllm is None:
                    errors.append(f"Engine '{engine_cfg.engine}' 需要 vLLM 配置")
                elif engine_cfg.vllm.mode == "managed" and not engine_cfg.vllm.model_path:
                    errors.append(f"Engine '{name}': managed 模式需要 model_path")
                elif engine_cfg.vllm.mode == "external" and not engine_cfg.vllm.endpoints:
                    errors.append(f"Engine '{name}': external 模式需要 endpoints")
        except Exception as e:
            errors.append(f"验证 Engine '{name}' 失败: {e}")
    
    return errors
