"""DocMill 配置系统。

提供统一的配置管理，支持 YAML + ENV + CLI 三层覆盖。
"""

from docmill.config.loader import load_config, validate_config
from docmill.config.schema import (
    DocMillConfig,
    EngineConfig,
    ServerConfig,
    VLLMConfig,
)

__all__ = [
    "DocMillConfig",
    "EngineConfig", 
    "ServerConfig",
    "VLLMConfig",
    "load_config",
    "validate_config",
]
