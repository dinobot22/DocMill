"""DocMill 应用启动器 — 配置驱动的一键启动。

用户只需执行一条命令即可拉起全部服务。
"""

from __future__ import annotations

import os as _os
import sys as _sys

# 将项目根目录加入 sys.path（无需 pip install 即可直接执行）
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import signal
import sys
from pathlib import Path

from docmill.config.loader import load_config, validate_config
from docmill.config.schema import DocMillConfig
from docmill.core import DocMill
from docmill.utils.logging import get_logger, setup_logging

logger = get_logger("launcher")


class AppLauncher:
    """DocMill 应用启动器。
    
    负责：
    - 解析配置
    - 注册引擎
    - 启动 vLLM sidecar
    - 启动 API Server
    - 优雅关闭
    """
    
    def __init__(self, config: DocMillConfig):
        self.config = config
        self.docmill: DocMill | None = None
    
    def start(self):
        """一键启动：注册引擎 → 拉起 vLLM → 启动 API Server"""
        setup_logging()
        
        # 创建数据目录
        data_dir = Path(self.config.server.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "uploads").mkdir(exist_ok=True)
        (data_dir / "results").mkdir(exist_ok=True)
        
        # 初始化 DocMill
        sidecar_log_dir = data_dir / "sidecar_logs"
        sidecar_log_dir.mkdir(exist_ok=True)
        
        self.docmill = DocMill(
            sidecar_log_dir=str(sidecar_log_dir),
        )
        
        # 1. 遍历配置，注册所有 Engine
        for name, engine_cfg in self.config.engines.items():
            vllm_config = None
            if engine_cfg.vllm:
                vllm_config = engine_cfg.vllm.model_dump(exclude_none=True)
            
            self.docmill.register_model(
                name=name,
                engine_name=engine_cfg.engine,
                vllm_config=vllm_config,
                **engine_cfg.engine_kwargs,
            )
            logger.info("注册模型: %s (engine=%s)", name, engine_cfg.engine)
        
        # 2. 自动启动 auto_start=True 的引擎（会自动拉起 vLLM sidecar）
        for name, engine_cfg in self.config.engines.items():
            if engine_cfg.auto_start:
                logger.info("启动模型: %s", name)
                try:
                    self.docmill.ensure_model_ready(name)
                except Exception as e:
                    logger.error("模型 '%s' 启动失败: %s", name, e)
        
        # 3. 启动 FastAPI Server
        self._start_server()
    
    def _start_server(self):
        """启动 FastAPI Server"""
        import uvicorn
        from docmill.server.main import create_app
        
        app = create_app(data_dir=self.config.server.data_dir)
        
        logger.info("启动 API Server: port=%d", self.config.server.api_port)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=self.config.server.api_port,
            log_level="info",
        )
    
    def shutdown(self, signum=None, frame=None):
        """优雅关闭"""
        logger.info("收到关闭信号，正在关闭...")
        if self.docmill:
            self.docmill.shutdown()
        sys.exit(0)


def main():
    """CLI 入口点"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DocMill - OCR & VLM Inference Runtime")
    parser.add_argument("command", choices=["serve"], help="Command to run")
    parser.add_argument("--config", default="docmill.yaml", help="Config file path")
    parser.add_argument("--validate-only", action="store_true", help="Only validate config and exit")
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    
    # 验证配置
    errors = validate_config(config)
    if errors:
        logger.error("配置验证失败:")
        for error in errors:
            logger.error("  - %s", error)
        sys.exit(1)
    
    if args.validate_only:
        logger.info("配置验证通过")
        sys.exit(0)
    
    # 启动应用
    launcher = AppLauncher(config)
    
    # 注册信号处理
    signal.signal(signal.SIGINT, launcher.shutdown)
    signal.signal(signal.SIGTERM, launcher.shutdown)
    
    launcher.start()


if __name__ == "__main__":
    main()
