#!/usr/bin/env python
"""DocMill 项目入口脚本 — 无需安装即可直接运行。

用法:
    python run.py serve                          # 使用默认配置启动
    python run.py serve --config docmill.yaml   # 指定配置文件
    python run.py serve --validate-only         # 仅验证配置
    python run.py dev                            # 开发模式（热重载）

    # 也可直接启动 uvicorn（开发调试）:
    python run.py uvicorn --port 8080 --reload
"""

from __future__ import annotations

import sys
import os

# ── 将项目根目录加入 sys.path，使 `docmill` 包可被直接导入 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── 加载完 sys.path 后，再导入项目内的模块 ──
from docmill.launcher.app_launcher import main as _launcher_main  # noqa: E402


def _dev_mode():
    """开发模式：uvicorn 热重载。"""
    import uvicorn
    uvicorn.run(
        "docmill.server.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=[PROJECT_ROOT],
    )


def _uvicorn_mode(args):
    """透传 uvicorn 参数直接启动。"""
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    opts = parser.parse_args(args)
    uvicorn.run(
        "docmill.server.main:app",
        host=opts.host,
        port=opts.port,
        reload=opts.reload,
        reload_dirs=[PROJECT_ROOT],
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == "dev":
        # python run.py dev
        sys.argv = [sys.argv[0]]
        _dev_mode()
    elif command == "uvicorn":
        # python run.py uvicorn [--port 9000] [--reload]
        _uvicorn_mode(sys.argv[2:])
    else:
        # python run.py serve [...] — 交给 app_launcher.main() 处理
        _launcher_main()
