"""DocMill FastAPI Server - 模块化路由"""

from docmill.server.routes import models, infer, history, files

__all__ = ["models", "infer", "history", "files"]