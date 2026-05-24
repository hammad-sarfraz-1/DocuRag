#!/usr/bin/env python
"""
Development entry point.

Starts the FastAPI server (with the MCP server thread auto-started inside
the FastAPI lifespan).
"""

import uvicorn
from backend.config import Config

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True,
    )
