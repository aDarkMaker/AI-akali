"""NoneBot service startup script"""

import asyncio
import sys
from pathlib import Path

import nonebot
from fastapi import Response
from nonebot.adapters.onebot.v11 import Adapter
from nonebot.log import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.network.ws_server import WebSocketServer  # noqa: E402

nonebot.init(_env_file=".env")

driver = nonebot.get_driver()
driver.register_adapter(Adapter)


@driver.on_startup
async def register_routes():
    """Register routes and start WebSocket server"""
    app = driver.server_app

    @app.get("/")
    async def root():
        """Root endpoint"""
        return Response(
            content="AI-Akali is running\n",
            media_type="text/plain; charset=utf-8",
        )

    ws_server = WebSocketServer(host="127.0.0.1", port=8080)
    asyncio.create_task(ws_server.start())


if __name__ == "__main__":
    logger.info("Starting NoneBot...")
    nonebot.run(host="127.0.0.1", port=8765)
