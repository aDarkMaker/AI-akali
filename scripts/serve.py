import asyncio
import sys
from pathlib import Path

import nonebot
from fastapi import Response
from nonebot.adapters.onebot.v11 import Adapter
from nonebot.log import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot.qq_protocol import QQProtocolClient  # noqa: E402
from src.network.ws_server import WebSocketServer  # noqa: E402
from src.utils import load_config  # noqa: E402

nonebot.init(_env_file=".env")

driver = nonebot.get_driver()
driver.register_adapter(Adapter)

# Global instances
ws_server: WebSocketServer = None
qq_client: QQProtocolClient = None


@driver.on_startup
async def register_routes():
    """Register routes and start WebSocket server"""
    global ws_server, qq_client

    app = driver.server_app

    @app.get("/")
    async def root():
        """Root endpoint"""
        return Response(
            content="AI-Akali is running\n",
            media_type="text/plain; charset=utf-8",
        )

    # Load config
    config = load_config("config/bot.yaml", project_root)
    uin = config.get("qq_uin", 0)
    password = config.get("qq_password", "")

    # Create WebSocket server
    ws_server = WebSocketServer(host="127.0.0.1", port=8080)

    # Create QQ Protocol Client
    qq_client = QQProtocolClient(
        uin=uin, password=password, ws_server=ws_server
    )

    # Register action handler (must be after qq_client is created)
    async def handle_send_msg(data: dict) -> dict:
        action = data.get("action")
        params = data.get("params", {})
        return await qq_client.send_message(action, params)

    ws_server.on_action("send_msg", handle_send_msg)

    # Start WebSocket server
    asyncio.create_task(ws_server.start())

    # Start QQ Protocol Client (login and message loop)
    asyncio.create_task(qq_client.start())

    logger.info("All services started")


@driver.on_shutdown
async def shutdown():
    if qq_client:
        await qq_client.stop()
        logger.info("QQ Protocol Client stopped")


if __name__ == "__main__":
    logger.info("Starting NoneBot...")
    nonebot.run(host="127.0.0.1", port=8765)
