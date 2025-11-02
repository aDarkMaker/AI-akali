import asyncio
import json
from typing import Any, Callable, Set

import websockets
from nonebot.log import logger


class WebSocketServer:

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.connected_clients: Set[Any] = set()
        self.handlers: dict[str, Callable] = {}

    def on_action(self, action: str, handler: Callable):
        self.handlers[action] = handler

    async def _handle_message(self, websocket, message: str) -> dict:
        try:
            data = json.loads(message)
            logger.info(f"Received: {data}")

            action = data.get("action")
            if action and action in self.handlers:
                result = await self.handlers[action](data)
                return result or self._create_response(data, success=True)
            elif action:
                return self._create_response(data, success=True)
            else:
                logger.info(f"Event: {data}")
                return self._create_response(data, success=True)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, message: {message}")
            return self._create_error_response("Invalid JSON", 1400)

    def _create_response(self, data: dict, success: bool = True) -> dict:
        return {
            "status": "ok" if success else "failed",
            "retcode": 0 if success else 1,
            "data": None,
            "echo": data.get("echo"),
        }

    def _create_error_response(self, msg: str, retcode: int = 1400) -> dict:
        return {
            "status": "failed",
            "retcode": retcode,
            "data": None,
            "msg": msg,
            "echo": None,
        }

    async def _client_handler(self, websocket, path: str = "/"):
        self.connected_clients.add(websocket)
        remote_address = websocket.remote_address
        logger.info(
            f"WebSocket client connected: {remote_address}, path: {path}"
        )

        try:
            async for message in websocket:
                try:
                    response = await self._handle_message(websocket, message)
                    await websocket.send(json.dumps(response))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    error_response = self._create_error_response(
                        f"Server error: {e}", 1500
                    )
                    await websocket.send(json.dumps(error_response))
                    raise

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed normally")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            raise
        finally:
            self.connected_clients.discard(websocket)
            logger.info("WebSocket client disconnected")

    async def start(self):
        async with websockets.serve(
            self._client_handler, self.host, self.port
        ):
            logger.info(
                f"WebSocket server started on ws://{self.host}:{self.port}"
            )
            await asyncio.Future()

    def broadcast(self, message: dict):
        if self.connected_clients:
            message_str = json.dumps(message)
            disconnected = set()
            for client in self.connected_clients:
                try:
                    asyncio.create_task(client.send(message_str))
                except Exception:
                    disconnected.add(client)
            self.connected_clients -= disconnected
