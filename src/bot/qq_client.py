import asyncio
from typing import Any, Callable, Dict, Optional

import websockets

from src.network import WSConnection, parse_message


class QQClient:

    def __init__(
        self, host: str = "127.0.0.1", port: int = 8765, access_token: str = ""
    ):
        self.host = host
        self.port = port
        self.access_token = access_token
        self.connection: Optional[WSConnection] = None
        self.running = False
        self._echo_counter = 0
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._pending_handlers: Dict[str, Callable] = {}

    def _get_ws_url(self, path: str = "/event") -> str:
        protocol = "ws"
        token_part = (
            f"?access_token={self.access_token}" if self.access_token else ""
        )
        return f"{protocol}://{self.host}:{self.port}{path}{token_part}"

    def _get_ws_headers(self) -> Dict[str, str]:
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    async def connect(self, path: str = "/event"):
        url = self._get_ws_url(path)
        headers = self._get_ws_headers()
        ws = await websockets.connect(url, additional_headers=headers)
        self.connection = WSConnection(ws)

        for event_type, handler in self._pending_handlers.items():
            self.connection.on(event_type, handler)
        self._pending_handlers.clear()

        self.running = True

    async def disconnect(self):
        self.running = False
        if self.connection and self.connection.ws:
            await self.connection.ws.close()
            self.connection = None

    async def send(
        self,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        echo: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.connection:
            raise RuntimeError("Not connected")

        if echo is None:
            self._echo_counter += 1
            echo = str(self._echo_counter)

        request = {
            "action": action,
            "params": params or {},
            "echo": echo,
        }

        future = asyncio.Future()
        self._pending_responses[echo] = future

        await self.connection.send(request)

        response = await future
        return response

    async def _handle_api_response(self, message: Dict[str, Any]):
        if "echo" in message and message["echo"] in self._pending_responses:
            future = self._pending_responses.pop(message["echo"])
            if not future.done():
                future.set_result(message)
            return True
        return False

    def on(self, event_type: str, handler):
        if self.connection:
            self.connection.on(event_type, handler)
        else:
            self._pending_handlers[event_type] = handler

    async def _recv_loop(self):
        try:
            while self.running and self.connection:
                try:
                    message = await self.connection.recv()
                    parsed = parse_message(message)

                    is_api_response = await self._handle_api_response(parsed)

                    if not is_api_response:
                        await self.connection.handle(parsed)
                except Exception as e:
                    print(f"Error in recv loop: {e}")
                    break

        finally:
            for future in self._pending_responses.values():
                if not future.done():
                    future.set_exception(RuntimeError("Connection closed"))
            self._pending_responses.clear()

    async def run(self, path: str = "/event"):
        await self.connect(path=path)
        await self._recv_loop()

    async def send_group_message(self, group_id: int, message: str):
        return await self.send(
            "send_msg",
            {
                "message_type": "group",
                "group_id": group_id,
                "message": message,
            },
        )

    async def send_private_message(self, user_id: int, message: str):
        return await self.send(
            "send_msg",
            {
                "message_type": "private",
                "user_id": user_id,
                "message": message,
            },
        )
