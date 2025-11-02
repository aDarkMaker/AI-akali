import json
from typing import Any, Callable, Dict

from websockets.client import WebSocketClientProtocol


class WSConnection:

    def __init__(self, ws: WebSocketClientProtocol):
        self.ws = ws
        self.handlers: Dict[str, Callable] = {}

    async def send(self, data: Dict[str, Any]):
        await self.ws.send(json.dumps(data))

    async def recv(self) -> Dict[str, Any]:
        msg = await self.ws.recv()
        return json.loads(msg)

    def on(self, event_type: str, handler: Callable):
        self.handlers[event_type] = handler

    async def handle(self, message: Dict[str, Any]):
        event_type = message.get("post_type")
        if event_type in self.handlers:
            await self.handlers[event_type](message)
