import asyncio
from datetime import datetime
from typing import Any, Callable, Dict


class QQMessageHandler:

    def __init__(self, self_id: int):
        self.self_id = self_id
        self.message_handlers: list[Callable] = []

    def register_handler(self, handler: Callable):
        self.message_handlers.append(handler)

    async def send_group_message(
        self, group_id: int, message: str
    ) -> Dict[str, Any]:
        result = await self._send_message("group", group_id, message)
        return {
            "message_id": result.get("message_id"),
            "retcode": 0 if result else 1,
        }

    async def send_private_message(
        self, user_id: int, message: str
    ) -> Dict[str, Any]:
        result = await self._send_message("private", user_id, message)
        return {
            "message_id": result.get("message_id"),
            "retcode": 0 if result else 1,
        }

    async def _send_message(
        self, message_type: str, target_id: int, message: str
    ) -> Dict[str, Any]:
        await asyncio.sleep(1)
        return {
            "message_id": int(datetime.now().timestamp() * 1000),
            "time": int(datetime.now().timestamp()),
        }

    async def handle_received_message(self, qq_message: Dict[str, Any]):
        event = {
            "type": "message",
            "time": qq_message.get("time", int(datetime.now().timestamp())),
            "self_id": self.self_id,
            "message_type": qq_message.get("message_type"),
            "user_id": qq_message.get("user_id"),
            "group_id": qq_message.get("group_id"),
            "message": qq_message.get("message", ""),
            "nickname": qq_message.get("nickname", ""),
        }

        for handler in self.message_handlers:
            try:
                await handler(event)
            except Exception as e:
                print(f"Error in message handler: {e}")
