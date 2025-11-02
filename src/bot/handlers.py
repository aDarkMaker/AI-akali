from typing import Callable, Dict


class MessageHandler:

    def __init__(self):
        self._group_handlers: list[Callable] = []
        self._private_handlers: list[Callable] = []

    def register_group_handler(self, handler: Callable):
        self._group_handlers.append(handler)

    def register_private_handler(self, handler: Callable):
        self._private_handlers.append(handler)

    async def handle_message(self, message: Dict) -> None:
        post_type = message.get("post_type")
        if post_type != "message":
            return

        message_type = message.get("message_type")
        raw_message = message.get("raw_message", "")
        sender = message.get("sender", {})

        if message_type == "group":
            await self.handle_group_message(message, raw_message, sender)
        elif message_type == "private":
            await self.handle_private_message(message, raw_message)

    async def handle_group_message(
        self, message: Dict, raw_message: str, sender: Dict
    ) -> None:
        group_id = message.get("group_id")
        user_id = sender.get("user_id")
        nickname = sender.get("nickname", "Unknown")

        print(
            f"Group Message: {group_id} | {nickname}({user_id}): {raw_message}"
        )

        for handler in self._group_handlers:
            try:
                await handler(message, raw_message, sender)
            except Exception as e:
                print(f"Error in group handler: {e}")

    async def handle_private_message(
        self, message: Dict, raw_message: str
    ) -> None:
        user_id = message.get("user_id")
        print(f"Private Message: {user_id}: {raw_message}")

        for handler in self._private_handlers:
            try:
                await handler(message, raw_message)
            except Exception as e:
                print(f"Error in private handler: {e}")


class NoticeHandler:

    async def handle_notice(self, message: Dict) -> None:
        notice_type = message.get("notice_type")
        print(f"Notice: {notice_type}: {message}")


class RequestHandler:

    async def handle_request(self, message: Dict) -> None:
        request_type = message.get("request_type")
        print(f"Request: {request_type}: {message}")
