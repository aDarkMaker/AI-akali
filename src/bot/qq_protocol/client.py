import asyncio
from typing import Any, Dict, Optional

from src.bot.qq_protocol.converter import ProtocolConverter
from src.bot.qq_protocol.login import QQLogin
from src.bot.qq_protocol.message import QQMessageHandler
from src.network.ws_server import WebSocketServer


class QQProtocolClient:

    def __init__(
        self,
        uin: int = 0,
        password: str = "",
        ws_server: Optional[WebSocketServer] = None,
    ):
        self.uin = uin
        self.login_handler = QQLogin(uin, password)
        self.message_handler = QQMessageHandler(uin)
        self.ws_server = ws_server
        self.running = False
        self._message_loop_task: Optional[asyncio.Task] = None

    async def start(self):
        if not self.login_handler.is_logged_in():
            success = await self.login_handler.login()
            if not success:
                raise RuntimeError("QQ login failed")

        self.running = True
        self.message_handler.register_handler(self._on_message_received)
        self._message_loop_task = asyncio.create_task(self._message_loop())
        print(f"QQ Protocol Client started (UIN: {self.uin})")

    async def stop(self):
        self.running = False
        if self._message_loop_task:
            self._message_loop_task.cancel()
            try:
                await self._message_loop_task
            except asyncio.CancelledError:
                pass

    async def _message_loop(self):
        while self.running:
            try:
                qq_message = await self._receive_message()
                if qq_message:
                    await self.message_handler.handle_received_message(
                        qq_message
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in message loop: {e}")
                await asyncio.sleep(1)

    async def _receive_message(self) -> Optional[Dict[str, Any]]:
        await asyncio.sleep(1)
        return None

    async def _on_message_received(self, qq_event: Dict[str, Any]):
        onebot_event = ProtocolConverter.qq_to_onebot_event(qq_event)
        if self.ws_server:
            self.ws_server.broadcast(onebot_event)

    async def send_message(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        qq_action = ProtocolConverter.onebot_to_qq_action(action, params)

        if qq_action["type"] == "send_message":
            message_type = qq_action.get("message_type")
            target_id = qq_action.get("target_id")
            message = qq_action.get("message", "")

            if message_type == "group":
                return await self.message_handler.send_group_message(
                    target_id, message
                )
            elif message_type == "private":
                return await self.message_handler.send_private_message(
                    target_id, message
                )

        return {"retcode": 1, "msg": "Unknown action"}
