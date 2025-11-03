import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot import QQClient  # noqa: E402
from src.bot.handlers import NoticeHandler  # noqa: E402
from src.bot.handlers import MessageHandler, RequestHandler  # noqa: E402
from src.utils import load_config  # noqa: E402


class QQBot:
    """QQ Bot main class"""

    def __init__(self, config_path: str = "config/bot.yaml"):
        base_dir = Path(__file__).parent.parent
        self.config = load_config(config_path, base_dir)
        self.client: QQClient = None

        self.message_handler = MessageHandler()
        self.notice_handler = NoticeHandler()
        self.request_handler = RequestHandler()

    def _setup_handlers(self):
        """Setup message handlers"""

        async def hello_handler(message: dict, raw_message: str, sender: dict):
            if "hello" in raw_message.lower() or "hi" in raw_message.lower():
                group_id = message.get("group_id")
                nickname = sender.get("nickname", "Unknown")
                await self.client.send_group_message(
                    group_id, f"Hello, {nickname}!"
                )

        self.message_handler.register_group_handler(hello_handler)

    async def start(self):
        """Start Bot"""
        qq_host = self.config.get("qq_host", "127.0.0.1")
        qq_port = self.config.get("qq_ws_port", 8080)
        qq_ws_path = self.config.get("qq_ws_path", "/event")
        access_token = self.config.get("access_token", "")

        print(f"Connecting to QQ service ({qq_host}:{qq_port}{qq_ws_path})...")

        self.client = QQClient(
            host=qq_host,
            port=qq_port,
            access_token=access_token,
        )

        self._setup_handlers()

        self.client.on("message", self.message_handler.handle_message)
        self.client.on("notice", self.notice_handler.handle_notice)
        self.client.on("request", self.request_handler.handle_request)

        try:
            await self.client.run(path=qq_ws_path)
        except KeyboardInterrupt:
            print("\nClosing bot...")
        finally:
            if self.client:
                await self.client.disconnect()


def main():
    """Main function"""
    bot = QQBot()

    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\nClosing bot...")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
