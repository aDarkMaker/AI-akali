import nonebot
from fastapi import Response
from nonebot.adapters.onebot.v11 import Adapter
from nonebot.log import logger

# Init
nonebot.init(
    _env_file=".env",
)

driver = nonebot.get_driver()

driver.register_adapter(Adapter)


@driver.on_startup
async def register_routes():
    app = driver.server_app

    @app.get("/")
    async def root():
        return Response(
            content="AI-Akali is running\n",
            media_type="text/plain; charset=utf-8",
        )


if __name__ == "__main__":
    logger.info("Starting NoneBot...")
    nonebot.run(host="127.0.0.1", port=8765)
