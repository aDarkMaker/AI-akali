from nonebot import get_driver


class Server:

    def __init__(self):
        self.driver = get_driver()

    def run(self, host: str = "127.0.0.1", port: int = 8765):
        self.driver.run_app(host=host, port=port)
