import httpx


class HttpClient:

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def post(self, endpoint: str, data: dict):
        return await self.client.post(f"{self.base_url}/{endpoint}", json=data)

    async def close(self):
        await self.client.aclose()
