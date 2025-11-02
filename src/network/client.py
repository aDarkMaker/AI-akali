from typing import Dict, Optional

import httpx


class HttpClient:

    def __init__(
        self, base_url: str, headers: Optional[Dict[str, str]] = None
    ):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0, headers=headers or {})

    async def post(
        self,
        endpoint: str,
        data: dict,
        headers: Optional[Dict[str, str]] = None,
    ):
        request_headers = {**self.client.headers}
        if headers:
            request_headers.update(headers)
        return await self.client.post(
            f"{self.base_url}/{endpoint}", json=data, headers=request_headers
        )

    async def close(self):
        await self.client.aclose()
