from typing import Any, Dict, Optional

from src.network import HttpClient


class QQHttpClient:

    def __init__(
        self, host: str = "127.0.0.1", port: int = 5700, access_token: str = ""
    ):
        base_url = f"http://{host}:{port}"
        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        self.client = HttpClient(base_url, headers=headers)
        self.access_token = access_token

    def _get_headers(self) -> Dict[str, str]:
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    async def request(
        self, action: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        endpoint = action
        data = params or {}
        response = await self.client.post(endpoint, data)
        return response.json()

    async def send_group_message(
        self, group_id: int, message: str
    ) -> Dict[str, Any]:
        return await self.request(
            "send_msg",
            {
                "message_type": "group",
                "group_id": group_id,
                "message": message,
            },
        )

    async def send_private_message(
        self, user_id: int, message: str
    ) -> Dict[str, Any]:
        return await self.request(
            "send_msg",
            {
                "message_type": "private",
                "user_id": user_id,
                "message": message,
            },
        )

    async def close(self):
        await self.client.close()
