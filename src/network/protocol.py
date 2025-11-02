from typing import Any, Dict


def parse_message(message: Dict[str, Any]) -> Dict[str, Any]:
    return message


def build_response(echo: str = None, data: Any = None) -> Dict[str, Any]:
    return {"status": "ok", "echo": echo, "data": data}


def build_message(event_type: str, **kwargs) -> Dict[str, Any]:
    return {"post_type": event_type, **kwargs}
