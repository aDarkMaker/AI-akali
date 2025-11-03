from typing import Any, Dict


class ProtocolConverter:

    @staticmethod
    def qq_to_onebot_event(qq_event: Dict[str, Any]) -> Dict[str, Any]:
        post_type = qq_event.get("type")
        if post_type == "message":
            return ProtocolConverter.__convert_message(qq_event)
        elif post_type == "notice":
            return ProtocolConverter.__convert_notice(qq_event)
        elif post_type == "request":
            return ProtocolConverter.__convert_request(qq_event)
        return qq_event

    @staticmethod
    def _convert_message(qq_event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "post_type": "message",
            "message_type": qq_event.get("message_type", "group"),
            "time": qq_event.get("time"),
            "self_id": qq_event.get("self_id"),
            "user_id": qq_event.get("user_id"),
            "group_id": qq_event.get("group_id"),
            "raw_message": qq_event.get("raw_message", ""),
            "message": qq_event.get("message", ""),
            "sender": {
                "user_id": qq_event.get("user_id"),
                "nickname": qq_event.get("nickname", ""),
            },
        }

    @staticmethod
    def _convert_notice(qq_event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "post_type": "notice",
            "notice_type": qq_event.get("notice_type", "unknown"),
            "time": qq_event.get("time"),
            "self_id": qq_event.get("self_id"),
            **qq_event,
        }

    @staticmethod
    def _convert_request(qq_event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "post_type": "request",
            "request_type": qq_event.get("request_type", "unknown"),
            "time": qq_event.get("time"),
            "self_id": qq_event.get("self_id"),
            **qq_event,
        }

    @staticmethod
    def onebot_to_qq_action(
        action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if action == "send_msg":
            return {
                "type": "send_message",
                "message_type": params.get("message_type"),
                "target_id": params.get("group_id") or params.get("user_id"),
                "message": params.get("message", ""),
            }
        return {"type": action, **params}
