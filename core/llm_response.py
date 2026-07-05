import json
import re


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def parse_llm_json_response(text):
    """解析 LLM 返回的 JSON，提取 answer 与 confidence。"""
    if not text:
        return {"answer": "", "confidence": 0.0}

    payload = text.strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(payload)
        if not match:
            return {"answer": payload, "confidence": 0.0}
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return {"answer": payload, "confidence": 0.0}

    if not isinstance(data, dict):
        return {"answer": str(data), "confidence": 0.0}

    answer = str(data.get("answer") or data.get("content") or "").strip()
    confidence = data.get("confidence", data.get("score", 0.0))
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "answer": answer or payload,
        "confidence": max(0.0, min(1.0, confidence)),
    }
