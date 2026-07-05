from threading import Lock

from base import Config, logger


class SessionStore:
    """In-memory chat session history."""

    def __init__(self, conf=None):
        self.conf = conf or Config()
        self.memory_store = {}
        self._lock = Lock()
        logger.info("Session store initialized (in-memory)")

    def ensure_session(self, session_id, user_id=None):
        with self._lock:
            self.memory_store.setdefault(session_id, [])

    def get_pairs(self, session_id, limit=10):
        with self._lock:
            return list(self.memory_store.get(session_id, [])[-limit:])

    def get_history_text(self, session_id, limit=3):
        pairs = self.get_pairs(session_id, limit=limit)
        return "\n".join(
            [f"用户: {item['question']}\n助手: {item['answer'][:200]}" for item in pairs]
        )

    def save_exchange(self, session_id, question, answer, source="rag", user_id=None):
        with self._lock:
            history = self.memory_store.setdefault(session_id, [])
            history.append({"question": question, "answer": answer, "source": source})
            if len(history) > 10:
                self.memory_store[session_id] = history[-10:]

    def clear_session(self, session_id):
        with self._lock:
            self.memory_store.pop(session_id, None)
