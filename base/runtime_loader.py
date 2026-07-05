import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - optional for minimal smoke tests
    yaml = None


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


class RuntimeLoader:
    """Load runtime policy from YAML with env overrides."""

    def __init__(self, project_root, runtime_file=None):
        self.project_root = Path(project_root)
        if runtime_file is not None:
            self.runtime_file = Path(runtime_file)
        else:
            env_path = os.environ.get("RUNTIME_CONFIG")
            if env_path:
                self.runtime_file = Path(env_path)
            else:
                self.runtime_file = self.project_root / "config" / "runtime.yaml"

        self.data = self._load_yaml(self.runtime_file) if self.runtime_file.exists() else {}

    @staticmethod
    def _load_yaml(path):
        if yaml is None:
            return {}
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _section(self, name):
        section = self.data.get(name, {})
        return section if isinstance(section, dict) else {}

    def get(self, section, key, fallback=None):
        env_name = f"RUNTIME_{section.upper()}_{key.upper()}"
        if env_name in os.environ:
            return os.environ[env_name]
        section_data = self._section(section)
        if key in section_data:
            return section_data[key]
        return fallback

    def get_bool(self, section, key, fallback):
        value = self.get(section, key, fallback)
        try:
            return _coerce_bool(value)
        except ValueError:
            return fallback

    def get_int(self, section, key, fallback):
        value = self.get(section, key, fallback)
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def get_float(self, section, key, fallback):
        value = self.get(section, key, fallback)
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def get_str(self, section, key, fallback):
        value = self.get(section, key, fallback)
        if value is None:
            return fallback
        return str(value)
