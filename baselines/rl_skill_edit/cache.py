from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


def _path_lock(path: Path) -> threading.RLock:
    with _LOCKS_GUARD:
        lock = _PATH_LOCKS.get(path)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[path] = lock
        return lock


class JsonFileCache:
    """A namespaced JSON cache with atomic, thread-safe writes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        self._lock = _path_lock(self.path)
        with self._lock:
            self._read()

    def get(self, namespace: str, key: str) -> Any | None:
        namespace = _nonempty_string("namespace", namespace)
        key = _nonempty_string("key", key)
        with self._lock:
            data = self._read()
            return data.get(namespace, {}).get(key)

    def set(self, namespace: str, key: str, value: Any) -> None:
        namespace = _nonempty_string("namespace", namespace)
        key = _nonempty_string("key", key)
        with self._lock:
            current = self._read()
            candidate = {
                existing_namespace: dict(entries)
                for existing_namespace, entries in current.items()
            }
            candidate.setdefault(namespace, {})[key] = value
            encoded = json.dumps(
                candidate,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            try:
                with os.fdopen(file_descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, self.path)
            except BaseException:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("cache root must be a JSON object")
        for namespace, entries in payload.items():
            if not isinstance(namespace, str) or not isinstance(entries, dict):
                raise ValueError("cache namespaces must map strings to JSON objects")
            if not all(isinstance(key, str) for key in entries):
                raise ValueError("cache keys must be strings")
        return payload


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


__all__ = ["JsonFileCache"]
