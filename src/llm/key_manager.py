"""Small, reusable API-key rotation helper.

Keys are read from environment variables (or an explicitly supplied mapping in
tests). Values are never logged, repr'ed, or included in error messages.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence


class KeyManager:
    """Round-robin key selection with optional cooldown after a failed call."""

    def __init__(
        self,
        env_vars: Sequence[str] = ("GROQ_API_KEY", "GROQ_API_KEY_2"),
        *,
        environ: Mapping[str, str] | None = None,
        cooldown: int = 1,
    ) -> None:
        self.env_vars = tuple(env_vars)
        self._environ = environ if environ is not None else os.environ
        self.cooldown = max(0, int(cooldown))
        self._keys = tuple(
            value.strip() for name in self.env_vars
            for value in [self._environ.get(name, "")]
            if value.strip()
        )
        self._index = 0
        self._failures: dict[int, int] = {}
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self._keys)

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def get(self) -> str:
        """Return the next usable key, raising without exposing key material."""
        with self._lock:
            if not self._keys:
                names = ", ".join(self.env_vars)
                raise RuntimeError(f"No API key configured; set one of: {names}.")
            for _ in range(len(self._keys)):
                index = self._index % len(self._keys)
                self._index += 1
                if self._failures.get(index, 0) == 0:
                    return self._keys[index]
                self._failures[index] -= 1
            raise NoUsableKeyError("All configured API keys are temporarily unavailable.")

    def mark_failed(self, key: str) -> None:
        """Put a matching key on a short cooldown (unknown values are ignored)."""
        with self._lock:
            try:
                index = self._keys.index(key)
            except ValueError:
                return
            self._failures[index] = self.cooldown

    def mark_rate_limited(self, key: str) -> None:
        """Disable a key only for a classified rate-limit/quota failure."""
        self.mark_failed(key)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()
            self._index = 0


class NoUsableKeyError(RuntimeError):
    """All keys are cooling down; callers should wait and retry later."""
