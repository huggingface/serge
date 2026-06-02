"""Opt-in context compression for LLM calls via the ``headroom-ai`` package.

Runs only when ``HEADROOM_COMPRESS`` is set and ``headroom`` is importable;
otherwise messages pass through unchanged. Any failure falls back to the
original messages so compression can never block a review.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float_env(name: str, default: float | None = None) -> float | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class CompressionConfig:
    # Defaults are tuned for a review agent: leave user/system messages alone
    # and protect recent turns, so only tool outputs and older assistant turns
    # get compressed. Fields mirror headroom.CompressConfig.
    enabled: bool = False
    compress_user_messages: bool = False
    compress_system_messages: bool = True
    protect_recent: int = 4
    target_ratio: Optional[float] = None
    min_tokens_to_compress: int = 250
    # None = headroom default Kompress model; "disabled" = no ML compression.
    kompress_model: Optional[str] = None
    # Model context window, forwarded as compress(model_limit=...).
    model_limit: int = 200_000

    @classmethod
    def from_env(cls) -> "CompressionConfig":
        return cls(
            enabled=_bool_env("HEADROOM_COMPRESS", False),
            compress_user_messages=_bool_env("HEADROOM_COMPRESS_USER_MESSAGES", False),
            compress_system_messages=_bool_env(
                "HEADROOM_COMPRESS_SYSTEM_MESSAGES", True
            ),
            protect_recent=_int_env("HEADROOM_PROTECT_RECENT", 4),
            target_ratio=_float_env("HEADROOM_TARGET_RATIO", None),
            min_tokens_to_compress=_int_env("HEADROOM_MIN_TOKENS", 250),
            kompress_model=(os.environ.get("HEADROOM_KOMPRESS_MODEL") or "").strip()
            or None,
            model_limit=_int_env("HEADROOM_MODEL_LIMIT", 200_000),
        )


class MessageCompressor:
    """Lazily-loaded wrapper around ``headroom.compress``."""

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self._headroom: Optional[tuple[Callable[..., Any], Any]] = None
        self._unavailable = False
        self._warned_unavailable = False

    @classmethod
    def from_env(cls) -> "MessageCompressor":
        return cls(CompressionConfig.from_env())

    @property
    def active(self) -> bool:
        return self.config.enabled

    def _load(self) -> Optional[tuple[Callable[..., Any], Any]]:
        # Import once; cache success or failure so we don't retry every call.
        if self._unavailable:
            return None
        if self._headroom is not None:
            return self._headroom
        try:
            from headroom import CompressConfig, compress  # type: ignore
        except Exception as exc:  # noqa: BLE001 — ImportError or broken install
            self._unavailable = True
            if not self._warned_unavailable:
                log.warning(
                    "HEADROOM_COMPRESS is set but 'headroom-ai' is not importable "
                    "(%s); sending messages uncompressed. Install with: "
                    "pip install 'reviewbot[headroom]'",
                    exc,
                )
                self._warned_unavailable = True
            return None
        self._headroom = (compress, CompressConfig)
        return self._headroom

    def compress(
        self, messages: list[dict[str, Any]], *, model: Optional[str] = None
    ) -> list[dict[str, Any]]:
        if not self.config.enabled or not messages:
            return messages
        loaded = self._load()
        if loaded is None:
            return messages
        compress_fn, compress_config_cls = loaded
        cfg = compress_config_cls(
            compress_user_messages=self.config.compress_user_messages,
            compress_system_messages=self.config.compress_system_messages,
            protect_recent=self.config.protect_recent,
            target_ratio=self.config.target_ratio,
            min_tokens_to_compress=self.config.min_tokens_to_compress,
            kompress_model=self.config.kompress_model,
        )
        try:
            # `model` is only for token counting / context limit — the request
            # still goes to the bot's own OpenAI-compatible endpoint.
            result = compress_fn(
                messages,
                model=model or "gpt-4o",
                model_limit=self.config.model_limit,
                config=cfg,
            )
        except Exception:  # noqa: BLE001 — never break a review on compression
            log.warning(
                "headroom compression failed; using original messages", exc_info=True
            )
            return messages

        saved = getattr(result, "tokens_saved", 0) or 0
        if saved > 0:
            log.info(
                "headroom compressed context: %d -> %d tokens (saved %d, %.0f%%)",
                getattr(result, "tokens_before", 0),
                getattr(result, "tokens_after", 0),
                saved,
                100.0 * (getattr(result, "compression_ratio", 0.0) or 0.0),
            )
        return getattr(result, "messages", messages) or messages
