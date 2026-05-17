from .headroom_config import HEADROOM_ENABLED, HEADROOM_MODEL


def _get_headroom_compress():
    try:
        import headroom
    except ImportError as exc:
        raise RuntimeError(
            "HEADROOM_ENABLED=true but headroom compression is unavailable. "
            "Install dependencies in the project-local .venv or add the headroom-ai package."
        ) from exc

    compress = getattr(headroom, "compress", None)
    if compress is None:
        raise RuntimeError(
            "HEADROOM_ENABLED=true but the installed headroom package does not export compress()."
        )

    return compress


def compress_messages(messages: list[dict]) -> list[dict]:
    if not HEADROOM_ENABLED:
        return messages

    compressed = _get_headroom_compress()(messages, model=HEADROOM_MODEL)

    if hasattr(compressed, "messages"):
        return compressed.messages

    return compressed
