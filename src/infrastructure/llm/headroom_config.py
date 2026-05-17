import os

HEADROOM_ENABLED = os.getenv("HEADROOM_ENABLED", "true").lower() == "true"
HEADROOM_MODEL = os.getenv("HEADROOM_MODEL", "gpt-4o")
