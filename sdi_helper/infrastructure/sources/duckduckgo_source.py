"""DuckDuckGo Images via HTTP VQD token API — no Selenium required.

Protocol (stable as of 2025-2026):
  1. GET https://duckduckgo.com/?q=<query>&ia=images  → extract vqd token
  2. GET https://duckduckgo.com/i.js?...&vqd=<token>  → JSON image results
     paginate via the `next` field until max_results reached.
"""

import re
import time
from typing import Iterator
from urllib.parse import quote_plus

import requests

from sdi_helper.domain.entities.candidate_url import CandidateUrl

_BAD_KEYWORDS = ("logo", "icon", "avatar", "profile", "banner", "ads", "sponsor")
_VECTOR_EXT = (".svg", ".ai", ".eps")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://duckduckgo.com/",
}
_VQD_RE = re.compile(r'vqd=[\'"]?([\w-]+)[\'"]?')
_SESSION_TIMEOUT = 10


def _is_bad_url(url: str) -> bool:
    u = url.lower()
    return any(bad in u for bad in _BAD_KEYWORDS) or u.endswith(_VECTOR_EXT)


class DuckDuckGoSource:
    name = "duckduckgo"

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _get_vqd(self, query: str) -> str | None:
        try:
            resp = self._session.get(
                "https://duckduckgo.com/",
                params={"q": query, "ia": "images"},
                timeout=_SESSION_TIMEOUT,
            )
            resp.raise_for_status()
        except Exception:
            return None
        m = _VQD_RE.search(resp.text)
        return m.group(1) if m else None

    def search(self, query: str, max_results: int) -> Iterator[CandidateUrl]:
        vqd = self._get_vqd(query)
        if not vqd:
            return

        emitted: set[str] = set()
        next_params: dict = {
            "l": "wt-wt",
            "o": "json",
            "q": query,
            "vqd": vqd,
            "f": ",,,",
            "p": "1",
        }
        base_url = "https://duckduckgo.com/i.js"
        search_page = f"https://duckduckgo.com/?q={quote_plus(query)}&ia=images"

        while len(emitted) < max_results:
            try:
                resp = self._session.get(base_url, params=next_params, timeout=_SESSION_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                break

            results = data.get("results", [])
            if not results:
                break

            for item in results:
                url = item.get("image") or item.get("thumbnail", "")
                if not url or not url.startswith("http"):
                    continue
                if _is_bad_url(url) or url in emitted:
                    continue
                emitted.add(url)
                yield CandidateUrl(
                    image_url=url,
                    source_page=search_page,
                    source_name=self.name,
                    query=query,
                )
                if len(emitted) >= max_results:
                    return

            # Paginate: DDG returns a `next` query string fragment.
            nxt = data.get("next", "")
            if not nxt:
                break
            # `next` is a query string like "l=wt-wt&o=json&q=...&vqd=...&s=50&..."
            from urllib.parse import parse_qs, urlencode
            try:
                parsed = {k: v[0] for k, v in parse_qs(nxt.lstrip("?")).items()}
            except Exception:
                break
            next_params = parsed
            time.sleep(0.5)  # be polite between pages

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
