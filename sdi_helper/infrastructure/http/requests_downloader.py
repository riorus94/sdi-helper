"""HTTP image downloader with retry / backoff.

Migration source: pipeline/agents/scraper.py:_make_session + run_pipeline.py:download_to_memory
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RequestsDownloader:
    def __init__(self, user_agent: str = "SdiHelperBot/1.0", timeout: int = 10) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            session = requests.Session()
            retry = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET"],
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._session = session
        return self._session

    def fetch(self, url: str) -> bytes | None:
        local = self._read_local(url)
        if local is not None:
            return local

        if not url.startswith("http"):
            return None
        try:
            resp = self._get_session().get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return None
            return resp.content
        except requests.RequestException:
            return None

    def _read_local(self, url: str) -> bytes | None:
        try:
            parsed = urlparse(url)
            if parsed.scheme == "file":
                path = Path(unquote(parsed.path.lstrip("/")))
                if path.exists() and path.is_file():
                    return path.read_bytes()
                return None

            raw = Path(url)
            if raw.exists() and raw.is_file():
                return raw.read_bytes()
        except Exception:
            return None
        return None
