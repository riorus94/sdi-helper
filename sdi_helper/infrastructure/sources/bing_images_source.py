"""Bing Images via headless Chrome."""

import json
import time
from typing import Any, Iterator

from selenium.webdriver.common.by import By

from sdi_helper.domain.entities.candidate_url import CandidateUrl

_BAD_KEYWORDS = ("logo", "icon", "avatar", "profile", "banner", "ads", "sponsor")
_VECTOR_EXT = (".svg", ".ai", ".eps")


def _is_bad_url(url: str) -> bool:
    u = url.lower()
    return any(bad in u for bad in _BAD_KEYWORDS) or u.endswith(_VECTOR_EXT)


class BingImagesSource:
    name = "bing"

    def __init__(self, driver: Any = None, driver_factory: Any = None) -> None:
        self._driver_factory = driver_factory
        self.driver = driver
        self._own_driver = driver is None

    def _ensure_driver(self) -> Any:
        if self.driver is None:
            if self._driver_factory is None:
                from sdi_helper.infrastructure.http.selenium_driver_factory import make_driver
                self.driver = make_driver()
            else:
                self.driver = self._driver_factory()
        return self.driver

    def search(self, query: str, max_results: int) -> Iterator[CandidateUrl]:
        driver = self._ensure_driver()
        search_url = (
            "https://www.bing.com/images/search?q="
            + query.replace(" ", "+")
            + "&form=HDRSC2&first=1"
        )
        driver.get(search_url)
        time.sleep(2)

        emitted: set[str] = set()
        scroll_attempts = 0
        max_scrolls = 12

        while len(emitted) < max_results and scroll_attempts < max_scrolls:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1.5)

            # Primary: <a class="iusc"> elements carry a JSON `m` attribute with
            # `murl` (full-size) and `turl` (thumbnail).
            for elem in driver.find_elements(By.CSS_SELECTOR, "a.iusc"):
                m_attr = elem.get_attribute("m")
                if not m_attr:
                    continue
                try:
                    data = json.loads(m_attr)
                except Exception:
                    continue
                url = data.get("murl") or data.get("turl", "")
                if not url or not url.startswith("http"):
                    continue
                if _is_bad_url(url) or url in emitted:
                    continue
                emitted.add(url)
                yield CandidateUrl(
                    image_url=url,
                    source_page=search_url,
                    source_name=self.name,
                    query=query,
                )
                if len(emitted) >= max_results:
                    return

            # Fallback: direct img.mimg thumbnails.
            for img in driver.find_elements(By.CSS_SELECTOR, "img.mimg"):
                src = img.get_attribute("src") or img.get_attribute("data-src") or ""
                if not src.startswith("http") or _is_bad_url(src) or src in emitted:
                    continue
                emitted.add(src)
                yield CandidateUrl(
                    image_url=src,
                    source_page=search_url,
                    source_name=self.name,
                    query=query,
                )
                if len(emitted) >= max_results:
                    return

            scroll_attempts += 1

    def close(self) -> None:
        if self.driver is not None and self._own_driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
