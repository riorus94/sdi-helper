"""Google Images via headless Chrome.

Migration source: pipeline/agents/scraper.py (search_google_images, _IMG_SELECTORS).
"""

import re
import time
from typing import Any, Iterator

from selenium.webdriver.common.by import By

from sdi_helper.domain.entities.candidate_url import CandidateUrl

_BAD_KEYWORDS = ("logo", "icon", "avatar", "profile", "banner", "ads", "sponsor")
_VECTOR_EXT = (".svg", ".ai", ".eps")

_IMG_SELECTORS = (
    "img.YQ4gaf",
    "img.rg_i",
    "div[data-tbnid] img",
)


def _is_bad_url(url: str) -> bool:
    u = url.lower()
    return any(bad in u for bad in _BAD_KEYWORDS) or u.endswith(_VECTOR_EXT)


class GoogleImagesSource:
    name = "google"

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
            "https://www.google.com/search?q="
            + query.replace(" ", "+")
            + "&tbm=isch&safe=off"
        )
        driver.get(search_url)
        time.sleep(2)

        emitted: set[str] = set()
        scroll_attempts = 0
        max_scrolls = 10

        while len(emitted) < max_results and scroll_attempts < max_scrolls:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1.5)

            imgs: list[Any] = []
            for sel in _IMG_SELECTORS:
                imgs = driver.find_elements(By.CSS_SELECTOR, sel)
                if imgs:
                    break

            for img in imgs:
                src = img.get_attribute("src") or img.get_attribute("data-src")
                if not src or not src.startswith("http"):
                    continue
                if _is_bad_url(src):
                    continue
                if src in emitted:
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

            for s in driver.find_elements(By.TAG_NAME, "script"):
                txt = s.get_attribute("innerHTML") or ""
                if "AF_initDataCallback" not in txt:
                    continue
                for url in re.findall(r'https?://[^"\\]+\.(?:jpg|jpeg|png|webp)', txt):
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

            scroll_attempts += 1

    def close(self) -> None:
        if self.driver is not None and self._own_driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
