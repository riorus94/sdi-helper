"""Headless Chrome driver factory.

Migration source: pipeline/agents/scraper.py:_make_driver
"""

import os
from typing import Any

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def make_driver() -> Any:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    override = os.environ.get("CHROMEDRIVER_PATH", "").strip()
    driver_path = (
        override
        if override and os.path.isfile(override)
        else ChromeDriverManager().install()
    )
    return webdriver.Chrome(service=Service(driver_path), options=options)
