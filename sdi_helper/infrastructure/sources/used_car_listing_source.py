"""OLX / Mobil123 / Carmudi gallery scraper - highest yield for SDI work."""

from typing import Iterator

from sdi_helper.domain.entities.candidate_url import CandidateUrl


class UsedCarListingSource:
    name = "used_car"

    def __init__(self, site: str = "olx") -> None:
        self.site = site

    def search(self, query: str, max_results: int) -> Iterator[CandidateUrl]:
        raise NotImplementedError("Backlog - Sprint 2 (per-site scraper)")

    def close(self) -> None:
        pass
