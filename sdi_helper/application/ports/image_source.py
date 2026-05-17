from typing import Iterator, Protocol

from sdi_helper.domain.entities.candidate_url import CandidateUrl


class ImageSource(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> Iterator[CandidateUrl]: ...

    def close(self) -> None: ...
