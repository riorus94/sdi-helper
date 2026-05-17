from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateUrl:
    image_url: str
    source_page: str
    source_name: str
    query: str
