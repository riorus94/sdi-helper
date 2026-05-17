from dataclasses import dataclass

from sdi_helper.domain.entities.scraped_image import ScrapedImage


@dataclass(frozen=True)
class StorageKeys:
    prefix: str = ""

    def _join(self, *parts: str) -> str:
        joined = "/".join(p.strip("/") for p in parts if p)
        return f"{self.prefix.rstrip('/')}/{joined}" if self.prefix else joined

    def image_key(self, img: ScrapedImage) -> str:
        return self._join("images", img.split.value, img.view.value, f"{img.uuid}.jpg")

    def label_key(self, img: ScrapedImage) -> str:
        return self._join("labels", img.split.value, img.view.value, f"{img.uuid}.txt")

    def manifest_key(self, img: ScrapedImage) -> str:
        return self._join("manifests", f"{img.uuid}.json")

    def dataset_yaml_key(self) -> str:
        return self._join("dataset.yaml")

    def urls_csv_key(self) -> str:
        return self._join("urls.csv")

    def quota_state_key(self) -> str:
        return self._join("state", "quota.json")

    def manifests_prefix(self) -> str:
        return self._join("manifests")
