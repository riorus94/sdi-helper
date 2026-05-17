from sdi_helper.domain.entities.bounding_box import BoundingBox
from sdi_helper.domain.entities.scraped_image import ScrapedImage
from sdi_helper.domain.services.storage_keys import StorageKeys
from sdi_helper.domain.value_objects.dataset_split import DatasetSplit
from sdi_helper.domain.value_objects.image_domain import ImageDomain
from sdi_helper.domain.value_objects.image_view import ImageView


def _sample(uuid: str = "abc123", view: ImageView = ImageView.SIDE) -> ScrapedImage:
    return ScrapedImage(
        uuid=uuid,
        image_url="https://example.com/x.jpg",
        source_name="google",
        query="test",
        view=view,
        domain=ImageDomain.REAL,
        bboxes=(BoundingBox(0.5, 0.5, 0.4, 0.3, 0.9),),
        view_confidence=0.9,
        split=DatasetSplit.TRAIN,
    )


def test_image_key_layout() -> None:
    keys = StorageKeys()
    assert keys.image_key(_sample()) == "images/train/side/abc123.jpg"


def test_prefix_is_applied() -> None:
    keys = StorageKeys(prefix="datasets/cars/v1")
    assert keys.image_key(_sample()) == "datasets/cars/v1/images/train/side/abc123.jpg"


def test_label_and_manifest_keys() -> None:
    keys = StorageKeys()
    img = _sample()
    assert keys.label_key(img) == "labels/train/side/abc123.txt"
    assert keys.manifest_key(img) == "manifests/abc123.json"


def test_dataset_yaml_key() -> None:
    assert StorageKeys().dataset_yaml_key() == "dataset.yaml"
    assert StorageKeys(prefix="d/c/v1").dataset_yaml_key() == "d/c/v1/dataset.yaml"


def test_quota_state_key() -> None:
    assert StorageKeys().quota_state_key() == "state/quota.json"


def test_view_separation_in_layout() -> None:
    keys = StorageKeys()
    front = _sample(view=ImageView.FRONT)
    rear = _sample(view=ImageView.REAR)
    assert "/front/" in keys.image_key(front)
    assert "/rear/" in keys.image_key(rear)
