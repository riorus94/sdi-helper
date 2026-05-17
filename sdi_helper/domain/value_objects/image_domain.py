from enum import Enum


class ImageDomain(str, Enum):
    REAL = "real"
    RENDER = "render"
    SKETCH = "sketch"
