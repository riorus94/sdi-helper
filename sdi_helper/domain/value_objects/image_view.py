from enum import Enum


class ImageView(str, Enum):
    FRONT = "front"
    SIDE = "side"
    REAR = "rear"
