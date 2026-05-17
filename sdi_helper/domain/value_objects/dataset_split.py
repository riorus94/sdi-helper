from enum import Enum


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
