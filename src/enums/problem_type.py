from enum import Enum


class ProblemType(Enum):
    Regression = 0
    BinaryClassification = 1
    MulticlassClassification = 2