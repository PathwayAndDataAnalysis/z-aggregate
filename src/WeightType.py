from enum import Enum


class WeightType(str, Enum):
    UNIFORM = "Uniform"
    CORRELATION = "Correlation"
    SPECIFICITY = "Specificity"
    NON_ZERO_RATE = "NonzeroRate"
    EXISTING = "Existing"
