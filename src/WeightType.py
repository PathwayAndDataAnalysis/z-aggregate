from enum import Enum


class WeightType(str, Enum):
    UNIFORM = "Uniform_Weight"
    CORRELATION = "Correlation_Weight"
    SPECIFICITY = "Specificity_Weight"
    NON_ZERO_RATE = "Non_Zero_Rate_Weight"
    EXISTING = "Existing_Weight"
