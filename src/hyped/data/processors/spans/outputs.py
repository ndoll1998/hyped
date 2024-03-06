from enum import Enum


class SpansOutputs(str, Enum):
    """Output column of data processors generating a sequence of spans"""

    BEGINS = "span_begins"
    """Output column of the sequence of begins to the generated spans"""

    ENDS = "span_ends"
    """Output column of the sequence of ends to the generated spans"""


class LabelledSpansOutputs(str, Enum):
    """Output column of data processors generating a sequence of labelled
    spans"""

    BEGINS = "span_begins"
    """Output column of the sequence of begins to the generated spans"""

    ENDS = "span_ends"
    """Output column of the sequence of ends to the generated spans"""

    LABELS = "span_labels"
    """Output column of the sequence of labels to the generated spans"""
