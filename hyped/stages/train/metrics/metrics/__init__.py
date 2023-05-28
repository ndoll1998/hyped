from .cls import ClassificationMetric, ClassificationMetricConfig
from .mlc import MlcMetricConfig, MlcMetric
from .seqeval import SeqEvalMetricConfig, SeqEvalMetric

AnyHypedMetricConfig = \
    ClassificationMetricConfig | \
    MlcMetricConfig | \
    SeqEvalMetricConfig
