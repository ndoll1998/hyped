import hyped
import pydantic
import dataclasses
from typing import Literal
from typing_extensions import Annotated
from hyped.metrics import metrics

@dataclasses.dataclass
class ClsMetricConfig(hyped.metrics.metrics.ClsMetricConfig):
    metric_type:Literal["cls"] = "cls"

@dataclasses.dataclass
class MlcMetricConfig(hyped.metrics.metrics.MlcMetricConfig):
    metric_type:Literal["mlc"] = "mlc"

@dataclasses.dataclass
class SeqEvalMetricConfig(hyped.metrics.metrics.SeqEvalMetricConfig):
    metric_type:Literal["seqeval"] = "seqeval"


MetricsConfig = dict[
    str,
    list[
        Annotated[
            (
                ClsMetricConfig |
                MlcMetricConfig |
                SeqEvalMetricConfig
            ),
            pydantic.Field(..., discriminator='metric_type')
        ]
    ]
]
