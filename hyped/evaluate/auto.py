from . import metrics
from .metrics.base import HypedMetrics
from .metrics.collection import HypedMetricsCollection
from transformers.adapters import heads
from hyped.utils.typedmapping import typedmapping

class HypedAutoMetrics(object):
    METRICS_MAPPING = typedmapping[
        type[heads.PredictionHead],
        type[HypedMetrics]
    ]()

    @classmethod
    def from_head(cls, head:heads.PredictionHead, **kwargs):
        # find metric type for given head
        for head_t, metrics_t in cls.METRICS_MAPPING.items():
            if isinstance(head, head_t):
                # create metric instance
                metrics = metrics_t(head, **kwargs)
                return metrics
        # no metric found for head of type
        raise ValueError("No metrics found for head of type %s." % type(head))

    @classmethod
    def from_model(
        cls,
        model:heads.ModelWithFlexibleHeadsAdaptersMixin,
        metrics_kwargs:dict ={},
        label_order:None|list[str] =None
    ) -> HypedMetrics:
        # type checking
        if not isinstance(model, heads.ModelWithFlexibleHeadsAdaptersMixin):
            raise ValueError("Expected model with `ModelWithFlexibleHeadsAdaptersMixin`, got %s." % type(model))
        if model.active_head is None:
            raise ValueError("No active head detected in model!")

        if isinstance(model.active_head, str):
            # single active head
            head = model.heads[model.active_head]
            return cls.from_head(head, **metrics_kwargs.get(model.active_head, {}))

        elif isinstance(model.active_head, list):
            # check if label order is given
            if label_order is None:
                raise ValueError("Label order is required for multi head models, got label_order=%s!" % label_order)
            # build metric for each head
            metrics = [
                cls.from_head(model.heads[head_name], **metrics_kwargs.get(head_name, {}))
                for head_name in model.active_head
            ]
            # build metrics collection and return
            return HypedMetricsCollection(metrics, model.active_head, label_order)

        raise Exception("Unexpected active head %s!" % model.active_head)

    @classmethod
    def register(cls, head_t:type[heads.PredictionHead], metrics_t:type[HypedMetrics]):
        cls.METRICS_MAPPING[head_t] = metrics_t

# register metrics
HypedAutoMetrics.register(heads.ClassificationHead, metrics.HypedClsMetrics)
HypedAutoMetrics.register(heads.MultiLabelClassificationHead, metrics.HypedMlcMetrics)
HypedAutoMetrics.register(heads.TaggingHead, metrics.HypedTaggingMetrics)
