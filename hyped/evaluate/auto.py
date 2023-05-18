from . import metrics
from .metrics.base import HypedMetric, HypedMetricConfig
from .collection import HypedMetricCollection
from transformers.adapters import heads
from hyped.utils.typedmapping import typedmapping

class HypedAutoMetric(object):
    METRICS_MAPPING = typedmapping[
        type[heads.PredictionHead],
        typedmapping
    ]()

    @classmethod
    def from_head(
        cls,
        head:heads.PredictionHead,
        config:HypedMetricConfig
    ) -> HypedMetric:
        # find metrics for head
        for head_t, metrics_mapping in cls.METRICS_MAPPING.items():
            if isinstance(head, head_t):
                # find specific metric
                metric_t = metrics_mapping.get(type(config), None)
                # check if metric type found
                if metric_t is None:
                    raise ValueError("Invalid metric type `%s`." % config.metric_type)
                # create metric instance
                metric = metric_t(head, config)
                return metric
        # no metric found for head
        raise ValueError("No metric registered for head of type `%s`." % type(head))

    @classmethod
    def from_model(
        cls,
        model:heads.ModelWithFlexibleHeadsAdaptersMixin,
        metric_configs:dict[str, list[HypedMetricConfig]],
        label_order:list[str]
    ) -> HypedMetricCollection:
        # type checking
        if not isinstance(model, heads.ModelWithFlexibleHeadsAdaptersMixin):
            raise ValueError("Expected model with `ModelWithFlexibleHeadsAdaptersMixin`, got %s." % type(model))
        if model.active_head is None:
            raise ValueError("No active head detected in model!")

        if isinstance(model.active_head, str):
            # single active head
            head = model.heads[model.active_head]
            # build metric collection
            return HypedMetricCollection(
                metrics=[cls.from_head(head, config) for config in metric_configs[head.name]],
                head_order=[model.active_head],
                label_order=label_order
            )

        elif isinstance(model.active_head, list):
            # check if label order is given
            if label_order is None:
                raise ValueError("Label order is required for multi head models, got label_order=%s!" % label_order)
            # build metric for each head
            metrics = [
                cls.from_head(model.heads[head_name], config)
                for head_name in model.active_head
                for config in metric_configs[head_name]
            ]
            # build metrics collection and return
            return HypedMetricCollection(metrics, model.active_head, label_order)

        raise Exception("Unexpected active head %s!" % model.active_head)

    @classmethod
    def register(
        cls,
        head_t:type[heads.PredictionHead],
        config_t:type[HypedMetricConfig],
        metrics_t:type[HypedMetric]
    ):
        if head_t not in cls.METRICS_MAPPING:
            cls.METRICS_MAPPING[head_t] = typedmapping[
                type[HypedMetricConfig], type[HypedMetric]
            ]()

        cls.METRICS_MAPPING[head_t][config_t] = metrics_t

HypedAutoMetric.register(
    head_t=heads.ClassificationHead,
    config_t=metrics.ClassificationMetricConfig,
    metrics_t=metrics.ClassificationMetric
)
HypedAutoMetric.register(
    head_t=heads.MultiLabelClassificationHead,
    config_t=metrics.MlcMetricConfig,
    metrics_t=metrics.MlcMetric
)
HypedAutoMetric.register(
    head_t=heads.TaggingHead,
    config_t=metrics.SeqEvalMetricConfig,
    metrics_t=metrics.SeqEvalMetric
)
