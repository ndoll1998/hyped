from . import metrics
from .metrics.base import (
    HypedMetric,
    HypedMetricConfig
)
from .collection import HypedMetricCollection
from hyped.utils.typedmapping import typedmapping
from hyped.modeling import (
    HypedModelWrapper,
    heads
)
from functools import cmp_to_key

class AutoHypedMetric(object):
    METRICS_MAPPING = typedmapping[
        type[heads.HypedHeadConfig],
        typedmapping
    ]()

    @classmethod
    def from_head_config(
        cls,
        h_config:heads.HypedHeadConfig,
        m_config:HypedMetricConfig
    ) -> HypedMetric:
        # find metrics for head
        key = cmp_to_key(lambda t, v: 2 * issubclass(v, t) - 1)
        for h_config_t in sorted(cls.METRICS_MAPPING, key=key):
            if isinstance(h_config, h_config_t):
                # find specific metric
                metrics_mapping = cls.METRICS_MAPPING[h_config_t]
                for m_config_t in sorted(metrics_mapping, key=key):
                    if isinstance(m_config, m_config_t):
                        metric_t = metrics_mapping[m_config_t]
                        return metric_t(h_config, m_config)

                raise ValueError(
                    "No metric registered for metric config type `%s` and head type `%s`." % (
                        type(m_config), type(h_config))
                )
        # no metric found for head
        raise ValueError("No metric registered for head of type `%s`." % type(head))

    @classmethod
    def from_model(
        cls,
        model:HypedModelWrapper,
        metric_configs:dict[str, list[HypedMetricConfig]],
        label_order:list[str]
    ) -> HypedMetricCollection:
        # type checking
        if not isinstance(model, HypedModelWrapper):
            raise ValueError("Expected model with `%s`, got %s." % (HypedModelWrapper, type(model)))

        # build metric for each head
        metrics = [
            cls.from_head_config(h_config, m_config)
            for h_config in model.head_configs
            for m_config in metric_configs.get(h_config.head_name, [])
        ]

        head_order = [h_config.head_name for h_config in model.head_configs]
        return HypedMetricCollection(metrics, head_order, label_order)

    @classmethod
    def register(
        cls,
        head_t:type[heads.HypedHeadConfig],
        config_t:type[HypedMetricConfig],
        metrics_t:type[HypedMetric]
    ):
        if head_t not in cls.METRICS_MAPPING:
            cls.METRICS_MAPPING[head_t] = typedmapping[
                type[HypedMetricConfig], type[HypedMetric]
            ]()

        cls.METRICS_MAPPING[head_t][config_t] = metrics_t

AutoHypedMetric.register(
    head_t=heads.HypedClsHeadConfig,
    config_t=metrics.ClsMetricConfig,
    metrics_t=metrics.ClsMetric
)
AutoHypedMetric.register(
    head_t=heads.HypedMlcHeadConfig,
    config_t=metrics.MlcMetricConfig,
    metrics_t=metrics.MlcMetric
)
AutoHypedMetric.register(
    head_t=heads.HypedTaggingHeadConfig,
    config_t=metrics.SeqEvalMetricConfig,
    metrics_t=metrics.SeqEvalMetric
)
