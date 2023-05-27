import transformers
from . import heads
from hyped.utils.typedmapping import typedmapping
from dataclasses import asdict
from copy import copy

class DummyPredictionHeadMixin:
    def build(self, model): pass
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

# dummy head types
class DummyClsHead(DummyPredictionHeadMixin, heads.HypedClsHead): pass
class DummyMlcHead(DummyPredictionHeadMixin, heads.HypedMlcHead): pass
class DummyTaggingHead(DummyPredictionHeadMixin, heads.HypedTaggingHead): pass
class DummyCausalLMHead(DummyPredictionHeadMixin, heads.HypedCausalLMHead): pass


class TransformerModelWrapper(transformers.PreTrainedModel, transformers.adapters.heads.ModelWithFlexibleHeadsAdaptersMixin):

    # sibling to the custom head mapping in hyped auto adapter model
    DUMMY_HEAD_MAPPING = typedmapping[str, type[DummyPredictionHeadMixin]]()

    def __init__(
        self,
        model:transformers.PreTrainedModel,
        head_name:str,
        head_config:heads.base.HypedPredictionHeadConfig
    ) -> None:
        assert not isinstance(model, transformers.adapters.heads.ModelWithFlexibleHeadsAdaptersMixin)

        # intiialize pretrained model and save model
        transformers.PreTrainedModel.__init__(self, copy(model.config))
        self.model = model

        # prepare for adding head later on
        self._init_head_modules()
        # create dummy head and add it to the wrapper
        if head_config.head_type not in type(self).DUMMY_HEAD_MAPPING:
            raise ValueError("No dummy head for head type `%s` found." % head_config.head_type)

        # prepare head config
        head_config = asdict(head_config)
        head_type = head_config.pop("head_type")
        # create dummy head
        head_t = type(self).DUMMY_HEAD_MAPPING[head_type]
        head = head_t(model, head_name=head_name, **head_config)
        # add prediction head and set it to the only active head
        self.add_prediction_head(head, overwrite_ok=False, set_active=False)
        self._active_heads = [head.name]

    def tie_weights(self):
        return None

    def forward(self, *args, **kwargs):
        # prepare kwargs, rename label column as expected by base
        kwargs = kwargs.copy()
        kwargs.update(self.heads[self.active_head].get_labels(kwargs))
        # apply model
        return self.model.forward(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def from_pretrained(self, *args, **kwargs):
        self.model.from_pretrained(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    @classmethod
    def register_dummy_head(cls, head_type:str, head_t:type[DummyPredictionHeadMixin]) -> None:
        cls.DUMMY_HEAD_MAPPING[head_type] = head_t

# register all dummy heads
TransformerModelWrapper.register_dummy_head(heads.HypedClsHeadConfig.head_type, DummyClsHead)
TransformerModelWrapper.register_dummy_head(heads.HypedMlcHeadConfig.head_type, DummyMlcHead)
TransformerModelWrapper.register_dummy_head(heads.HypedTaggingHeadConfig.head_type, DummyTaggingHead)
TransformerModelWrapper.register_dummy_head(heads.HypedCausalLMHeadConfig.head_type, DummyCausalLMHead)
