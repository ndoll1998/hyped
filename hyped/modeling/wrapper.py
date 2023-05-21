import transformers
from . import heads
from copy import copy

class DummyPredictionHeadMixin:
    def build(self, model): pass
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

class DummyClsHead(DummyPredictionHeadMixin, heads.HypedClsHead): pass
class DummyMlcHead(DummyPredictionHeadMixin, heads.HypedMlcHead): pass

def create_dummy_head_for_model(model, head_name:str ="default"):

    model_t = type(model)
    model_n = model_t.__name__

    if model_n.endswith("SequenceClassification"):

        if model.config.problem_type == "single_label_classification":
            return DummyClsHead(
                model,
                head_name=head_name,
                label_column=transformers.utils.find_labels(type(model))[0],
                num_labels=model.config.num_labels,
                id2label=model.config.id2label
            )

        elif model.config.problem_type == "multi_label_classification":
            return DummyMlcHead(
                model,
                head_name=head_name,
                label_column=transformers.utils.find_labels(type(model))[0],
                num_labels=model.config.num_labels,
                id2label=model.config.id2label
            )

    raise ValueError("Could not infer head type from model type `%s`" % model_t)

class TransformerModelWrapper(transformers.PreTrainedModel, transformers.adapters.heads.ModelWithFlexibleHeadsAdaptersMixin):

    def __init__(self, model:transformers.PreTrainedModel, head_name:str ="default") -> None:
        assert not isinstance(model, transformers.adapters.heads.ModelWithFlexibleHeadsAdaptersMixin)

        # intiialize pretrained model and save model
        transformers.PreTrainedModel.__init__(self, copy(model.config))
        self.model = model

        # prepare for adding head later on
        self._init_head_modules()
        # create dummy head and add it to the wrapper
        head = create_dummy_head_for_model(model, head_name=head_name)
        self.add_prediction_head(head, overwrite_ok=False, set_active=False)
        # set the heaid to be the only active head
        self._active_heads = [head.name]

    def tie_weights(self):
        return None

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    def from_pretrained(self, *args, **kwargs):
        self.model.from_pretrained(*args, **kwargs)

