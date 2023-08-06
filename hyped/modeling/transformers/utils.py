from transformers import (
    PreTrainedModel,
    AutoModel
)

def get_pretrained_module(model:PreTrainedModel) -> PreTrainedModel:
        # find base model class
        model_class = AutoModel._model_mapping.get(type(model.config), None)
        # check if is registered
        if model_class is None:
            raise TypeError("Could not infer pretrained weights from model type. Make sure base model class if registered in `transformers.AutoModel`")

        # search for model class
        for module in model.children():
            if isinstance(module, model_class):
                return module

        else:
            # member not found
            raise AttributeError("Pretrained model member of type `%s` not found in `%s`" % (model_class, model.__class__))
