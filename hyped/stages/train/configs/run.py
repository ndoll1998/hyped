import pydantic
from .trainer import TrainerConfig
from .metrics import MetricsConfig
from .model.adapters import AdapterTransformerModelConfig


class RunConfig(pydantic.BaseModel):
    """Run Configuration Model"""
    # run name
    name:str
    # model and trainer configuration
    model:AdapterTransformerModelConfig
    trainer:TrainerConfig
    metrics:MetricsConfig

    # TODO
    #@pydantic.validator('model', pre=True)
    def _infer_model_library(cls, value):
        if 'library' not in value:

            if ('heads' in value) and ('task' in value):
                raise ValueError("Could not infer library from model config, both `heads` and `task` field specified!")

            if 'heads' in value:
                # if heads are present then this is an adapter model
                value['library'] = "adapter-transformers"

            elif 'task' in value:
                # if task is specified then this is a pure transformer model
                value['library'] = "transformers"

            else:
                raise ValueError("Could not infer library from model config, neither `heads` nor `task` field specified!")

        return value

    @pydantic.validator('trainer', pre=True)
    def _pass_name_to_trainer_config(cls, v, values):
        assert 'name' in values
        if isinstance(v, pydantic.BaseModel):
            return v.copy(update={'name': values.get('name')})
        elif isinstance(v, dict):
            return v | {'name': values.get('name')}
