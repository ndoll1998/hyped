import pydantic
from .trainer import TrainerConfig
from .metrics import MetricsConfig
from .model.base import ModelConfig

class RunConfig(pydantic.BaseModel):
    """Run Configuration Model"""
    # run name
    name:str
    # model and trainer configuration
    model:ModelConfig
    trainer:TrainerConfig
    metrics:MetricsConfig

    @pydantic.validator('model', pre=True)
    def _parse_model_config(cls, value):

        # try to infer library from model configuration
        if 'backend' not in value:

            if ('heads' in value) and ('task' in value):
                raise ValueError("Could not infer library from model config, both `heads` and `task` field specified!")

            if 'heads' in value:
                # if heads are present then this is an adapter model
                value['backend'] = "adapter-transformers"

            elif 'task' in value:
                # if task is specified then this is a pure transformer model
                value['backend'] = "transformers"

            else:
                raise ValueError("Could not infer library from model config, neither `heads` nor `task` field specified!")

        # must have backend specification at this point
        assert 'backend' in value

        if value['backend'] == 'transformers':
            from .model.transformers import TransformerModelConfig
            return TransformerModelConfig(**value)

        if value['backend'] == 'adapter-transformers':
            from .model.adapters import AdapterTransformerModelConfig
            return AdapterTransformerModelConfig(**value)

        raise ValueError("Invalid backend %s" % value['backend'])

    @pydantic.validator('trainer', pre=True)
    def _pass_name_to_trainer_config(cls, v, values):
        assert 'name' in values
        if isinstance(v, pydantic.BaseModel):
            return v.copy(update={'name': values.get('name')})
        elif isinstance(v, dict):
            return v | {'name': values.get('name')}
