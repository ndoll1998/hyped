import pydantic
from .trainer import TrainerConfig
from .metrics import MetricsConfig
from .model.base import ModelConfig

class ExperimentConfig(pydantic.BaseModel):
    """Experiment Configuration Model"""
    # run name
    name:str
    # model and trainer configuration
    model:ModelConfig
    trainer:TrainerConfig
    metrics:MetricsConfig

    @pydantic.field_validator('model', mode='before')
    def _parse_model_config(cls, value):

        # default backend is transformers
        value['backend'] = value.get('backend', 'transformers')

        # must have backend specification at this point
        assert 'backend' in value

        if value['backend'] == 'transformers':
            from .model.transformers_config import TransformerModelConfig
            return TransformerModelConfig(**value)

        if value['backend'] == 'adapter-transformers':
            from .model.adapters_config import AdapterTransformerModelConfig
            return AdapterTransformerModelConfig(**value)

        if value['backend'] == 'peft':
            from .model.peft_config import PeftModelConfig
            return PeftModelConfig(**value)

        raise ValueError("Invalid backend %s" % value['backend'])

    @pydantic.model_validator(mode='before')
    def _pass_name_to_trainer_config(cls, values):
        return values | {
            'trainer': values['trainer'] | {'name': values.get('name')}
        }
