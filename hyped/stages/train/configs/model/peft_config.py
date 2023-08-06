import hyped
import pydantic
import dataclasses
import datasets
import transformers
from typing import Literal
from typing_extensions import Annotated
from .transformers_config import TransformerModelConfig, Task
# import peft backend
import peft
import hyped.modeling.peft

@dataclasses.dataclass
class AdaptionPromptConfig(peft.AdaptionPromptConfig):
    peft_type: Literal[peft.PeftType.ADAPTION_PROMPT]

@dataclasses.dataclass
class LoraConfig(peft.LoraConfig):
    peft_type: Literal[peft.PeftType.LORA]

@dataclasses.dataclass
class IA3Config(peft.IA3Config):
    peft_type: Literal[peft.PeftType.IA3]

@dataclasses.dataclass
class AdaLoraConfig(peft.AdaLoraConfig):
    peft_type: Literal[peft.PeftType.ADALORA]

def get_peft_task_type(task:Task) -> peft.TaskType:
    return {
        Task.CLASSIFICATION:              peft.TaskType.SEQ_CLS,
        Task.MULTI_LABEL_CLASSIFICATION:  peft.TaskType.SEQ_CLS,
        Task.TOKEN_CLASSIFICATION:        peft.TaskType.TOKEN_CLS,
        Task.CAUSAL_LANGUAGE_MODELING:    peft.TaskType.CAUSAL_LM
    }[task if isinstance(task, Task) else Task(task)]

class PeftModelConfig(TransformerModelConfig):
    """ PEFT Adapter Model Configuration """
    backend:Literal['peft'] = 'peft'
    # peft adapter configuration
    adapter_name:str = "default"
    peft_config:Annotated[
        (
            AdaptionPromptConfig |
            LoraConfig |
            IA3Config |
            AdaLoraConfig
        ),
        pydantic.Field(..., discriminator='peft_type')
    ]

    @pydantic.root_validator(pre=True)
    def _infer_peft_task_type_from_model(cls, values):
        task = values['task']
        conf = values['peft_config']
        # set task type if not explicitly defined
        conf['task_type'] = conf.get('task_type', get_peft_task_type(task))
        return values | {'peft_config': conf}

    def build(self, info:datasets.DatasetInfo) -> transformers.PreTrainedModel:
        # build the model and unwrap it
        wrapped = super(PeftModelConfig, self).build(info)
        model, h_config = wrapped.__wrapped__, wrapped.head_configs[0]
        # build the peft model and wrap it in the peft wrapper
        model = peft.get_peft_model(model, self.peft_config, adapter_name=self.adapter_name)
        return hyped.modeling.peft.HypedPeftModelWrapper(
            model=model, h_config=h_config
        )
