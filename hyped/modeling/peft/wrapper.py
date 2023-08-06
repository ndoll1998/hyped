import peft
import transformers
from wrapt import CallableObjectProxy
from ..transformers.wrapper import HypedTransformerModelWrapper
from ..heads import HypedHeadConfig

class HypedPeftModelWrapper(HypedTransformerModelWrapper):
    BASE_MODEL_TYPE:type[peft.PeftModel] = peft.PeftModel
