from .cls import HypedClsHead, HypedClsHeadConfig
from .mlc import HypedMlcHead, HypedMlcHeadConfig
from .tagging import HypedTaggingHead, HypedTaggingHeadConfig
from .causal_lm import HypedCausalLMHead, HypedCausalLMHeadConfig

AnyHypedHeadConfig = \
    HypedClsHeadConfig | \
    HypedMlcHeadConfig | \
    HypedTaggingHeadConfig
