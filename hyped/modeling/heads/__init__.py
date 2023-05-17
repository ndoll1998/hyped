from .cls import HypedClsHead, HypedClsHeadConfig
from .mlc import HypedMlcHead, HypedMlcHeadConfig
from .tagging import HypedTaggingHead, HypedTaggingHeadConfig

AnyHypedHeadConfig = \
    HypedClsHeadConfig | \
    HypedMlcHeadConfig | \
    HypedTaggingHeadConfig
