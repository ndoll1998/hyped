import hyped
import pydantic
import datasets
import transformers

class DataConfig(pydantic.BaseModel):
    """Data Configuration Model"""
    # dataset config
    dataset:str
    splits:list[str] = [datasets.Split.TRAIN, datasets.Split.TEST]
    # data processing config
    processor:hyped.AnyProcessorConfig = pydantic.Field(..., discriminator='type')
    filter:hyped.AnyFilterConfig = pydantic.Field(...) #, discriminator='type')

    @pydantic.validator('dataset')
    def validate_dataset(cls, v):
        if v is None:
            raise ValueError("No Dataset provided by configuration!")
        try:
            # try to load dataset builder
            builder = datasets.load_dataset_builder(v)
            return v
        except FileNotFoundError as e:
            # raise exception if dataset builder cannot be found
            raise ValueError("Dataset not found: %s" % v) from e

    @property
    def info(self) -> datasets.DatasetInfo:
        return datasets.load_dataset_builder(self.dataset)._info()

class ModelConfig(pydantic.BaseModel):
    """Model Configuration Model"""
    pretrained_ckpt:str

    @pydantic.validator('pretrained_ckpt')
    def _check_pretrained_ckpt(cls, value):
        try:
            # check if model is valid by loading config
            transformers.AutoConfig.from_pretrained(value)
        except OSError as e:
            # handle model invalid
            raise ValueError("Unkown pretrained checkpoint: %s" % value) from e

        return value

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        return transformers.AutoTokenizer.from_pretrained(self.pretrained_ckpt, use_fast=True, add_prefix_space=True)


class RunConfig(pydantic.BaseModel):
    """Run Configuration Model"""
    data:DataConfig
    model:ModelConfig
