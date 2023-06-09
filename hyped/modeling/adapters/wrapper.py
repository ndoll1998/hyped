import torch
from ..heads import HypedHeadConfig
from ..wrapper import HypedModelWrapper
from transformers.adapters.heads import (
    MultiHeadOutput,
    ModelWithFlexibleHeadsAdaptersMixin
)

class HypedAdapterModelWrapper(HypedModelWrapper):

    def __init__(self, model:ModelWithFlexibleHeadsAdaptersMixin) -> None:
        # check model type
        if not isinstance(model, ModelWithFlexibleHeadsAdaptersMixin):
            raise TypeError("Model must inherit type `%s`, got `%s`" % (type(model), ModelWithFlexibleHeadsAdaptersMixin))

        # initialize wrapper
        super(HypedAdapterModelWrapper, self).__init__(model)

    def __call__(self, *args, **kwargs):
        # apply model
        out = super(HypedAdapterModelWrapper, self).__call__(*args, **kwargs)
        # compute combined loss for parallel heads
        if isinstance(out, MultiHeadOutput) and (out.get('loss', None) is None):
            h_outs = out['head_outputs']
            # check if all heads computed a loss
            if all("loss" in out and out["loss"] is not None for out in h_outs):
                loss_weights = torch.FloatTensor([
                    h_config.loss_coeff for h_config in self.head_configs
                ])
                # compute combined loss
                h_losses = torch.stack([out["loss"] for out in h_outs])
                h_losses = (loss_weights.to(h_losses.device) * h_losses).sum()
                out['loss'] = h_losses.sum()
        # return output
        return out

    @property
    def head_configs(self) -> list[HypedHeadConfig]:
        # get active head names
        head_names = self.active_head
        head_names = [head_names] if isinstance(head_names, str) else head_names
        # TODO: create head configs from adapter-transformers heads
        # collect heads to the active names
        return [self.heads[name].h_config for name in head_names]
