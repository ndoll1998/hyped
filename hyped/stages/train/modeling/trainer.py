import torch
from transformers import Trainer
from transformers.adapters import AdapterTrainer
from transformers.adapters.heads import MultiHeadOutput

class MultiHeadLossMixin(object):
    def compute_loss(self, model, inputs, return_outputs=False):

        # apply model and
        output = model(**inputs)

        if isinstance(output, MultiHeadOutput) and (output.get('loss', None) is None):
            # check if all heads computed a loss
            head_outputs = output['head_outputs']
            if all("loss" in out and out["loss"] is not None for out in head_outputs):
                # compute combined loss
                output['loss'] = torch.sum(torch.stack([out["loss"] for out in head_outputs]))

        if isinstance(output, dict) and (output.get('loss', None) is None):
            raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )

        # get loss from output and return
        loss = output['loss'] if isinstance(output, dict) else output[0]
        return (loss, output) if return_outputs else loss

class MultiHeadTrainer(MultiHeadLossMixin, Trainer):
    """ Transformers trainer for multi-head models """

class MultiHeadAdapterTrainer(MultiHeadLossMixin, AdapterTrainer):
    """ Adapter trainer for multi-head models """

