import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_qkvpacked_func

import transformers
from dataclasses import dataclass

@dataclass
class bllmOutput(transformers.utils.ModelOutput):

    logits: torch.FloatTensor
    loss: None | torch.FloatTensor = None
 

class bllmConfig(transformers.BertConfig):

    @property
    def dtype(self):
        return torch.float16


class bllmEmbedding(nn.Module):
    
    def __init__(self, config: bllmConfig) -> None:
        super(bllmEmbedding, self).__init__()

        # create the token embedding matrix, note that the padding index is set but during training
        # and inference the sequence is filled up with mask tokens instead of padding tokens
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, dtype=config.dtype
        )
        # create the position emebdding matrix
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size, dtype=config.dtype)

        # layer normalization and dropout
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=config.dtype)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # helper to easily get the input ids
        self.register_buffer(
            "pos_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        # get the position ids to the input
        _, length = input_ids.size()
        pos_ids = self.pos_ids[:, :length]
        # compute the embeddings
        tok_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(pos_ids)
        # compute final embedding
        embeds = tok_embeds + pos_embeds
        embeds = self.dropout(embeds)
        embeds = self.ln(embeds)

        return embeds


class bllmActivation(nn.Module):

    def __init__(self, config: bllmConfig) -> None:
        super(bllmActivation, self).__init__()
        self.fn = (
            transformers.activations.ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str) else
            config.hidden_act
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.fn(x)

class bllmLayer(nn.Module):

    def __init__(self, config: bllmConfig) -> None:
        super(bllmLayer, self).__init__()

        assert config.hidden_size % config.num_attention_heads == 0
        # read config values
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.n_heads
        self._attn_dropout_prob = config.attention_probs_dropout_prob

        # self attention components
        # query, key and value transformations all in one operation
        self.attn_qkv = nn.Linear(
            config.hidden_size, config.hidden_size * 3, dtype=config.dtype
        )
        self.attn_dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False, dtype=config.dtype
        )
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attn_ln = nn.LayerNorm(config.hidden_size, dtype=config.dtype)

        # feed forward with non-linear activation
        self.feed_forward = nn.Sequential( 
            nn.Linear(config.hidden_size, config.intermediate_size, dtype=config.dtype),
            bllmActivation(config),
            nn.Linear(config.intermediate_size, config.hidden_size, dtype=config.dtype),
        )
        # dropout and layernorm
        self.feed_forward_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.feed_forward_ln = nn.LayerNorm(config.hidden_size, dtype=config.dtype)

    @property
    def attn_dropout_prob(self) -> float:
        return self._attn_dropout_prob if self.training else 0.0

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        b, s, d = x.size()
        # apply self-attention
        x = self.attn_ln(
            x + self.attn_dropout(
                self.attn_dense(
                    # use flash attention implementation
                    flash_attn_qkvpacked_func(
                        qkv=self.attn_qkv(x).reshape(
                            b, s, 3, self.n_heads, self.head_dim
                        ),
                        dropout_p=self.attn_dropout_prob,
                        causal=False
                    ).reshape(b, s, d)
                )
            )
        )
        # pass through feed forward network
        x = self.feed_forward_ln(
            x + self.feed_forward_dropout(
                self.feed_forward(x)
            )
        )

        return x

class bllmEncoder(nn.Module):
    def __init__(self, config: bllmConfig) -> None:
        super(bllmEncoder, self).__init__()
        # create attention blocks
        self.layers = nn.ModuleList([
            bllmLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # pass input through each block
        for layer in self.layers:
            x = layer(x)

        return x

class bllmClassifier(nn.Linear):
    def __init__(self, config: bllmConfig) -> None:
        super(bllmClassifier, self).__init__(
            config.hidden_size, config.vocab_size, dtype=config.dtype
        )


class bllm(transformers.PreTrainedModel):

    config_class = bllmConfig

    def __init__(self, config: bllmConfig) -> None:
        super(bllm, self).__init__(config=config)
        # create components
        self.embedding = bllmEmbedding(config)
        self.encoder = bllmEncoder(config)
        self.classifier = bllmClassifier(config)
        # initialize weights
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: None | torch.LongTensor = None,
        attention_mask: None | torch.LongTensor = None
    ):

        emb = self.embedding(input_ids)
        enc = self.encoder(emb)
        logits = self.classifier(enc)

        if labels is not None:

            loss = F.cross_entropy(
                logits.flatten(end_dim=-2),
                labels.flatten(),
                ignore_index=-100
            )

            return bllmOutput(loss=loss, logits=logits)

        return bllmOutput(loss=None, logits=logits)


if __name__ == '__main__':

    model = bllm(
        bllmConfig(
            vocab_size=32000,
            max_position_embeddings=2048,
            hidden_size=24*128,
            num_hidden_layers=24,
            num_attention_heads=24,
            intermediate_size=2048,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
    ).cuda()
    
    out = model.forward(
        input_ids = torch.randint(0, 32000, size=(8, 128)).cuda(),
        labels = torch.randint(0, 32000, size=(8, 128)).cuda()
    )
