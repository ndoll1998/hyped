import os
from dataclasses import dataclass
from typing import Any

import torch
import datasets
import transformers
# data preprocessing
import hyped.data.io.datasets
from hyped.data.pipe import DataPipe
from hyped.data.processors.tokenizers.hf import HuggingFaceTokenizerConfig, HuggingFaceTokenizer, HuggingFaceTokenizerOutputs
from hyped.data.processors.sequence.chunk import ChunkSequenceConfig, ChunkSequence
from hyped.data.processors.sequence.extend import ExtendSequenceConfig, ExtendSequence
# modelling
from hyped.modelling.backends.hf.collators import HuggingFaceDataCollatorWithPadding

from bllm import bllmConfig, bllm
from datasets.distributed import split_dataset_by_node


@dataclass
class DataCollatorForPrefixLanguageModelling(HuggingFaceDataCollatorWithPadding):

    mean: float = 0.5
    std: float = 0.2

    def mask_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:

        # compute sequence length of each example in the current batch
        lengths = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
        # we should always mask the eos token as the model needs to
        # learn when to stop the sequence
        input_ids[input_ids == self.tokenizer.eos_token_id] = self.tokenizer.mask_token_id

        # compute prefix cutoffs
        cutoff = torch.minimum(
            lengths - 1,
            torch.normal(
                mean=lengths * self.mean,
                std=lengths * self.std
            ).long()
        )

        # compute prefix-tuning mask, i.e. which tokens should
        # be masked in the input
        mask = (
            torch.arange(input_ids.size(1)).reshape(1, -1) >= cutoff.reshape(-1, 1)
        )
        # mask the input
        input_ids[mask] = tokenizer.mask_token_id

        return input_ids

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # pad sequences in batch
        batch = super(DataCollatorForPrefixLanguageModelling, self).__call__(batch)
        # copy input ids as labels
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        # mask tokens in the input
        input_ids = self.mask_input_ids(input_ids)
        # model can attend to the whole input sequence including padding tokens
        # but only gets feedback from masked tokens
        batch["input_ids"] = input_ids
        batch["attention_mask"][:, :] = 1
        batch["labels"] = torch.where(
            input_ids != self.tokenizer.pad_token_id, labels, -100
        )
        # return collated batch
        return batch


if __name__ == '__main__':
    
    from argparse import ArgumentParser
    # create argument parser
    parser = ArgumentParser()
    parser.add_argument("--data-files", type=str, nargs="+", required=True)
    parser.add_argument("--data-seed", type=int, default=42, required=False)
    parser.add_argument("--output-dir", type=str, required=True)
    # parse arguments
    args = parser.parse_args()
    
    # set up distributed environment
    #torch.distributed.init_process_group(backend='nccl')
    # get worker information
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    print("### WORKER INIT (%i, %i, %i)" % (local_rank, global_rank, world_size))

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "gpt2",
        bos_token="<b>",
        eos_token="<e>",
        pad_token="<p>",
        mask_token="<m>",
        model_max_length=2048
    )

    # stream dataset from disk
    ds = datasets.load_dataset(
        "hyped.data.io.datasets.typed_json",
        data_files=args.data_files,
        features=datasets.Features({"text": datasets.Value("string")}),
        streaming=True
    )["train"]
    # split dataset by each node to avoid duplicate items
    ds = split_dataset_by_node(
        ds, rank=global_rank, world_size=world_size
    )

    # define data preprocessing pipeline
    pipe = DataPipe(
        [
            HuggingFaceTokenizer(
                HuggingFaceTokenizerConfig(
                    tokenizer=tokenizer,
                    keep_input_features=False,
                    add_special_tokens=False
                )
            ),
            ChunkSequence(
                ChunkSequenceConfig(
                    sequence=HuggingFaceTokenizerOutputs.INPUT_IDS,
                    # allocate one token for initial bos token
                    chunk_size=tokenizer.model_max_length - 1,
                    # chunk_stride=2048
                )
            ),
            ExtendSequence(
                ExtendSequenceConfig(
                    sequence=HuggingFaceTokenizerOutputs.INPUT_IDS,
                    output=HuggingFaceTokenizerOutputs.INPUT_IDS,
                    prepend=[tokenizer.bos_token_id]
                )
            )
        ]
    )

    # apply preprocessing pipeline and shuffle dataset
    ds = pipe.apply(ds)
    ds = ds.shuffle(buffer_size=10000, seed=args.data_seed)

    # create the model
    model = bllm(
        bllmConfig(
            vocab_size=len(tokenizer),
            max_position_embeddings=tokenizer.model_max_length,
            hidden_size=24*128,
            num_hidden_layers=24,
            num_attention_heads=24,
            intermediate_size=2048,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
    )

    # compute the number of trainable parameters
    n_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Number of parameters: %i" % n_params)

    # specify training arguments
    args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        # dataloader_num_workers=2,

        per_device_train_batch_size=10,
        gradient_accumulation_steps=8,
        # approximate number of steps for one epoc
        # with the current setup
        max_steps=(386 * 160000000 * 1.25) // (10 * 8 * 8),
    
        adam_epsilon=1e-6,
        learning_rate=5e-4,
        weight_decay=0.1,        
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=dict(
            use_reentrant=True
        ),
        bf16=True,
        
        save_strategy="steps",
        save_steps=15000,
        save_total_limit=5,

        ddp_find_unused_parameters=False,

        remove_unused_columns=True,

        disable_tqdm=True,
        logging_strategy="steps",
        logging_steps=5,
        report_to="wandb",
        skip_memory_metrics=True,
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
        
        local_rank=local_rank
    )

    # move model to gpu
    model = model.to("cuda:%i" % local_rank)

    # create the trainer instance
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForPrefixLanguageModelling(
            tokenizer=tokenizer,
            padding="max_length",
            max_length=tokenizer.model_max_length
        ),
    )

    # train the model
    trainer.train()
