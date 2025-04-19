# workflow.py

# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainingArguments, TrainerCallback
from peft import PeftModel
from ...extras.ploting import plot_loss
from ...extras import logging
from ...model import load_model, load_tokenizer
from .trainer import CustomIQLTrainer
from .utils import load_transitions_from_goldsequences, collate
from ...hparams import DataArguments, FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def run_iql(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    callbacks: Optional[List[TrainerCallback]] = None,
) -> None:
    """
    Run the Implicit Q-Learning (IQL) training process.

    Args:
        model_args: Arguments pertaining to model configuration.
        data_args: Arguments pertaining to data processing.
        training_args: Arguments pertaining to training configuration.
        finetuning_args: Arguments pertaining to finetuning configuration.
        callbacks: Optional list of trainer callbacks.
    """
    # Load model and tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    transitions = load_transitions_from_goldsequences(data_args.dataset[0])
    train_dataset = transitions
    eval_dataset = None  # No evaluation dataset for now

    # Load model without value head - our implementation handles value function separately
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # Log model type for debugging
    if isinstance(model, PeftModel):
        logger.info_rank0(f"Using PeftModel type: {type(model).__name__}")
    else:
        logger.info_rank0(f"Using model type: {type(model).__name__}")

    # Initialize the IQL trainer
    trainer = CustomIQLTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=collate,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        processor=tokenizer_module["processor"],
        callbacks=callbacks,
        optimizers=(None, None),  # We'll create custom optimizers in the trainer
    )

    batch_size = training_args.per_device_train_batch_size
    dl: DataLoader = DataLoader(transitions, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=32)

    accelerator = Accelerator()
    trainer, dl, trainer.optimizer, trainer.critic1_optimizer, trainer.critic2_optimizer, trainer.value_optimizer = accelerator.prepare(trainer, dl, trainer.optimizer, trainer.critic1_optimizer, trainer.critic2_optimizer, trainer.value_optimizer)

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "actor_loss", "critic_loss", "value_loss"])

    # Evaluation
    if training_args.do_eval and eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
