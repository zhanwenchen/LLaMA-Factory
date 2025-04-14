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

from typing import TYPE_CHECKING, List, Optional, Tuple

from ...data import MultiModalDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomIQLTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_iql(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
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
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="iql", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)

    data_collator = MultiModalDataCollatorForSeq2Seq(template=template, **tokenizer_module)

    # Create reference model if needed
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True) if finetuning_args.use_ref_model else None

    # Initialize our Trainer with all required arguments
    trainer = CustomIQLTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module.get("eval_dataset", None),
        tokenizer=tokenizer,
        model_init=None,
        compute_metrics=None,
        callbacks=callbacks,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        processor=tokenizer_module.get("processor", None),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.iql_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
