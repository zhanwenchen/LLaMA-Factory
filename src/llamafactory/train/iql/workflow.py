# workflow.py

from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union

from transformers import TrainingArguments, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorWithPadding

from .trainer import CustomIQLTrainer
from ...data import get_dataset, get_template_and_fix_tokenizer
from ...model import load_model, load_tokenizer, load_adapter
from ...extras.callbacks import LogCallback
from ..trainer_utils import create_modelcard_and_push

if TYPE_CHECKING:
    from transformers import PreTrainedModel, TrainerCallback
    from ...hparams import ModelArguments, DataArguments, FinetuningArguments


def run_iql(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: TrainingArguments,
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
) -> "PreTrainedModel":
    """Run the Implicit Q-Learning (IQL) training process.

    IQL is an offline RL algorithm that learns a policy from a fixed dataset of experiences
    without explicit policy optimization steps. Instead, it uses conservative Q-learning
    with value-function-based policy extraction via advantage weighting.

    The workflow follows these steps:
    1. Load and prepare model, tokenizer, and datasets
    2. Set up the IQL trainer with all necessary components
    3. Run training and evaluation
    4. Save the model and optionally push to Hugging Face Hub

    Args:
        model_args: Arguments for model configuration
        data_args: Arguments for dataset and preprocessing
        training_args: Arguments for training process
        finetuning_args: Arguments for fine-tuning method
        callbacks: Optional list of callbacks for training events

    Returns:
        The fine-tuned model
    """
    # Load tokenizer and template for data formatting
    tokenizer = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load dataset with appropriate transformations for IQL
    dataset = get_dataset(template, model_args, data_args, training_args, stage="iql")

    # Load the base model (policy model)
    is_trainable = training_args.do_train and not training_args.no_cuda
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable)

    # Load reference model if specified (typically a pre-trained/SFT model)
    ref_model = None
    if finetuning_args.ref_model_name_or_path is not None and finetuning_args.use_ref_model:
        ref_model = load_model(
            tokenizer,
            model_args,
            finetuning_args,
            False,  # Reference model doesn't need to be trainable
            finetuning_args.ref_model_name_or_path
        )
        # Load LoRA adapter or other adapters if using parameter-efficient fine-tuning
        if finetuning_args.finetuning_type != "full" and finetuning_args.ref_adapter_path is not None:
            ref_model = load_adapter(ref_model, finetuning_args.ref_adapter_path, model_args, finetuning_args)

    # Prepare data collator with appropriate padding
    data_collator_kwargs: Dict[str, Any] = {
        "pad_to_multiple_of": 8 if training_args.fp16 or training_args.bf16 else None
    }

    # Ensure the tokenizer has a pad token
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = 0  # Use a default value if not set

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, **data_collator_kwargs)

    # Set up callbacks for logging and other training events
    if callbacks is None:
        callbacks = []

    if not any(isinstance(cb, LogCallback) for cb in callbacks):
        callbacks.append(LogCallback())

    # Initialize the IQL trainer with all components
    trainer = CustomIQLTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        train_dataset=dataset.get("train_dataset"),
        eval_dataset=dataset.get("eval_dataset"),
        tokenizer=tokenizer,
        callbacks=callbacks,
        # Pass default values for required arguments
        model_init=None,
        compute_metrics=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        processor=None,
    )

    # Run the IQL training process
    if training_args.do_train:
        trainer.iql_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.save_state()

        # Push to Hugging Face Hub if requested
        if finetuning_args.push_to_hub:
            create_modelcard_and_push(
                model_args,
                data_args,
                finetuning_args,
                training_args
            )

    # Run evaluation if requested
    if training_args.do_eval:
        trainer.evaluate()

    return model
