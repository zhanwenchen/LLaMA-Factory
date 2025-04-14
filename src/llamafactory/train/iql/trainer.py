# trainer.py

import os
import math
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union, cast
from transformers import Trainer, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.optimization import get_scheduler
from tqdm import tqdm
from typing_extensions import override
from types import MethodType

from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from ...hparams import FinetuningArguments


class CustomIQLTrainer(Trainer):
    """Custom Trainer implementing Implicit Q-Learning (IQL) algorithm.

    IQL is an offline RL algorithm that learns a policy from fixed offline data
    without explicit policy optimization steps. Instead, it uses a conservative
    Q-learning objective combined with a value-function-based policy extraction.

    Key components:
      1. Q-function learning via TD learning
      2. Value function learning via expectile regression
      3. Policy extraction via advantage-weighted regression

    Attributes:
        finetuning_args: Configuration for fine-tuning process
        ref_model: Reference model for target computation
        tau: Expectile parameter for value function regression (default 0.7)
        beta: Temperature parameter for advantage weighting (default 1.0)
        alpha: Weight for the value loss component (default 0.005)
        _stored_metrics: Dictionary to track training metrics
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        args: TrainingArguments,
        data_collator: Any,
        train_dataset: Optional[torch.utils.data.Dataset],
        eval_dataset: Optional[torch.utils.data.Dataset],
        tokenizer: Any,
        model_init: Optional[Any],
        compute_metrics: Optional[Any],
        callbacks: Optional[List[Any]],
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR],
        preprocess_logits_for_metrics: Optional[Any],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
    ) -> None:
        """Initialize the IQL trainer with all required components.

        Args:
            model: The model to train
            ref_model: Optional reference model for target value computation
            args: Training arguments
            data_collator: Function to create batches
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer for processing text
            model_init: Optional function to initialize the model
            compute_metrics: Function to compute evaluation metrics
            callbacks: List of callbacks for training events
            optimizers: Tuple of (optimizer, scheduler)
            preprocess_logits_for_metrics: Function to preprocess logits before metric computation
            finetuning_args: Arguments specific to the fine-tuning process
            processor: Optional processor for multimodal inputs
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.finetuning_args = finetuning_args
        self.state = TrainerState()
        self.control = TrainerControl()
        self.ref_model = ref_model
        self.callback_handler = CallbackHandler(callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler)

        # IQL hyperparameters with default values if not specified
        self.tau = getattr(finetuning_args, "iql_tau", 0.7)  # Expectile value for value function regression
        self.beta = getattr(finetuning_args, "iql_beta", 1.0)  # Temperature for advantage weighting
        self.alpha = getattr(finetuning_args, "iql_alpha", 0.005)  # Weight for value loss

        # Dictionary to store metrics during training
        self._stored_metrics: Dict[str, List[float]] = {}

        # Set up processor callback if provided
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # Set up BAdam optimizer if requested
        if getattr(finetuning_args, "use_badam", False):
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # Prepare reference model if provided
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

    @override
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Creates a custom optimizer for the IQL training process.

        Returns:
            The configured optimizer instance
        """
        if self.optimizer is None and self.finetuning_args is not None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Creates a learning rate scheduler for IQL training.

        Args:
            num_training_steps: Total number of training steps
            optimizer: Optimizer to schedule, uses self.optimizer if None

        Returns:
            The configured learning rate scheduler
        """
        if self.finetuning_args is not None:
            create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def expectile_loss(
        self,
        values: torch.Tensor,
        targets: torch.Tensor,
        tau: float = 0.7
    ) -> torch.Tensor:
        """Computes the expectile regression loss for value function training.

        Expectile regression gives asymmetric weights to underestimation and
        overestimation errors, controlled by the tau parameter.

        Args:
            values: Predicted value function outputs
            targets: Target values (typically Q-values)
            tau: Expectile parameter (default 0.7)

        Returns:
            The expectile loss tensor
        """
        delta = targets - values
        weight = torch.where(delta > 0, tau, 1 - tau)
        return torch.mean(weight * (delta ** 2))

    def compute_q_loss(
        self,
        q_values: torch.Tensor,
        target_q_values: torch.Tensor
    ) -> torch.Tensor:
        """Computes the Q-function TD loss.

        Args:
            q_values: Estimated Q-values from the model
            target_q_values: Target Q-values from target network

        Returns:
            The TD loss for the Q-function
        """
        return F.mse_loss(q_values, target_q_values)

    def compute_value_loss(
        self,
        state_values: torch.Tensor,
        q_values: torch.Tensor
    ) -> torch.Tensor:
        """Computes the value function loss using expectile regression.

        Args:
            state_values: Estimated values from the value network
            q_values: Q-values used as targets

        Returns:
            The expectile loss for value function
        """
        return self.expectile_loss(state_values, q_values, self.tau)

    def compute_policy_loss(
        self,
        logits: torch.Tensor,
        advantages: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Computes the policy loss using advantage-weighted regression.

        This implements the key policy extraction mechanism of IQL, where
        the policy is trained to maximize actions with high advantage values.

        Args:
            logits: Model logits for next token prediction
            advantages: Estimated advantages (Q-values - V-values)
            input_ids: Token IDs from the batch
            attention_mask: Attention mask indicating valid positions

        Returns:
            The advantage-weighted policy loss
        """
        # Get log probabilities for all possible actions
        log_probs = F.log_softmax(logits, dim=-1)

        # Create a mask for valid action positions (exclude padding tokens)
        action_mask = (input_ids != IGNORE_INDEX).unsqueeze(-1)

        # Extract log probs for the actions actually taken in the dataset
        chosen_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=input_ids.unsqueeze(-1).clamp(min=0)
        ) * action_mask

        # Weight actions by their exponential advantages
        # This is the key IQL mechanism: better actions get higher weights
        exp_advantages = torch.exp(advantages / self.beta).detach() * action_mask
        # Clamp weights to avoid numerical instability
        exp_advantages = torch.clamp(exp_advantages, max=100.0)

        # Policy loss with advantage weighting
        # We want to maximize log-probs weighted by advantages, so we negate
        policy_loss = -(chosen_log_probs * exp_advantages).sum() / (action_mask.sum() + 1e-8)

        return policy_loss

    def forward_pass(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Any:
        """Performs a forward pass through the model to get all outputs.

        Args:
            model: The model to perform the forward pass with
            batch: Input batch containing tokens and masks

        Returns:
            The complete model outputs
        """
        outputs = model(**batch, return_dict=True, output_hidden_states=True)
        return outputs

    def compute_iql_loss(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes all IQL loss components.

        This function implements the core IQL algorithm by computing the three
        key components: value loss, Q-function loss, and policy loss.

        Args:
            model: The model to use for computing losses
            batch: Input batch of training data

        Returns:
            Tuple of (total loss, metrics dictionary)
        """
        metrics: Dict[str, float] = {}

        # Get model outputs
        outputs = self.forward_pass(model, batch)

        # Handle different output formats
        # If outputs is a tuple (older transformer models might return tuples)
        if isinstance(outputs, tuple):
            logits = outputs[0]
            hidden_states = outputs[1] if len(outputs) > 1 else None
        else:
            # Standard case where outputs is an object
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None

        # Extract value and Q-function estimates
        # These would normally come from separate heads in the model
        state_values = getattr(outputs, "values", None) if not isinstance(outputs, tuple) else None
        q_values = getattr(outputs, "q_values", None) if not isinstance(outputs, tuple) else None

        # Fallback implementation if model doesn't have separate heads
        if state_values is None or q_values is None:
            # Use reference model to get target values if available
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_outputs = self.forward_pass(self.ref_model, batch)

                    # Handle different output formats for reference model too
                    if isinstance(ref_outputs, tuple):
                        ref_logits = ref_outputs[0]
                    else:
                        ref_logits = ref_outputs.logits

                    # Compute log probabilities from reference model
                    log_probs, valid_length = get_batch_logps(
                        logits=ref_logits,
                        labels=batch["labels"]
                    )

                    # Use log probabilities as reward signals
                    target_values = log_probs.unsqueeze(-1)
            else:
                # If no reference model, use a simple placeholder
                target_values = torch.zeros(
                    (batch["input_ids"].size(0), 1),
                    device=logits.device
                )

            # Simple placeholder implementations if model lacks the proper heads
            # For a full implementation, the model architecture should be modified
            # to include these value and Q-function heads
            if hidden_states is None:
                # If hidden_states is not available, use the last dimension of logits
                hidden_states = torch.mean(logits, dim=1)

            # Check if hidden_states is already 2D (batch_size, hidden_size)
            if hidden_states.dim() > 2:
                # Average across sequence length to get (batch_size, hidden_size)
                hidden_states = torch.mean(hidden_states, dim=1)

            # Create simple scalar values from hidden states
            state_values = torch.mean(hidden_states, dim=-1, keepdim=True)
            q_values = state_values + 0.1  # Simple offset as placeholder

        # Compute advantages for policy improvement: Q(s,a) - V(s)
        advantages = q_values - state_values.detach()

        # Compute the three IQL loss components
        value_loss = self.compute_value_loss(state_values, q_values.detach())
        q_loss = self.compute_q_loss(q_values, target_values.detach())
        policy_loss = self.compute_policy_loss(
            logits=logits,
            advantages=advantages,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        # Combine losses with appropriate weights
        total_loss = self.alpha * value_loss + q_loss + policy_loss

        # Record metrics for logging
        metrics["value_loss"] = value_loss.item()
        metrics["q_loss"] = q_loss.item()
        metrics["policy_loss"] = policy_loss.item()
        metrics["advantages_mean"] = advantages.mean().item()
        metrics["state_values_mean"] = state_values.mean().item()
        metrics["q_values_mean"] = q_values.mean().item()

        return total_loss, metrics

    @override
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute the IQL loss for the given inputs.

        This method is called by the Trainer during training and evaluation.

        Args:
            model: The model to compute the loss for
            inputs: Input batch containing tokens and masks
            return_outputs: Whether to return outputs in addition to the loss

        Returns:
            Either the loss tensor alone or a tuple of (loss, outputs)
        """
        loss, metrics = self.compute_iql_loss(model, inputs)

        # Store metrics for logging
        for key, value in metrics.items():
            if key not in self._stored_metrics:
                self._stored_metrics[key] = []
            self._stored_metrics[key].append(value)

        if return_outputs:
            return loss, metrics
        return loss

    def iql_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Runs the IQL training process.

        This method is the main entry point for IQL training, utilizing the
        standard Trainer.train method which calls our overridden compute_loss.

        Args:
            resume_from_checkpoint: Optional path to checkpoint to resume from
        """
        # Use the standard Trainer.train method
        self.train(resume_from_checkpoint=resume_from_checkpoint)

    @override
    def log(self, logs: Dict[str, float]) -> None:
        """Log training metrics, including stored IQL-specific metrics.

        Args:
            logs: Dictionary of metrics to log
        """
        # Add stored metrics to logs
        for key, values in self._stored_metrics.items():
            if values:
                logs[key] = sum(values) / len(values)

        # Clear stored metrics
        self._stored_metrics = {}

        # Call standard logging
        super().log(logs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        """Saves the model checkpoint.

        Args:
            output_dir: Directory to save the model to, defaults to args.output_dir
            _internal_call: Whether this is an internal call to save during training
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.args.should_save:
            self._save(output_dir, state_dict=self.model.state_dict())
