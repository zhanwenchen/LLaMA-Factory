"""
Trainer for IQL (Implicit Q-Learning) fine-tuning.
Adapted from unified implementation with improvements to organization and clarity.
"""

from typing import Dict, Optional, Tuple, Union
from torch import (
    bfloat16 as torch_bfloat16,
    vstack as torch_vstack,
    zeros,
    stack,
    Tensor,
    as_tensor,
    no_grad,
    min as torch_min,
    exp as torch_exp,
    where,
    addcmul as torch_addcmul,
)
from torch.nn.functional import mse_loss
from torch.nn import Module
from torch.optim import Adam, Optimizer
from transformers import Trainer, PreTrainedTokenizer, PreTrainedModel, ProcessorMixin, TrainerState
from peft import PeftModel
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer
from .utils import mlp, llama3_prompt
from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomIQLTrainer(Trainer):
    """
    Trainer for IQL (Implicit Q-Learning) fine-tuning.

    This implementation follows the canonical IQL algorithm with separate networks for:
    - Actor (LLM policy model)
    - Critic (Q-function, two networks for stability)
    - Value function

    The training follows the standard IQL update procedure:
    1. Update Value Network using expectile regression
    2. Update Actor Network based on advantages
    3. Update Critic Networks using TD learning
    4. Periodically sync target networks
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, Module],
        finetuning_args: FinetuningArguments,
        processor: Optional[ProcessorMixin] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)

        self.finetuning_args = finetuning_args
        self.tokenizer: PreTrainedTokenizer = kwargs["tokenizer"]

        # IQL hyperparameters
        self.hard_update_every = 10  # How often to sync target networks
        self.gamma = 0.99  # Discount factor
        dtype = next(model.parameters()).dtype
        self.temperature = as_tensor(3.0, device=self.args.device, dtype=dtype)
        self.expectile = 0.8  # Expectile value for value learning

        # Set up the transformer model and get embedding dimension
        self.model = model  # Actor model (LLM)
        self.actor_embedding = self._get_input_embeddings(model)
        emb_dim = self.actor_embedding.embedding_dim
        hidden_dim = 256
        out_dim = 1
        layers = 2

        accelerator = self.accelerator
        device = accelerator.device

        # Create critic and value networks
        dtype = model.dtype
        self.critic1 = mlp(emb_dim, hidden_dim, out_dim, layers, device, dtype)
        self.critic2 = mlp(emb_dim, hidden_dim, out_dim, layers, device, dtype)
        self.critic1_target = mlp(emb_dim, hidden_dim, out_dim, layers, device, dtype)
        self.critic2_target = mlp(emb_dim, hidden_dim, out_dim, layers, device, dtype)
        self.value = mlp(emb_dim, hidden_dim, out_dim, layers, device, dtype)

        # Initialize target networks
        self.sync_targets()

        # Create optimizers for critic and value networks
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=1e-5)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=1e-5)
        self.value_optimizer = Adam(self.value.parameters(), lr=1e-5)

    def _get_input_embeddings(self, model: Union[PreTrainedModel, Module]):
        """Get input embeddings from model, handling different model types correctly."""
        if isinstance(model, PeftModel):
            return model.base_model.get_input_embeddings()
        else:
            return model.get_input_embeddings()

    @no_grad()
    def sync_targets(self) -> Tuple[Module, Module]:
        """Synchronize target networks with current networks."""
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        return self.critic1_target, self.critic2_target

    def embed_pair(self, state: str, action: str) -> Tensor:
        """Embed a state-action pair using the model's embeddings."""
        device = self.args.device
        ids = self.tokenizer(state + " " + action, return_tensors="pt").to(device).input_ids
        return self.actor_embedding(ids).mean(dim=1)

    def expectile_loss(self, diff: Tensor, expectile: float = 0.8) -> Tensor:
        """
        Computes the expectile loss for value function regression.

        Args:
            diff: Difference between Q-values and value predictions
            expectile: Expectile parameter (tau in paper)

        Returns:
            Expectile loss tensor
        """
        expectile_tensor = as_tensor(expectile, device=diff.device, dtype=diff.dtype)
        weight = where(diff > 0, expectile_tensor, 1 - expectile_tensor)
        return weight * diff * diff

    def create_optimizer(self) -> Optimizer:
        """Create optimizer for the policy model."""
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    # def iql_train(self, resume_from_checkpoint: Optional[str] = None) -> TrainerState:
    #     """
    #     Main training entry point for IQL, similar to Trainer.train.

    #     Args:
    #         resume_from_checkpoint: Optional checkpoint to resume from

    #     Returns:
    #         The final TrainerState
    #     """
    #     # Prepare all auxiliary networks with accelerator
    #     # self.accelerator.prepare(
    #     #     self.critic1, self.critic2, self.value,
    #     #     self.critic1_target, self.critic2_target,
    #     #     self.critic1_optimizer, self.critic2_optimizer, self.value_optimizer
    #     # )

    #     # Start normal trainer training loop which will call our custom training_step
    #     return super().train(resume_from_checkpoint=resume_from_checkpoint)

    def training_step(self, model, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Perform a single IQL training step.

        This method implements the core IQL algorithm update:
        1. Update Value Network using expectile regression
        2. Update Actor Network based on advantages
        3. Update Critic Networks using TD learning
        4. Periodically sync target networks

        Args:
            model: The model to train
            inputs: The inputs to the model

        Returns:
            The loss tensor
        """
        step = self.state.global_step

        # Extract data from batch
        # assert isinstance(inputs, dict)
        # Handle the format from dataset
        obs, next_obs, acts, next_acts, rewards, dones, _ = inputs

        # obs = inputs["observation"]
        # next_obs = inputs["next_observation"]
        # acts = inputs["action"]
        # next_acts = inputs["next_action"]
        # rewards = inputs["reward"]
        # dones = inputs["done"]

        # Embed state-action pairs
        accelerator = self.accelerator
        device = accelerator.device
        # dtype = model.dtype
        dtype = torch_bfloat16

        # Embed state-action pairs - using your optimized approach
        crit_in = torch_vstack([self.embed_pair(s, a) for s, a in zip(obs, acts)])
        crit_next = torch_vstack([self.embed_pair(s, a) for s, a in zip(next_obs, next_acts)])

        # Convert rewards and dones to tensors - using as_tensor for efficiency
        # assert isinstance(rewards, Tensor)
        rewards = as_tensor(rewards, device=device, dtype=dtype).unsqueeze(-1)
        dones = as_tensor(dones, device=device, dtype=dtype).unsqueeze(-1)

        # 1. Update Value Network
        self.value_optimizer.zero_grad(set_to_none=True)

        # Compute Q values using TARGET networks for value update
        with no_grad():
            q1_target = self.critic1_target(crit_in)
            q2_target = self.critic2_target(crit_in)
            min_q = torch_min(q1_target, q2_target)

        v_pred = self.value(crit_in)
        loss_v = self.expectile_loss(min_q - v_pred, self.expectile).mean()

        accelerator.backward(loss_v)
        self.value_optimizer.step()

        # 2. Update Actor Network (Policy)
        # Recompute advantage using updated value network (with no gradient flow)
        with no_grad():
            v_pred = self.value(crit_in)
            # Apply exponential weighting to advantages using temperature - using torch_exp
            temperature = self.temperature
            exp_a = torch_exp((min_q - v_pred) * temperature).clamp(max=temperature).squeeze(-1)

        # Actor loss (weighted by advantages) - using your approach from iql_agent_mine.py
        actor_losses = []
        for s, a, adv in zip(obs, acts, exp_a):
            prompt = llama3_prompt(s, a)
            enc = self.tokenizer(prompt, return_tensors="pt").to(device)
            labels = enc.input_ids.clone()

            # Set labels for system and user parts to -100 (ignore in loss)
            action_ids = self.tokenizer(a, add_special_tokens=False, return_tensors="pt")["input_ids"]
            labels[:, :-action_ids.shape[1]] = IGNORE_INDEX

            outputs = model(**enc, labels=labels)
            actor_losses.append(adv * outputs.loss)

        loss_a = stack(actor_losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss_a)
        self.optimizer.step()

        # 3. Update Critic Networks
        # Compute targets using VALUE network for next states - using optimized calculation
        with no_grad():
            next_v = self.value(crit_next)
            # Using torch_addcmul for optimized reward + gamma * (1-done) * next_v calculation
            q_target = torch_addcmul(input=rewards, tensor1=(1 - dones), tensor2=next_v, value=self.gamma)

        # Update critic 1
        self.critic1_optimizer.zero_grad(set_to_none=True)
        q1 = self.critic1(crit_in)
        loss_critic1 = mse_loss(q1, q_target)
        accelerator.backward(loss_critic1)
        self.critic1_optimizer.step()

        # Update critic 2
        self.critic2_optimizer.zero_grad(set_to_none=True)
        q2 = self.critic2(crit_in)
        loss_critic2 = mse_loss(q2, q_target)
        accelerator.backward(loss_critic2)
        self.critic2_optimizer.step()

        # Periodically update target networks
        if step % self.hard_update_every == 0:
            self.sync_targets()

        # Combine losses for logging
        combined_loss = loss_a + 0.5 * (loss_critic1 + loss_critic2) + loss_v

        self.log({
            "actor_loss": loss_a.item(),
            "critic_loss": 0.5 * (loss_critic1.item() + loss_critic2.item()),
            "value_loss": loss_v.item(),
            "total_loss": combined_loss.item()
        })

        return combined_loss
