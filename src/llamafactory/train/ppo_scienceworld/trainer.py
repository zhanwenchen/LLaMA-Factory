"""
PPO trainer implementation that interfaces with Agentboard's ScienceWorld environment
to collect external rewards for reinforcement learning.
"""

import math
import os
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import (
    dump_layernorm,
    restore_layernorm,
    replace_model,
    prepare_interaction_for_reward,
    get_scienceworld_rewards,
)


logger = logging.get_logger(__name__)


class ScienceworldPPOTrainer(PPOTrainer, Trainer):
    """
    PPO Trainer specialized for Agentboard's ScienceWorld environment.
    Directly interfaces with the environment to compute rewards.
    """

    def __init__(
        self,
        model_args,
        training_args,
        finetuning_args,
        generating_args,
        callbacks,
        model,
        reward_model=None,
        ref_model=None,
        tokenizer=None,
        processor=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        reward_weights=None,
        scienceworld_env=None,
    ) -> None:
        """
        Initialize ScienceworldPPOTrainer.

        Args:
            model_args: Model configuration parameters.
            training_args: Training configuration parameters.
            finetuning_args: Fine-tuning configuration parameters.
            generating_args: Text generation configuration parameters.
            callbacks: List of callbacks for training events.
            model: Model to be trained.
            reward_model: Optional reward model (not used if using environment rewards).
            ref_model: Optional reference model for KL divergence calculation.
            tokenizer: Tokenizer for the model.
            processor: Optional processor for multimodal inputs.
            data_collator: Data collator for batching.
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
            reward_weights: Optional dictionary of weights for different reward components.
            scienceworld_env: Optional pre-initialized ScienceWorld environment.
        """
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        # Store reward weights - how much each metric contributes to the final reward
        self.reward_weights = reward_weights or {
            "progress_rate": 0.6,
            "success_rate": 0.3,
            "grounding_accuracy": 0.1,
        }

        # Check if we're using environment rewards (based on the finetuning_args)
        self.use_env_rewards = (
            finetuning_args.use_scienceworld_env or
            finetuning_args.reward_model_type == "env"
        )

        # Store or initialize the ScienceWorld environment
        self.scienceworld_env = scienceworld_env
        if self.use_env_rewards and self.scienceworld_env is None:
            try:
                from agentboard.environment import load_environment

                # Find the correct path for ScienceWorld data
                # First check in the agentboard installation directory
                agentboard_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))), "agentboard", "data", "scienceworld")

                # If not found there, try the current working directory
                if not os.path.exists(agentboard_data_path):
                    agentboard_data_path = os.path.join(os.getcwd(), "agentboard", "data", "scienceworld")

                # Fallback to the first available test file
                label_path = None
                if os.path.exists(agentboard_data_path):
                    test_files = [f for f in os.listdir(agentboard_data_path) if f.endswith('.jsonl')]
                    if test_files:
                        label_path = os.path.join(agentboard_data_path, test_files[0])
                        logger.info_rank0(f"Using ScienceWorld test file: {label_path}")

                # Default configuration for ScienceWorld
                env_config = {
                    "serverPath": None,  # Use default
                    "envStepLimit": 50,
                    "label_path": label_path,
                    "simplefied": False  # This is intentionally misspelled to match Agentboard's API
                }

                # Try to initialize the environment with the configuration
                try:
                    self.scienceworld_env = load_environment("scienceworld", env_config)
                    logger.info_rank0("Successfully loaded ScienceWorld environment")
                except FileNotFoundError:
                    # If that fails, try initializing without a label path
                    env_config.pop("label_path", None)
                    self.scienceworld_env = load_environment("scienceworld", env_config)
                    logger.info_rank0("Successfully loaded ScienceWorld environment without label file")

            except ImportError:
                logger.warning_rank0(
                    "ScienceWorld environment not found. Make sure agentboard is installed: "
                    "pip install -e /path/to/agentboard"
                )
                self.scienceworld_env = None
            except Exception as e:
                logger.error(f"Failed to initialize ScienceWorld environment: {str(e)}")
                self.scienceworld_env = None

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()  # patch for deepspeed training

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)

        if finetuning_args.reward_model_type == "full" and not self.use_env_rewards:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.

        Args:
            resume_from_checkpoint: Optional path to a checkpoint to resume training from.

        Returns:
            The final trainer state.
        """
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = float("inf")
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running PPO training with ScienceWorld rewards *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {:,}".format(
                total_train_batch_size
            )
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")
        logger.info_rank0(f"  Using environment rewards: {self.use_env_rewards}")

        if self.use_env_rewards:
            logger.info_rank0(f"  Reward weights = {self.reward_weights}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # Get inputs
            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch_queries, mini_batch_responses = self.get_inputs(
                    batch[idx : idx + self.config.mini_batch_size]
                )
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses)
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)

            # Run PPO step
            self.model.train()
            stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch_for_logging = {
                        "query": self.tokenizer.batch_decode(queries, skip_special_tokens=True),
                        "response": self.tokenizer.batch_decode(responses, skip_special_tokens=True),
                    }
                    self.log_stats(stats, batch_for_logging, rewards)
                except Exception as e:
                    logger.warning_rank0(f"Failed to save stats due to error: {e}")

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)
        return self.state

    def create_optimizer(
        self,
        model,
        training_args,
        finetuning_args,
    ):
        """
        Create a custom optimizer for the model.

        Args:
            model: The model to optimize.
            training_args: Training arguments.
            finetuning_args: Fine-tuning arguments.

        Returns:
            The optimizer.
        """
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    def create_scheduler(
        self, training_args, num_training_steps: int, optimizer
    ):
        """
        Create a learning rate scheduler.

        Args:
            training_args: Training arguments.
            num_training_steps: Total number of training steps.
            optimizer: The optimizer.

        Returns:
            The learning rate scheduler.
        """
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: Dict[str, "torch.Tensor"]) -> Tuple[List["torch.Tensor"], List["torch.Tensor"]]:
        """
        Generates model's responses given queries.

        Args:
            batch: Batch of data.

        Returns:
            Tuple of query tensors and response tensors.
        """
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            generate_output: "torch.Tensor" = unwrapped_model.generate(
                generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            )
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right

        return queries, responses

    @torch.no_grad()
    def get_rewards(
        self,
        queries: List["torch.Tensor"],
        responses: List["torch.Tensor"],
    ) -> List["torch.Tensor"]:
        """
        Computes rewards either using the environment or a reward model.

        Args:
            queries: List of tensor queries.
            responses: List of tensor responses.

        Returns:
            List of reward tensors.
        """
        # If using environment rewards
        if self.use_env_rewards:
            # Decode the token IDs to get text
            decoded_queries = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
            decoded_responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            # Process each query-response pair
            interactions = []
            for query, response in zip(decoded_queries, decoded_responses):
                interactions.append(prepare_interaction_for_reward(
                    query=query,
                    response=response
                ))

            # Get rewards from ScienceWorld environment
            rewards_list = get_scienceworld_rewards(
                interactions=interactions,
                reward_weights=self.reward_weights,
                env=self.scienceworld_env
            )

            # Convert to tensors
            rewards = [torch.tensor(r, dtype=torch.float) for r in rewards_list]

        # Otherwise use the reward model
        else:
            device = self.current_device
            queries_on_device = [query.to(device) for query in queries]
            responses_on_device = [response.to(device) for response in responses]

            scores = []
            for query, response in zip(queries_on_device, responses_on_device):
                if self.finetuning_args.reward_model_type == "lora":
                    score = self.reward_model(
                        input_ids=torch.cat([query, response])[None, :],
                        return_dict=True
                    ).logits.detach().float().mean()
                elif self.finetuning_args.reward_model_type == "full":
                    score = self.reward_model(
                        input_ids=torch.cat([query, response])[None, :].to(device),
                        attention_mask=torch.ones_like(torch.cat([query, response])[None, :], device=device),
                        return_dict=True
                    ).logits.detach().float().mean()
                else:  # api
                    score = self.reward_model(query, response)

                scores.append(score)

            rewards = scores

        return rewards

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model,
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: Dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        """
        Calculates model outputs in multiple batches.

        Args:
            model: The model to use for forward pass.
            queries: List of query tensors.
            responses: List of response tensors.
            model_inputs: Dict of model inputs.
            return_logits: Whether to return logits.
            response_masks: Optional masks for responses.

        Returns:
            Tuple of logprobs, logits, values, and masks.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """
        Saves model checkpoint.

        Args:
            output_dir: Directory to save the model to.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
