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

import sys
from typing import Any, Dict, List, Optional
# from argparse import ArgumentParser, Namespace
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.utils.logging import get_logger
from llamafactory.train.tuner import run_exp  # use absolute import
# from transformers.utils import logging
# logging.set_verbosity_info()

# from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
# import os


class DynamicSaveStepsCallback(TrainerCallback):
    """
    A callback that manages checkpoint saving based on specified target steps.

    This callback will save checkpoints at specified target steps
    (e.g., 10, 100, 1000) regardless of the original max_samples_search configuration.
    """

    def __init__(self, target_steps: List[int]):
        """
        Initialize the DynamicSaveStepsCallback with target steps.

        Args:
            target_steps: List of specific steps where we want to save checkpoints
        """
        self.target_steps = sorted(target_steps)
        self.next_save_idx = 0
        self.logger = get_logger('launcher')
        self.logger.error(f"launcher: DynamicSaveStepsCallback initialized with target steps: {self.target_steps}")

    def on_save(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs
    ) -> "TrainerControl":
        """
        Called when the trainer is about to save a checkpoint.

        This method checks if the current step is our next target step to save.
        If it is, we allow the save to proceed. Otherwise, we check if we need
        to force a save for our target steps.

        Args:
            args: Training arguments
            state: Current trainer state
            control: Control object
            **kwargs: Additional keyword arguments

        Returns:
            Control object determining whether a save should happen
        """
        # Skip if we've used all our target steps
        if self.next_save_idx >= len(self.target_steps):
            self.logger.error(f"Out of target_steps. {self.next_save_idx=}, {self.target_steps=}")
            return control

        current_step = state.global_step
        next_target = self.target_steps[self.next_save_idx]

        # If we've passed our target step without saving, force a save now
        if current_step >= next_target:
            self.logger.error(f"launcher: Target step {next_target} reached or passed at step {current_step}, saving checkpoint")

            # Advance to the next target step
            self.next_save_idx += 1

            # Ensure save happens
            control.should_save = True

            # If we're at exactly the target step, log it specially
            if current_step == next_target:
                self.logger.error(f"launcher: Checkpoint saved at step {current_step} (precise target step)")
        else:
            # Skip the save
            control.should_save = False
            self.logger.error(f"launcher: Skipping save. {current_step=}, {next_target=}, {self.target_steps=}")

        return control


def extract_custom_args() -> List[int]:
    """
    Extract and remove custom max_samples_search argument from sys.argv.

    This function looks for the --max_samples_search argument in the command line,
    parses its value, removes it from sys.argv to prevent conflicts with
    LlamaFactory's argument parser, and returns the parsed values.

    Returns:
        List[int]: The list of target steps parsed from --max_samples_search argument
    """
    logger = get_logger('launcher')
    target_steps = []

    # Look for --max_samples_search argument in sys.argv
    i = 1
    # breakpoint()
    while i < len(sys.argv):
        # if sys.argv[i] in ('--max_samples_search', '--max_samples_search_list'):
        if sys.argv[i] == '--max_samples_search':
            if i + 1 < len(sys.argv):
                try:
                    # Parse the comma-separated list of steps
                    steps_str = sys.argv[i + 1]
                    target_steps = [int(step.strip()) for step in steps_str.split(',')]
                    logger.error(f"launcher: Parsed custom save steps: {target_steps}")

                    # Remove both the argument and its value from sys.argv
                    sys.argv.pop(i)  # Remove --max_samples_search
                    sys.argv.pop(i)  # Remove its value
                    continue  # Don't increment i after removal
                except ValueError:
                    logger.warning(f"launcher: Invalid value for --max_samples_search: '{sys.argv[i + 1]}'. Expected comma-separated integers.")
                    # Leave it for LlamaFactory to handle the error
            else:
                logger.warning("launcher: --max_samples_search argument provided without a value")
                # Remove the dangling argument
                sys.argv.pop(i)
                continue
        i += 1

    return target_steps


def launch(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    """
    Launch the training experiment with specified args and callbacks.

    Args:
        args: Dictionary of arguments to pass to run_exp
        callbacks: List of callbacks to pass to run_exp
    """
    run_exp(args=args, callbacks=callbacks)


if __name__ == "__main__":
    logger = get_logger('launcher')
    # Extract and remove custom max_samples_search before LlamaFactory's parser sees it
    target_steps = extract_custom_args()

    # Initialize callbacks list
    callbacks = []

    # Add DynamicSaveStepsCallback if target steps were provided
    # if target_steps:
    logger.error(f"launcher: Using DynamicSaveStepsCallback with steps: {target_steps}")
    callbacks.append(DynamicSaveStepsCallback(target_steps=target_steps))

    # Launch with remaining args and callbacks
    launch(args=None, callbacks=callbacks)
