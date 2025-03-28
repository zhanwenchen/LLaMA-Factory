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
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.utils.logging import set_verbosity_info
# from transformers.utils.logging import get_logger
from logging import getLogger
from llamafactory.train.tuner import run_exp  # use absolute import


set_verbosity_info()


class DynamicSaveStepsCallback(TrainerCallback):
    """
    A callback that manages checkpoint saving based on specified target steps.

    This callback will save checkpoints ONLY at specified target steps
    (e.g., 10, 100, 1000) regardless of the original training configuration.
    It blocks all other checkpoint saving to ensure no unwanted checkpoints are created.
    """

    def __init__(self, target_steps: List[int], logger) -> None:
        """
        Initialize the callback with target steps for checkpoint saving.

        Args:
            target_steps: List of training steps at which checkpoints should be saved.
        """
        super().__init__()
        self.target_steps: list[int] = list(dict.fromkeys(target_steps))
        # self.logger = getLogger("launcher")
        self.logger = logger
        self.logger.info(f"DynamicSaveStepsCallback initialized with target steps: {self.target_steps}")

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs
    ) -> "TrainerControl":
        """
        Called at the beginning of each step to control checkpoint saving.

        This is where we decide if we should allow checkpoint saving for this step,
        before the trainer has a chance to set should_save.

        Args:
            args: Training arguments
            state: Current trainer state
            control: Control object
            kwargs: Additional keyword arguments

        Returns:
            Updated control object
        """
        # logger = getLogger("launcher.DynamicSaveStepsCallback.on_step_end")
        current_step = state.global_step
        # self.logger.info(f"DynamicSaveStepsCallback.on_step_end: {current_step=}, {self.target_steps=}")
        # print(f"DynamicSaveStepsCallback.on_step_end: {current_step=}, {self.target_steps=}")
        # breakpoint()
        # if current_step in self.target_steps:
        #     control.should_save = True
        #     self.logger.info(f"DynamicSaveStepsCallback.on_step_end: Will allow checkpoint save at target step {current_step}")
        #     return control
        # if control.should_save is False:
        #     return control
        # if current_step in [5, 10, 20, 50]:
        #     breakpoint()
        # breakpoint()
        control.should_save = False
        # Only allow saving at target steps or the final step
        if current_step in self.target_steps:
            control.should_save = True
            self.logger.info(f"DynamicSaveStepsCallback.on_step_end: Will allow checkpoint. {current_step=}, {self.target_steps=}, {control.should_save=}, {state.max_steps=}")
            # print(f"DynamicSaveStepsCallback.on_step_end: Will allow checkpoint. {current_step=}, {self.target_steps=}, {control.should_save=}, {state.max_steps=}")
            return control
            # Don't modify any existing should_save value - let the trainer decide
        elif current_step >= state.max_steps:
            # control.should_save = False
            # self.logger.info(f"DynamicSaveStepsCallback.on_step_end: Will allow final checkpoint save at step {current_step}")
            # print(f"DynamicSaveStepsCallback.on_step_end: Will allow final checkpoint save at step {current_step}")
            return control
            # Don't modify any existing should_save value - let the trainer decide
        # else:
            # Explicitly block saving at other steps
            # self.logger.error(f"DynamicSaveStepsCallback.on_step_end: Will block checkpoint save at step {current_step} (not a target step)")
            # print(f"DynamicSaveStepsCallback.on_step_end: Will block checkpoint save at step {current_step} (not a target step)")
            # control.should_save = False

        return control

    def on_save(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs
    ) -> "TrainerControl":
        """
        Called after a checkpoint has been saved.

        This method is called after the save has already occurred. We use
        this mainly for logging purposes.

        Args:
            args: Training arguments
            state: Current trainer state
            control: Control object
            kwargs: Additional keyword arguments

        Returns:
            Updated control object
        """
        current_step = state.global_step
        # self.logger.info(f"DynamicSaveStepsCallback.on_save: {current_step=}, {self.target_steps=}, {control.should_save=}")
        # print(f"DynamicSaveStepsCallback.on_save: {current_step=}, {self.target_steps=}, {control.should_save=}")

        # Log the save operation
        if current_step in self.target_steps:
            self.logger.info(f"DynamicSaveStepsCallback.on_save: Completed checkpoint save at target step {current_step}")
            # print(f"DynamicSaveStepsCallback.on_save: Completed checkpoint save at target step {current_step}")
        elif current_step >= state.max_steps:
            breakpoint()
            self.logger.info(f"DynamicSaveStepsCallback.on_save: Completed final checkpoint save at step {current_step}")
            # print(f"DynamicSaveStepsCallback.on_save: Completed final checkpoint save at step {current_step}")
        else:
            # This should not happen if on_step_end is working correctly
            breakpoint()
            raise RuntimeError(f"DynamicSaveStepsCallback.on_save: Unexpected checkpoint saved at step {current_step} (not a target step)")

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
    logger = getLogger('launcher')
    target_steps = []

    # Look for --max_samples_search argument in sys.argv
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ('--max_samples_search', '--orig_max_samples_search'):
            if i + 1 < len(sys.argv):
                try:
                    # Parse the comma-separated list of steps
                    steps_str = sys.argv[i + 1]
                    # Ensure we parse all values correctly
                    target_steps = [int(step.strip()) for step in steps_str.split(',')]
                    logger.info(f"launcher: Parsed custom save steps: {target_steps}")
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

    # Ensure we have a complete list of steps
    if target_steps:
        logger.error(f"launcher: Will create checkpoints ONLY at steps: {target_steps}")
    else:
        logger.warning("launcher: No checkpoint steps provided with --max_samples_search")

    return target_steps


def launch(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None) -> None:
    """
    Launch the training experiment with specified args and callbacks.

    Args:
        args: Dictionary of arguments to pass to run_exp
        callbacks: List of callbacks to pass to run_exp
    """
    callbacks = callbacks or []
    run_exp(args=args, callbacks=callbacks)


if __name__ == "__main__":
    logger = getLogger('launcher')

    # Extract and remove custom max_samples_search before LlamaFactory's parser sees it
    target_steps = extract_custom_args()

    # Initialize callbacks list
    callbacks = []

    # Add DynamicSaveStepsCallback only if target steps were provided
    # logger.info(f"launcher: Using DynamicSaveStepsCallback with steps: {target_steps}")
    callbacks.append(DynamicSaveStepsCallback(target_steps=target_steps, logger=logger))

    # Launch with remaining args and callbacks
    launch(args=None, callbacks=callbacks)
