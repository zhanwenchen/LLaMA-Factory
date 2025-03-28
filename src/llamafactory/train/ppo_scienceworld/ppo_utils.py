"""
Utility functions for PPO training with ScienceWorld environment rewards.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, Set, cast
from dataclasses import dataclass
import os
import torch
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from scienceworld import ScienceWorldEnv
from agentboard.environment.scienceworld_env import Scienceworld
from ...extras import logging


logger = logging.get_logger(__name__)


class ScienceWorldError(Exception):
    """Base exception for ScienceWorld-related errors."""
    pass


class TaskNotFoundError(ScienceWorldError):
    """Exception raised when a task could not be found or identified."""
    pass


class TaskLoadError(ScienceWorldError):
    """Exception raised when a task could not be loaded."""
    pass


class InvalidActionError(ScienceWorldError):
    """Exception raised when an action was invalid."""
    pass


@dataclass
class RewardResult:
    """Data class to hold reward computation results."""
    progress_rate: float
    success_rate: float
    grounding_accuracy: float
    valid_actions: int
    total_actions: int
    final_reward: float


# Valid ScienceWorld task names based on the error message
VALID_SCIENCEWORLD_TASKS: Set[str] = {
    'boil', 'change-the-state-of-matter-of', 'chemistry-mix',
    'chemistry-mix-paint-secondary-color', 'chemistry-mix-paint-tertiary-color',
    'find-animal', 'find-living-thing', 'find-non-living-thing', 'find-plant',
    'freeze', 'grow-fruit', 'grow-plant', 'identify-life-stages-1',
    'identify-life-stages-2', 'inclined-plane-determine-angle',
    'inclined-plane-friction-named-surfaces', 'inclined-plane-friction-unnamed-surfaces',
    'lifespan-longest-lived', 'lifespan-longest-lived-then-shortest-lived',
    'lifespan-shortest-lived', 'measure-melting-point-known-substance',
    'measure-melting-point-unknown-substance', 'melt', 'mendelian-genetics-known-plant',
    'mendelian-genetics-unknown-plant', 'power-component',
    'power-component-renewable-vs-nonrenewable-energy', 'test-conductivity',
    'test-conductivity-of-unknown-substances', 'use-thermometer'
}

# Default fallback task if no valid task is identified
DEFAULT_TASK_NAME: str = 'use-thermometer'

# Mapping of common task keywords to their formal task names
TASK_KEYWORD_MAPPING: Dict[str, str] = {
    'thermometer': 'use-thermometer',
    'temperature': 'use-thermometer',
    'temperature reading': 'use-thermometer',
    'measuring': 'use-thermometer',
    'measure': 'use-thermometer',

    'melt': 'melt',
    'melting': 'melt',
    'melting point': 'measure-melting-point-known-substance',

    'freeze': 'freeze',
    'freezing': 'freeze',
    'ice': 'freeze',

    'boil': 'boil',
    'boiling': 'boil',

    'conductivity': 'test-conductivity',
    'test conductivity': 'test-conductivity',
    'conduct electricity': 'test-conductivity',

    'mix': 'chemistry-mix',
    'mixing': 'chemistry-mix',

    'paint': 'chemistry-mix-paint-secondary-color',
    'color mixing': 'chemistry-mix-paint-secondary-color',
    'secondary color': 'chemistry-mix-paint-secondary-color',
    'tertiary color': 'chemistry-mix-paint-tertiary-color',

    'inclined plane': 'inclined-plane-determine-angle',
    'angle': 'inclined-plane-determine-angle',
    'friction': 'inclined-plane-friction-named-surfaces',

    'grow': 'grow-plant',
    'growing': 'grow-plant',
    'plant': 'grow-plant',
    'fruit': 'grow-fruit',

    'animal': 'find-animal',
    'living': 'find-living-thing',
    'non-living': 'find-non-living-thing',
    'plant': 'find-plant',

    'life stages': 'identify-life-stages-1',
    'life cycle': 'identify-life-stages-1',

    'lifespan': 'lifespan-longest-lived',
    'longest lived': 'lifespan-longest-lived',
    'shortest lived': 'lifespan-shortest-lived',

    'genetics': 'mendelian-genetics-known-plant',
    'mendel': 'mendelian-genetics-known-plant',

    'power': 'power-component',
    'energy': 'power-component',
    'renewable': 'power-component-renewable-vs-nonrenewable-energy',
    'non-renewable': 'power-component-renewable-vs-nonrenewable-energy',

    'state of matter': 'change-the-state-of-matter-of',
    'state change': 'change-the-state-of-matter-of'
}


class LabelFreeScienceworld:
    """
    A wrapper around ScienceWorldEnv that doesn't require labels.
    This class replicates the necessary functionality of the Scienceworld class
    but without depending on label files.
    """

    def __init__(
        self,
        server_path: Optional[str] = None,
        env_step_limit: int = 50
    ) -> None:
        """
        Initialize the LabelFreeScienceworld wrapper.

        Args:
            server_path: Optional path to the ScienceWorld server executable.
            env_step_limit: Maximum number of steps allowed in the environment.
        """
        self.env = ScienceWorldEnv("", server_path, envStepLimit=env_step_limit)
        self.reward = 0.0
        self.done = False
        self.modified_goal = ""
        self.selected_obs = []
        self.finished_sub_goal = []

    def load(self, task_name: str, var: int, simplification: str) -> Any:
        """
        Load a ScienceWorld task without requiring labels.

        Args:
            task_name: The name of the task to load.
            var: The task variation.
            simplification: The difficulty level.

        Returns:
            The loaded environment.
        """
        env = self.env.load(task_name, var, simplificationStr=simplification)
        # Extract goal information from the environment
        task_description = self.env.getTaskDescription()
        gold_action_sequence = self.env.getGoldActionSequence()

        # Set default selected_obs based on the goal description
        self.selected_obs = [self.env.getGoalProgressStr()]
        self.modified_goal = task_description
        self.finished_sub_goal = [0 for _ in range(len(self.selected_obs))]

        logger.info(f"Loaded task without labels: {task_name} (var: {var}, simplification: {simplification})")
        logger.info(f"Task description: {task_description[:100]}...")

        return env

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: The action to perform.

        Returns:
            A tuple of (observation, reward, done, info)
        """
        action = action.strip()
        if action == "check valid actions":
            valid_actions = ", ".join(self.get_action_space())
            observation = f"Choose an action from these valid actions: {valid_actions}"
            info = {"valid": True}
            return observation, self.reward, self.done, info

        observation, _, _, info = self.env.step(action)
        self.reward = self.get_reward()
        self.done = self._check_is_done(None)  # We don't need selected_obs here

        return observation, self.reward, self.done, info

    def get_action_space(self, abstract: bool = True) -> List[str]:
        """
        Get the valid actions in the current environment state.

        Args:
            abstract: Whether to return abstract action types.

        Returns:
            A list of valid action strings.
        """
        svalid_actions = []
        if abstract:
            for a in self.env.getPossibleActions():
                if "reset" not in a:
                    svalid_actions.append(a)
        else:
            valid_actions = self.env.getValidActionObjectCombinationsWithTemplates()
            forbidden_words = [
                "teleport", "connect", "dunk", "eat", "flush", "close door"
            ]
            for valid_action in valid_actions:
                v = valid_action['action']
                should_skip = False
                for fw in forbidden_words:
                    if fw in v:
                        should_skip = True
                        break
                if not should_skip:
                    svalid_actions.append(valid_action['action'])

        if "check valid actions" not in svalid_actions:
            svalid_actions.append("check valid actions")

        return svalid_actions

    def reset(self) -> str:
        """
        Reset the environment.

        Returns:
            The initial observation.
        """
        self.reward = 0.0
        self.done = False
        return self.env.reset()

    def get_reward(self) -> float:
        """
        Get the current reward based on task progress.

        Returns:
            A reward value between 0 and 1.
        """
        # Use the built-in task progress as reward
        progress_str = self.env.getGoalProgressStr()

        # Parse progress percentage if possible
        import re
        match = re.search(r"(\d+)%", progress_str)
        if match:
            progress = int(match.group(1)) / 100.0
            return progress

        # Fallback to gold action sequence completion
        gold_actions = self.env.getGoldActionSequence()
        if gold_actions:
            # Check how many actions have been completed
            current_step = self.env.getStepsTaken()
            return min(1.0, current_step / len(gold_actions))

        return 0.0

    def _check_is_done(self, _: Optional[List[str]] = None) -> bool:
        """
        Check if the task is completed.

        Returns:
            True if the task is completed, False otherwise.
        """
        # Use the environment's isDone method
        return self.env.isDone()

    def inventory(self) -> List[str]:
        """
        Get the current inventory contents.

        Returns:
            A list of items in the inventory.
        """
        return self.env.inventory()

    def getTaskDescription(self) -> str:
        """
        Get the task description.

        Returns:
            The task description string.
        """
        return self.env.getTaskDescription()

    def getGoalProgressStr(self) -> str:
        """
        Get the goal progress string.

        Returns:
            A string describing the current progress toward the goal.
        """
        return self.env.getGoalProgressStr()

    def getGoldActionSequence(self) -> List[str]:
        """
        Get the gold action sequence for the current task.

        Returns:
            A list of actions that solve the task.
        """
        return self.env.getGoldActionSequence()


def dump_layernorm(model: "torch.nn.Module") -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Dumps LayerNorm parameters for later restore.

    Args:
        model: The model containing LayerNorm parameters.

    Returns:
        A list of tuples containing the LayerNorm parameters.
    """
    layernorm_params = []
    for module in model.modules():
        if isinstance(module, ALL_LAYERNORM_LAYERS):
            layernorm_params.append((module.weight.data.detach().clone(), module.bias.data.detach().clone()))

    return layernorm_params


def restore_layernorm(model: "torch.nn.Module", layernorm_params: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
    """
    Restores LayerNorm parameters.

    Args:
        model: The model to restore LayerNorm parameters to.
        layernorm_params: The LayerNorm parameters to restore.
    """
    i = 0
    for module in model.modules():
        if isinstance(module, ALL_LAYERNORM_LAYERS):
            weight, bias = layernorm_params[i]
            module.weight.data.copy_(weight)
            module.bias.data.copy_(bias)
            i += 1


def replace_model(model_a: "torch.nn.Module", model_b: "torch.nn.Module") -> None:
    """
    Replaces parameters of model_a with parameters of model_b.

    Args:
        model_a: The model to replace parameters.
        model_b: The model to get parameters from.
    """
    model_a_params = dict(model_a.named_parameters())
    for name, param in model_b.named_parameters():
        if name in model_a_params:
            model_a_params[name].data.copy_(param.data)


def extract_task_name_from_query(query: str) -> str:
    """
    Extract the ScienceWorld task name from the query.

    Args:
        query: The instruction/query text.

    Returns:
        A valid ScienceWorld task name.
    """
    # First, check if any valid task name is directly mentioned
    query_lower = query.lower()

    # Direct match: Look for exact task names in the query
    for task in VALID_SCIENCEWORLD_TASKS:
        if task in query_lower:
            return task

    # Keyword match: Check each keyword mapping
    for keyword, task_name in TASK_KEYWORD_MAPPING.items():
        if keyword in query_lower:
            return task_name

    # Structured extraction: Try to find task information from structured prompts
    if "task:" in query_lower:
        task_section = query_lower.split("task:")[1].strip().split("\n")[0]

        # Check if any valid task name is in the task section
        for task in VALID_SCIENCEWORLD_TASKS:
            if task in task_section:
                return task

        # Check for keywords in the task section
        for keyword, task_name in TASK_KEYWORD_MAPPING.items():
            if keyword in task_section:
                return task_name

    # Paragraph search: Check for specific content indicators in entire query
    if "thermometer" in query_lower or "temperature" in query_lower:
        return "use-thermometer"
    if "melt" in query_lower or "melting" in query_lower:
        return "melt"
    if "freeze" in query_lower or "freezing" in query_lower:
        return "freeze"
    if "conduct" in query_lower and ("electricity" in query_lower or "electric" in query_lower):
        return "test-conductivity"
    if "mix" in query_lower and "paint" in query_lower:
        return "chemistry-mix-paint-secondary-color"
    if "inclined plane" in query_lower:
        return "inclined-plane-determine-angle"
    if "grow" in query_lower and "plant" in query_lower:
        return "grow-plant"
    if "animal" in query_lower and ("find" in query_lower or "locate" in query_lower):
        return "find-animal"
    if "life" in query_lower and ("stage" in query_lower or "cycle" in query_lower):
        return "identify-life-stages-1"

    # If no task could be identified, use default but log the error
    logger.warning(f"Could not determine task name from query. Using '{DEFAULT_TASK_NAME}' as default.")
    return DEFAULT_TASK_NAME


def clean_task_name(task_name: str) -> str:
    """
    Clean a task name to ensure it's valid for the ScienceWorld environment.

    Args:
        task_name: The task name to clean.

    Returns:
        A cleaned task name.

    Raises:
        TaskNotFoundError: If the cleaned task name is not in the list of valid tasks.
    """
    # Remove any trailing numbers/underscores from the task name
    clean_name = task_name
    if '_' in task_name:
        clean_name = task_name.split('_')[0]

    # Check if the task name is valid
    if clean_name not in VALID_SCIENCEWORLD_TASKS:
        raise TaskNotFoundError(
            f"Task '{clean_name}' is not a valid ScienceWorld task. Valid tasks are: {sorted(list(VALID_SCIENCEWORLD_TASKS))}"
        )

    return clean_name


def prepare_interaction_for_reward(
    query: str,
    response: str,
    task_name: Optional[str] = None,
    var: int = 1,
    simplification: str = "easy"
) -> Dict[str, Any]:
    """
    Prepares an interaction for reward calculation.

    Args:
        query: The query/instruction from the user.
        response: The response from the model.
        task_name: The name of the ScienceWorld task (optional, will be extracted from query if None).
        var: The task variant number.
        simplification: The difficulty level of the task.

    Returns:
        A dictionary containing the formatted interaction.
    """
    # Extract task name from query if not provided
    if task_name is None:
        task_name = extract_task_name_from_query(query)

    return {
        "query": query,
        "response": response,
        "task_name": task_name,
        "var": var,
        "simplification": simplification
    }


def execute_in_environment(
    env: Any,
    response: str,
) -> RewardResult:
    """
    Executes a response in the ScienceWorld environment and calculates rewards.

    Args:
        env: The ScienceWorld environment instance.
        response: The model's response to execute.

    Returns:
        A RewardResult dataclass with the computed reward components.
    """
    # Execute the response in the environment
    actions = response.strip().split("\n")
    valid_actions = 0
    total_actions = len(actions) if len(actions) > 0 else 1

    for action in actions:
        action = action.strip()
        if not action:
            continue

        # Execute the action in the environment
        _, step_reward, done, info = env.step(action)

        if info.get("valid", False):
            valid_actions += 1

        if done:
            break

    # Calculate the reward components
    progress_rate = env.get_reward()  # Get the progress reward from environment
    success_rate = 1.0 if env._check_is_done(None) else 0.0  # Check if task was completed
    grounding_accuracy = valid_actions / total_actions  # Ratio of valid actions

    # Calculate the final reward
    final_reward = (
        0.6 * progress_rate +
        0.3 * success_rate +
        0.1 * grounding_accuracy
    )

    return RewardResult(
        progress_rate=progress_rate,
        success_rate=success_rate,
        grounding_accuracy=grounding_accuracy,
        valid_actions=valid_actions,
        total_actions=total_actions,
        final_reward=final_reward
    )


def get_scienceworld_rewards(
    interactions: List[Dict[str, Any]],
    reward_weights: Optional[Dict[str, float]] = None,
    env_step_limit: int = 50,
    env: Optional[Any] = None,
    server_path: Optional[str] = None
) -> List[float]:
    """
    Gets rewards from ScienceWorld environment.

    Args:
        interactions: A list of interaction dictionaries.
        reward_weights: A dictionary of weights for different reward components.
        env_step_limit: The maximum number of steps in the environment.
        env: A pre-initialized ScienceWorld environment object.
        server_path: The path to the ScienceWorld server.

    Returns:
        A list of reward values.
    """
    if reward_weights is None:
        reward_weights = {
            "progress_rate": 0.6,
            "success_rate": 0.3,
            "grounding_accuracy": 0.1,
        }

    rewards = []

    # Initialize the ScienceWorld environment if not provided
    # assert env is None
    if env is None: # typically False
        breakpoint()
        # Import the agentboard environment directly - let any errors propagate

        # Create a default path to the label file
        label_path = None

        # Try to find a label path in the agentboard data directory
        agentboard_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "agentboard", "data", "scienceworld")

        if os.path.exists(agentboard_data_path):
            for file in os.listdir(agentboard_data_path):
                if file.endswith('.jsonl'):
                    label_path = os.path.join(agentboard_data_path, file)
                    logger.info(f"Using label file: {label_path}")
                    break

        if label_path is None:
            # If not found in the project directory, look in the workspace
            possible_label_paths = [
                "/home/zhanwen/finetuners/agentboard/data/scienceworld/test.jsonl",
                "/home/zhanwen/finetuners/data/scienceworld/scienceworld_consolidated_goalset.jsonl",
                "/home/zhanwen/finetuners/data/scienceworld/test_60_90.jsonl"
            ]

            for path in possible_label_paths:
                if os.path.exists(path):
                    label_path = path
                    logger.info(f"Using label file: {label_path}")
                    break

        if label_path is None:
            raise FileNotFoundError(
                "Could not find a valid ScienceWorld label file. "
                "Please specify a path to a .jsonl file containing ScienceWorld task labels."
            )

        # Initialize the environment - let any errors propagate to stop the program
        env = Scienceworld(serverPath=server_path, envStepLimit=env_step_limit, label_path=label_path)
        logger.info(f"Initialized ScienceWorld environment with label file: {label_path}")

    for interaction in interactions:
        query = interaction.get("query", "")
        response = interaction.get("response", "")
        task_name = interaction.get("task_name")
        var = interaction.get("var", 1)
        simplification = interaction.get("simplification", "easy")
        breakpoint()

        # Extract task name from query if not provided
        if task_name is None:
            task_name = extract_task_name_from_query(query)

        # Clean and validate the task name
        clean_name = clean_task_name(task_name)

        # Load the task in the environment - let any errors propagate
        logger.info(f"Loading task: {clean_name} (var: {var}, simplification: {simplification})")
        env.load(clean_name, var, simplification)
        env.reset()

        # Execute the response in the environment and get rewards
        result = execute_in_environment(env, response)

        # Apply reward weights
        final_reward = (
            reward_weights.get("progress_rate", 0.6) * result.progress_rate +
            reward_weights.get("success_rate", 0.3) * result.success_rate +
            reward_weights.get("grounding_accuracy", 0.1) * result.grounding_accuracy
        )

        logger.info(
            f"Task: {clean_name}, Progress: {result.progress_rate:.2f}, Success: {result.success_rate:.2f}, "
            f"Grounding: {result.grounding_accuracy:.2f}, Actions: {result.valid_actions}/{result.total_actions}, "
            f"Final reward: {final_reward:.4f}"
        )

        rewards.append(float(final_reward))

    return rewards
