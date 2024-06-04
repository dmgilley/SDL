#!/usr/bin/env python3


from typing import *


class Action:

    def __init__(self, name: str, category: str, inputs: dict = None, outputs: dict = None, reward: float = 0.0) -> None:
        self.name = name
        self.category = category
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.reward = reward

    def get_outputs(self) -> List[float]:
        return [v for k,v in sorted(self.outputs.items())]


class Sample:

    def __init__(self) -> None:
        self.actions: List[Action] = []
        self.stability: float = 0.0
        self.closed: bool = False

    def add_action(self, action: Action) -> None:
        if len(action.outputs) == 0:
            raise ValueError("Error! Attempting to add a material.Action instance that has no outputs to a material.Sample instance.")
        if self.closed:
            raise ValueError("Error! Attempting to run an experiment on a closed sample.")
        self.actions.append(action)
        stability_value = action.outputs.get("stability")
        if stability_value is not None:
            self.stability = stability_value
            self.closed = True
        if action.category == "turn_back":
            self.closed = True

    def pull_outputs(self, action_name: str, requested_output_names: Union[str,List[str]]) -> List[float]:
        if isinstance(requested_output_names, str):
            requested_output_names = [requested_output_names]
        matching_actions = [a.outputs for a in self.actions if a.name == action_name]
        if not matching_actions:
            return [0.0] * len(requested_output_names)
        values = [matching_actions[0].get(output_name, 0.0) for output_name in requested_output_names]
        if len(values) != len(requested_output_names):
            missing_keys = [name for name in requested_output_names if name not in matching_actions[0]]
            print(f"Warning! Attempting to extract non-existent outputs: {', '.join(missing_keys)} from {action_name}.")
        return values