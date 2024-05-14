#!/usr/bin/env python3


class Action:

    def __init__(self, name, category, parameters={}):
        self.name = name
        self.category = category
        self.parameters = parameters


class State:

    def __init__(self, name, category, outputs={}, reward=0.0):
        self.name = name
        self.category = category
        self.outputs = outputs
        self.reward = reward

    def get_outputs(self):
        return [v for k,v in sorted(self.outputs.items())]


class Material:

    def __init__(self):
        self.states = []
        self.stability = 0.0
        self.closed = False

    def add_state(self, state):
        if self.closed:
            print('Error! Attempting to run an experiment on a material that is "closed" (i.e. stability was already measured).')
            return
        self.states.append(state)
        if 'stability' in state.outputs.keys():
            self.stability = state.outputs['stability']
            self.closed = True
        if state.category == 'turn_back':
            self.closed = True

    def pull_outputs(self, state_name, output_names):
        if type(output_names) == str:
            output_names = [output_names]
        outputs = [_.outputs for _ in self.states if _.name == state_name]
        if not outputs:
            return [0.0 for _ in output_names]
        values = [outputs[0].get(_,0.0) for _ in output_names]
        if len(values) != len(output_names):
            bad_keys = [_ for _ in output_names if _ not in outputs[0]]
            print('Warning! Attempting to extract a non-existent output. Requested {} from {}.'.format(','.join(bad_keys),state_name))
        return values