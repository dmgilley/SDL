#!/usr/bin/env python3


from SDL import material
import numpy as np


def f1(params):
    p1 = params
    if isinstance(params, material.Material):
        p1 = params.pull_outputs('BladeCoat','temperature')[0]/400
    return 2*p1/3*(1-np.sin(4*np.pi*p1))


def f2(params, p2=None):
    p1 = params
    if isinstance(params, material.Material):
        p1 = params.pull_outputs('BladeCoat','temperature')[0]/400
        p2 = params.pull_outputs('BladeCoat','anneal_time')[0]/300
    return p1/8*(1-np.sin(4*np.pi*p1)) - ((p2-0.2)**2)/4 + (1/2)


def g1(params, p2=None):
    p1 = params
    if isinstance(params, material.Material):
        p1 = params.pull_outputs('BladeCoat','temperature')[0]/400
        p2 = params.pull_outputs('BladeCoat','anneal_time')[0]/300
    if isinstance(params, np.ndarray) and not p2:
        p1 = params[:,0]/400
        p2 = params[:,0]/300
    return 80*(f1(p1) + f2(p1,p2))


class Experiment:

    def __init__(self, category='base_class', action_space=[], inputs={}, cost=0.0):
        self.category = category
        self.action_space = action_space
        self.inputs = inputs
        self.cost = cost

    def calculate_outputs(self, material, action):
        return None

    def get_input_space(self, length=100):
        if not self.inputs:
            return None
        input_labels = sorted(list(self.inputs.keys()))
        input_ranges = [np.linspace(self.inputs[_][0], self.inputs[_][1], num=length) for _ in input_labels]
        input_values = np.array(np.meshgrid(*input_ranges)).T.reshape(-1,len(input_ranges))
        return (input_labels, input_values)


class BladeCoat(Experiment):

    def __init__(self, action_space=[], inputs={}, cost=0.0):
        super().__init__(
            category='processing',
            action_space=action_space,
            inputs=inputs,
            cost=cost
            )
        
    def calculate_outputs(self, sample, action):
        return action.parameters
    

class Anneal(Experiment):

    def __init__(self, action_space=[], inputs={}, cost=0.0):
        super().__init__(
            category='post-processing',
            action_space=action_space,
            inputs=inputs,
            cost=cost
            )
        
    def calculate_outputs(self, sample, action):
        return action.parameters
    

class CV(Experiment):

    def __init__(self, action_space=[], inputs={}, cost=0.0):
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost
            )
        
    def calculate_outputs(self, sample, action):
        return {'onset': f1(sample)}
    

class ECQM(Experiment):

    def __init__(self, action_space=[], inputs={}, cost=0.0):
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost
            )
        
    def calculate_outputs(self, sample, action):
        return {'mass_change': f2(sample)}
    

class Stability(Experiment):

    def __init__(self, action_space=[], inputs={}, cost=0.0):
        super().__init__(
            category='stability',
            action_space=action_space,
            inputs=inputs,
            cost=cost
            )
        
    def calculate_outputs(self, sample, action):
        return {'stability': g1(sample)}
    
    def calc_stability(self, params, p2=None):
        return g1(params, p2=p2)


class VSDLEnvironment:

    def __init__(self, experiments={}):
        self.experiments = experiments

    def add_experiment(self, experiment):
        if type(experiment) == list:
            overwrite_list = [_.__class__.__name__ for _ in experiment if _.__class__.__name__ in self.experiments.keys()]
            if overwrite_list:
                print('Warning! Overwriting experiment(s) in VSDLEnvironment: {}'.format(overwrite_list))
            self.experiments.update({_.__class__.__name__:_ for _ in experiment})
        else:
            if experiment.__class__.__name__  in self.experiments:
                print('Warning! Overwriting experiment in VSDLEnvironment: {}'.format(experiment.__class__.__name__))
            self.experiments[experiment.__class__.__name__] = experiment

    def get_action_space(self, sample):
        if not sample.states:
            return sorted([k for k,v in self.experiments.items() if v.category == 'processing'])
        for name,experiment in self.experiments.items():
            if name == sample.states[-1].name:
                action_space = [_ for _ in experiment.action_space if _ not in [i.name for i in sample.states]]
                continue
        if sample.states[-1].category == 'stability':
            if len(action_space) != 0:
                print('Error! Last experiment was a stability measurement, but VSDLEnvironment.get_actions() is attempting to return a non-empty list.')
                return None
            print('Warning! Getting action space for a material that has had stability performed.')
        return action_space
    
    def get_all_experiments(self):
        return sorted(list(self.experiments.keys()))
    
    def get_experiment_names(self, category=''):
        return sorted([k for k,v in self.experiments.items() if v.category == category])