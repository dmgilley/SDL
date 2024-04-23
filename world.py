#!/usr/bin/env python3


from sdlabs import material
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

    def __init__(
            self,
            action_space=[],
            inputs={
                'coating_temperature': [0.0,500.0], # C
                'speed': [1.0,20.0], # mm/s
                'solvent_to_polymer_ratio': [0.0,1.0],
                'relative_humidity': [0.0,100.0], # %
                'solvent_BP': [60.0,180.0], # C
                'annealing_time': [0.0,14.0], # hr
                'annealing_temperature': [25.0,250.0], # C
            },
            cost=1.0+5.0): # 1 hr and 5/10 difficulty
        super().__init__(
            category='processing',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return action.parameters


class SpinCoat(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={
                'coating_temperature': [0.0,500.0], # C
                'speed': [500.0,5000.0], # rpm
                'solvent_to_polymer_ratio': [0.0,1.0],
                'relative_humidity': [0.0,100.0], # %
                'solvent_BP': [60.0,180.0], # C
                'annealing_time': [0.0,14.0], # hr
                'annealing_temperature': [25.0,250.0], # C
            },
            cost=0.83+3.0): # 50 min and 3/10 difficulty
        super().__init__(
            category='processing',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return action.parameters
    

class RamanSpectroscopy(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            cost=1.2): # for now, assume same as UV/Vis
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'peak_position': f1(sample), # expected 4000 - 400 cm-1
        }


class UVVisSpectroscopy(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            cost=0.16+1.0): # 10 min and 1/10 difficulty
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'peak_position': f1(sample), # expected 200 - 700 nm
            'peak_width': f1(sample), # expected 0 - 30 nm
            'absorbance': f1(sample), # expected 0 - 1.5
        }


class EQCM(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            cost=13.0+8.0): # 13 hr and 8/10 difficulty
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'change_in_mass': f1(sample), # expected 0 - 100 %
        }


class CV(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            cost=1.0+5.0): # best guess is 1 hr and 5/10 difficulty, but this is just a guess
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'capacitance': f1(sample), # expected 10 - 1000 F cm-3
        }


class SpectroElectroChemistry(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            cost=2.0+6.0): # best guess is 2 hr and 6/10 difficulty, but this is just a guess
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'peak_position_change': f1(sample), # expected 0 - 100 nm
            'peak_width_change': f1(sample), # expected 0 - 30 nm
            'absorbance_change': f1(sample), # expected 0 - 100 %
        }


class AFM(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            cost=12.0+6.0): # 30 min to 1 day, 6/10 difficulty
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'RMS_surface_roughness': [], # expected 0 - 50 nm
            'phase_angle': [], # expected -90 to 90 degrees
            'pore_size': [], # expected 0 - 1500 nm
        }


class GIWAXS(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            cost=12.0+6.0): # 30 min to 1 day, 6/10 difficulty, total guess
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'stacking_distance': [], # expected 1.5 to 6.0 Angstrom
            'lamellar_size': [], # expected 5 to 80 degrees
        }
    

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
