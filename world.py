#!/usr/bin/env python3


from sdlabs import material
from sdlabs.utility import parse_function_inputs
import numpy as np
from itertools import product


def f1(inputs):
    p1 = parse_function_inputs(inputs, [('BladeCoat','coating_temperature')])
    p1 = p1/500 # normalize to expected value
    return 1-(2*p1-1)**2


def f2(inputs):
    p1 = parse_function_inputs(inputs, [('BladeCoat','speed')])
    p1 = (p1-1)/19 # normalize to expected value
    return 6.8*(p1**2 - p1**3)


def f3(inputs):
    p1 = parse_function_inputs(inputs, [('BladeCoat','solvent_to_polymer_ratio')])
    return (1+np.sin(10*p1))/2


def f4(inputs):
    p1 = parse_function_inputs(inputs, [('BladeCoat','relative_humidity')])
    p1 = p1/100 # normalize to expected value
    return (1+np.cos(20*p1))/2


def f5(inputs):
    p1 = parse_function_inputs(inputs, [('BladeCoat','solvent_BP')])
    p1 = (p1-60)/120 # normalize to expected value
    return 1-np.log(p1+1)


def f6(inputs):
    p1 = parse_function_inputs(inputs, [('BladeCoat','annealing_time')])
    p1 = p1/14 # normalize to expected value
    return np.sqrt(p1)


def f7(inputs):
    p1 = parse_function_inputs(inputs, [('BladeCoat','annealing_temperature')])
    p1 = (p1-25)/225 # normalize to expected value
    return np.exp(p1-1)


def g1(inputs):
    return 6000/7*( f1(inputs) + f2(inputs) + f3(inputs) + f4(inputs) + f5(inputs) + f6(inputs) + f7(inputs) )

def g2(inputs):
    return 6000/4*( f1(inputs) + f2(inputs) + f3(inputs) + f4(inputs) )

def g3(inputs):
    return 6000/2*( f1(inputs) + f5(inputs) )


class Experiment:

    def __init__(self, category='base_class', action_space=[], inputs={}, cost=0.0):
        self.category = category
        self.action_space = action_space
        self.inputs = inputs
        self.cost = cost

    def calculate_outputs(self, material, action):
        return None

    def get_input_space(self, length=20):
        if not self.inputs:
            return None
        input_labels = sorted(list(self.inputs.keys()))
        input_ranges = [np.linspace(self.inputs[_][0], self.inputs[_][1], num=length) for _ in input_labels]
        input_values = np.array(np.meshgrid(*input_ranges)).T.reshape(-1,len(input_ranges))
        return (input_labels, input_values)
    
    def yield_input_spaces(self, length=50, chunk_size=25):
        if not self.inputs:
            return None
        input_labels = sorted(list(self.inputs.keys()))
        input_ranges = [np.linspace(self.inputs[_][0], self.inputs[_][1], num=length) for _ in input_labels]
        indices = [range(0, len(sublist), chunk_size) for sublist in input_ranges]
        for indexes in product(*indices):
            subranges = [sublist[idx:idx+chunk_size] for sublist, idx in zip(input_ranges, indexes)]
            input_values = np.array(np.meshgrid(*subranges)).T.reshape(-1,len(subranges))
            yield (input_labels,input_values)


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
            #cost=1.2): # for now, assume same as UV/Vis
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'peak_position': f1(sample)*3600 + 400, # expected 4000 - 400 cm-1
        }


class UVVisSpectroscopy(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            #cost=0.16+1.0): # 10 min and 1/10 difficulty
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'peak_position': f2(sample)*500 + 200, # expected 200 - 700 nm
            'peak_width': f3(sample)*30, # expected 0 - 30 nm
            'absorbance': f4(sample)*1.5, # expected 0 - 1.5
        }


class EQCM(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            #cost=13.0+8.0): # 13 hr and 8/10 difficulty
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'change_in_mass': f5(sample), # expected 0 - 100 %
        }


class CV(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            cost=1.0+4.0): # best guess is 1 hr and 4/10 difficulty, but this is just a guess
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'capacitance': f5(sample), # expected 10 - 1000 F cm-3
        }


class SpectroElectroChemistry(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            #cost=2.0+6.0): # best guess is 2 hr and 6/10 difficulty, but this is just a guess
            cost=2.0+8.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'peak_position_change': f2(sample)*100, # expected 0 - 100 nm
            'peak_width_change': f3(sample)*30, # expected 0 - 30 nm
            'absorbance_change': f4(sample)*100, # expected 0 - 100 %
        }


class AFM(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            #cost=12.0+6.0): # 30 min to 1 day, 6/10 difficulty
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'RMS_surface_roughness': f2(sample), # expected 0 - 50 nm
            'phase_angle': f4(sample), # expected -90 to 90 degrees
            'pore_size': f6(sample), # expected 0 - 1500 nm
        }


class GIWAXS(Experiment):

    def __init__(
            self,
            action_space=[],
            inputs={},
            #cost=1.0+4.0): # 1 hr, 5/10 difficulty, total guess
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            inputs=inputs,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'stacking_distance': f6(sample), # expected 1.5 to 6.0 Angstrom
            'lamellar_size': f7(sample), # expected 5 to 80 degrees
        }
    

class Stability(Experiment):

    def __init__(self, action_space=[], inputs={}, cost=0.0, stability_calc=g1):
        super().__init__(
            category='stability',
            action_space=action_space,
            inputs=inputs,
            cost=cost,
            )
        self.stability_calc=stability_calc
        
    def calculate_outputs(self, sample, action):
        return {'stability': self.stability_calc(sample)}
    
    def calc_stability(self, inputs):
        return self.stability_calc(inputs)


class TurnBack(Experiment):

    def __init__(self, action_space=[], inputs={}, cost=0.0):
        super().__init__(
            category='turn_back',
            action_space=action_space,
            inputs=inputs,
            cost=cost,
            )
        
    def calculate_outputs(self, sample, action):
        return {'None':None}


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
