#!/usr/bin/env python3


from sdlabs.utility import *
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


def f8(inputs):
    p1 = parse_function_inputs(inputs, [('BladeCoat','speed')])
    p2 = parse_function_inputs(inputs, [('BladeCoat','solvent_to_polymer_ratio')])
    # normalize to expected values
    p1 = (p1-1)/19
    p2 = (p2-0)/1
    return (1-np.sqrt( (p1-0.2)**2 + (p2-0.2)**2 ) ) / 1.1 + 0.12


def f9(inputs):
    p1 = parse_function_inputs(inputs, [('BladeCoat','speed')])
    p3 = parse_function_inputs(inputs, [('BladeCoat','annealing_temperature')])
    # normalize to expected values
    p1 = (p1-1)/19
    p3 = (p3-25)/225
    return ( - p1**3 + p1**2 - (p3*0.7-0.5)**3 - (p3*0.7-0.5)**2 ) / 0.27 + 0.5


def f10(inputs):
    p2 = parse_function_inputs(inputs, [('BladeCoat','solvent_to_polymer_ratio')])
    p3 = parse_function_inputs(inputs, [('BladeCoat','annealing_temperature')])
    # normalize to expected values
    p2 = (p2-0)/1
    p3 = (p3-25)/225
    return ( 1 - (np.sin(10*p2-0.5))**2/(p3+0.9)**1 + (p3-0.5)**3 - (p3-0.5)**2)/1.5 + 0.33


def g1(inputs):
    return 6000/4*( f1(inputs) + f2(inputs) + f3(inputs) + f4(inputs) )

def g2(inputs):
    return 6000/4*( f1(inputs) + f8(inputs) + f9(inputs) + f10(inputs) )

def g3(inputs):
    return 6000/3*( f2(inputs) + f3(inputs) + f4(inputs) )


class Experiment:

    def __init__(self, category='base_class', action_space=[], parameters={}, cost=0.0):
        self.category = category
        self.action_space = action_space
        self.parameters = parameters
        self.cost = cost

    def calculate_outputs(self, sample, action):
        return None

    def get_input_space(self, length=10, noise=0.1):
        if not self.parameters:
            return None
        input_labels = sorted(list(self.parameters.keys()))
        input_ranges = [np.linspace(self.parameters[_][0], self.parameters[_][1], num=length) for _ in input_labels]
        input_values = np.array(np.meshgrid(*input_ranges)).T.reshape(-1,len(input_ranges))
        input_values += np.random.normal(loc=0.0, scale=noise*np.mean(input_values), size=input_values.shape)
        for idx,val in enumerate(input_labels):
            input_values[:,idx] = np.where(input_values[:,idx] < self.parameters[val][0],self.parameters[val][0],input_values[:,idx])
            input_values[:,idx] = np.where(input_values[:,idx] > self.parameters[val][1],self.parameters[val][1],input_values[:,idx])
        return (input_labels, input_values)
    
    def yield_input_spaces(self, length=15, chunk_size=15, noise=0.1):
        if not self.parameters:
            return None
        input_labels = sorted(list(self.parameters.keys()))
        input_ranges = [np.linspace(self.parameters[_][0], self.parameters[_][1], num=length) for _ in input_labels]
        indices = [range(0, len(sublist), chunk_size) for sublist in input_ranges]
        for indexes in product(*indices):
            subranges = [sublist[idx:idx+chunk_size] for sublist, idx in zip(input_ranges, indexes)]
            input_values = np.array(np.meshgrid(*subranges)).T.reshape(-1,len(subranges))
            input_values += np.random.normal(loc=0.0, scale=noise*np.mean(input_values), size=input_values.shape)
            for idx,val in enumerate(input_labels):
                input_values[:,idx] = np.where(input_values[:,idx] < self.parameters[val][0],self.parameters[val][0],input_values[:,idx])
                input_values[:,idx] = np.where(input_values[:,idx] > self.parameters[val][1],self.parameters[val][1],input_values[:,idx])
            yield (input_labels,input_values)


class BladeCoat(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={
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
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return action.inputs


class SpinCoat(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={
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
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return action.inputs
    

class RamanSpectroscopy(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={},
            #cost=1.2): # for now, assume same as UV/Vis
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'peak_position': f1(sample)*3600 + 400, # expected 4000 - 400 cm-1
        }


class UVVisSpectroscopy(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={},
            #cost=0.16+1.0): # 10 min and 1/10 difficulty
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'peak_position': f2(sample)*500 + 200, # expected 200 - 700 nm
            'peak_width': f3(sample)*30, # expected 0 - 30 nm
            'absorbance': f4(sample)*1.5, # expected 0 - 1.5
        }
    
class UVVisSpectroscopy2(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={},
            #cost=0.16+1.0): # 10 min and 1/10 difficulty
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'peak_position': f8(sample)*500 + 200, # expected 200 - 700 nm
            'peak_width': f9(sample)*30, # expected 0 - 30 nm
            'absorbance': f10(sample)*1.5, # expected 0 - 1.5
        }


class EQCM(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={},
            #cost=13.0+8.0): # 13 hr and 8/10 difficulty
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'change_in_mass': f5(sample), # expected 0 - 100 %
        }


class CV(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={},
            cost=1.0+4.0): # best guess is 1 hr and 4/10 difficulty, but this is just a guess
        super().__init__(
            category='characterization',
            action_space=action_space,
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'capacitance': f5(sample), # expected 10 - 1000 F cm-3
        }


class SpectroElectroChemistry(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={},
            #cost=2.0+6.0): # best guess is 2 hr and 6/10 difficulty, but this is just a guess
            cost=8.0+32.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'peak_position_change': f2(sample)*100, # expected 0 - 100 nm
            'peak_width_change': f3(sample)*30, # expected 0 - 30 nm
            'absorbance_change': f4(sample)*100, # expected 0 - 100 %
        }
    
class SpectroElectroChemistry2(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={},
            #cost=2.0+6.0): # best guess is 2 hr and 6/10 difficulty, but this is just a guess
            cost=8.0+32.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'peak_position_change': f8(sample)*100, # expected 0 - 100 nm
            'peak_width_change': f9(sample)*30, # expected 0 - 30 nm
            'absorbance_change': f10(sample)*100, # expected 0 - 100 %
        }


class AFM(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={},
            #cost=12.0+6.0): # 30 min to 1 day, 6/10 difficulty
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'RMS_surface_roughness': f2(sample), # expected 0 - 50 nm
            'phase_angle': f4(sample), # expected -90 to 90 degrees
            'pore_size': f6(sample), # expected 0 - 1500 nm
        }
    
class AFM2(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={},
            #cost=12.0+6.0): # 30 min to 1 day, 6/10 difficulty
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        var1 = f2(sample)
        var2 = f3(sample)
        var3 = f4(sample)
        return {
            'RMS_surface_roughness': np.array(np.random.rand(var1.shape[0],var1.shape[1])*50), # expected 0 - 50 nm
            'phase_angle': np.array(np.random.rand(var2.shape[0],var2.shape[1])*180-90), # expected -90 to 90 degrees
            'pore_size': np.array(np.random.rand(var3.shape[0],var3.shape[1])*1500), # expected 0 - 1500 nm
        }


class GIWAXS(Experiment):

    def __init__(
            self,
            action_space=[],
            parameters={},
            #cost=1.0+4.0): # 1 hr, 5/10 difficulty, total guess
            cost=1.0+4.0): # temporary while adjusting experiment rewards
        super().__init__(
            category='characterization',
            action_space=action_space,
            parameters=parameters,
            cost=cost)
        
    def calculate_outputs(self, sample, action):
        return {
            'stacking_distance': f6(sample), # expected 1.5 to 6.0 Angstrom
            'lamellar_size': f7(sample), # expected 5 to 80 degrees
        }
    

class Stability(Experiment):

    def __init__(self, action_space=[], parameters={}, cost=5.0, stability_calc=g1):
        super().__init__(
            category='stability',
            action_space=action_space,
            parameters=parameters,
            cost=cost,
            )
        self.stability_calc=stability_calc
        
    def calculate_outputs(self, sample, action):
        return {'stability': self.stability_calc(sample)}
    
    def calc_stability(self, inputs):
        return self.stability_calc(inputs)


class TurnBack(Experiment):

    def __init__(self, action_space=[], parameters={}, cost=0.0):
        super().__init__(
            category='turn_back',
            action_space=action_space,
            parameters=parameters,
            cost=cost,
            )
        
    def calculate_outputs(self, sample, action):
        return {'None': np.array([0.0])}


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
        if not sample.actions:
            return self.get_experiment_names(category='processing')
        last_action = sample.actions[-1]
        experiment = self.experiments.get(last_action.name, None)
        if experiment == None:
            raise KeyError("Error! The name of the sample's last action isn't listed in the environment's experiments.")
        action_space = [_ for _ in experiment.action_space if _ not in [i.name for i in sample.actions]]
        if last_action.category == 'stability':
            if len(action_space) != 0:
                raise ValueError('Error! Last experiment was a stability measurement, but world.VSDLEnvironment.get_action_space() is attempting to return a non-empty list.')
            print('Warning! Getting action space for a material that has had stability performed.')
        return action_space
    
    def get_experiment_names(self, category: Union[str, List[str]] = 'all', exclude: bool = False) -> List[str]:
        if category == 'all':
            category = sorted(list(set([exp.category for exp in self.experiments.values()])))
        elif type(category) == str:
            category = [category]
        if exclude:
            return sorted([k for k,v in self.experiments.items() if v.category not in category])
        return sorted([k for k,v in self.experiments.items() if v.category in category])
    
    def get_input_dimensionality(self):
        experiment_name = self.get_experiment_names(category='processing')[0]
        return len(self.experiments[experiment_name].parameters)
    
    def get_input_keys(self,
                       key_type: str = 'all',
                       category: Union[str, List[str]] = 'all',
                       exclude: bool = False,
                       include_stability: bool = False) -> List[Tuple[str]]:
        
        if key_type == 'single_exp':
            return [tuple([_]) for _ in self.get_experiment_names(category=category, exclude=exclude)]

        if key_type == 'all':
            system = {experiment_name:experiment.action_space for experiment_name,experiment in self.experiments.items()}
            input_keys = []
            for start in self.get_experiment_names(category='processing'):
                for end in self.get_experiment_names(category=['stability','turn_back']):
                    input_keys += find_paths(system, start, end, include_endpoint=False)
                    input_keys += find_paths(system, start, end, include_endpoint=include_stability)
            input_keys = list(set(input_keys))
            input_keys.sort(key=lambda x: str(x))
            input_keys.sort(key=lambda x: len(x))
            return input_keys