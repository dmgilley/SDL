#!/usr/bin/env python3


import gc, warnings
from sdlabs import material,world
from scipy import optimize
from copy import deepcopy
from typing import *
import numpy as np
import sklearn.gaussian_process as GP
import scipy.stats as sps


class MLStrategy:
    def __init__(
            self,
            environment: Union[ None, world.VSDLEnvironment ] = None,
            discount: float = 0.1,
            epsilon: float = 0.5,
            include_MAE: bool = False,
            optimizer_method: str = 'Nelder-Mead',
            BO_acq_func_name: str = 'UCB',
            savf_update_method: str = 'MC',
            kernel: Union[ None, GP.kernels.Kernel ] = None,
            GPRs: Union[ None, GP.GaussianProcessRegressor ] = None, 
            GPR_dict: Union[ None, Dict[str, Union[ int, float, GP.kernels.Kernel ] ] ] = None,
            savfs: Union[ Tuple[int,float], Dict[str,Tuple[int,float]] ] = (0,0.0)) -> None:
        self.epsilon = epsilon # larger epsilon corresponds to stronger exploitation
        self.discount = discount
        self.include_MAE = include_MAE
        self.optimizer_method = optimizer_method
        self.actions = {}
        self.stabilities = {}
        self.sample_number = 1
        self.processed_samples = 0
        self.environment_added = False
        self.initialize_GPRs(kernel=kernel, GPR_dict=GPR_dict, GPRs=GPRs)
        self.initialize_savfs(savfs=savfs)
        if environment != None:
            self.add_environment(environment)
        self.set_BO_acq_func(BO_acq_func_name)
        self.set_savf_update_method(savf_update_method)
        self.initialize_MAE()

    def custom_optimizer(self, obj_func, initial_theta, bounds=(1e-5,1e5)):
        if type(bounds) == tuple:
            bounds = np.log(np.vstack([bounds]).reshape(-1,2))    
        def obj_func_wrapper(theta):
            return obj_func(theta, eval_gradient=False)
        opt_res = optimize.minimize(obj_func_wrapper, initial_theta, bounds=bounds, method=self.optimizer_method)
        return opt_res.x, opt_res.fun
    
    def initialize_MAE(self):
        if not self.environment_added: # this catches when the MLStrategy is initialized without an environment
            return
        if self.include_MAE == False:
            setattr(self, 'MAE', None)
            return
        setattr(self, 'MAE', {})
        for p_exp_n in self.environment.get_experiment_names(category='processing'):
            self.MAE[tuple([tuple([p_exp_n]),tuple(['Stability'])])] = []
        return
        
    def initialize_savfs(
            self,
            savfs: Union[ Tuple[int,float], Dict[str,Tuple[int,float]]] = (0,0.0),
            environment: Union[ None, world.VSDLEnvironment ] = None):
        if not hasattr(self, 'savfs'):
            setattr(self, 'savfs', savfs)
        if environment != None and type(self.savfs) != dict:
            self.savfs = {n:deepcopy(self.savfs) for n in environment.get_experiment_names()}
        
    def initialize_GPRs(
            self,
            kernel: Union[ None, GP.kernels.Kernel ] = None,
            GPR_dict: Union[ None, Dict[str, Union[ int, float, GP.kernels.Kernel ]] ] = None,
            GPRs: Union[ None, GP.GaussianProcessRegressor ] = None, 
            environment: Union[ None, world.VSDLEnvironment ] = None):
        if not hasattr(self, 'GPRs'):
            if kernel == None:
                kernel = GP.kernels.Matern()
            if GPR_dict == None:
                GPR_dict = {
                    'kernel': kernel,
                    'alpha': 1e-9,
                    'n_restarts_optimizer': 10,
                }
            if 'kernel' not in GPR_dict.keys():
                GPR_dict['kernel'] = kernel
            GPR_dict['optimizer'] = self.custom_optimizer
            if GPRs == None:
                GPRs = GP.GaussianProcessRegressor(**GPR_dict)
            setattr(self, 'GPRs', GPRs)
        if environment != None and type(self.GPRs) != dict:
            experiment_names = environment.get_experiment_names(category=['stability','turn_back'], exclude=True)
            self.GPRs = {tuple([tuple([k]),tuple(['Stability'])]):deepcopy(self.GPRs) for k in experiment_names}

    def add_environment(self, environment: Union[ None, world.VSDLEnvironment ] = None, overwrite: bool = False):
        if self.environment_added and not overwrite:
            warnings.warn('Warning! Tried to add environment to {} instance, but an environment has already been added.'.format(self))
            return
        self.initialize_savfs(environment=environment)
        self.initialize_GPRs(environment=environment)
        setattr(self, 'environment', environment)
        self.environment_added = True
        self.initialize_MAE()

    def get_sample_numbers(self, unprocessed_only=True):
        starting_idx = 0
        if unprocessed_only:
            starting_idx = self.processed_samples
        return sorted(self.actions.keys())[starting_idx:]

    def BO_probability_of_improvement(self, mean, std):
        # Snoeck et al., "Practical Bayesian Optimization of Machine Learning Algorithms"
        fbest = np.max([alist[-1].outputs['stability'] for alist in self.actions.values() if alist[-1].outputs.get('stability',None) != None])
        return sps.norm.cdf((mean - fbest) / std)

    def BO_expected_improvement(self, mean, std):
        # Frazier and Wang, "Bayesian Optimization for Materials Design"
        fbest = np.max([alist[-1].outputs['stability'] for alist in self.actions.values() if alist[-1].outputs.get('stability',None) != None])
        gamma = (mean - fbest) / std
        return (mean - fbest) * sps.norm.cdf(gamma) + std * sps.norm.pdf(gamma)

    def BO_UCB(self, mean, std):
        # Snoeck et al., "Practical Bayesian Optimization of Machine Learning Algorithms"
        fbest = np.max([alist[-1].outputs['stability'] for alist in self.actions.values() if alist[-1].outputs.get('stability',None) != None])
        return mean + (1-self.epsilon) * std

    def set_BO_acq_func(self, BO_acq_func_name):
        BO_acq_func_map = {
            'Probability of Improvement': self.BO_probability_of_improvement,
            'Expected Improvement': self.BO_expected_improvement,
            'UCB': self.BO_UCB,
        }
        if BO_acq_func_name not in BO_acq_func_map:
            raise KeyError("Invalid BO_acq_func_name specified")
        setattr(self, 'BO_acq_func', BO_acq_func_map[BO_acq_func_name])

    def MC_savf_update(self):
        for sample_number in self.get_sample_numbers():
            reward_list = [a.reward for a in self.actions[sample_number]]
            for action_idx,action in enumerate(self.actions[sample_number]):
                count = self.savfs[action.name][0] + 1
                Gt = np.sum([(self.discount**reward_idx)*reward for reward_idx,reward in enumerate(reward_list[action_idx:])])
                new_savf = self.savfs[action.name][1] + (Gt - self.savfs[action.name][1])/(count)
                self.savfs[action.name] = (count, new_savf)

    def set_savf_update_method(self, savf_update_method):
        method_map = {
            'MC': self.MC_savf_update
        }
        if savf_update_method not in method_map:
            raise KeyError("Invalid savf_update_method specified")
        setattr(self, 'update_savfs', method_map[savf_update_method])

    def select_action(self, sample: material.Sample) -> material.Action:
        raise NotImplementedError

    def take_action(self, sample: material.Sample, action: material.Action) -> material.Action:
        outputs = self.environment.experiments[action.name].calculate_outputs(sample, action)
        return material.Action(
            deepcopy(action.name),
            deepcopy(action.category),
            outputs={k:v.flatten()[0] for k,v in outputs.items()})
    
    def update_after_sample(self, sample):
        self.actions[self.sample_number] = sample.actions
        self.stabilities[self.sample_number] = [self.predict_stability(action) for action in sample.actions]
        self.calculate_rewards()
        self.sample_number += 1

    def predict_stability(
            self,
            action: material.Action,
            sample_number: Union[None, int] = None,
            ) -> Tuple[Union[None,float],Union[None,float]]:
        if sample_number == None:
            sample_number = self.sample_number
        if action.category == 'stability':
            return (action.outputs['stability'].flatten()[0],0.0)
        if action.category == 'turn_back':
            return (None,None)
        ignore_stability_target = False
        if self.actions[sample_number][-1].category == 'turn_back':
            ignore_stability_target = True
        GPR_key_list = [key
                    for key in self.GPRs.keys()
                    if action.name in key[0]
                    and sample_number in self.get_GPR_sample_numbers(key, ignore_stability_target=ignore_stability_target)]
        GPR_key = max(GPR_key_list, key=len)
        inputs = self.get_GPR_inputs(GPR_key, [sample_number])
        mean, std = self.GPRs[GPR_key].predict(inputs, return_std=True)
        return mean.flatten()[0], std.flatten()[0]

    def calculate_rewards(self, sample_number: Union[None, int] = None) -> None:
        if sample_number == None:
            sample_number = self.sample_number
        raise NotImplementedError

    def update_after_episode(self):
        self.update_epsilon()
        self.update_savfs()
        self.update_GPRs()
        self.processed_samples = len(self.actions)
        if self.include_MAE != False:
            self.calculate_MAE()

    def update_epsilon(self):
        return None
    
    def update_GPRs(self) -> None:
        for GPR_key in self.GPRs.keys():
            sample_numbers = self.get_GPR_sample_numbers(GPR_key)
            if len(sample_numbers) == 0:
                return None
            inputs = self.get_GPR_inputs(GPR_key, sample_numbers)
            targets = self.get_GPR_targets(GPR_key, sample_numbers)
            self.GPRs[GPR_key].fit(inputs, targets)

    def get_GPR_sample_numbers(
            self,
            GPR_key: Tuple[Tuple[str], Tuple[str]],
            ignore_stability_target: bool = False
            ) -> List[int]:
        input_names = deepcopy(GPR_key[0])
        target_names = deepcopy(GPR_key[1])
        if ignore_stability_target:
            target_names = tuple([exp_name for exp_name in target_names if exp_name != 'Stability'])
        return [sam_num
                for sam_num in self.get_sample_numbers(unprocessed_only=False)
                if  np.all(np.isin( input_names,[a.name for a in self.actions[sam_num]]))
                and np.all(np.isin(target_names,[a.name for a in self.actions[sam_num]]))]
    
    def get_GPR_inputs(
            self,
            GPR_key: Tuple[Tuple[str], Tuple[str]],
            sample_numbers: List[int]) -> np.ndarray:
        inputs = []
        for sam_num in sample_numbers:
            temp_map = {action.name:idx for idx,action in enumerate(self.actions[sam_num])}
            action_idxs = [temp_map[name] for name in GPR_key[0]]
            inputs.append([_ for l in [self.actions[sam_num][idx].get_outputs() for idx in action_idxs] for _ in l])
        return np.array(inputs).reshape(len(sample_numbers),-1)

    def get_GPR_targets(
            self,
            GPR_key: Tuple[Tuple[str], Tuple[str]],
            sample_numbers: List[int]) -> np.ndarray:
        targets = []
        for sam_num in sample_numbers:
            temp_map = {action.name:idx for idx,action in enumerate(self.actions[sam_num])}
            action_idxs = [temp_map[name] for name in GPR_key[1]]
            targets.append([_ for l in [self.actions[sam_num][idx].get_outputs() for idx in action_idxs] for _ in l])
        return np.array(targets).reshape(len(sample_numbers),-1)
    
    def calculate_MAE(self):
        for key in self.MAE:
            input_labels, input_values = self.environment.experiments[key[0][0]].get_input_space()
            prediction = self.GPRs[key].predict(input_values).flatten()
            truth = self.environment.experiments[key[1][0]].calc_stability({(key[0][0],label):input_values[:,lidx].reshape(-1,1) for lidx,label in enumerate(input_labels)}).flatten()
            self.MAE[key].append(
                (np.mean(np.absolute(truth-prediction)),
                 np.mean(np.abs(truth-prediction)/truth),
                 self.GPRs[key].score(input_values,truth)
                 ))
        return
    

class BO1(MLStrategy):

    def __init__(
            self,
            environment: Union[ None, world.VSDLEnvironment ] = None,
            discount: float = 0.1,
            epsilon: float = 0.5,
            include_MAE: bool = True,
            optimizer_method: str = 'Nelder-Mead',
            BO_acq_func_name: str = 'UCB',
            savf_update_method: str = 'MC',
            kernel: Union[ None, GP.kernels.Kernel ] = None,
            GPRs: Union[ None, GP.GaussianProcessRegressor ] = None, 
            GPR_dict: Union[ None, Dict[str, Union[ int, float, GP.kernels.Kernel ] ] ] = None,
            savfs: Union[ Tuple[int,float], Dict[str,Tuple[int,float]] ] = (0,0.0)) -> None:

        super().__init__(
            environment=environment,
            discount=discount,
            epsilon=epsilon,
            include_MAE=include_MAE,
            optimizer_method=optimizer_method,
            BO_acq_func_name=BO_acq_func_name,
            savf_update_method=savf_update_method,
            kernel=kernel,
            GPRs=GPRs,
            GPR_dict=GPR_dict,
            savfs=savfs)
        
    def update_epsilon(self):
        self.epsilon = np.min([ 0.90, self.epsilon + np.abs(0.2*self.epsilon) ])
        return
    
    def MC_savf_update(self):
        for sample_number in self.get_sample_numbers():
            if self.actions[sample_number][-1].category != 'stability':
                action = self.actions[sample_number][-1]
                count = self.savfs[action.name][0] + 1
                Gt = action.reward
                new_savf = self.savfs[action.name][1] + (Gt - self.savfs[action.name][1])/(count)
                self.savfs[action.name] = (count, new_savf)
                continue
            reward_list = [a.reward for a in self.actions[sample_number]]
            for action_idx,action in enumerate(self.actions[sample_number]):
                count = self.savfs[action.name][0] + 1
                Gt = np.sum([(self.discount**reward_idx)*reward for reward_idx,reward in enumerate(reward_list[action_idx:])])
                new_savf = self.savfs[action.name][1] + (Gt - self.savfs[action.name][1])/(count)
                self.savfs[action.name] = (count, new_savf)
    
    def get_processing_actions(self, number_of_samples=1):
        best = []
        for experiment_name in self.environment.get_experiment_names(category='processing'):
            GPR_key = tuple([tuple([experiment_name]),tuple(['Stability'])])
            for input_labels,input_values in self.environment.experiments[experiment_name].yield_input_spaces():
                mean, std = self.GPRs[GPR_key].predict(input_values, return_std=True)
                mean = mean.reshape(-1,1)
                std = std.reshape(-1,1)
                acq_func_output = self.BO_acq_func(mean,std)
                best = sorted(
                    [(
                        acq_func_output[idx][0],
                        input_labels,
                        input_values[idx],
                        experiment_name) for idx in range(len(input_values))] + best,
                    key = lambda x: x[0],
                    reverse=True)[:number_of_samples]
        return [
            material.Action(
                _[3],
                self.environment.experiments[_[3]].category,
                {variable_name:_[2][variable_idx] for variable_idx,variable_name in enumerate(_[1])}
            )
            for _ in best]

    def select_action(self, sample: material.Sample) -> material.Action:
        action_space = self.environment.get_action_space(sample)
        if not sample.actions:
            action = self.BO_selection(action_space)
        else:
            action = self.RL_selection(action_space)
        if action.category == 'stability' or action.category == 'turn_back':
            self.done = True
        return action

    def BO_selection(self, action_space):
        best = (None,{},-np.inf) # experiment name, {input labels:input values}, acq func value
        for experiment_name in action_space:
            GPR_key = tuple([tuple([experiment_name]),tuple(['Stability'])])
            for input_labels,input_values in self.environment.experiments[experiment_name].yield_input_spaces():
                mean, std = self.GPRs[GPR_key].predict(input_values, return_std=True)
                mean = mean.reshape(-1,1)
                std = std.reshape(-1,1)
                acq_func_output = self.BO_acq_func(mean,std)
                max_idx = np.argmax(acq_func_output)
                max_val = acq_func_output[max_idx]
                if max_val > best[2]:
                    best = (
                        experiment_name,
                        {var:input_values[max_idx,var_idx] for var_idx,var in enumerate(input_labels)},
                        max_val)
        return material.Action(
            best[0],
            self.environment.experiments[best[0]].category,
            inputs=best[1]
            )

    def RL_selection(self, actions):
        if np.random.rand() > self.epsilon:
            action = np.random.choice(actions)
            return material.Action(
                action,
                self.environment.experiments[action].category)
        else:
            temp_savfs = {_:self.savfs[_][1] for _ in actions}
            action = max(temp_savfs, key=temp_savfs.get)
            return material.Action(
                action,
                self.environment.experiments[action].category)

    def calculate_rewards(self, sample_number: Union[None, int] = None) -> None:

        if sample_number == None:
            sample_number = self.sample_number

        if self.stabilities[sample_number][-1][0] == None:

            for action_idx,action in enumerate(self.actions[sample_number][:-1]):
                self.actions[sample_number][action_idx].reward = -self.environment.experiments[action.name].cost

            stability_cost = self.environment.experiments[self.environment.get_experiment_names(category='stability')[0]].cost
            uncertainty = self.stabilities[sample_number][-2][1]/self.stabilities[sample_number][-2][0]
            self.actions[sample_number][-1].reward = stability_cost*(np.log(0.2/uncertainty)/2 - 1)

            return None
        
        true_stability = self.stabilities[sample_number][-1][0]
        for action_idx,action in enumerate(self.actions[sample_number]):

            if action.category in ['processing','stability']:
                self.actions[sample_number][action_idx].reward = -self.environment.experiments[action.name].cost

            if action.category == 'characterization':
                error = np.abs((self.stabilities[sample_number][action_idx][0] - true_stability)/(true_stability))
                self.actions[sample_number][action_idx].reward = self.environment.experiments[action.name].cost*np.log(0.3/error)/2

        return None


        mean_stabilities = [_[0] for _ in self.stabilities[sample_number]]
        std_stabilities =  [_[1] for _ in self.stabilities[sample_number]]
        
        for action_idx,action in enumerate(self.actions[sample_number]):

            if action.category == 'turn_back':
                self.actions[sample_number][action_idx].reward = 100*std_stabilities[-2]/mean_stabilities[-2]
                continue

            if action.category == 'processing':
                self.actions[sample_number][action_idx].reward = -self.environment.experiments[action.name].cost
                continue

            deltaRSD = 0
            mean_stabilities_1 = [mean_stabilities[action_idx-1], mean_stabilities[action_idx]]
            std_stabilities_1 = [std_stabilities[action_idx-1], std_stabilities[action_idx]]
            if 0 not in mean_stabilities_1:
                deltaRSD = (std_stabilities_1[0]/mean_stabilities_1[0]) - (std_stabilities_1[1]/mean_stabilities_1[1])
            self.actions[sample_number][action_idx].reward = 100*deltaRSD - self.environment.experiments[action.name].cost

        return None