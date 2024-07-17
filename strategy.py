#!/usr/bin/env python3


import gc, warnings
from sdlabs import material,world
from sdlabs.utility import *
from scipy import optimize
from copy import deepcopy
from typing import *
import numpy as np
import sklearn.gaussian_process as GP
import scipy.stats as sps


class MLStrategy:
    def __init__(
            self,
            epsilon: float = 0.5, # larger epsilon corresponds to stronger exploitation
            discount: float = 0.1,
            include_MAE: bool = False,
            optimizer_method: str = 'Nelder-Mead',
            environment: Union[ None, world.VSDLEnvironment ] = None,
            savfs: Union[ Tuple[int,float], Dict[str,Tuple[int,float]] ] = (0,0.0),
            kernel: Union[ None, GP.kernels.Kernel ] = None,
            GP_dict: Union[ None, Dict[str, Union[ int, float, GP.kernels.Kernel ] ] ] = None,
            GPs: Union[ None, GP.GaussianProcessRegressor ] = None,
            BO_acq_func_name: str = 'UCB',
            savf_update_method_name: str = 'MC',
            ) -> None:
        self.actions = {}
        self.stabilities = {}
        self.sample_number = 1
        self.processed_samples = 0
        self.epsilon = epsilon
        self.discount = discount
        self.include_MAE = include_MAE
        self.optimizer_method = optimizer_method
        self.environment = environment
        self.savfs = savfs
        self.kernel = kernel
        self.GP_dict = GP_dict
        self.GPs = GPs
        self.MAEs = None
        self.clean_savfs()
        self.clean_GPs()
        self.clean_MAEs()
        self.set_BO_acq_func(BO_acq_func_name)
        self.set_savf_update_method(savf_update_method_name)

    def clean_savfs(self) -> None:
        if self.environment != None and type(self.savfs) != dict:
            self.savfs = {_:deepcopy(self.savfs) for _ in self.environment.get_input_keys(include_stability=True)}
        return

    def clean_GPs(self) -> None:
        if self.kernel == None:
            self.kernel = GP.kernels.Matern()
        if self.GP_dict == None:
            self.GP_dict = {
                'kernel': deepcopy(self.kernel),
                'alpha': 1e-9,
                'n_restarts_optimizer': 10,
            }
        if 'kernel' not in self.GP_dict.keys():
            self.GP_dict['kernel'] = deepcopy(self.kernel)
        self.GP_dict['optimizer'] = self.custom_optimizer
        if self.GPs == None:
            self.GPs = GP.GaussianProcessRegressor(**self.GP_dict)
        if self.environment != None and type(self.GPs) != dict:
            self.GPs = {input_key:deepcopy(self.GPs) for input_key in self.environment.get_input_keys(include_stability=False)}
        return 

    def clean_MAEs(self):
        if self.include_MAE == False:
            self.MAEs = None
            return 
        if self.environment == None or type(self.GPs) != dict:
            return
        self.MAEs = {_:[] for _ in self.GPs.keys()}
        return

    def BO_probability_of_improvement(self, mean, std):
        # Snoeck et al., "Practical Bayesian Optimization of Machine Learning Algorithms"
        #fbest = np.max([alist[-1].outputs['stability'] for alist in self.actions.values() if alist[-1].outputs.get('stability',None) != None])
        fbest = np.max([action_list[-1].outputs.get('stability',-np.inf) for action_list in self.actions.values()])
        return sps.norm.cdf((mean - fbest) / std)

    def BO_expected_improvement(self, mean, std):
        # Frazier and Wang, "Bayesian Optimization for Materials Design"
        #fbest = np.max([alist[-1].outputs['stability'] for alist in self.actions.values() if alist[-1].outputs.get('stability',None) != None])
        fbest = np.max([action_list[-1].outputs.get('stability',-np.inf) for action_list in self.actions.values()])
        gamma = (mean - fbest) / std
        return (mean - fbest) * sps.norm.cdf(gamma) + std * sps.norm.pdf(gamma)

    def BO_UCB(self, mean, std):
        # Snoeck et al., "Practical Bayesian Optimization of Machine Learning Algorithms"
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
        return

    def MC_savf_update(self):
        for sample_number in self.get_sample_numbers():
            reward_list = [a.reward for a in self.actions[sample_number]]
            for action_idx,action in enumerate(self.actions[sample_number]):
                action_key = assemble_action_key(self.actions[sample_number][:action_idx+1],include_stability=True)
                count = self.savfs[action_key][0] + 1
                Gt = np.sum([(self.discount**reward_idx)*reward for reward_idx,reward in enumerate(reward_list[action_idx:])])
                new_savf = self.savfs[action_key][1] + (Gt - self.savfs[action_key][1])/(count)
                self.savfs[action_key] = (count, new_savf)
        return

    def set_savf_update_method(self, savf_update_method):
        method_map = {
            'MC': self.MC_savf_update
        }
        if savf_update_method not in method_map:
            raise KeyError("Invalid savf_update_method specified")
        setattr(self, 'update_savfs', method_map[savf_update_method])
        return

    def add_environment(self, environment: world.VSDLEnvironment, overwrite: bool = False):
        if isinstance(self.environment, world.VSDLEnvironment) and not overwrite:
            warnings.warn('Warning! Tried to add environment to {} instance, but an environment has already been added.'.format(self))
            return
        self.environment = environment
        self.clean_GPs()
        self.clean_savfs()
        self.clean_MAEs()
        return

    def custom_optimizer(self, obj_func, initial_theta, bounds=(1e-5,1e5)):
        if type(bounds) == tuple:
            bounds = np.log(np.vstack([bounds]).reshape(-1,2))    
        def obj_func_wrapper(theta):
            return obj_func(theta, eval_gradient=False)
        opt_res = optimize.minimize(obj_func_wrapper, initial_theta, bounds=bounds, method=self.optimizer_method)
        return opt_res.x, opt_res.fun

    def get_sample_numbers(self, unprocessed_only=True):
        starting_idx = 0
        if unprocessed_only:
            starting_idx = self.processed_samples
        return sorted(self.actions.keys())[starting_idx:]

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
        return

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
        GP_key = assemble_action_key(self.actions[sample_number][:int(np.argwhere(np.array([_.name for _ in self.actions[sample_number]]) == action.name))+1])
        inputs = self.get_GP_inputs(GP_key, [sample_number])
        mean, std = self.GPs[GP_key].predict(inputs, return_std=True)
        return mean.flatten()[0], std.flatten()[0]

    def calculate_rewards(self, sample_number: Union[None, int] = None) -> None:
        if sample_number == None:
            sample_number = self.sample_number
        raise NotImplementedError

    def update_after_episode(self):
        self.update_epsilon()
        self.update_savfs()
        self.update_GPs()
        self.processed_samples = len(self.actions)
        if self.include_MAE != False:
            self.calculate_MAE()
        return

    def update_epsilon(self):
        return None
    
    def update_GPs(self) -> None:
        for GP_key in self.GPs.keys():
            sample_numbers = self.get_GP_sample_numbers(GP_key)
            if len(sample_numbers) == 0:
                return None
            inputs = self.get_GP_inputs(GP_key, sample_numbers)
            targets = self.get_GP_targets(sample_numbers)
            self.GPs[GP_key].fit(inputs, targets)
        return

    def get_GP_sample_numbers(self, GP_key: Tuple[str]) -> List[int]:
        input_names = deepcopy(GP_key)
        return [sam_num
                for sam_num in self.get_sample_numbers(unprocessed_only=False)
                if  np.all(np.isin( input_names,[a.name for a in self.actions[sam_num]]))
                and self.actions[sam_num][-1].category == 'stability']
    
    def get_GP_inputs(
            self,
            GP_key: Tuple[str],
            sample_numbers: List[int]) -> np.ndarray:
        inputs = []
        for sam_num in sample_numbers:
            temp_map = {action.name:idx for idx,action in enumerate(self.actions[sam_num])}
            action_idxs = [temp_map[name] for name in GP_key]
            inputs.append([_ for l in [self.actions[sam_num][idx].get_outputs() for idx in action_idxs] for _ in l])
        return np.array(inputs).reshape(len(sample_numbers),-1)

    def get_GP_targets(
            self,
            sample_numbers: List[int]) -> np.ndarray:
        targets = [self.actions[sam_num][-1].get_outputs() for sam_num in sample_numbers]
        return np.array(targets).reshape(len(sample_numbers),-1)
    
    def calculate_MAE(self):
        processing_experiment_name = self.environment.get_experiment_names(category='processing')[0]
        stability_experiment_name = self.environment.get_experiment_names(category='stability')[0]
        input_space_labels, input_space_values = self.environment.experiments[processing_experiment_name].get_input_space()
        input_space_dict = {(processing_experiment_name,prop):input_space_values[:,idx].reshape(-1,1) for idx,prop in enumerate(input_space_labels)}
        experiment_outputs = {}
        for experiment_name in self.environment.get_experiment_names():
            if self.environment.experiments[experiment_name].category == 'processing':
                experiment_outputs[experiment_name] = input_space_values
            if self.environment.experiments[experiment_name].category == 'characterization':
                this_experiment_outputs = self.environment.experiments[experiment_name].calculate_outputs(input_space_dict,None)
                experiment_outputs[experiment_name] = np.concatenate(tuple([v.reshape(-1,1) for k,v in sorted(this_experiment_outputs.items())]), axis=1)
            if self.environment.experiments[experiment_name].category == 'turn_back':
                experiment_outputs[experiment_name] = np.zeros((input_space_values.shape[0],1))
        truth = self.environment.experiments[stability_experiment_name].calc_stability(input_space_dict).flatten()
        for key in self.MAEs.keys():
            GP_input_matrix = np.concatenate(tuple([experiment_outputs[experiment_name] for experiment_name in key]),axis=1)
            prediction = self.GPs[key].predict(GP_input_matrix).flatten()
            self.MAEs[key].append(
                (np.mean(np.absolute(truth-prediction)),
                 np.mean(np.abs(truth-prediction)/truth),
                 self.GPs[key].score(GP_input_matrix,truth)
                 ))
        return
    

class BO1(MLStrategy):
        
    def __init__(
            self,
            epsilon: float = 0.5, # larger epsilon corresponds to stronger exploitation
            discount: float = 0.1,
            include_MAE: bool = False,
            optimizer_method: str = 'Nelder-Mead',
            environment: Union[ None, world.VSDLEnvironment ] = None,
            savfs: Union[ Tuple[int,float], Dict[str,Tuple[int,float]] ] = (0,0.0),
            kernel: Union[ None, GP.kernels.Kernel ] = None,
            GP_dict: Union[ None, Dict[str, Union[ int, float, GP.kernels.Kernel ] ] ] = None,
            GPs: Union[ None, GP.GaussianProcessRegressor ] = None,
            BO_acq_func_name: str = 'UCB',
            savf_update_method_name: str = 'MC',
            ) -> None:

        super().__init__(
            epsilon=epsilon,
            discount=discount,
            include_MAE=include_MAE,
            optimizer_method=optimizer_method,
            environment=environment,
            savfs=savfs,
            kernel=kernel,
            GP_dict=GP_dict,
            GPs=GPs,
            BO_acq_func_name=BO_acq_func_name,
            savf_update_method_name=savf_update_method_name)
        
    def update_epsilon(self):
        self.epsilon = np.min([ 0.90, self.epsilon + np.abs(0.2*self.epsilon) ])
        return
    
    def get_processing_actions(self, number_of_samples=1):
        best = []
        for experiment_name in self.environment.get_experiment_names(category='processing'):
            GP_key = tuple([experiment_name])
            for input_labels,input_values in self.environment.experiments[experiment_name].yield_input_spaces():
                mean, std = self.GPs[GP_key].predict(input_values, return_std=True)
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
            raise NotImplementedError('BO Selection not currently implemented. Expected to choose processing actions using get_processing_actions instead.')
        else:
            action = self.RL_selection(action_space, sample)
        return action

    def BO_selection(self, action_space):
        raise NotImplementedError('BO Selection not currently implemented. Expected to choose processing actions using get_processing_actions instead.')
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

    def RL_selection(self, action_space, sample):
        if np.random.rand() > self.epsilon:
            action_name = np.random.choice(action_space)
            return material.Action(
                action_name,
                self.environment.experiments[action_name].category)
        partial_savf_key_listtype = [_ for _ in assemble_action_key(sample.actions)]
        temp_savfs = {_:self.savfs[tuple(partial_savf_key_listtype + [_])][1] for _ in action_space}
        action_name = max(temp_savfs, key=temp_savfs.get)
        return material.Action(
            action_name,
            self.environment.experiments[action_name].category)
    
    def MC_savf_update(self):
        for sample_number in self.get_sample_numbers():
            if self.actions[sample_number][-1].category != 'stability':
                action_key = assemble_action_key(self.actions[sample_number],include_stability=True)
                count = self.savfs[action_key][0] + 1
                Gt = self.actions[sample_number][-1].reward
                new_savf = self.savfs[action_key][1] + (Gt - self.savfs[action_key][1])/(count)
                self.savfs[action_key] = (count, new_savf)
                continue
            reward_list = [a.reward for a in self.actions[sample_number]]
            for action_idx,action in enumerate(self.actions[sample_number]):
                if action.category == 'processing':
                    continue
                action_key = assemble_action_key(self.actions[sample_number][:action_idx+1],include_stability=True)
                count = self.savfs[action_key][0] + 1
                Gt = np.sum([(self.discount**reward_idx)*reward for reward_idx,reward in enumerate(reward_list[action_idx:])])
                new_savf = self.savfs[action_key][1] + (Gt - self.savfs[action_key][1])/(count)
                self.savfs[action_key] = (count, new_savf)
        return

    def calculate_rewards(self, sample_number: Union[None, int] = None) -> None:

        if sample_number == None:
            sample_number = self.sample_number

        if self.stabilities[sample_number][-1][0] == None: # last experiment was TurnBack

            for action_idx,action in enumerate(self.actions[sample_number][:-1]):
                self.actions[sample_number][action_idx].reward = -self.environment.experiments[action.name].cost

            stability_cost = self.environment.experiments[self.environment.get_experiment_names(category='stability')[0]].cost
            uncertainty = self.stabilities[sample_number][-2][1]/self.stabilities[sample_number][-2][0]
            self.actions[sample_number][-1].reward = stability_cost*(np.log(0.2/uncertainty)/2 - 1)

            return
        
        true_stability = self.stabilities[sample_number][-1][0]
        for action_idx,action in enumerate(self.actions[sample_number]):

            if action.category in ['processing','stability']:
                self.actions[sample_number][action_idx].reward = -self.environment.experiments[action.name].cost

            if action.category == 'characterization':
                error = np.abs((self.stabilities[sample_number][action_idx][0] - true_stability)/(true_stability))
                self.actions[sample_number][action_idx].reward = self.environment.experiments[action.name].cost*np.log(0.3/error)/2

        return


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

        return