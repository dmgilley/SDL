#!/usr/bin/env python3


import gc
from sdlabs import material
import numpy as np
import sklearn.gaussian_process as GP
import scipy.stats as sps


import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


class MLStrategy:
    def __init__(self, epsilon, BO_acq_func_name='UCB'):
        self.epsilon = epsilon
        self.states = {}
        self.rewards = {}
        self.stabilities = {}
        self.sample_number = 0
        self.set_BO_acq_func(BO_acq_func_name)

    def BO_probability_of_improvement(self, mean, std):
        # Snoeck et al., "Practical Bayesian Optimization of Machine Learning Algorithms"
        fbest = np.max([_ for _ in self.stabilities.values() if type(_) != list])
        return sps.norm.cdf((mean - fbest) / std)

    def BO_expected_improvement(self, mean, std):
        # Frazier and Wang, "Bayesian Optimization for Materials Design"
        fbest = np.max([_ for _ in self.stabilities.values() if type(_) != list])
        gamma = (mean - fbest) / std
        return (mean - fbest) * sps.norm.cdf(gamma) + std * sps.norm.pdf(gamma)

    def BO_UCB(self, mean, std):
        # Snoeck et al., "Practical Bayesian Optimization of Machine Learning Algorithms"
        fbest = np.max([_ for _ in self.stabilities.values() if type(_) != list])
        return mean + self.epsilon * std

    def set_BO_acq_func(self, BO_acq_func_name):
        BO_acq_func_map = {
            'Probability of Improvement': self.BO_probability_of_improvement,
            'Expected Improvement': self.BO_expected_improvement,
            'UCB': self.BO_UCB,
        }
        if BO_acq_func_name not in BO_acq_func_map:
            raise ValueError("Invalid BO_acq_func_name specified")
        setattr(self, 'BO_acq_func', BO_acq_func_map[BO_acq_func_name])

    def select_action(self, sample, environment):
        action = material.Action('name', 'category', parameters={})
        return action

    def take_action(self, action, sample, environment):
        outputs = environment.experiments[action.name].calculate_outputs(sample, action)
        reward = self.calculate_reward(action, outputs)
        state = material.State(action.name, action.category, outputs=outputs, reward=reward)
        return state
    
    def calculate_reward(self, action, outputs):
        return 0.0
    
    def update_after_step(self, sample, environment):
        return None
    
    def update_after_sample(self, sample, environment):
        return None

    def update_after_episode(self, sample, environment):
        return None


class ArchitectureOne(MLStrategy):

    def __init__(self, discount=0.1, epsilon=0.5):
        super().__init__(epsilon)
        self.predicted_stabilities = {}
        self.savfs = {}
        self.GPRs = {}
        self.discount = discount
        self.processed_samples = 0
        self.done = False

    def initialize_savfs(self, environment):
        self.savfs = {_:(0,0.0) for _ in environment.experiments.keys()}

    def initialize_GPRs(self, environment, kernel_type='RBF', kernel_dict={}, GPR_dict={'n_restarts_optimizer':5}):
        for experiment_name, experiment in environment.experiments.items():
            if experiment.category == 'stability' or experiment.category == 'turn_back':
                continue
            kernel_class = getattr(GP.kernels, kernel_type)
            kernel = kernel_class(**kernel_dict)
            self.GPRs[experiment_name] = GP.GaussianProcessRegressor(kernel=kernel, **GPR_dict)

    def reset(self):
        self.sample_number += 1
        self.states[self.sample_number] = []
        self.rewards[self.sample_number] = []
        self.stabilities[self.sample_number] = []
        self.predicted_stabilities[self.sample_number] = []
        self.done = False

    def update_after_sample(self, sample, environment):
        if sample.states[-1].category == 'turn_back':
            self.states[self.sample_number] = sample.states
            self.rewards[self.sample_number] = [_.reward for _ in sample.states]
            mean, _ = self.predict_stability(sample.states[-2])
            self.stabilities[self.sample_number] = mean
            return
        self.states[self.sample_number] = sample.states
        self.rewards[self.sample_number] = [_.reward for _ in sample.states]
        self.stabilities[self.sample_number] = sample.stability

    def update_after_episode(self, sample, environment):
        self.update_savfs()
        self.update_GPRs()
        self.processed_samples = len(self.states)

    def update_savfs(self): # RL algorithm dependent
        for sample_number in sorted(self.states.keys())[self.processed_samples:]:
            state_list = self.states[sample_number]
            reward_list = self.rewards[sample_number]
            for state_idx,state in enumerate(state_list):
                count = self.savfs[state.name][0] + 1
                Gt = np.sum([(self.discount**reward_idx)*reward for reward_idx,reward in enumerate(reward_list[state_idx:])])
                new_savf = self.savfs[state.name][1] + (Gt - self.savfs[state.name][1])/(count)
                self.savfs[state.name] = (count, new_savf)

    def get_unprocessed_sample_numbers(self):
        return sorted(self.states.keys())[self.processed_samples:]
    
    def get_GPR_inputs(self, existing_inputs, state):
        add_inputs = np.array([state.outputs[k] for k in sorted(state.outputs.keys())]).reshape(1,-1)
        if existing_inputs is None:
            return add_inputs
        return np.append(existing_inputs, add_inputs, axis=0)
    
    def get_GPR_targets(self, existing_targets, sample_number, state):
        
        stability = self.stabilities[sample_number]
        predicted_stability_characterizations = [
            _ for idx,_ in enumerate(self.predicted_stabilities[sample_number])
            if self.states[sample_number][idx].category == 'characterization']

        if state.category == 'processing':
            #value = stability
            #value = np.sum([_[0] for _ in predicted_stability_characterizations]) + stability
            #value = np.sum([_[0]-_[1] for _ in predicted_stability_characterizations]) + stability
            value = np.sum([_[0]-_[1] for _ in predicted_stability_characterizations])/np.max([1,len(predicted_stability_characterizations)]) + stability
            #total_unc = np.sum([_[1] for _ in predicted_stability_characterizations])
            #value = np.sum([_[0]*(1-_[1]/total_unc) for _ in predicted_stability_characterizations]) + stability
            #value = np.sum([_[0]*(1-_[1]/_[0]) for _ in predicted_stability_characterizations])/len(predicted_stability_characterizations) + stability

            add_targets = np.array([value]).reshape(1,-1)
            if existing_targets is None:
                return add_targets
            return np.append(existing_targets, add_targets, axis=0)
        
        if state.category == 'characterization':
            add_targets = np.array([stability]).reshape(1,-1)
            if existing_targets is None:
                return add_targets
            return np.append(existing_targets, add_targets, axis=0)
        
        if state.category == 'stability':
            add_targets = np.array([None]).reshape(1,-1)
            if existing_targets is None:
                return add_targets
            return np.append(existing_targets, add_targets, axis=0)
            

    def update_GPRs(self):
        for state_name in self.GPRs.keys():
            new_inputs, new_targets = None, None
            for sample_number in self.get_unprocessed_sample_numbers():
                state = {_.name:_ for _ in self.states[sample_number]}.get(state_name,None)
                if not state: continue
                new_inputs = self.get_GPR_inputs(new_inputs, state)
                new_targets = self.get_GPR_targets(new_targets, sample_number, state)
            if new_inputs is None: continue
            if hasattr(self.GPRs[state_name], 'X_train_'):
                new_inputs = np.append(self.GPRs[state_name].X_train_, new_inputs, axis=0)
                new_targets = np.append(self.GPRs[state_name].y_train_, new_targets, axis=0)
            self.GPRs[state_name].fit(new_inputs, new_targets)
            del new_inputs,new_targets
            gc.collect()

    def select_action(self, sample, environment):
        action_space = environment.get_action_space(sample)
        if not sample.states:
            action = self.BO_selection(action_space, environment)
        else:
            action = self.RL_selection(action_space, environment)
        if action.category == 'stability' or action.category == 'turn_back':
            self.done = True
        return action

    def BO_selection(self, action_space, environment):
        best = (None,{},-np.inf) # experiment name, {input labels:input values}, acq func value
        for experiment_name in action_space:
            for input_labels,input_values in environment.experiments[experiment_name].yield_input_spaces():
                mean, std = self.GPRs[experiment_name].predict(input_values, return_std=True)
                mean = mean.reshape(-1,1)
                std = std.reshape(-1,1)
                acq_func_output = self.BO_acq_func(mean,std)
                max_idx = np.argmax(acq_func_output)
                max_val = acq_func_output[max_idx]
                if max_val > best[2]:
                    best = (
                        experiment_name,
                        {var:input_values[max_idx,var_idx] for var_idx,var in enumerate(input_labels)},
                        max_val
                        )
                del input_values
                gc.collect()
        return material.Action(
            best[0],
            environment.experiments[best[0]].category,
            parameters=best[1]
            )

    def RL_selection(self, actions, environment):
        if np.random.rand() > self.epsilon:
            return self.RL_explore(actions, environment)
        else:
            return self.RL_exploit(actions, environment)

    def RL_explore(self, actions, environment):
        action = np.random.choice(actions)
        return material.Action(
            action,
            environment.experiments[action].category
            )

    def RL_exploit(self, actions, environment):
        temp_savfs = {_:self.savfs[_][1] for _ in actions}
        action = max(temp_savfs, key=temp_savfs.get)
        return material.Action(
            action,
            environment.experiments[action].category
            )
    
    def take_action(self, action, sample, environment):
        outputs = environment.experiments[action.name].calculate_outputs(sample, action)
        faux_reward = 0.0
        state = material.State(action.name, action.category, outputs=outputs, reward=faux_reward)
        if action.name != 'Stability' and action.category != 'turn_back':
            mean, std = self.predict_stability(state)
        elif action.category == 'turn_back':
            mean, std = self.predicted_stabilities[self.sample_number][-1]
        else:
            mean, std = outputs['stability'], 0.0
        self.predicted_stabilities[self.sample_number].append((mean,std))
        state.reward = self.calculate_reward(action, sample, environment)
        return state
    
    def predict_stability(self, state):
        X = np.array([v for k,v in sorted(state.outputs.items())]).reshape(1,-1)
        temp = self.GPRs[state.name].predict(X,return_std=True)
        return (temp[0].flatten()[0], temp[1].flatten()[0])

    def calculate_reward(self, action, sample, environment):
        if action.category == 'turn_back':
            mean, std = self.predicted_stabilities[self.sample_number][-1]
            return -std/mean*100
        if len(self.predicted_stabilities[self.sample_number]) < 2:
            return -environment.experiments[action.name].cost
        pred_stab = self.predicted_stabilities[self.sample_number]
        deltaRSD = 0
        if not 0 in [pred_stab[_][0] for _ in [-2,-1]]:
            deltaRSD = (pred_stab[-2][1]/pred_stab[-2][0]) - (pred_stab[-1][1]/pred_stab[-1][0])
        return 100*deltaRSD - environment.experiments[action.name].cost