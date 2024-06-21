#!/usr/bin/env python3


###################################################################################################
# Imports
from sdlabs import material,world,strategy
from sdlabs.analysis import OutputData
from sdlabs.utility import *
import datetime, inspect, gc, pickle
import sklearn as skl
import sklearn.gaussian_process as GP
from typing import *


###################################################################################################
# Functions for MAE calculations
def assemble_GPR_data(data,run):
    inputs, targets = [],[]
    for batch_idx in range(data.sampling_procedure[1][1]+1):
        inputs.append(np.array([_[0].get_outputs() for _ in data.states[run][batch_idx]]))
        targets.append(np.array([_[-1].get_outputs()[0] for _ in data.states[run][batch_idx]]).reshape(-1,1))
    return inputs, targets


def calc_cumulative_GPRs(inputs, targets, GPRbase):
    cumulative_GPRs = [deepcopy(GPRbase) for _ in (inputs)]
    for batch_idx in range(len(inputs)):
        temp_inputs = np.concatenate(inputs[:batch_idx+1])
        temp_outputs = np.concatenate(targets[:batch_idx+1])
        cumulative_GPRs[batch_idx].fit(temp_inputs,temp_outputs)
    return cumulative_GPRs


def calc_cumulative_MAEs(reference_inputs,reference_targets,cumulative_GPRs):
    predicted_targets = [_.predict(reference_inputs) for _ in cumulative_GPRs]
    cumulative_MAEs = [
        skl.metrics.mean_absolute_error(reference_targets,_)
        for _ in predicted_targets
    ]
    return cumulative_MAEs


def get_cumulative_MAE(data,run,reference_inputs,reference_targets,GPRbase=None):
    if not GPRbase:
        GPRbase = GP.GaussianProcessRegressor(GP.kernels.RBF(),n_restarts_optimizer=5)
    measured_inputs, measured_outputs = assemble_GPR_data(data,run)
    cumulative_GPRs = calc_cumulative_GPRs(measured_inputs, measured_outputs, GPRbase)
    cumulative_MAEs = calc_cumulative_MAEs(reference_inputs,reference_targets,cumulative_GPRs)
    return cumulative_MAEs


###################################################################################################
# Data writing
def dump_to_pkl_output(
        filename: str,
        environment: world.VSDLEnvironment,
        agent: strategy.MLStrategy,
        initial_datapoints_per_parameter: int,
        number_of_batches: int,
        samples_per_batch: int,
        run: Union[None, int] = None,
        ) -> None:

    if run == None:
        timestamp = "written {}".format(datetime.datetime.now())
        sampling_procedure = [
            ("initial_datapoints_per_parameter",initial_datapoints_per_parameter),
            ("number_of_batches",number_of_batches),
            ("samples_per_batch",samples_per_batch),
            ]
        with open(filename, 'wb') as f:
            pickle.dump(["timestamp",timestamp], f)
            pickle.dump(["sampling_procedure",sampling_procedure],f)
            pickle.dump(["environment",environment],f)
            pickle.dump(["agent",agent],f)
    
    elif run:
        with open(filename,'ab') as f:
            pickle.dump(["run",run],f)
            pickle.dump(["agent",agent],f)

    return


###################################################################################################
# Main class for coordinating and running a campaign
class Campaign():

    def __init__(self,
                name: str = 'default_{}'.format(datetime.datetime.now()),
                runs: int = 0,
                environment: world.VSDLEnvironment = world.VSDLEnvironment(),
                agent: strategy.MLStrategy = strategy.MLStrategy(),
                initial_datapoints_per_parameter: int = 0,
                number_of_batches: int = 0,
                samples_per_batch: int = 0,
                ):
        agent.add_environment(environment=environment)
        self.name = name
        self.runs = runs
        self.environment = environment
        self.agent = agent
        self.agent_base = agent
        self.initial_datapoints_per_parameter = initial_datapoints_per_parameter
        self.number_of_batches = number_of_batches
        self.samples_per_batch = samples_per_batch
    
    def run(self, verbose=False):
        dump_to_pkl_output(
            self.name + '.out.pkl', # filename
            self.environment, # environment
            self.agent_base, # base agent used; this is pre-campaign info
            run = None,
            initial_datapoints_per_parameter = self.initial_datapoints_per_parameter,
            number_of_batches = self.number_of_batches,
            samples_per_batch = self.samples_per_batch
            )
        for run in range(1,self.runs+1):
            if verbose:
                print('\nRun {} ({})'.format(run,datetime.datetime.now()))
            self.agent = deepcopy(self.agent_base)
            self.initial_VSDL_exploration(verbose=verbose)
            self.run_campaign(verbose=verbose)
            dump_to_pkl_output(
                self.name + '.out.pkl', # filename
                self.environment, # environment
                self.agent, # dump final agent information
                run=run,
                initial_datapoints_per_parameter = self.initial_datapoints_per_parameter,
                number_of_batches = self.number_of_batches,
                samples_per_batch = self.samples_per_batch
                )
        return

    def initial_VSDL_exploration(self, verbose: bool = False) -> None:
        for p_experiment_name in self.environment.get_experiment_names(category='processing'):
            parameter_labels, parameter_values = self.environment.experiments[p_experiment_name].get_input_space(
                length = self.initial_datapoints_per_parameter)
            if verbose:
                print('\n  initial VSDL exploration of {} ({})...'.format(p_experiment_name,datetime.datetime.now()))
            for row in parameter_values:
                sample = material.Sample()
                action = material.Action(
                    p_experiment_name,
                    'processing',
                    inputs={label:row[idx] for idx,label in enumerate(parameter_labels)})
                action = self.agent.take_action(sample, action)
                sample.add_action(action)
                for c_experiment_name in self.environment.get_experiment_names(category='characterization'):
                    action = material.Action(c_experiment_name, 'characterization')
                    action = self.agent.take_action(sample, action)
                    sample.add_action(action)
                for s_experiment_name in self.environment.get_experiment_names(category='stability'):
                    action = material.Action(s_experiment_name,'stability')
                    action = self.agent.take_action(sample, action)
                    sample.add_action(action)
                self.agent.update_after_sample(sample)
        self.agent.update_after_episode()

    def run_campaign(self, verbose: bool = False) -> None:
        for batch in range(1,self.number_of_batches+1):
            if verbose:
                print('\n  running batch {} ({})...'.format(batch,datetime.datetime.now()))
            processing_actions = self.agent.get_processing_actions(number_of_samples=self.samples_per_batch)
            for sample_idx in range(self.samples_per_batch):
                if verbose:
                    print('    running sample {} ({})...'.format(sample_idx+1,datetime.datetime.now()))
                sample = material.Sample()
                action = self.agent.take_action(sample, processing_actions[sample_idx])
                sample.add_action(action)
                while not sample.closed:
                    selected_action = self.agent.select_action(sample)
                    action = self.agent.take_action(sample, selected_action)
                    sample.add_action(action)
                self.agent.update_after_sample(sample)
            self.agent.update_after_episode()

    def dump_to_MAE(self, MAE: Union[None, Dict[int,float]] = None) -> None:
        filename = self.name + '.MAEout.txt'
        if not MAE:    
            with open(filename,'w') as f:
                f.write('Written {}\n\n'.format(datetime.datetime.now()))
            return
        with open(filename,'a') as f:
            for run in sorted(MAE.keys()):
                f.write('Run {}\n\n'.format(run))
                f.write('{}\n\n'.format(json.dumps(MAE[run])))
        return

    def calculate_MAE(self):
        data = read_output(
            self.name + '.out.txt',
            data=OutputData(
                    self.name, 
                    calc_stability = self.environment.experiments['Stability'].calc_stability))
        self.dump_to_MAE()
        for experiment_name in self.environment.get_experiment_names(category='processing'):
            reference_inputs = self.environment.experiments[experiment_name].get_input_space(length=25)
            reference_targets = data.calc_stability({k:reference_inputs[1][:,idx] for idx,k in enumerate(reference_inputs[0])})
            MAE = {run: get_cumulative_MAE(data,run,reference_inputs[1],reference_targets) for run in range(1,data.runs+1)}
            del reference_inputs,reference_targets
            gc.collect()
            self.dump_to_MAE(MAE)
        return