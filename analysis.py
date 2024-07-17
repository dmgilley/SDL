
from sdlabs import world,strategy
from sdlabs.utility import *
import datetime
from typing import *


class OutputData():

    def __init__(
            self,
            name: str = 'default_{}'.format(datetime.datetime.now()),
            timestamp = None,
            color: str = '#000000',
            runs: int = 0,
            initial_datapoints_per_parameter: Union[None, int] = None,
            number_of_batches: Union[None, int] = None,
            samples_per_batch: Union[None, int] = None,
            environment: world.VSDLEnvironment = world.VSDLEnvironment(),
            agents: Dict[int, strategy.MLStrategy] = {0: strategy.MLStrategy()},
            calc_stability: Union[None, object] = None,
            ) -> None:

        self.name = name
        self.timestamp = timestamp
        self.color = color
        self.runs = runs
        self.initial_datapoints_per_parameter= initial_datapoints_per_parameter
        self.number_of_batches = number_of_batches
        self.samples_per_batch = samples_per_batch
        self.environment = environment
        self.agents = agents
        self.calc_stability = calc_stability
        return
    
    def read_output(self) -> None:
        run = 0
        with open(self.name+'.out.pkl','rb') as file:
            for data in pickleLoader(file):
                if data[0] == 'timestamp':
                    self.timestamp = data[1]
                    continue
                if data[0] == 'sampling_procedure':
                    self.initial_datapoints_per_parameter = data[1][0][1]
                    self.number_of_batches = data[1][1][1]
                    self.samples_per_batch = data[1][2][1]
                    continue
                if data[0] == 'environment':
                    self.environment = data[1]
                    continue
                if data[0] == 'agent':
                    self.agents[run] = data[1]
                    continue
                if data[0] == 'run':
                    run = data[1]
                    continue
        self.runs = run
        return
    
    def compile_sample_numbers(self):
        initial_sample_numbers = list(range(
            1,
            self.initial_datapoints_per_parameter**(self.environment.get_input_dimensionality())+1))
        return [initial_sample_numbers] + [
            list(range(
                batch_idx*self.samples_per_batch+initial_sample_numbers[-1]+1,
                (batch_idx+1)*self.samples_per_batch+initial_sample_numbers[-1]+1))
                for batch_idx in range(self.number_of_batches)]

    def average_stability_over_runs(self, include_initial_exploration=False):
        starting_batch_idx = 1
        if include_initial_exploration:
            starting_batch_idx = 0
        compiled_data = [
            np.array([[
                self.agents[run].stabilities[sample_number][-1][0]
                if self.agents[run].stabilities[sample_number][-1][0] != None
                else self.agents[run].stabilities[sample_number][-2][0]
                for sample_number in list_of_sample_numbers ]#if self.agents[run].stabilities[sample_number][-1][0] != None]
                    for run in sorted(self.agents.keys()) if run != 0]).flatten()
                        for list_of_sample_numbers in self.compile_sample_numbers()[starting_batch_idx:]]
        return np.array([np.mean(_) for _ in compiled_data]).flatten(), np.array([np.std(_) for _ in compiled_data]).flatten()
    
    def average_MAE_over_runs(self, include_initial_exploration=False):
        starting_batch_idx = 1
        if include_initial_exploration:
            starting_batch_idx = 0
        batch_idxs = list(range(starting_batch_idx,self.number_of_batches+1))
        MAE_keys = sorted(list(self.agents[0].MAEs.keys()))
        compiled = {
            MAE_key:[
                [
                    self.agents[run].MAEs[MAE_key][b_idx] for run in range(self.runs+1) if run!= 0
                ]
                for b_idx in batch_idxs
            ]
            for MAE_key in MAE_keys
        }
        return {
            MAE_key:tuple(
                np.array([
                    [np.mean([_[idx] for _ in list_]) for list_ in compiled[MAE_key]],
                    [np.std([_[idx] for _ in list_])  for list_ in compiled[MAE_key]],]
                ).transpose()
            for idx in range(3))
        for MAE_key in MAE_keys}