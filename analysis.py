
from sdlabs import material,world,strategy
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
                    self.initial_datapoints_per_parameter = data[1][0]
                    self.number_of_batches = data[1][1]
                    self.samples_per_batch = data[1][2]
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


def compare_variable(data1, data2, variable):
    list_1 = data1.combine_runs(variable)
    list_2 = data2.combine_runs(variable)
    return [(sum(1 for x, y in zip(sublist1, sublist2) if x >= y) / len(sublist1))
            for sublist1, sublist2 in zip(list_1, list_2)]


def read_MAEoutput(MAEfile):
    run = 0
    MAEdata = {}
    with open(MAEfile) as f:
        for line in f:
            fields = line.split()
            if not fields: continue
            if fields[0] == '#': continue
            if fields[0].lower() == 'run':
                run = int(fields[1])
                continue
            if is_jsonable(line):
                MAEdata[run] = json.loads(line)
    return MAEdata