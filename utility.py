#!/usr/bin/env python3


from sdlabs import material
import inspect, datetime, json, pickle, gc
import numpy as np
import sklearn.gaussian_process as GP
import sklearn as skl
from copy import deepcopy


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


def initial_VSDL_exploration(
        agent, environment,
        number_of_initial_datapoints=3, number_of_batches=20, samples_per_batch=5, verbose=False):

    # Get lists of experiments
    processing_names = environment.get_experiment_names(category='processing')
    processing_parameters = [environment.experiments[_].get_input_space(length=number_of_initial_datapoints) for _ in processing_names]

    # Loop over all processing options
    for pname_idx,pname in enumerate(processing_names):
        if verbose:
            print('\n  initial VSDL exploration of {} ({})...'.format(pname,datetime.datetime.now()))
        for row_idx in range(len(processing_parameters[pname_idx][1])):

            # Create a new material
            agent.reset()
            sample = material.Material()

            # Process
            action = material.Action(
                pname,
                'processing',
                parameters={
                    label:processing_parameters[pname_idx][1][row_idx,col_idx]
                    for col_idx,label in enumerate(processing_parameters[pname_idx][0])
                }
            )
            state = agent.take_action(action, sample, environment)
            sample.add_state(state)

            # Run all characterizations
            for name in environment.get_experiment_names(category='characterization'):
                action=material.Action(name,'characterization')
                state = agent.take_action(action, sample, environment)
                sample.add_state(state)

            # Run all stabilities
            for name in environment.get_experiment_names(category='stability'):
                action = material.Action(name,'stability')
                state = agent.take_action(action, sample, environment)
                sample.add_state(state)

            # Update the sample info
            agent.update_after_sample(sample, environment)

    # Update the agent with all of the new information
    agent.update_after_episode(sample, environment)

    return agent


def run_VSDL_campaign(
        agent, environment,
        number_of_initial_datapoints=3, number_of_batches=20, samples_per_batch=5, verbose=False):

    for batch in range(number_of_batches):
        if verbose:
            print('\n  running batch {} ({})...'.format(batch,datetime.datetime.now()))
        agent.epsilon = np.min([0.95,agent.epsilon*1.3])
        for sample in range(samples_per_batch):
            if verbose:
                print('    running sample {}...'.format(sample))
            agent.reset()
            sample = material.Material()
            while not agent.done:
                if verbose:
                    print('      selecting action...')
                action = agent.select_action(sample, environment)
                if verbose:
                    print('      taking action...')
                state = agent.take_action(action, sample, environment)
                sample.add_state(state)
                agent.update_after_step(sample, environment)
            agent.update_after_sample(sample, environment)
        agent.update_after_episode(sample, environment)

    return agent   


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


def dump_to_output(
        filename, environment, agent,
        number_of_initial_datapoints=3, number_of_batches=20, samples_per_batch=5, run=None):
    
    if not run:
        with open(filename,'w') as f:

            # Metadata
            f.write('Written {}\n\n'.format(datetime.datetime.now()))

            # Sampling Procedure
            f.write('~ SamplingProcedure ~\n\n')
            f.write('  Initial Datapoints per Processing Variable: {}\n'.format(number_of_initial_datapoints))
            f.write('  Number of Batches: {}\n'.format(number_of_batches))
            f.write('  Samples per Batch: {}\n\n'.format(samples_per_batch))

            # Environment
            f.write('~ Environment ~\n\n')
            for experiment in [v for k,v in sorted(environment.experiments.items())]:
                f.write('  Class -- {}\n'.format(experiment.__class__.__name__))
                f.write('  Category -- {}\n'.format(experiment.category))
                f.write('  Action Space -- {}\n'.format(experiment.action_space))
                f.write('  Inputs -- {}\n'.format(experiment.inputs))
                f.write('  Cost -- {}\n'.format(experiment.cost))
                f.write('{}\n\n'.format(inspect.getsource(experiment.calculate_outputs)))

            # Agent
            f.write('~ Agent ~\n\n')
            f.write('  discount -- {}\n'.format(agent.discount))
            f.write('  epsilon -- {}\n'.format(agent.epsilon))

    elif run:
        with open(filename,'a') as f:

            f.write('\n\n~ Run ~\n\n')
            f.write('  {}\n'.format(run))
            
            # Results
            for k,v in sorted(agent.__dict__.items()):
                if k.lower() in ['states']:
                    f.write('\n')
                    f.write('~ {} ~\n\n'.format(k.capitalize()))
                    for sample in sorted(v.keys()):
                        f.write('  {} {}\n'.format(sample,json.dumps([_.__dict__ for _ in v[sample]],sort_keys=True)))


                if k.lower() in ['rewards','predicted_stabilities','stabilities','savfs']:
                    f.write('\n')
                    f.write('~ {} ~\n\n'.format(k.capitalize()))
                    for key in sorted(v.keys()):
                        f.write('  {} {}\n'.format(key,json.dumps(v[key],sort_keys=True)))

                if k.lower() in ['gprs']:
                    f.write('\n')
                    f.write('~ {} ~\n\n'.format(k.capitalize()))
                    for name in sorted(v.keys()):
                        temp_dict = {prop:make_jsonable(deepcopy(val)) for prop,val in v[name].__dict__.items()}
                        f.write('  {} {}\n'.format(name,json.dumps(temp_dict)))

            # MAE
            #if MAE:


    return


def make_jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except:
        try:
            json.dumps(obj.__dict__)
            return obj.__dict__
        except:
            return 'Not jsonable ({})'.format(type(obj))
        

def is_jsonable(obj):
    try:
        json.loads(obj)
        return obj
    except:
        return None


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


def get_batch_sample_numbers(start,samples_per_batch,batches):
    if start != 0:
        return [list(range(1,start+1))] + [[sample+samples_per_batch*batch+start for sample in range(1,samples_per_batch+1)] for batch in range(0,batches)]
    return [[sample+samples_per_batch*batch+start for sample in range(1,samples_per_batch+1)] for batch in range(0,batches)]


class SDLOutputData():

    def __init__(self, name, color='#', calc_stability=None):
        self.name = name
        self.color = color
        self.calc_stability = calc_stability
        self.runs = 0
        self.sampling_procedure = []
        self.states = {}
        self.rewards = {}
        self.predicted_stabilities = {}
        self.stabilities = {}
        self.GPRs = {}
        self.savfs = {}

    def get_batch_sample_numbers(self, power=2):
        return get_batch_sample_numbers(
            self.sampling_procedure[0][1]**power,
            self.sampling_procedure[2][1],
            self.sampling_procedure[1][1],
            )
    
    def combine_batches(self, variable):
        return {run:[(np.mean(_),np.std(_)) for _ in getattr(self,variable)[run]]
            for run in getattr(self,variable).keys()}
    
    def combine_runs(self, variable):
        return [ [i for run in sorted(getattr(self,variable).keys()) for i in getattr(self,variable)[run][batch_idx]]
            for batch_idx in range(self.sampling_procedure[1][1]+1)
        ]
    
    def average_over_batches(self, variable):
        by_run_dict = self.combine_batches(variable)
        result = [[by_run_dict[run][batch_idx][0] for run in sorted(by_run_dict.keys())]
            for batch_idx in range(self.sampling_procedure[1][1]+1)
        ]
        return (
            np.array([np.mean(_) for _ in result]).flatten(),
            np.array([np.std(_)/np.sqrt(self.runs) for _ in result]).flatten(),
            )


def read_output(filename, data=None, power=2):

    if not data:
        data = SDLOutputData(filename[:-8])
    temp_dict = {_:{} for _ in ['predicted_stabilities','rewards','stabilities','states']}

    flag,run = None,None
    with open(filename,'r') as f:
        for line in f:
            fields = line.split()
            if not fields: continue
            if fields[0] == '#': continue
            if fields[0] == '~':
                flag = fields[1].lower()
                continue

            if flag == 'samplingprocedure':
                data.sampling_procedure.append((' '.join(fields[:-1])[:-1],int(float(fields[-1]))))
                continue
            
            if flag == 'run':
                run = int(float(fields[0]))
                for _ in temp_dict.keys():
                    temp_dict[_][run] = {}
                data.GPRs[run] = {}
                data.savfs[run] = {}
                data.runs += 1
                continue

            if flag in ['predicted_stabilities','rewards','stabilities','states']:
                temp_dict[flag][run][int(float(fields[0]))] = json.loads(' '.join(fields[1:]))
                continue

            if flag == 'savfs':
                data.savfs[run][fields[0]] = json.loads(' '.join(fields[1:]))
                continue

            if flag == 'gprs':
                data.GPRs[run][fields[0]] = json.loads(' '.join(fields[1:]))
                continue

    samples = data.get_batch_sample_numbers(power=power)
    for run in range(1,data.runs+1):
        data.predicted_stabilities[run] = [[temp_dict['predicted_stabilities'][run][_] for _ in sample_number_list] for sample_number_list in samples]
        data.stabilities[run] = [[temp_dict['stabilities'][run][_] for _ in sample_number_list] for sample_number_list in samples]
        data.rewards[run] = [[temp_dict['rewards'][run][_] for _ in sample_number_list] for sample_number_list in samples]
        data.states[run] = [[[material.State(**i) for i in temp_dict['states'][run][_]] for _ in sample_number_list] for sample_number_list in samples]

    return data


def compare_variable(data1, data2, variable):
    list_1 = data1.combine_runs(variable)
    list_2 = data2.combine_runs(variable)
    return [(sum(1 for x, y in zip(sublist1, sublist2) if x >= y) / len(sublist1))
            for sublist1, sublist2 in zip(list_1, list_2)]


def create_savename(architectures, test_numbers):
    return ' -- '.join(['_'.join([architectures[idx],test_numbers[idx]]) for idx in range(len(architectures))])


class CampaignInfo():

    def __init__(self,
                name='',
                runs=1,
                environment=None,
                agent = None,
                sampling_procedure = [
                    ('number_of_initial_datapoints', 3),
                    ('number_of_batches', 2),
                    ('samples_per_batch', 2)]
                ):
        self.name = name
        self.runs = runs
        self.environment = environment
        self.agent = agent
        self.sampling_procedure = sampling_procedure

    def dump_to_output(self, agent=None, run=None):
        filename = self.name + '.out.txt'
        if not agent:
            dump_to_output(
                filename, self.environment, self.agent,
                run=run, **{_[0]:_[1] for _ in self.sampling_procedure})
            return
        dump_to_output(
            filename, self.environment, agent,
            run=run, **{_[0]:_[1] for _ in self.sampling_procedure})
        return
    
    def dump_to_MAE(self, MAE=None):
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

    
    def run(self, verbose=False):
        self.dump_to_output()
        for run in range(1,self.runs+1):
            if verbose:
                print('Run {} ({})'.format(run,datetime.datetime.now()))
            agent = deepcopy(self.agent)
            agent.initialize_GPRs(self.environment)
            agent.initialize_savfs(self.environment)        
            agent = initial_VSDL_exploration(agent, self.environment, verbose=verbose, **{_[0]:_[1] for _ in  self.sampling_procedure})
            agent = run_VSDL_campaign(agent, self.environment, **{_[0]:_[1] for _ in  self.sampling_procedure}, verbose=verbose)
            if verbose:
                print('  dumping to output ({})...'.format(datetime.datetime.now()))
            self.dump_to_output(agent=agent, run=run)
        return
    
    def run_and_dump_MAE(self):
        data = read_output(
            self.name + '.out.txt',
            data=SDLOutputData(
                    self.name, 
                    calc_stability=self.environment.experiments['Stability'].calc_stability
            )
        )
        self.dump_to_MAE()
        for experiment_name in [k for k,v in self.environment.experiments.items() if v.category == 'processing']:
            reference_inputs = self.environment.experiments[experiment_name].get_input_space()
            reference_targets = data.calc_stability({k:reference_inputs[1][:,idx] for idx,k in enumerate(reference_inputs[0])})
            MAE = {run: get_cumulative_MAE(data,run,reference_inputs[1],reference_targets) for run in range(1,data.runs+1)}
            del reference_inputs,reference_targets
            gc.collect()
            self.dump_to_MAE(MAE)
        return
    

def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass


def dump_campaign_list(list_, filename):
    with open(filename, 'wb') as f:
        for _ in list_:
            pickle.dump(_,f)
    return


def read_campaign_list(filename):
    with open(filename, 'rb') as f:
        return [_ for _ in pickleLoader(f)]
    

def parse_function_inputs(inputs,variables):
    # input s/b Material instance or a dict
    # variables s/b [(processing experiment name,variable name), ...]
    if isinstance(inputs, material.Material):
        if len(variables) == 1:
            return inputs.pull_outputs(variables[0][0],variables[0][1])[0]
        return tuple([inputs.pull_outputs(_[0],_[1])[0] for _ in variables])
    if len(variables) == 1:
        return inputs.get(variables[0][1],None).reshape(-1,1)
    return tuple([inputs.get(_[1],None).reshape(-1,1) for _ in variables])
