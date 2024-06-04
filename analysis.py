from sdlabs.utility import *

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



class OutputData():

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