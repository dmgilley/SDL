#!/usr/bin/env python3


###################################################################################################
# Imports
import json, pickle, linecache, tracemalloc
import numpy as np
from copy import deepcopy
from typing import *
from sdlabs import material


###################################################################################################
# Memory investigations 
def display_top(snapshot, key_type='lineno', limit=10):
    """
    Display memory allocation statistics for the top lines of code.

    Args:
        snapshot: A memory snapshot obtained using tracemalloc.take_snapshot().
        key_type (str): Specifies the sorting key for the statistics (default is 'lineno').
        limit (int): The number of top lines to display (default is 10).
    """
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print(f"Top {limit} lines")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(f"#{index}: {frame.filename}:{frame.lineno}: {stat.size / 1024:.1f} KiB")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print(f"    {line}")

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print(f"{len(other)} other: {size / 1024:.1f} KiB")
    total = sum(stat.size for stat in top_stats)
    print(f"Total allocated size: {total / 1024:.1f} KiB")


###################################################################################################
# File reading and writing
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


###################################################################################################
# General utility 
def get_batch_sample_numbers(start,samples_per_batch,batches):
    if start != 0:
        return [list(range(1,start+1))] + [[sample+samples_per_batch*batch+start for sample in range(1,samples_per_batch+1)] for batch in range(0,batches)]
    return [[sample+samples_per_batch*batch+start for sample in range(1,samples_per_batch+1)] for batch in range(0,batches)]


def parse_function_inputs(
        inputs: Union[float, Dict[str, np.ndarray], material.Sample],
        variables: Union[Tuple[str, ...], List[Tuple[str, ...]]] ) -> np.ndarray:
    """
    Parses function inputs and constructs an output array.

    Args:
        inputs ( float | Dict[str, np.ndarray] | material.Material ):
            Input data (float, dictionary, or material.Material instance).
        variables ( Tuple[str, ...] | List[Tuple[str, ...]] ):
            Variables (tuple(s)) specifying which outputs to extract.

    Returns:
        np.ndarray: Constructed output array.

    Notes:
        - If inputs is a float, variables should be either a single tuple or a list of a single tuple.
        - If inputs is a material.Material instance, it extracts outputs based on the specified action names.
        - If inputs is a dictionary, it concatenates arrays based on the specified keys.
    """
    if isinstance(inputs, float):
        if isinstance(variables, tuple):
            return np.array([[inputs]])
        elif isinstance(variables, list) and len(variables) == 1:
            return np.array([[inputs]])
        else:
            raise ValueError("Invalid variables format for float input.")

    elif isinstance(inputs, material.Sample):
        if not isinstance(variables, (tuple, list)):
            raise ValueError("Invalid variables format for material.Sample input.")
        output_values = []
        for action_name, output_name in variables:
            output_value = inputs.pull_outputs(action_name, output_name)[0]
            output_values.append(output_value)
        return np.array(output_values).reshape(1, -1)

    elif isinstance(inputs, dict):
        if not isinstance(variables, (tuple, list)):
            raise ValueError("Invalid variables format for dictionary input.")
        output_values = []
        for key in variables:
            if key in inputs:
                output_values.append(inputs[key])
            else:
                raise ValueError(f"Key '{key}' not found in the input dictionary.")
        return np.concatenate(output_values, axis=1)

    else:
        raise ValueError("Invalid input type. Supported types: float, dict, material.Material")