import json
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from multiprocessing import Pool
import os
import sys

cores = int(os.environ['SLURM_JOB_NUM_NODES']) * int(os.environ['SLURM_CPUS_PER_TASK'])

def analyze(audio):
    '''
    Run pYAAPT, and return None if error
    '''
    try:
        return list(pYAAPT.yaapt(basic.SignalObj(audio)).samp_values)
    except:
        return None

def make_data(piece):
    '''
    Make data for the portion of the data in the current stride (set by sys arg, so that the script can be run in parallel, otherwise it would take days on a good cpu)

    Stride is set in tandem with the sys args so the whole dataset is used. When there is no more dev or test data, it will just skip it
    '''
    try:
        stride = 550 
        subset = list(piece.items())[stride*int(sys.argv[1]):stride*int(sys.argv[1])+stride]
        res = {audio_name: {'wav_path': info['wav_path'], 'label': analyze(info['wav_path'])} for audio_name, info in subset}        
        return res
    except:
        return {}

data = json.load(open('data.json', 'r'))

with Pool(cores) as pool:
    output = pool.map(make_data, data.values())
    data['train'] = output[0]
    data['valid'] = output[1]
    data['test'] = output[2]

with open(f'data/data_pitch_p{sys.argv[1]}.json','w') as f:
    f.write(json.dumps(data))
