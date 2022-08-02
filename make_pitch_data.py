import json
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from multiprocessing import Pool
import os
import sys

lang_id = sys.argv[1]

# Conform to model - this is for HuBERT

def analyze(audio):
    '''
    Run pYAAPT, and return None if error
    '''
    try:
        return list(pYAAPT.yaapt(basic.SignalObj(audio), **{'frame_length' : 25.0, 'frame_space' : 20.0}).values) # Conform parameters to model - this is for HuBERT
    except:
        return None

def make_data(piece):
    '''
    Make data for a specific language.

    When there is no more dev or test data, it will just skip it.
    '''
    try:
        res = {audio_name: analyze(info['wav_path']) for audio_name, info in piece.items() if info['label'] == lang_id}        
        return res
    except:
        return {}

data = json.load(open('data.json', 'r'))

cores = int(os.environ['SLURM_JOB_NUM_NODES']) * int(os.environ['SLURM_CPUS_PER_TASK'])
with Pool(cores) as pool:
    output = pool.map(make_data, data.values())
    data['train'] = output[0]
    data['valid'] = output[1]
    data['test'] = output[2]

open(f'data/data_pitch{sys.argv[1]}.json','w').write(json.dumps(data))
