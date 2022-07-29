from datasets import load_dataset
import json
import os
import glob

def make_data(feature = None):
    '''
    Gather data from FLEURS and write it to a json file in the correct format for S3PRL

    feature: None | a label id from FLEURS (ie. lang_id, transcription, ...)
    '''

    dataset = load_dataset('google/fleurs', 'all')
    
    path = '/data/jtrmal1/jsalt22/smill229/audio/'
    open('data_gender1.json','w').write(json.dumps({ 
        'train': {sentence['audio']['path']: {
            'wav_path': path + 'train/' + sentence['audio']['path'], 
            'label': str(sentence[feature]) if feature else ''
        } for sentence in dataset['train'] if sentence['gender'] == 1},
        'valid': {sentence['audio']['path']: {
            'wav_path': path + 'dev/' + sentence['audio']['path'], 
            'label': str(sentence[feature]) if feature else ''
        } for sentence in dataset['validation'] if sentence['gender'] == 1},
        'test': {sentence['audio']['path']: {
            'wav_path': path + 'test/' + sentence['audio']['path'], 
            'label': str(sentence[feature]) if feature else ''
        } for sentence in dataset['test'] if sentence['gender'] == 1}
    }))

def get_paths(path_start, new_location):
    '''
    Symbolic link each audio file to a specified dirctory to be used later so the paths are cleaner. 
    '''

    # Train
    paths = ' '.join(glob.glob(f'{path_start}/huggingface/datasets/downloads/extracted/**/train/', recursive = True))
    os.system(f'ln -s {paths} {new_location}/audio/')

    # Dev
    paths = ' '.join(glob.glob(f'{path_start}/huggingface/datasets/downloads/extracted/**/dev/', recursive = True))
    os.system(f'ln -s {paths} {new_location}/audio/')

    # Test
    paths = ' '.join(glob.glob(f'{path_start}/huggingface/datasets/downloads/extracted/**/test/', recursive = True))
    os.system(f'ln -s {paths} {new_location}/audio/')