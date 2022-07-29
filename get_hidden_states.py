import torch
from torch.nn.utils.rnn import pad_sequence
from s3prl.nn.upstream import S3PRLUpstream
import json
from scipy.io.wavfile import read
import sys
import numpy as np

batches = 25

name = 'hubert' # update with desired model
model = S3PRLUpstream(name).cuda()
model.eval()

data = {}
with open(f'data_pitch_all_filt.json', 'r') as f:
    data = json.load(f)

lang_id = {}
with open(f'data.json', 'r') as f:
    lang_id = {name: info['label'] for name, info in json.load(f)['test'].items()}

states = {}
with torch.no_grad():
    wavs = [torch.FloatTensor(read(audio['wav_path'])[1]).reshape(-1, 1) for name, audio in data['test'].items() if lang_id[name] == sys.argv[1]]
    wav_lens = torch.IntTensor([int(len(wav)) for wav in wavs]).cuda()
    wavs = pad_sequence(wavs, True, 500000).cuda()

    print(wavs[0])
    wavs = wavs.chunk(batches)
    wav_lens = wav_lens.chunk(batches)

    print(wavs[0])
    print(wav_lens[0])

    names = np.split(np.array([k for k in data['test'].keys() if lang_id[k] == sys.argv[1]]), [len(wav_len) for wav_len in wav_lens])

    for wav_batch, wav_batch_len, names_batch in zip(wavs, wav_lens, names):
        hidden_states, hidden_states_len = model(wav_batch, wav_batch_len).slice(2)
        
        states = {**states, **{name: state.tolist() for name, state in zip(names_batch, hidden_states)}}

with open(f'states/states{sys.argv[1]}.json', 'w') as f:
    f.write(json.dumps(states))
