import torch
from s3prl.nn.upstream import S3PRLUpstream
import json
from scipy.io.wavfile import read
import sys
from numpy import pad

def get_states(name, lang_id):
    model = S3PRLUpstream(name).cuda()
    model.eval()
    model_layers = 13

    data = {}
    with open(f'data.json', 'r') as f:
        data = json.load(f)

    train_states = [{}] * model_layers
    test_states = [{}] * model_layers
    for data_set, states in zip([data['train'], data['test']], [train_states, test_states]):
        with torch.no_grad():
            wavs = [read(audio['wav_path'])[1] for name, audio in data_set.items() if data_set[name]['label'] == lang_id]
            wavs = [torch.FloatTensor(wav).reshape(1, -1, 1).cuda() for wav in wavs]
            wav_lens = [torch.IntTensor([wav.size()[1]]).cuda() for wav in wavs]

            names = [name for name in data_set.keys() if data_set[name]['label'] == lang_id]

            for wav, wav_len, name in zip(wavs, wav_lens, names):
                hidden_states, hidden_states_len = model(wav, wav_len).slice(2)

                for layer in range(model_layers):
                    states[layer][name] = hidden_states[layer][0].tolist()

    return train_states, test_states
