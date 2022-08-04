import torch
from s3prl.nn.upstream import S3PRLUpstream
import json
from scipy.io.wavfile import read
import sys
from numpy import pad

def get_states(name, lang_id, data_set, layers):
    model = S3PRLUpstream(name).cuda()
    model.eval()

    data = {}
    with open(f'data.json', 'r') as f:
        data = json.load(f)[data_set]

    all_states = {layer: {} for layer in layers}
    for states in all_states:
        with torch.no_grad():
            wavs = [read(audio['wav_path'])[1] for name, audio in data.items() if data[name]['label'] == lang_id]
            wavs = [torch.FloatTensor(wav).reshape(1, -1, 1).cuda() for wav in wavs]
            wav_lens = [torch.IntTensor([wav.size()[1]]).cuda() for wav in wavs]

            names = [name for name in data.keys() if data[name]['label'] == lang_id]

            for wav, wav_len, name in zip(wavs, wav_lens, names):
                hidden_states = model(wav, wav_len)['hidden_states']

                for layer in layers:
                    states[layer][name] = hidden_states[layer][0].tolist()

    return states
