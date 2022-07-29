import json
from functools import reduce

jsons = []
for i in range(500):
    with open(f'data/data_pitch_p{i}.json', 'r') as jsn:
        jsons.append(json.load(jsn))

out = {}
out['train'] = reduce(lambda a, b: {**a, **b}, [jsn['train'] for jsn in jsons])
out['valid'] = reduce(lambda a, b: {**a, **b}, [jsn['valid'] for jsn in jsons])
out['test'] = reduce(lambda a, b: {**a, **b}, [jsn['test'] for jsn in jsons])
jsons = None

with open('data/data_pitch_all.json', 'w') as jsn:
    jsn.write(json.dumps(out))
