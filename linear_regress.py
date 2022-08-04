from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import json
import sys
import pitch_recons
from math import ceil

lang_id = sys.argv[1]
layers = range(13)
model = 'hubert'

x_train = pitch_recons.get_states(model, lang_id, 'train', layers)
x_train = [[x_train[layer][name] for name in sorted(x_train[layer])] for layer in layers]
x_train = [[utter_states for aud_states in layer for utter_states in aud_states] for layer in x_train]

x_test = pitch_recons.get_states(model, lang_id, 'test', layers)
x_test = [[x_test[layer][name] for name in sorted(x_test[layer])] for layer in layers]
x_test = [[utter_states for aud_states in layer for utter_states in aud_states] for layer in x_test]

#remove 2 data points
y_train = []
y_test = []
with open(f'data/data_pitch{lang_id}.json', 'r') as f:
    jsn = json.load(f)

    for name in sorted(jsn['train']):
        y_train += jsn['train'][name][::2] # Get every other since pYAAPT uses 160 samples instead of Hubert's 320

    for name in sorted(jsn['test']):
        y_test += jsn['test'][name][::2]

# Align mismatched frames if number of frames mismatch
# x_train = [x[(len(x) - len(y_train)) // 2:ceil((len(x) - len(y_train)) / 2)] for x in x_train]
# x_test = [x[(len(x) - len(y_test)) // 2:ceil((len(x) - len(y_test)) / 2)] for x in x_test]

mses = []
for layer in layers:
    print('---------')

    regr = LinearRegression()
    regr.fit(x_train[layer], y_train)

    y_pred = regr.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'layer {layer} mse: {mse}')
    mses.append(str(mse))

print('All MSE:')
print('\t'.join(mses))
print(f'lang_id: {lang_id}')
