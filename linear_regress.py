from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import json
import sys
import pitch_recons
from math import ceil

lang_id = sys.argv[1]

x_train, x_test = pitch_recons.get_states('hubert', lang_id)

x_train = [[x_train[layer][name] for name in sorted(x_train[layer])] for layer in range(13)]
x_train = [[utter_states for aud_states in layer for utter_states in aud_states] for layer in x_train]

x_test = [[x_test[layer][name] for name in sorted(x_test[layer])] for layer in range(13)]
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

for layer in range(13):
    print('---------')

    regr = LinearRegression()
    regr.fit(x_train[layer], y_train)

    scr = regr.score(x_test[layer], y_test)
    print(f'layer {layer} scr: {scr}')

    y_pred = regr.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'layer {layer} mse: {mse}')

    r2 = r2_score(y_test, y_pred)
    print(f'layer {layer} r2: {r2}')
