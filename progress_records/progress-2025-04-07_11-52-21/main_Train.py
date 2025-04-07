from __future__ import unicode_literals
import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tkinter.filedialog import askopenfilename

from ENNC import ENNC

# Set seed for repeatability
np.random.seed(7)

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join("Models", f"ENNC_Run_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Read options from command line
arguments = sys.argv

### LOAD DATASET ###

trainFile = askopenfilename(initialdir=os.path.abspath('../../../Dataset/'), title='Load Training Data',
                            defaultextension='mat', filetypes=(("mat file", "*.mat"), ("All Files", "*.*"))) \
    if '-tr' not in arguments else arguments[arguments.index('-tr')+1]
dataTr = scipy.io.loadmat(trainFile)
tr_Time = dataTr['tr_Time']
tr_Iin = dataTr['tr_Iin']
tr_SoC = dataTr['tr_SoC']
tr_Vout = dataTr['tr_Vout']
tr_Temp = dataTr['Temp'] if 'Temp' in dataTr else np.zeros(tr_Iin.shape)
tr_Cn = dataTr['Cn'] if 'Cn' in dataTr else np.array([[1]])
tr_Ts = float(dataTr['Ts']) if 'Ts' in dataTr else 1

testFile = askopenfilename(initialdir=os.path.abspath('../Dataset/'), title='Load Test Data',
                           defaultextension='mat', filetypes=(("mat file", "*.mat"), ("All Files", "*.*"))) \
    if '-ts' not in arguments else arguments[arguments.index('-ts')+1]
dataTs = scipy.io.loadmat(testFile)
ts_Time = dataTs['ts_Time']
ts_Iin = dataTs['ts_Iin']
ts_SoC = dataTs['ts_SoC']
ts_Vout = dataTs['ts_Vout']
ts_Temp = dataTs['Temp'] if 'Temp' in dataTs else np.zeros(ts_Iin.shape)
ts_Cn = dataTs['Cn'] if 'Cn' in dataTs else np.array([[1]])
ts_Ts = float(dataTs['Ts']) if 'Ts' in dataTs else 1

assert tr_Ts == ts_Ts

### NORMALIZATION ###
headroom = 0.1
maxInputVTr = np.max(tr_Vout)
minInputVTr = np.min(tr_Vout)
delta = headroom * (maxInputVTr - minInputVTr)
maxInputVTr += delta / 2
minInputVTr -= delta / 2

maxInputITr = (1 + headroom) * np.max(np.abs(tr_Iin))
maxInputTTr = 60

tr_Vout = (tr_Vout - minInputVTr) / (maxInputVTr - minInputVTr)
tr_Iin /= maxInputITr
tr_Temp /= maxInputTTr

ts_Vout = (ts_Vout - minInputVTr) / (maxInputVTr - minInputVTr)
ts_Iin /= maxInputITr
ts_Temp /= maxInputTTr

### RESHAPE INPUTS ###
inputTr = [np.reshape(tr_Iin, (-1, 1, 1)), np.reshape(tr_SoC, (-1, 1, 1))]
inputTs = [np.reshape(ts_Iin, (-1, 1, 1)), np.reshape(ts_SoC, (-1, 1, 1))]
outputTr = np.reshape(tr_Vout, (-1, 1, 1))
outputTs = np.reshape(ts_Vout, (-1, 1, 1))

Temp_in = 'Temp' in dataTr and np.any(tr_Temp)
if Temp_in:
    inputTr.append(np.reshape(tr_Temp, (-1, 1, 1)))
    inputTs.append(np.reshape(ts_Temp, (-1, 1, 1)))

### DEFINE MODEL ###
nEpoch = 2000
batchSize = 64
optimizer = 'Nadam'
loss_fn = 'mse'

ennc_model = ENNC(Cn=tr_Cn, Ts=tr_Ts, cRate_in=True, SoC_in=True, Temp_in=Temp_in,
                  num_neurons_Ist=15, num_hidden_Ist=1, hiddenActivation_Ist='relu', outputActivation_Ist='sigmoid',
                  num_tau=1, num_neurons_Dyn=15, num_hidden_Dyn=1, maxTau=10000, minTau=10,
                  hiddenActivation_Dyn='relu', outputActivation_Dyn='sigmoid',
                  num_cheby=0, num_bernstein=20, num_trig=10, num_bspline=0,
                  outputActivation_Qst='sigmoid')

### TRAINING ###
start_time = time.time()
history = ennc_model.fit(inputTr, outputTr, nEpoch=nEpoch, batchSize=batchSize, optimizer=optimizer, loss=loss_fn)
trainingTime = time.time() - start_time

### EVALUATION ###
pred_train = ennc_model.net.predict(inputTr).reshape(-1)
pred_test = ennc_model.net.predict(inputTs).reshape(-1)
true_train = outputTr.reshape(-1)
true_test = outputTs.reshape(-1)

mse_train = ennc_model.net.test_on_batch(inputTr, outputTr)
mse_test = ennc_model.net.test_on_batch(inputTs, outputTs)

mae = mean_absolute_error(true_test, pred_test)
rmse = np.sqrt(mean_squared_error(true_test, pred_test))
r2 = r2_score(true_test, pred_test)

### SAVE MODEL AND RESULTS ###
with open(os.path.join(output_dir, "model.json"), 'w') as f:
    json.dump(json.loads(ennc_model.net.to_json()), f)
ennc_model.net.save_weights(os.path.join(output_dir, "model_weights.weights.h5"))

scipy.io.savemat(os.path.join(output_dir, "results.mat"), {
    'tr_Time': tr_Time, 'ts_Time': ts_Time,
    'tr_Iin': tr_Iin, 'ts_Iin': ts_Iin,
    'tr_SoC': tr_SoC, 'ts_SoC': ts_SoC,
    'tr_Temp': tr_Temp, 'ts_Temp': ts_Temp,
    'tr_Vout': tr_Vout, 'ts_Vout': ts_Vout,
    'trainingTime': trainingTime, 'lossHistory': history.history['loss'],
    'mse_train': mse_train, 'mse_test': mse_test,
    'mae': mae, 'rmse': rmse, 'r2': r2
})

### PLOTS ###
plt.figure()
plt.plot(true_test, label="Actual")
plt.plot(pred_test, label="Predicted", linestyle='--')
plt.title("Predicted vs Actual (Test Set)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, f"predicted_vs_actual_{timestamp}.png"))

plt.figure()
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(os.path.join(output_dir, f"training_loss_{timestamp}.png"))

### SAVE LOG FILE ###
log_path = os.path.join(output_dir, "log.txt")
with open(log_path, 'w') as log:
    log.write(f"ENNC Model Training Log\n")
    log.write(f"Run Timestamp: {timestamp}\n")
    log.write(f"Training File: {os.path.basename(trainFile)}\n")
    log.write(f"Test File: {os.path.basename(testFile)}\n")
    log.write(f"Training Time: {trainingTime:.2f} seconds\n")
    log.write(f"Hyperparameters:\n")
    log.write(f"  Epochs: {nEpoch}\n")
    log.write(f"  Batch Size: {batchSize}\n")
    log.write(f"  Optimizer: {optimizer}\n")
    log.write(f"  Loss Function: {loss_fn}\n")
    log.write(f"MSE Train: {mse_train:.6f}\n")
    log.write(f"MSE Test: {mse_test:.6f}\n")
    log.write(f"MAE: {mae:.6f}\n")
    log.write(f"RMSE: {rmse:.6f}\n")
    log.write(f"R2 Score: {r2:.6f}\n")

print(f"Model saved in: {output_dir}")
print(f"Train MSE: {mse_train}, Test MSE: {mse_test}")
print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")
