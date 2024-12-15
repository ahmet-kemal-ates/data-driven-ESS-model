from __future__ import unicode_literals
import os
import sys
import json
import time
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename, asksaveasfilename

from ENNC import ENNC

# Set seed for repeatability
np.random.seed(7)

# Read options from command line
arguments = sys.argv

### LOAD DATASET ###

# Read Training Set
trainFile = askopenfilename(initialdir=os.path.abspath('../../../Dataset/'), title='Load Training Data', defaultextension='mat',
                            filetypes=(("mat file", "*.mat"), ("All Files", "*.*"))) \
    if not '-tr' in arguments else arguments[arguments.index('-tr')+1]

dataTr = scipy.io.loadmat(trainFile)
tr_Time = dataTr['Time']
tr_Iin = dataTr['Iin']
tr_SoC = dataTr['SoC']
tr_Vout = dataTr['Vout']
tr_Temp = dataTr['Temp'] if 'Temp' in dataTr else np.zeros(tr_Iin.shape)
tr_Cn = dataTr['Cn']
tr_Ts = float(dataTr['Ts']) if 'Ts' in dataTr else 1

# Read Test Set
testFile = askopenfilename(initialdir=os.path.abspath('../DataSet/'), title='Load Test Data', defaultextension='mat',
                           filetypes=(("mat file", "*.mat"), ("All Files", "*.*"))) \
    if not '-ts' in arguments else arguments[arguments.index('-ts')+1]

dataTs = scipy.io.loadmat(testFile)
ts_Time = dataTs['Time']
ts_Iin = dataTs['Iin']
ts_SoC = dataTs['SoC']
ts_Vout = dataTs['Vout']
ts_Temp = dataTs['Temp'] if 'Temp' in dataTs else np.zeros(ts_Iin.shape)
ts_Cn = dataTs['Cn']
ts_Ts = float(dataTs['Ts']) if 'Ts' in dataTs else 1

assert tr_Ts == ts_Ts

### NORMALIZE ###

# Set parameters for normalization
headroom = 0.1
maxInputVTr = np.max(tr_Vout)
minInputVTr = np.min(tr_Vout)

delta = headroom*(maxInputVTr-minInputVTr)
maxInputVTr += delta/2
minInputVTr -= delta/2

maxInputITr = (1+headroom)*np.max(np.abs(tr_Iin))

maxInputTTr = 60

# Normalize training set data
tr_Vout = (tr_Vout - minInputVTr) / (maxInputVTr - minInputVTr)
tr_Iin /= maxInputITr
tr_Temp /= maxInputTTr

# Normalize test set data
ts_Vout = (ts_Vout - minInputVTr) / (maxInputVTr - minInputVTr)
ts_Iin /= maxInputITr
ts_Temp /= maxInputTTr

### DEFINE INPUTS AND OUTPUTS

# Reshape Training set for final evaluation
inputTr_Iin = np.reshape(tr_Iin.transpose(), (tr_Iin.shape[1], tr_Iin.shape[0], 1))
inputTr_SoC = np.reshape(tr_SoC.transpose(), (tr_SoC.shape[1], tr_SoC.shape[0], 1))
inputTr_Temp = np.reshape(tr_Temp.transpose(), (tr_Temp.shape[1], tr_Temp.shape[0], 1))
outputTr = np.reshape(tr_Vout.transpose(), (tr_Vout.shape[1], tr_Vout.shape[0], 1))

# Reshape Test Set as a tuple fo temporal sequences
inputTs_Iin = np.reshape(ts_Iin.transpose(), (ts_Iin.shape[1], ts_Iin.shape[0], 1))
inputTs_SoC = np.reshape(ts_SoC.transpose(), (ts_SoC.shape[1], ts_SoC.shape[0], 1))
inputTs_Temp = np.reshape(ts_Temp.transpose(), (ts_Temp.shape[1], ts_Temp.shape[0], 1))
outputTs = np.reshape(ts_Vout.transpose(), (ts_Vout.shape[1], ts_Vout.shape[0], 1))

Ts = tr_Ts

#arguments Define metaparameters
num_neurons_Ist = 15
num_hidden_Ist = 1
hiddenActivation_Ist = 'relu'
outputActivation_Ist = 'sigmoid'

num_tau = 1
num_Neurons_Dyn = 15
num_hidden_Dyn = 1
maxTau = 10000
minTau = 10
hiddenActivation_Dyn = 'relu'
outputActivation_Dyn = 'sigmoid'

num_cheby = 0
num_bernstein = 20
num_trig = 10
outputActivation_Qst = 'sigmoid'

cRate_in = True if '-crate' not in arguments else bool(int(arguments[arguments.index('-crate')+1]))
SoC_in = True if '-soc' not in arguments else bool(int(arguments[arguments.index('-soc')+1]))
Temp_in = True if '-temp' not in arguments else bool(int(arguments[arguments.index('-temp')+1]))
Temp_in *= tr_Temp.any()

ennc_model = ENNC(Cn=tr_Cn, Ts=Ts, cRate_in=cRate_in, SoC_in=SoC_in, Temp_in=Temp_in,
                  num_neurons_Ist=num_neurons_Ist, num_hidden_Ist=num_hidden_Ist,
                  hiddenActivation_Ist=hiddenActivation_Ist, outputActivation_Ist=outputActivation_Ist,
                  num_tau=num_tau, num_neurons_Dyn=num_Neurons_Dyn, num_hidden_Dyn=num_hidden_Dyn, maxTau=maxTau, minTau=minTau,
                  hiddenActivation_Dyn=hiddenActivation_Dyn, outputActivation_Dyn=outputActivation_Dyn,
                  num_cheby=num_cheby, num_bernstein=num_bernstein, num_trig=num_trig, num_bspline=0,
                  outputActivation_Qst=outputActivation_Qst
                  )

### START TRAINING ###
startWatch = time.time()
nEpoch = 2000
batchSize = 1

if Temp_in:
    history = ennc_model.fit([inputTr_Iin, inputTr_SoC, inputTr_Temp], outputTr,
                             nEpoch=nEpoch, batchSize=batchSize,
                             optimizer='Nadam', loss='mse')
else:
    history = ennc_model.fit([inputTr_Iin, inputTr_SoC], outputTr,
                             nEpoch=nEpoch, batchSize=batchSize,
                             optimizer='Nadam', loss='mse')

### END OF TRAINING ###
trainingTime = time.time()-startWatch
print('Elapsed Time: ', trainingTime)

## Get Network Weights
netWeights = ennc_model.GetWeights()
netWeights['maxInputITr'] = float(maxInputITr)
netWeights['maxInputVTr'] = float(maxInputVTr)
netWeights['minInputVTr'] = float(minInputVTr)
netWeights['maxInputTTr'] = float(maxInputTTr)

### PERFORMANCE EVALUATION ###
# Training Set
if Temp_in:
    Vout_Train = ennc_model.net.predict_on_batch([inputTr_Iin, inputTr_SoC, inputTr_Temp])

    # Evaluate loss on training set
    mse_train = ennc_model.net.test_on_batch([inputTr_Iin, inputTr_SoC, inputTr_Temp],
                                             outputTr)

    # Evaluate single components
    Vist_train = ennc_model.VistFnc([inputTr_Iin, inputTr_SoC, inputTr_Temp])[0]  # Ist Net
    Vdyn_train = ennc_model.VdynFnc([inputTr_Iin, inputTr_SoC, inputTr_Temp])[0]  # Dyn Net
    Vdyns_train = ennc_model.VdynsFnc([inputTr_Iin, inputTr_SoC, inputTr_Temp])[0][0]  # Dyn Net
    Vqst_train = ennc_model.VqstFnc([inputTr_SoC, inputTr_Temp])[0]  # Qst Net

    Rist_train = ennc_model.RistFnc([inputTr_Iin, inputTr_SoC, inputTr_Temp])[0]  # Rist Net
    Rdyn_train = ennc_model.RdynFnc([inputTr_Iin, inputTr_SoC, inputTr_Temp])[0][0]  # Rdyn Net
    tauDyn_train = ennc_model.TauDynFnc([inputTr_Iin, inputTr_SoC, inputTr_Temp])[0][0]  # tauDyn Net

    # Test Set
    Vout_test = ennc_model.net.predict([inputTs_Iin, inputTs_SoC, inputTs_Temp])

    # Evaluate loss on test set
    mse_test = ennc_model.net.test_on_batch([inputTs_Iin, inputTs_SoC, inputTs_Temp],
                                            outputTs)

    # Evaluate single components
    Vist_test = ennc_model.VistFnc([inputTs_Iin, inputTs_SoC, inputTs_Temp])[0]  # Ist Net
    Vdyns_test = ennc_model.VdynsFnc([inputTs_Iin, inputTs_SoC, inputTs_Temp])[0][0]  # Dyn Net
    Vdyn_test = ennc_model.VdynFnc([inputTs_Iin, inputTs_SoC, inputTs_Temp])[0]  # Dyn Net
    Vqst_test = ennc_model.VqstFnc([inputTs_SoC, inputTs_Temp])[0]  # Qst Net

    Rist_test = ennc_model.RistFnc([inputTs_Iin, inputTs_SoC, inputTs_Temp])[0]  # Rist Net
    Rdyn_test = ennc_model.RdynFnc([inputTs_Iin, inputTs_SoC, inputTs_Temp])[0][0]  # Rdyn Net
    tauDyn_test = ennc_model.TauDynFnc([inputTs_Iin, inputTs_SoC, inputTs_Temp])[0][0]  # tauDyn Net
else:
    Vout_Train = ennc_model.net.predict_on_batch([inputTr_Iin, inputTr_SoC])

    # Evaluate loss on training set
    mse_train = ennc_model.net.test_on_batch([inputTr_Iin, inputTr_SoC],
                                             outputTr)

    # Evaluate single components
    Vist_train = ennc_model.VistFnc([inputTr_Iin, inputTr_SoC])[0]  # Ist Net
    Vdyn_train = ennc_model.VdynFnc([inputTr_Iin, inputTr_SoC])[0]  # Dyn Net
    Vdyns_train = ennc_model.VdynsFnc([inputTr_Iin, inputTr_SoC])[0][0]  # Dyn Net
    Vqst_train = ennc_model.VqstFnc([inputTr_SoC])[0]  # Qst Net

    Rist_train = ennc_model.RistFnc([inputTr_Iin, inputTr_SoC])[0]  # Rist Net
    Rdyn_train = ennc_model.RdynFnc([inputTr_Iin, inputTr_SoC])[0][0]  # Rdyn Net
    tauDyn_train = ennc_model.TauDynFnc([inputTr_Iin, inputTr_SoC])[0][0]  # tauDyn Net

    # Test Set
    Vout_test = ennc_model.net.predict([inputTs_Iin, inputTs_SoC])

    # Evaluate loss on test set
    mse_test = ennc_model.net.test_on_batch([inputTs_Iin, inputTs_SoC],
                                            outputTs)

    # Evaluate single components
    Vist_test = ennc_model.VistFnc([inputTs_Iin, inputTs_SoC])[0]  # Ist Net
    Vdyns_test = ennc_model.VdynsFnc([inputTs_Iin, inputTs_SoC])[0][0]  # Dyn Net
    Vdyn_test = ennc_model.VdynFnc([inputTs_Iin, inputTs_SoC])[0]  # Dyn Net
    Vqst_test = ennc_model.VqstFnc([inputTs_SoC])[0]  # Qst Net

    Rist_test = ennc_model.RistFnc([inputTs_Iin, inputTs_SoC])[0]  # Rist Net
    Rdyn_test = ennc_model.RdynFnc([inputTs_Iin, inputTs_SoC])[0][0]  # Rdyn Net
    tauDyn_test = ennc_model.TauDynFnc([inputTs_Iin, inputTs_SoC])[0][0]  # tauDyn Net

# Save results
if not os.path.exists('Models'):
    os.mkdir('Models')

log = {'Training_Data': os.path.splitext(os.path.basename(trainFile))[0],
       'Test_Data': os.path.splitext(os.path.basename(testFile))[0],
       'Loss_Function': ennc_model.net.loss,
       'Optimizer': ennc_model.net.optimizer.__str__(),
       'Num_of_Epoch': nEpoch,
       'numCheby': num_cheby,
       'numTrig': num_trig,
       'numBernstein': num_bernstein,
       'cRate_in': cRate_in,
       'SoC_in': SoC_in}

# Save Model
defaultFileName = os.path.splitext(os.path.basename(trainFile))[0] + '_' + time.ctime().replace(':', '-')
outModelFile = asksaveasfilename(title='Save Model', initialdir=os.path.abspath('Models'), initialfile=defaultFileName,
                                 defaultextension='json', filetypes=(("json file", "*.json"), ("All Files", "*.*")))\
    if not '-os' in arguments else arguments[arguments.index('-os')+1]+defaultFileName
basenameOutPath = os.path.splitext(outModelFile)

model_json_string = ennc_model.net.to_json()
jsonModel = json.loads(model_json_string)
jsonModel['input_scaling'] = {'maxInputVTr': maxInputVTr,
               'minInputVTr': minInputVTr,
               'maxInputITr': maxInputITr}
model_json_string = json.dumps(jsonModel)
with open(outModelFile, 'w') as outJsonFile:
    outJsonFile.write(model_json_string)
ennc_model.net.save_weights(basenameOutPath[0] + '_Weights.h5')

# Save results in matlab file
scipy.io.savemat(basenameOutPath[0]+'_Matlab.mat', {
    # Log
    'log': log,
    # Normalization Data
    'maxInputVTr': maxInputVTr,
    'minInputVTr': minInputVTr,
    'maxInputITr': maxInputITr,
    # Real Inputs and Outputs Train
    'tr_Time': tr_Time,
    'tr_Iin': tr_Iin,
    'tr_SoC': tr_SoC,
    'tr_Vout': tr_Vout,
    'tr_Temp': tr_Temp,
    # Real Inputs and Outputs Test
    'ts_Time': ts_Time,
    'ts_Iin': ts_Iin,
    'ts_SoC': ts_SoC,
    'ts_Vout': ts_Vout,
    'ts_Temp': ts_Temp,
    # Network Inputs
    'inputTr_Iin': inputTr_Iin,
    'inputTr_SoC': inputTr_SoC,
    'inputTr_Temp': inputTr_Temp,
    'inputTs_Iin': inputTs_Iin,
    'inputTs_SoC': inputTs_SoC,
    'inputTs_Temp': inputTs_Temp,
    # Network Outputs
    'Vout_train': Vout_Train,
    'Vout_test': Vout_test,
    'Vqst_train': Vqst_train,
    'Vist_train': Vist_train,
    'Vdyn_train': Vdyn_train,
    'Vdyns_train': Vdyns_train,
    'Vqst_test': Vqst_test,
    'Vist_test': Vist_test,
    'Vdyn_test': Vdyn_test,
    'Vdyns_test': Vdyns_test,
    'Rist_train': Rist_train,
    'Rist_test': Rist_test,
    'Rdyn_train': Rdyn_train,
    'Rdyn_test': Rdyn_test,
    'tauDyn_train': tauDyn_train,
    'tauDyn_test': tauDyn_test,
    # Network Weights
    'netWeights': netWeights,
    # Network Performance
    'mse_train': mse_train,
    'mse_test': mse_test,
    'trainingTime': trainingTime,
    'lossHistory': history.history['loss']
    })

# Plotting training set results
plotEnable = True if len(arguments) == 1 else False
if plotEnable:
    plt.figure()
    plt.plot(outputTr[0, :])
    plt.plot(Vout_Train[0, :])
    plt.title('Total Estimation - Training Set\n mse: %.4e' % mse_train)
    plt.legend(['Real Voltage', 'Estimated Voltage'], loc=4)
    plt.grid = True
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Voltage')
    plt.show()

    plt.figure()
    plt.plot(Vist_train[0,])
    plt.title('Instantaneous Contribution Estimation - Training Set')
    plt.legend(['Real Voltage', 'Estimated Voltage'], loc=4)
    plt.grid = True
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Voltage')
    plt.show()

    plt.figure()
    plt.plot(Vdyn_train[0,])
    plt.title('Dynamic Contribution Estimation - Training Set')
    plt.legend(['Real Voltage', 'Estimated Voltage'], loc=4)
    plt.grid = True
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Voltage')
    plt.show()

    plt.figure()
    plt.plot(Vqst_train[0,])
    plt.title('Quasi-stationary Contribution Estimation - Training Set')
    plt.legend(['Real Voltage', 'Estimated Voltage'], loc=3)
    plt.grid = True
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Voltage')
    plt.show()

    # Plotting test set results
    plt.figure()
    plt.plot(outputTs[0,])
    plt.plot(Vout_test[0,])
    plt.title('Total Estimation - Test Set\n mse: %.4e' % mse_test)
    plt.legend(['Real Voltage', 'Estimated Voltage'], loc=4)
    plt.grid = True
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Voltage')
    plt.show()

    plt.figure()
    plt.plot(Vist_test[0,])
    plt.title('Instantaneous Contribution Estimation - Test Set')
    plt.legend(['Real Voltage', 'Estimated Voltage'], loc=4)
    plt.grid = True
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Voltage')
    plt.show()

    plt.figure()
    plt.plot(Vdyn_test[0,])
    plt.title('Dynamic Contribution Estimation - Test Set')
    plt.legend(['Real Voltage', 'Estimated Voltage'], loc=4)
    plt.grid = True
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Voltage')
    plt.show()

    plt.figure()
    plt.plot(Vqst_test[0,])
    plt.title('Quasi-stationary Contribution Estimation - Test Set')
    plt.legend(['Real Voltage', 'Estimated Voltage'], loc=4)
    plt.grid = True
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Voltage')
    plt.show()
