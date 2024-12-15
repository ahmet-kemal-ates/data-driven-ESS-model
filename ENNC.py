import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Lambda, RNN
from keras.layers.merge import Add, Multiply, Concatenate
from keras import backend as K

import sys
sys.path.append('../../../Libraries/Python')

from customLayers import FunctionalLink
from circuitRNNcells import VdynState


class ENNC:
    def __init__(self, Cn, Ts, cRate_in, SoC_in, Temp_in,
                 num_neurons_Ist=15, num_hidden_Ist=1,
                 hiddenActivation_Ist='relu', outputActivation_Ist='sigmoid',
                 num_tau=2, num_neurons_Dyn=15, num_hidden_Dyn=1, maxTau=10000, minTau=10,
                 hiddenActivation_Dyn='relu', outputActivation_Dyn='sigmoid',
                 num_cheby=0, num_bernstein=20, num_trig=10, num_bspline=0,
                 outputActivation_Qst='sigmoid',
                 ):

        # Save parameters
        # General parameters
        self.Cn = Cn
        self.Ts = Ts
        self.cRate_in = cRate_in
        self.SoC_in = SoC_in
        self.Temp_in = Temp_in

        # Vist net
        self.num_neurons_Ist = num_neurons_Ist
        self.num_hidden_Ist = num_hidden_Ist
        self.hiddenActivation_Ist = hiddenActivation_Ist
        self.outputActivation_Ist = outputActivation_Ist

        # Vdyn net
        self.num_tau = num_tau
        self.maxTau = maxTau
        self.minTau = minTau
        self.num_neurons_Dyn = num_neurons_Dyn
        self.num_hidden_Dyn = num_hidden_Dyn
        self.hiddenActivation_Dyn = hiddenActivation_Dyn
        self.outputActivation_Dyn = outputActivation_Dyn

        # Vqst net
        self.num_cheby = num_cheby
        self.num_bernstein = num_bernstein
        self.num_trig = num_trig
        self.num_bspline = num_bspline
        self.outputActivation_Qst = outputActivation_Qst


        ### BUILD NETWORK ###
        # Define input layers
        Iin = Input(shape=(None, 1), name='Iin')
        SoC = Input(shape=(None, 1), name='SoC')
        Temp = Input(shape=(None, 1), name='Temp')

        # Build parameteric input
        if Temp_in:
            if cRate_in:
                if SoC_in:
                    componentInput = Concatenate()([Iin, Temp, SoC])
                else:
                    componentInput = Concatenate()([Iin, Temp])
            else:
                if SoC_in:
                    componentInput = Concatenate()([Temp, SoC])
                else:
                    componentInput = Temp
        else:
            if cRate_in:
                if SoC_in:
                    componentInput = Concatenate()([Iin, SoC])
                else:
                    componentInput = Iin
            else:
                if SoC_in:
                    componentInput = SoC
                else:
                    componentInput = Lambda(lambda x: x**0)(Iin)

        # Initialize Input of each network
        Rist = componentInput
        Rdyn = componentInput
        tauDyn = componentInput

        if Temp_in:
            Vqst = Concatenate()([SoC, Temp])
        else:
            Vqst = SoC

        # Build Rist Net
        # Hidden Layers
        for n in range(num_hidden_Ist):
            Rist = TimeDistributed(Dense(num_neurons_Ist, activation=hiddenActivation_Ist, kernel_initializer='glorot_normal'),
                                   name='HidRistNet_' + n.__str__())(Rist)

        # Output Layer
        Rist = TimeDistributed(Dense(1, activation=outputActivation_Ist, kernel_initializer='glorot_normal'), name='OutRistNet')(Rist)
        # Vist = Rist*Iin
        Vist = Multiply(name='OutIstNet')([Rist, Iin])

        # Build dynamic network
        ## Rdyn Net
        # Replicate input current for each RC dipole
        Iin_p = Iin
        for n in range(num_tau - 1):
            Iin_p = Concatenate()([Iin_p, Iin])

        # Hidden Layers
        for n in range(num_hidden_Dyn):
            Rdyn = TimeDistributed(Dense(num_neurons_Dyn, activation=hiddenActivation_Dyn, kernel_initializer='glorot_normal'),
                                   name='HidRdynNet_' + n.__str__())(Rdyn)
        # Output Layer
        Rdyn = TimeDistributed(Dense(num_tau, activation=outputActivation_Dyn, kernel_initializer='glorot_normal'),
                               name='OutRdynNet')(Rdyn)
        # Get the value Rdyn*Iin
        RdynI = Multiply()([Rdyn, Iin_p])

        # tauDyn Net
        # Hidden Layers
        for n in range(num_hidden_Dyn):
            tauDyn = TimeDistributed(Dense(num_neurons_Dyn, activation=hiddenActivation_Dyn, kernel_initializer='glorot_normal'),
                                     name='HidTauDynNet_' + n.__str__())(tauDyn)
        # Output Layer
        tauDyn = TimeDistributed(
            Dense(num_tau, activation='sigmoid', kernel_initializer='glorot_normal',
                  ), name='OutTauDynNet')(tauDyn)

        # Concatenate tau and Rdyn*Iin as inputs of the recurrent layer
        Vdyn = Concatenate()([tauDyn, RdynI])
        #Create the reccurent layer
        cell_dyn = VdynState(num_tau, Ts=Ts, maxTau=maxTau, minTau=minTau)
        Vdyn = RNN(cell_dyn, return_sequences=True, name='VdynState')(Vdyn)
        # Mix all the Vdyn of each RC dipole
        trainable = True if self.num_tau != 1 else False
        init = 'glorot_normal' if self.num_tau != 1 else 'one'
        Vdyn = TimeDistributed(Dense(1, activation='linear', kernel_initializer=init, use_bias=False),
                               trainable=trainable, name='OutDynNet')(Vdyn)

        # Build Vqst network
        # Functional Reservoir
        Vqst = TimeDistributed(FunctionalLink(num_cheby=num_cheby, num_trig=num_trig, num_bernstein=num_bernstein, num_bspline=num_bspline),
                               name='HidQstNet')(Vqst)
        # Output Layer
        Vqst = TimeDistributed(Dense(1, activation=outputActivation_Qst, kernel_initializer='zero'), name='OutQstNet')(Vqst)

        # Mix the three network outputs
        Vout = Add()([Vist, Vqst, Vdyn])

        #Define the overall network
        if Temp_in:
            self.net = Model(inputs=[Iin, SoC, Temp], outputs=Vout)

            # Define the output functions
            self.VistFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input,
                                       self.net.get_layer(name='Temp').input],
                                      [self.net.get_layer(name='OutIstNet').output])

            self.VdynFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input,
                                       self.net.get_layer(name='Temp').input],
                                      [self.net.get_layer(name='OutDynNet').output])

            self.VdynsFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input,
                                        self.net.get_layer(name='Temp').input],
                                       [self.net.get_layer(name='OutDynNet').input])

            self.VqstFnc = K.function([self.net.get_layer(name='SoC').input, self.net.get_layer(name='Temp').input],
                                      [self.net.get_layer(name='OutQstNet').output])

            self.RistFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input,
                                       self.net.get_layer(name='Temp').input],
                                      [self.net.get_layer(name='OutRistNet').output])

            self.RdynFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input,
                                       self.net.get_layer(name='Temp').input],
                                      [self.net.get_layer(name='OutRdynNet').output])

            self.TauDynFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input,
                                         self.net.get_layer(name='Temp').input],
                                        [self.net.get_layer(name='OutTauDynNet').output])
        else:
            self.net = Model(inputs=[Iin, SoC], outputs=Vout)

            # Define the output functions
            self.VistFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input],
                                      [self.net.get_layer(name='OutIstNet').output])

            self.VdynFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input],
                                      [self.net.get_layer(name='OutDynNet').output])

            self.VdynsFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input],
                                       [self.net.get_layer(name='OutDynNet').input])

            self.VqstFnc = K.function([self.net.get_layer(name='SoC').input],
                                      [self.net.get_layer(name='OutQstNet').output])

            self.RistFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input],
                                      [self.net.get_layer(name='OutRistNet').output])

            self.RdynFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input],
                                      [self.net.get_layer(name='OutRdynNet').output])

            self.TauDynFnc = K.function([self.net.get_layer(name='Iin').input, self.net.get_layer(name='SoC').input],
                                        [self.net.get_layer(name='OutTauDynNet').output])


    def fit(self, x_tr, y_tr, nEpoch=2000, batchSize=1, optimizer='Nadam', loss='mse'):
        self.net.compile(optimizer='Nadam', loss='mse')
        history = self.net.fit(x_tr, y_tr, epochs=nEpoch, batch_size=batchSize, verbose=2)

        return history

    def GetWeights(self):
        ## Get Network Weights
        Rist_hidden = []
        for n in range(self.num_hidden_Ist):
            Rist_hidden.append(self.net.get_layer(name='HidRistNet_' + n.__str__()).get_weights())

        Rist_out = self.net.get_layer(name='OutRistNet').get_weights()

        Rdyn_hidden = []
        for n in range(self.num_hidden_Dyn):
            Rdyn_hidden.append(self.net.get_layer(name='HidRdynNet_' + n.__str__()).get_weights())
        Rdyn_out = self.net.get_layer(name='OutRdynNet').get_weights()

        tauDyn_hidden = []
        for n in range(self.num_hidden_Dyn):
            tauDyn_hidden.append(self.net.get_layer(name='HidTauDynNet_' + n.__str__()).get_weights())
        tauDyn_out = self.net.get_layer(name='OutTauDynNet').get_weights()
        tauDyn_gain = self.net.get_layer(name='VdynState').get_weights()

        Vdyn_out = self.net.get_layer(name='OutDynNet').get_weights()

        Vqst_out = self.net.get_layer(name='OutQstNet').get_weights()

        # Pad with zeros for the not active inputs
        if self.Temp_in:
            if self.cRate_in is True and self.SoC_in is False:
                Rist_hidden[0][0] = np.concatenate((Rist_hidden[0][0], np.zeros((1, Rist_hidden[0][0].shape[1]))),
                                                   axis=0)
                Rdyn_hidden[0][0] = np.concatenate((Rdyn_hidden[0][0], np.zeros((1, Rdyn_hidden[0][0].shape[1]))),
                                                   axis=0)
                tauDyn_hidden[0][0] = np.concatenate((tauDyn_hidden[0][0], np.zeros((1, tauDyn_hidden[0][0].shape[1]))),
                                                     axis=0)
            elif self.cRate_in is False and self.SoC_in is True:
                Rist_hidden[0][0] = np.concatenate((np.zeros((1, Rist_hidden[0][0].shape[1])), Rist_hidden[0][0]),
                                                   axis=0)
                Rdyn_hidden[0][0] = np.concatenate((np.zeros((1, Rdyn_hidden[0][0].shape[1])), Rdyn_hidden[0][0]),
                                                   axis=0)
                tauDyn_hidden[0][0] = np.concatenate((np.zeros((1, tauDyn_hidden[0][0].shape[1])), tauDyn_hidden[0][0]),
                                                     axis=0)
            elif self.cRate_in is False and self.SoC_in is False:
                Rist_hidden[0][0] = np.concatenate((np.zeros((1, Rist_hidden[0][0].shape[1])), Rist_hidden[0][0],
                                                    np.zeros((1, Rist_hidden[0][0].shape[1]))), axis=0)
                Rdyn_hidden[0][0] = np.concatenate((np.zeros((1, Rdyn_hidden[0][0].shape[1])), Rdyn_hidden[0][0],
                                                    np.zeros((1, Rdyn_hidden[0][0].shape[1]))), axis=0)
                tauDyn_hidden[0][0] = np.concatenate((np.zeros((1, tauDyn_hidden[0][0].shape[1])), tauDyn_hidden[0][0],
                                                      np.zeros((1, tauDyn_hidden[0][0].shape[1]))), axis=0)
        else:
            if self.cRate_in is True and self.SoC_in is False:
                Rist_hidden[0][0] = np.concatenate((Rist_hidden[0][0], np.zeros((1, Rist_hidden[0][0].shape[1]))), axis=0)
                Rdyn_hidden[0][0] = np.concatenate((Rdyn_hidden[0][0], np.zeros((1, Rdyn_hidden[0][0].shape[1]))), axis=0)
                tauDyn_hidden[0][0] = np.concatenate((tauDyn_hidden[0][0], np.zeros((1, tauDyn_hidden[0][0].shape[1]))),
                                                     axis=0)
            elif self.cRate_in is False and self.SoC_in is True:
                Rist_hidden[0][0] = np.concatenate((np.zeros((1, Rist_hidden[0][0].shape[1])), Rist_hidden[0][0]), axis=0)
                Rdyn_hidden[0][0] = np.concatenate((np.zeros((1, Rdyn_hidden[0][0].shape[1])), Rdyn_hidden[0][0]), axis=0)
                tauDyn_hidden[0][0] = np.concatenate((np.zeros((1, tauDyn_hidden[0][0].shape[1])), tauDyn_hidden[0][0]),
                                                 axis=0)

        Rist_w = {'hiddenActivation': self.hiddenActivation_Ist, 'outputActivation': self.outputActivation_Ist,
                  'W_i2h': Rist_hidden, 'W_h2o': Rist_out}
        Rdyn_w = {'hiddenActivation': self.hiddenActivation_Dyn, 'outputActivation': self.outputActivation_Dyn,
                  'W_i2h': Rdyn_hidden, 'W_h2o': Rdyn_out}
        tauDyn_w = {'hiddenActivation': self.hiddenActivation_Dyn, 'outputActivation': 'sigmoid',
                    'W_i2h': tauDyn_hidden, 'W_h2o': tauDyn_out, 'gain': tauDyn_gain, 'maxTau': float(self.maxTau),
                    'minTau': float(self.minTau)}
        Vdyn_w = {'W_mix': Vdyn_out}
        Vqst_w = {'outputActivation': self.outputActivation_Qst,
                  'num_cheby': float(self.num_cheby), 'num_trig': float(self.num_trig), 'num_bernstein': float(self.num_bernstein),
                  'W_h2o': Vqst_out}

        netWeights = {'Cn': self.Cn, 'Ts': float(self.Ts), 'Rist_w': Rist_w, 'Rdyn_w': Rdyn_w, 'tauDyn_w': tauDyn_w, 'Vdyn_w': Vdyn_w, 'Vqst_w': Vqst_w}
        self.netWeights = netWeights

        return netWeights

    def SetTrainableIst(self, trainable):
        for n in range(self.num_hidden_Ist):
            self.net.get_layer(name='HidRistNet_' + n.__str__()).trainable = trainable

        self.net.get_layer(name='OutRistNet').trainable = trainable

    def SetTrainableDyn(self, trainable):
        # Rdyn
        for n in range(self.num_hidden_Ist):
            self.net.get_layer(name='HidRdynNet_' + n.__str__()).trainable = trainable

        self.net.get_layer(name='OutRdynNet').trainable = trainable

        # tauDyn
        for n in range(self.num_hidden_Ist):
            self.net.get_layer(name='HidTauDynNet_' + n.__str__()).trainable = trainable

        self.net.get_layer(name='OutTauDynNet').trainable = trainable

        # Tau dyn gain
        self.net.get_layer(name='VdynState').trainable = trainable

        self.net.get_layer(name='OutDynNet').trainable = trainable

    def SetTrainableQst(self, trainable):
        self.net.get_layer(name='OutQstNet').trainable = trainable
