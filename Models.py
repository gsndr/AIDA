from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten
from keras import optimizers
from keras.models import Model
from keras import regularizers
import numpy as np

from keras.layers import LeakyReLU

np.random.seed(12)
from tensorflow import set_random_seed

set_random_seed(12)


class Models():
    def __init__(self, n_classes):
        self._nClass = n_classes

    # model with three hidden layers: 80, 40, 20 neurons + 1 softmax layer
    def baselineModel(self, x_train, params):
        input_dim = Input(shape=(x_train.shape[1],))
        l1 = Dense(80, activation=params['first_activation'], kernel_initializer=params['kernel_initializer'])(
            input_dim)
        #l1= BatchNormalization()(l1)
       # l1 = Dropout(.5)(l1)
        l2 = Dense(40, activation=params['second_activation'], kernel_initializer=params['kernel_initializer'])(l1)
      #  l2 = Dropout(.5)(l2)
        l3 = Dense(20, activation=params['third_activation'], kernel_initializer=params['kernel_initializer'])(l2)
        l3 = Dropout(.5)(l3)
        # softmax = Dense(self._nClass, activation='softmax', kernel_initializer=params['kernel_initializer'],
        #                kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(l3)

        softmax = Dense(self._nClass, activation='softmax', kernel_initializer=params['kernel_initializer'])(l3)

        model = Model(input_dim, softmax)

        model.summary()
        model.compile(loss=params['losses'],
                      optimizer=params['optimizer'](),
                      metrics=['acc'])

        return model



    # A deep autoecndoer model
    def deepAutoEncoder(self, x_train, params):
        n_col = x_train.shape[1]
        input = Input(shape=(n_col,))
        # encoder_layer
        # Dropoout?
        #  input1 = Dropout(.2)(input)
        encoded = Dense(params['first_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'],
                        name='encoder1')(input)
        encoded = Dense(params['second_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'],

                        name='encoder2')(encoded)
        encoded = Dense(params['third_layer'], activation=params['first_activation'],
                        kernel_initializer=params['kernel_initializer'],  activity_regularizer=regularizers.l1(10e-5),

                        name='encoder3')(encoded)
       # l1 = BatchNormalization()(encoded)
       # encoded = Dropout(.5)(encoded)
        #LR = LeakyReLU(alpha=0.1)
        decoded = Dense(params['second_layer'], activation=params['first_activation'], kernel_initializer=params['kernel_initializer'], name='decoder1')(encoded)
        decoded = Dense(params['first_layer'], activation=params['second_activation'], kernel_initializer=params['kernel_initializer'], name='decoder2')(decoded)
        decoded = Dense(n_col, activation=params['third_activation'], kernel_initializer=params['kernel_initializer'], name='decoder')(decoded)
        # serve per L2 normalization?
        # encoded1_bn = BatchNormalization()(encoded)

        autoencoder = Model(input=input, output=decoded)
        autoencoder.summary
        learning_rate = 0.001
        decay = learning_rate / params['epochs']
        autoencoder.compile(loss=params['losses'],
                            optimizer=params['optimizer']()#(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8, amsgrad=False)#
                             , metrics=['accuracy'])

        return autoencoder





