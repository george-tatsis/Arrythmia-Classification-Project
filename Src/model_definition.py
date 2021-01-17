import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from keras.optimizers import Adam
from sklearn.utils import resample
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Softmax, Add, Flatten, Activation, MaxPool1D
from keras.layers.merge import concatenate
import time
import numpy as np

class Res_Conv_NN:
    def __init__(self):
        self._name_ = None
        self._time_ = {'with_resampling': 0, 'without_resampling': 0}
        self.blocks = 5
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                        ModelCheckpoint(filepath='best_model_inception.h5', monitor='val_loss', save_best_only=True)]
        self.input_shape = None
        self.history = None
        self.optimizer = Adam(lr = 0.001)

    def residual_block(self,prev_layer):
        conv5_1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(prev_layer)
        activ_1 = Activation("relu")(conv5_1)
        conv5_2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(activ_1)
        add = Add()([conv5_2, prev_layer])
        activ_2 = Activation("relu")(add)
        layer_out = MaxPooling1D(pool_size=5, strides=2)(activ_2)

        return layer_out

    def model(self):
        inp = Input(shape=self.input_shape)
        _block_ = Conv1D(filters=32, kernel_size=5, strides=1)(inp)

        for _ in range(self.blocks):
            _block_ = self.residual_block(_block_)

        F1 = Flatten()(_block_)
        D1 = Dense(32)(F1)
        A6 = Activation("relu")(D1)
        D2 = Dense(32)(A6)
        D3 = Dense(5)(D2)
        A7 = Softmax()(D3)

        model = Model(inputs=inp, outputs=A7)

        model.summary()
        print("\n")
        return model

    def classify(self, x_train, x_test, y_train, y_test):
        model = self.model()

        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        self.history = model.fit(x_train, y_train,
                            epochs = 30, batch_size = 100, 
                            validation_data=(x_test, y_test),
                            callbacks=self.callbacks)

        pred = model.predict(x_test)
        return pred
    
    def no_resampling(self,data_obj):
        self._name_ = 'Res_Conv_NN__no_resampling'
        start = time.time()

        print("\n\nRunning a convolutional neural network with {} residual blocks without resampling the data...".format(self.blocks))

        data_obj = data_obj.prepare(one_hot=True)
        x_train, x_test = data_obj.feat_train, data_obj.feat_test
        x_train = np.expand_dims(x_train, 2)
        x_test = np.expand_dims(x_test, 2)

        y_train, y_test = data_obj.labels_train, data_obj.labels_test

        self.input_shape = x_train.shape[1:]

        pred = self.classify(x_train, x_test, y_train, y_test)
        self._time_['without_resampling'] = time.time() - start

        print("Process Completwith_resamplinged.")

        return (y_test, pred)

    def with_resampling(self,data_obj):

        self._name_ = 'Res_Conv_NN__with_resampling'
        start = time.time()

        print("\n\nRunning a convolutional neural network with {} residual blocks with resampling the data...".format(self.blocks))
        
        data_obj = data_obj.prepare(one_hot=True,resample=True)
        x_train, x_test = data_obj.feat_train, data_obj.feat_test
        x_train = np.expand_dims(x_train, 2)
        x_test = np.expand_dims(x_test, 2)

        y_train, y_test = data_obj.y_train, data_obj.y_test

        self.input_shape = x_train.shape[1:]

        pred = self.classify(x_train, x_test, y_train, y_test)
        self._time_['with_resampling'] = time.time() - start

        print("Process Completwith_resamplinged.")

        return (y_test, pred)


class Inc_Conv_NN:
    def __init__(self):
        self._name_ = None
        self._time_ = {'with_resampling': 0, 'without_resampling': 0}
        self.blocks = 4
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                        ModelCheckpoint(filepath='best_model_inception.h5', monitor='val_loss', save_best_only=True)]
        self.input_shape = None
        self.history = None
        self.optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

    def inception_block(self,prev_layer):   
        conv1=Conv1D(filters = 64, kernel_size = 1, activation='relu', padding = 'same')(prev_layer)
        
        conv3=Conv1D(filters = 64, kernel_size = 1, activation='relu', padding = 'same')(prev_layer)
        conv3=Conv1D(filters = 64, kernel_size = 3, activation='relu', padding = 'same')(conv3)
        
        conv5=Conv1D(filters = 64, kernel_size = 1, activation='relu', padding = 'same')(prev_layer)
        conv5=Conv1D(filters = 64, kernel_size = 5, activation='relu', padding = 'same')(conv5)
        
        pool= MaxPool1D(pool_size=3, strides=1, padding='same')(prev_layer)
        convmax=Conv1D(filters = 64, kernel_size = 1, activation='relu', padding = 'same')(pool)
        
        layer_out = concatenate([conv1, conv3, conv5, convmax], axis=1)
        return layer_out

    def model(self):
        X_input=Input(input_shape = self.input_shape)
        
        _block_ = Conv1D(filters = 64, kernel_size = 7, activation='relu', padding = 'same')(X_input)
        _block_ = MaxPool1D(pool_size=3, strides=2, padding='same')(_block_)
        
        _block_ = Conv1D(filters = 64, kernel_size = 1, activation='relu', padding = 'same')(_block_)

        for _ in range(self.blocks):
            _block_ = self.inception_block(_block_)        
        
        _block_ = MaxPool1D(pool_size=7, strides=2, padding='same')(_block_)
        
        _block_ = Flatten()(_block_)
        output = Dense(5,activation='softmax')(_block_)
        
        model = Model(inputs = X_input, outputs = output, name='Inception')

        model.summary()
        print("\n")

        return model

    def classify(self, x_train, x_test, y_train, y_test):
        model = self.model()

        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        self.history = model.fit(x_train, y_train,
                            epochs = 5, batch_size = 100, 
                            validation_data=(x_test, y_test),
                            callbacks=self.callbacks)

        pred = model.predict(x_test)
        return pred
    
    def no_resampling(self,data_obj):
        self._name_ = 'Inc_Conv_NN__no_resampling'
        start = time.time()

        print("\n\nRunning a convolutional neural network with {} residual blocks without resampling the data...".format(self.blocks))

        data_obj = data_obj.prepare(one_hot=True)
        x_train, x_test = data_obj.feat_train, data_obj.feat_test
        x_train = np.expand_dims(x_train, 2)
        x_test = np.expand_dims(x_test, 2)

        y_train, y_test = data_obj.labels_train, data_obj.labels_test

        self.input_shape = x_train.shape[1:]

        pred = self.classify(x_train, x_test, y_train, y_test)
        self._time_['without_resampling'] = time.time() - start

        print("Process Completwith_resamplinged.")

        return (y_test, pred)

    def with_resampling(self,data_obj):
        self._name_ = 'Inc_Conv_NN__with_resampling'
        start = time.time()

        print("\n\nRunning a convolutional neural network with {} residual blocks with resampling the data...".format(self.blocks))
        
        data_obj = data_obj.prepare(one_hot=True,resample=True)
        x_train, x_test = data_obj.feat_train, data_obj.feat_test
        x_train = np.expand_dims(x_train, 2)
        x_test = np.expand_dims(x_test, 2)

        y_train, y_test = data_obj.labels_train, data_obj.labels_test

        self.input_shape = x_train.shape[1:]

        pred = self.classify(x_train, x_test, y_train, y_test)
        self._time_['with_resampling'] = time.time() - start

        print("Process Completwith_resamplinged.")

        return (y_test, pred)


class SVM_rbf:
    def __init__(self):
        self._name_ = None
        self._time_ = {'with_resampling': 0, 'without_resampling': 0}

    def classify(self, x_train, x_test, y_train, PCA=False):
        if PCA:
            model = Pipeline(steps=[('pca', PCA()), 
                        ('svm', SVC(C=45,gamma=0.1))])
        else:
            model = Pipeline(steps=[('svm', SVC(C=45,gamma=0.1))])

        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        return pred
    
    def no_resampling(self,data_obj,PCA=False):
        if PCA:
            self._name_ = 'PCA__SVM_rbf__no_resampling'
        else:
            self._name_ = 'SVM_rbf__no_resampling'

        start = time.time()

        print("\n\nRunning a Support Vector Machine classifier without resampling the data...")

        data_obj = data_obj.prepare()
        x_train, x_test = data_obj.feat_train, data_obj.feat_test

        y_train, y_test = data_obj.labels_train, data_obj.labels_test

        pred = self.classify(x_train,x_test,y_train, PCA)
        self._time_['without_resampling'] = time.time() - start

        print("Process Completwith_resamplinged.")

        return (y_test, pred)

    def with_resampling(self,data_obj,PCA=False):
        if PCA:
            self._name_ = 'PCA__SVM_rbf__with_resampling'
        else:
            self._name_ = 'SVM_rbf__with_resampling'

        start = time.time()

        print("\n\nRunning a Support Vector Machine classifier with resampling the data...")
        
        data_obj = data_obj.prepare(resample=True)
        x_train, x_test = data_obj.feat_train, data_obj.feat_test

        y_train, y_test = data_obj.labels_train, data_obj.labels_test

        pred = self.classify(x_train,x_test,y_train, PCA)
        self._time_['with_resampling'] = time.time() - start

        print("Process Completwith_resamplinged.")

        return (y_test, pred)