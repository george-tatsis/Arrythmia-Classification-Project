import pandas as pd
from config import data_folder, classes
from keras.utils import to_categorical
from sklearn.utils import resample
import numpy as np
import os

class ProcessData:

    def __init__(self):
        self.path = data_folder
        self.data_train = pd.read_csv(os.path.join(self.path,'mitbih_train.csv'), header=None, sep=',')
        self.data_test = pd.read_csv(os.path.join(self.path,'mitbih_test.csv'), header=None, sep=',')
        self.classes = classes
        self.feat_train = None
        self.labels_train = None
        self.feat_test = None
        self.labels_test = None

    def feat_labels(self):
        self.feat_train = self.data_train.iloc[:,:-1]
        self.labels_train = self.data_train.iloc[:,-1]
        self.feat_test = self.data_test.iloc[:,:-1]
        self.labels_test = self.data_test.iloc[:,-1]
        
    def stats(self):
        self.feat_labels()

        print("The training set consists of {} subjects".format(self.feat_train.shape[0]))
        print("The training set consists of {} subjects".format(self.feat_test.shape[0]))
        print("The subjects are divided into {} classes".format(len(self.labels_train.unique())))
        print("\n")
        
        print("Number of subjects in each class:")
        for ind in range(len(self.labels_train.unique())):
            total = sum(self.labels_train == ind) + sum(self.labels_test == ind)
            print("Class {} : {}".format(self.classes[ind],total))

    def one_hot_encoding(self):
        num_classes = len(self.labels_train.unique())
        self.labels_train = to_categorical(y=self.labels_train, 
                                            num_classes=num_classes)
        self.labels_test = to_categorical(y=self.labels_test, 
                                            num_classes=num_classes)

        print("One hot enconding the labels column...")
        print('\n')

    def resampling(self):
        dataframes = []
        df = (self.data_train[self.data_train[187]==0]).sample(n=20000,random_state=42)
        dataframes.append(df)
        for label in range(1,5):
            df = self.data_train[self.data_train[187]==label]

            df = resample(df,replace=True,n_samples=20000,random_state=123)
            dataframes.append(df)

        self.data_train = pd.concat(dataframes)

        print("Resampling the dataset to achieve a balance between the classes...")
        print('\n')

    def prepare(self,resample=False,one_hot=False,):
        if resample:
            self.resampling()

        self.stats()

        if one_hot:
            self.one_hot_encoding()
        

