import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

class VoiceDatasetSVM:
    def __init__(self):
        self.__preprocess__()

    def __preprocess__(self):
        male_df = pd.read_csv('data/male_train.csv').drop(['Unnamed: 0', 'medfreq'], 1).dropna()
        female_df = pd.read_csv('data/female_train.csv').drop(['Unnamed: 0', 'medfreq'], 1).dropna()
        self.train_df = pd.concat([male_df, female_df])
        X = self.train_df.drop(['gender'], 1)
        y = self.train_df['gender']
        undersample = RandomUnderSampler(sampling_strategy='majority')
        X, self.y = undersample.fit_resample(X, y)
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        self.X = pd.DataFrame(self.scaler.transform(X))

    def scale(self, X):
        return pd.DataFrame(self.scaler.transform(X))


    def X_train(self):
        return self.X


    def y_train(self):
        return self.y



class VoiceDatasetGMM:
    def __init__(self):
        self.male = np.loadtxt('data/male_train.txt')
        self.female = np.loadtxt('data/female_train.txt')

    def male_train(self):
        return self.male

    def female_train(self):
        return self.female



