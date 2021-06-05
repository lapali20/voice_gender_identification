import librosa
import scipy as sp
import numpy as np
import pandas as pd
import entropy as ent
from sklearn import preprocessing
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import mode


def peak_freq(sound, sr):
    ft = sp.fft.fft(sound)
    magnitude = np.abs(ft)
    frequency = np.linspace(0, sr, len(magnitude))

    p_index = np.argmax(magnitude[:sr//2])
    peakfreq = frequency[p_index]
    return peakfreq / 1000


def med_freq(sound, rate):
    spec = np.fft.fft(sound)
    magnitude = np.abs(spec)
    frequency = np.linspace(0, rate, len(magnitude))
    power = np.sum(magnitude ** 2)

    mid = 0
    i = 0
    while mid < (power / 2):
        mid += magnitude[i] ** 2
        i += 1

    return frequency[i] / 1000


def create_time_frequency(sound, frame_size, hop_length, rate):
    s_scale = librosa.stft(sound, n_fft=frame_size, hop_length=hop_length, center=False)
    stft = []
    frequency = np.arange(0, 1 + frame_size/2) * rate / frame_size

    for i in range(s_scale.shape[1]):
        seg = s_scale.transpose()[i]
        magnitude = np.abs(seg)

        index = np.argmax(magnitude)
        stft.append(frequency[index] / 1000)

    return np.asarray(stft)


def create_features(sound, sr):
    X = pd.DataFrame()
    N_FFT = 2048
    HOP_LENGTH = 512
    FRAME_SIZE = int(sr/2)
    b, a = sp.signal.butter(3, [.01, .05], 'band')
    sound = sp.signal.filtfilt(b, a, sound)

    for i in range(0, len(sound) - FRAME_SIZE, FRAME_SIZE):
        segment = sound[i:i + FRAME_SIZE]

        tf = create_time_frequency(segment, frame_size=N_FFT, hop_length=HOP_LENGTH, rate=sr)
        centroid = librosa.feature.spectral_centroid(segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH).transpose()

        index = len(X)

        try:
            X.loc[index, 'centroid'] = np.min(centroid) / 1000
            X.loc[index, 'sd'] = np.std(tf)
            X.loc[index, 'kurt'] = kurtosis(tf)
            X.loc[index, 'skew'] = skew(tf)
            X.loc[index, 'mode'] = mode(tf).mode[0]
            X.loc[index, 'peakfreq'] = peak_freq(sound, sr)
            X.loc[index, 'Q25'] = q25 = np.quantile(tf, 0.25)
            X.loc[index, 'Q75'] = q75 = np.quantile(tf, 0.75)
            X.loc[index, 'IQR'] = q75 - q25
            X.loc[index, 'sp.ent'] = ent.spectral_entropy(sound, sf=sr)
            X.loc[index, 'sfm'] = np.std(librosa.feature.spectral_flatness(sound, n_fft=N_FFT, hop_length=128))
            X.loc[index, 'mindom'] = np.min(tf)
            X.loc[index, 'maxdom'] = np.max(tf)
            X.loc[index, 'meandom'] = np.mean(tf)

        except:
            pass

    return X


def get_MFCC(sr, audio):
    features = librosa.feature.mfcc(audio, sr, n_mfcc=13, hop_length=512, n_fft=2048)
    features = preprocessing.scale(features)

    return features

def create_gmm_features(sound, sr):
    features = []

    b, a = sp.signal.butter(3, [.01, .05], 'band')
    sound = sp.signal.filtfilt(b, a, sound)

    FRAME_SIZE = int(sr / 2)

    for i in range(0, len(sound) - FRAME_SIZE, FRAME_SIZE):
        segment = sound[i:i + FRAME_SIZE]
        vector = get_MFCC(sr, segment)
        if len(features) == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

    return features