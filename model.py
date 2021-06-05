import pickle
import os

from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture as GMM

from data import VoiceDatasetSVM, VoiceDatasetGMM
from feature_process import *

# class VoiceModelGMM:
#     def __init__(self, model_path='voicemodelgmm'):
#         if os.path.exists(model_path + 'male.gz'):
#             self.male_model = pickle.load(open(model_path + 'male.gz', 'rb'))
#         else:
#             self.male_model = self.__train_male_model()
#             pickle.dump(self.male_model, open(model_path + 'male.gz', 'wb'))

#         if os.path.exists(model_path + 'female.gz'):
#             self.female_model = pickle.load(open(model_path + 'female.gz', 'rb'))
#         else:
#             self.female_model = self.__train_female_model()
#             pickle.dump(self.female_model, open(model_path + 'female.gz', 'wb'))

#     def __train_male_model(self):
#         data = VoiceDatasetGMM()
#         male_train = data.male_train()
#         model = GMM(n_components = 8, max_iter = 200, covariance_type = 'diag', n_init = 3)
#         model.fit(male_train)
#         return model


#     def __train_female_model(self):
#         data = VoiceDatasetGMM()
#         female_train = data.female_train()
#         model = GMM(n_components = 8, max_iter = 200, covariance_type = 'diag', n_init = 3)
#         model.fit(female_train)
#         return model

#     def predict(self, sound, sr):
#         features = create_features(sound, sr)
#         output = []

#         for f in features:
#             try:
#                 log_likelihood_male = np.array(self.male_model.score([f])).sum()
#                 log_likelihood_female = np.array(self.female_model.score([f])).sum()

#                 if log_likelihood_male > log_likelihood_female:
#                     output.append(1)
#                 else:
#                     output.append(0)

#             except:
#                 pass
#         return round(sum(output)/len(output))

#     def get_target_name(self, y):
#         return ['Female', 'Male'][y]

class VoiceModelSVM:
    def __init__(self, model_path='voicemodelsvm.gz'):
        if os.path.exists(model_path):
            self.model = pickle.load(open(model_path, 'rb'))
        else:
            self.model = self.__train_model()
            pickle.dump(self.model, open(model_path, 'wb'))

    def __train_model(self):
        data = VoiceDatasetSVM()
        X_train = data.X_train()
        y_train = data.y_train()
        model = SVC().fit(X_train, y_train)
        return model

    def predict(self, sound, sr):
        features = create_features(sound, sr)
        scaler = VoiceDatasetSVM().scaler
        features = pd.DataFrame(data=scaler.transform(features))
        pred = self.model.predict(features)
        return round(sum(pred) / len(pred))

    def get_target_name(self, y):
        return ['Female', 'Male'][y]


