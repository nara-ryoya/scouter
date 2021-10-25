import numpy as np
import pandas as pd

face_features_stats = pd.read_csv("face_features_stats.csv", index_col=0)

eye_mean = face_features_stats.loc["landmark_1", "mean"]
eye_std = face_features_stats.loc["landmark_1", "std"]
mouth_mean = face_features_stats.loc["landmark_2", "mean"]
mouth_std = face_features_stats.loc["landmark_2", "std"]
theta_mean = face_features_stats.loc["landmark_3", "mean"]
theta_std = face_features_stats.loc["landmark_3", "std"]

class Feature:
    def __init__(self, _mean, _std):
        self._mean = _mean
        self._std = _std
    
    def standardize(self, param):
        return (param - self._mean) / self._std

eye = Feature(eye_mean, eye_std)
mouth = Feature(mouth_mean, mouth_std)
theta = Feature(theta_mean, theta_std)

def calc_strongness(strongness_list):
    arr = np.array(strongness_list)
    f1 = eye.standardize(np.mean(arr[:, 0]))
    f2 = mouth.standardize(np.mean(arr[:, 1]))
    f3 = theta.standardize(np.mean(arr[: , 2]))
    print(f1, f2, f3)
    return int((10 ** ((f1 + f2 - f3 + 3) * 0.8)) * 1.5)




