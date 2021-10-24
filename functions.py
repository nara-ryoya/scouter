import numpy as np
import pandas as pd

criterion = pd.read_csv("criterion.csv")

class Feature:
    def __init__(self, _mean, _std):
        self._mean = _mean
        self._std = _std
    
    def standardize(self, param):
        return (param - self._mean) / self._std


def calc_strongness_by_distance(eye, mouth, eyebrow):
    alpha = 100
    criterion["distance"] = criterion.apply(
        lambda x: alpha * np.linalg.norm(np.array([x["landmark_1"] - eye, x["landmark_2"] - mouth, x["landmark_3"] - eyebrow])), 
        axis=1
    )
    distance_max = criterion["distance"].max()
    criterion["weight_sum"] = criterion["strongness"] * (distance_max - criterion["distance"])
    strong_sum = criterion["weight_sum"].sum()
    distance_sum = (distance_max - criterion["distance"]).sum()
    s = strong_sum / distance_sum
    print(s)

    return int(np.power(10, s))
    # print(np.exp(s))

def calc_strongness(strongness_list):
    arr = np.array(strongness_list)
    eye = Feature(0.345473, 0.093)
    mouth = Feature(0.09344, 0.076)
    eyebrow = Feature(-0.000979, 0.106)
    eye_mean = eye.standardize(np.mean(arr[:, 0]))
    mouth_mean = mouth.standardize(np.mean(arr[:, 1]))
    eyebrow_mean = mouth.standardize(np.mean(arr[: , 2]))
    return calc_strongness_by_distance(eye_mean, mouth_mean, eyebrow_mean)





