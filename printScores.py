import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas
import numpy as np

with open('modelSaves/ModelOutput.pickle', 'rb') as handle:
    data = pickle.load(handle)

data_3D.keys()