# Hand-Written-Digit-Prediction
Objective:
The objective of this project is to develop a machine learning model that can accurately predict handwritten digits using the MNIST dataset.
Data Source:
The MNIST dataset can be downloaded from Kaggle or directly from popular Python libraries like sklearn and tensorflow.
Steps for the Project:
1. Import Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
