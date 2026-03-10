from generator import generate_test_data
import pandas as pd
import numpy as np

k, b = generate_test_data()

data = pd.read_csv("../data/linear_regression.csv")
X = data[["x"]].values
Y = data["y"].values
X = np.column_stack((np.ones(len(X)), X))
print(X.shape)
print(X)