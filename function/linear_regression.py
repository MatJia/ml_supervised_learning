from generator import generate_test_data
import pandas as pd
import numpy as np
import random as rd

real_k, real_b = generate_test_data()

data = pd.read_csv("../data/linear_regression.csv")
X = data[["x"]].values
Y = data["y"].values
X = np.column_stack((np.ones(len(X)), X))
beta = np.linalg.inv(X.T @ X) @ X.T @ Y
print(f"closed form: {list(beta)}")#closed_form_solution
#Gradient Descent
w, b = round(rd.uniform(-10,10),3), rd.randint(-10000,10000)
learning_rate = 0.1
learning_round = 10000
train_rec = open("../train_rec.txt","w")
list_x, list_y = list(data["x"] / 100000), list(data["y"])
for epoch in range(learning_round):
    total_error = 0
    total_error_dw = 0
    total_error_db = 0
    for i, j in zip(list_x, list_y):
        total_error += pow((w*i + b - j), 2)
        total_error_dw += (w*i + b - j) * i
        total_error_db += (w*i + b - j)
    loss = total_error / len(list_y)
    loss_dw = 2 * total_error_dw / len(list_y)
    loss_db = 2 * total_error_db / len(list_y)
    #renew w,b
    w = w - learning_rate * loss_dw
    b = b - learning_rate * loss_db

    train_rec.write(f"round {epoch}: loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}\n")
    if epoch % 5 == 0:
        print(f"round {epoch}: loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}, lr = {learning_rate:.4f}")
        if learning_rate >= 0.0001:
            learning_rate *= 0.995#lr decay
    if loss < 10:
        break
train_rec.close()
print(f"closed form: {beta.tolist()}")#closed_form_solution
print(f"GD result: {[b,w/100000]}") #return scaling
print(f"real result: {[real_b, real_k]}")