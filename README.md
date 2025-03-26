# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions - y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)
theta=linear_regression(X1_scaled,Y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![WhatsApp Image 2025-03-26 at 10 06 31_ec51cb32](https://github.com/user-attachments/assets/46475239-a177-4ba1-881b-a1cda593b5c2)

![WhatsApp Image 2025-03-26 at 10 06 45_37871c5a](https://github.com/user-attachments/assets/d4e417dd-9bf9-4d17-bd0c-df74fd1608ae)

![image](https://github.com/user-attachments/assets/81e15583-6b4d-435a-8b08-65a797e0b89a)

![WhatsApp Image 2025-03-26 at 10 06 45_3407f267](https://github.com/user-attachments/assets/5336af95-d163-4a87-b16d-fad2088f9379)

![WhatsApp Image 2025-03-26 at 10 06 44_4faf3dd0](https://github.com/user-attachments/assets/fe395872-6f8d-4915-928a-8d181b0110db)

![WhatsApp Image 2025-03-26 at 10 06 45_f366091e](https://github.com/user-attachments/assets/bffc28d3-0e89-4740-be2c-24977fcd2ad8)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
