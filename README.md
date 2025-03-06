# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Mourya G
RegisterNumber: 212224230170
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        #Calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        #Calculate errors
        errors=(predictions-y).reshape(-1,1)
        #update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("50_Startups.csv")
data.head()

#Assuming rhe last column is your target variable 'y' and the preceding columns.
X = (data.iloc[1:,:-2].values)
X1 =X.astype(float)

scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn model Parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target calue for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value:{pre}")
```

## Output:
DATA.HEAD()
![364217302-e156cbdc-9313-4bda-af2b-0146f86c36ef](https://github.com/user-attachments/assets/7f66e59b-133a-4a90-9ea2-fbe7d035c008)

X VALUE
![364217413-2722cf57-3445-4927-9dbb-91fb29afa605](https://github.com/user-attachments/assets/1bcd5c83-f6f3-4324-adaf-48b47a6e6c14)

X1_VALUE
![364217645-4a1661da-780c-4e4e-8dd5-e33ca78726a1](https://github.com/user-attachments/assets/1f71017f-a18b-490d-9b85-90ec02fc72ed)

PREDICTED VALUE
![364217878-2d549d59-bb8a-46fe-8598-34acbc3c0e6e](https://github.com/user-attachments/assets/d242835a-26fa-4d37-a3b7-ceb60482021d)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
