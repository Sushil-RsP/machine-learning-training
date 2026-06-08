# Decision Tree
# Problem
''' You have been tasked with creating a decision tree model to predict whether a person is
   likely to purchase a new smartphone based on their age, income and education level. You are
   provided with a dataset containing these attributes and the target variable indicating whether
   the person made a purchase or not '''

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#data = pd.read_csv(r"c:\Users\LENOVO\Documents\machine learning\Scikit Learn\Decision\phone_pur.csv")
#x = data[['age', 'income', 'education']]
#y = data['purchase']

x = np.array([[30,50000,1],[25,30000,0],[40,70000,2],[35,60000,1],[20,20000,0],[45,80000,2],
              [28,45000,1],[32,55000,0],[50,90000,3],[22,25000,0],[38,65000,1],[33,58000,2],
              [27,40000,1],[48,85000,3],[26,35000,0],[42,72000,2],[29,48000,1],[31,52000,0],
              [47,88000,2],[23,28000,0],[39,67000,1],[34,60000,2],[21,22000,0],[46,82000,3],
              [24,32000,1],[41,75000,2],[36,62000,1],[37,64000,2],[49,89000,3],[25,30000,0],
              [28,45000,1],[30,50000,2],[32,55000,0],[35,60000,1],[40,70000,3],[22,25000,0],
              [24,32000,1],[26,35000,2],[28,45000,1],[30,50000,0],[33,58000,2],[35,60000,1],
              [38,65000,3],[27,40000,1],[29,48000,2],[31,52000,1],[34,60000,0],[36,62000,2],
              [39,67000,1],[42,72000,3],[44,78000,2]])
# 0 is for High school, 1 is for Bachelor's, 2 is for Master's, 3 is for PhD
y = np.array([1,0,1,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1])
# 1 is for Yes, 0 is for No
x_train, x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy of model is : {accuracy}')

# user input
age = float(input("Enter your age : "))
salary = float(input("Enter your salary : "))
education = int(input("Enter your education level : "))
user_input = np.array([[age,salary,education]])
prediction = model.predict(user_input)
if prediction[0] == 1:
    print("customer will buy smartphone")
else:
    print("customer don't buy smartphone")