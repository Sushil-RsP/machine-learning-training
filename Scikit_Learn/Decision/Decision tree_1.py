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

data = pd.read_csv(r"C:\Users\LENOVO\Documents\machine learning\Scikit Learn\Decision\smart_phone.csv")
x = data[['age', 'income', 'education']]
y = data['purchase']

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
# 0 is for High school, 1 is for Bachelor's, 2 is for Master's, 3 is for PhD
user_input = np.array([[age,salary,education]])
prediction = model.predict(user_input)
if prediction[0] == 1:
    print("customer will buy smartphone")
else:
    print("customer will not buy smartphone")