# Support Vector Machines (SVM)
# Problem
''' A telecommunications company wants to reduce customer churn by identifying customers at a risk of leaving. They have 
    historical data on customer behavior and want to build a model to predict which customers are most likely to churn '''
''' the churn rate would refer to the percentage of customers who close their contract or subscription with your company 
    in any given time period, means rechage '''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = {
    'age' : [20,25,30,35,40,45,50,55,60],
    'monthly_plan' : [200,250,299,150,120,100,200,240,299],
    'churn' : [1,1,1,0,0,0,0,0,1]
}

df =  pd.DataFrame(data)

x = df[['age', 'monthly_plan']]
y = df[['churn']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', C=1.0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
#print(accuracy)

report = classification_report(y_test, y_pred)
#print(report)

# user input
user_age = eval(input("enter customers age : "))
user_monthly_plan = float(input("enter customers monthly plan : "))
user_data = np.array([[user_age, user_monthly_plan]])

presiction = model.predict(user_data)
if presiction[0] == 1:
    print("The customer will stay")
else:
    print("The customer is likely to churn (in a risk)")