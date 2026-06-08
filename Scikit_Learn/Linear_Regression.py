# Problem
''' Predict a Student's final exam score based on the number of hours they study '''


# Step no. 1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step no. 2
# make a Data
study_time = {
    "hour_study" : [2,3,4,5,6,7,8,9,10],
    "exam_score" : [50,60,70,75,80,85,90,95,98]
}
df = pd.DataFrame(study_time)

# Step no. 3
# extract the features from data, columns are also known as features
x = df[["hour_study"]]                 # because this is a 2D array
y = df[["exam_score"]]

# Step no. 4
# train test data here we 80% of data we train and 20% of data we test
x_train , x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
# in test_size value 0.2 means 20% of data

# Step no. 5
# use linear regression
model = LinearRegression()              # this stape in not compulsory
model.fit(x_train, y_train)             # for model fit the data from x_train and y_train 

# Step no. 6
# take input from use two train data
user_input = eval(input("enter a number of hours you study : "))
prediction = model.predict([[user_input]])
ass = np.round(prediction[0],2)                    #rounding off the prediction to 2 decimal places
ash = float(ass)                                  
print(f'predicted exam score is : {ash}')