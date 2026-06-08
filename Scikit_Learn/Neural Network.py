# Neural Network
# Problem
''' You are tasked with creating a neural network model to whether a student will pass or fail
  an exam based on two features: hours studied and previous exam scores. The dataset should contains
  information on hours studied and previous exam scores for a group of students, along with thair
  exam outcome (pass or fail) '''

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

data = pd.read_csv(r"C:\Users\LENOVO\Documents\machine learning\Scikit Learn\neural.csv")
# print(data)

LabelEncoder = LabelEncoder()
encoded_exam_outcome = LabelEncoder.fit_transform(data['Exam_Outcome'])
# print(encoded_exam_outcome.dtype)

x = data[['Hours_Studied','Previous_Exam_Score']]
# print(x)

y = encoded_exam_outcome
# print(y)

cls = MLPClassifier(hidden_layer_sizes=(4,),activation='logistic',max_iter=1000,random_state=42)
cls.fit(x, y)

# user input
Hours_Studied = float(input("enter the number of Hours Studied : "))
Previous_Exam_Score = float(input("Enter Previous Exam Score : "))
user_input = np.array([[Hours_Studied, Previous_Exam_Score]])

prediction = cls.predict(user_input)
# print(prediction)

# decode prediction
decoded_prediction = LabelEncoder.inverse_transform(prediction)
print("exam score prediction of new student : {}".format(decoded_prediction[0]))