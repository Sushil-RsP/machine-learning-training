# Random forest classifer
# Problem
''' Use random forest classifer to predict whether a person is likely to purchase a product
   based on certain features like age, gender and estimated salary '''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv(r"C:\Users\LENOVO\Documents\machine learning\Scikit Learn\random forest.csv")
x = data.drop('purchased',axis=1)  
print(x.dtypes)
# you can also do this by using x = data[['age','gender','estimated salary']]
y = data['purchased']

labelEncoder = LabelEncoder()
x['gender'] = labelEncoder.fit_transform(x['gender'])
print(f"changed data type is : {x['gender'].dtype}")

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

# User input
age = int(input("enter age : "))
gender = input("Enter gender : ")
salary = int(input("enter  estimated salary : "))

user_gender = labelEncoder.fit_transform([gender])[0]
user_input = [[age, user_gender, salary]]

prediction = model.predict(user_input)
print(prediction)
if prediction[0] == 1:
    print("this person is likely to purchase product")
else:
    print("this person is not purchase product")




'''user_input = {
    'Car_Model': 'sonet',
    'Body_Type': 'SUV',
    'Manufacturer': 'Kia',
    'Fuel_Type': 'Petrol',
    'Safety_Rating': 4.5,
    'Price': 15000,
    'Budget': 20000,
    'Desired_Features': 'Airbags',
    'Age': 30,
    'Family_Size': 3
}'''

# Convert to DataFrame and encode
'''user = pd.DataFrame([user_input])
for col in ['Car Model', 'Body Type', 'Manufacturer', 'Fuel Type', 'Desired Features']:
    user[col] = label_encoder.fit_transform(user[col]) '''

# Predict
''' user_input_array = user.values
prediction = model.predict(user_input_array) 

print(prediction)
if prediction[0] == 1:
    print("this person is likely to purchase product")
else:
    print("this person is not purchase product") '''