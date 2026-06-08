# Problem
''' In e-commerce company, the management wants to predict whether a customer will purchase a high-value product based on their age,
 time spent on the website, and whether they have added items to their cart. The goal is to optimize marketing stratergies by
   targeting potential customers more effectively, thereby increasing sales and revenue '''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = np.array([[25,30,0],[20,60,1],[35,25,0],[30,30,1]])
# here in every list first is age, second is time, third is add to cart
y = np.array([0,1,0,1])
# this is a third part of list in x which is customer add to cart or not

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

# accuracy test id 1.0 mean traing and testing data is good
accuracy = model.score(x_test, y_test)
print(f'accuracy : {accuracy}')

user_age = float(input("enter age of customer : "))
user_time_spent = float(input("enter customer time spent on website : "))
user_added_to_cart = int(input("enter 1 if added else 0 : "))
user_data = np.array([[user_age, user_time_spent, user_added_to_cart]])

prediction = model.predict(user_data)
if prediction[0] == 1:
    print("customer buy")
else:
    print("customer not buy")