''' A retail company wants to predict customer purchasing behavior based on their age, salary and past purchase history.
   The company aim to use K-Nearest Neighbors (KNN) algorithm to classify customers into potential buying groups to 
   personalize marketing statergies. This predictive model will help the company understand and target specific customer 
   segments more effectively, there by increasing sales and customer satisfaction '''
# KNN Because of classification

import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = np.array([[35,50000,1],[40,65000,1],[20,30000,1],[38,10000,1],[55,120000,3],[45,80000,3],[50,90000,2],[60,75000,2],[70,110000,2]])
label = np.array([2,2,0,0,1,1,2,2,2])

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)

# Standardizing the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)         
x_test = scaler.fit_transform(x_test)

knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(x_train,y_train)

accuracy = knn.score(x_test, y_test)
print(accuracy)

age = float(input("enter age of customer : "))
salary = float(input("enter salary of customer : "))
buy_product_by_customer = int(input("enter number buy product by customer : "))
user_data = np.array([[age, salary, buy_product_by_customer]])
prediction = knn.predict(user_data)
print(int(prediction))