# Gradient Boosting
# Problem
''' Create a predictive  model using gradient boosting to forecast housing prices based on
  various features such as square footage, number of bedrooms, number of bathrooms and location '''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"C:\Users\LENOVO\Documents\machine learning\Scikit Learn\square_footage,num_bedrooms,num_bat [MConverter.eu].csv")
print(data.dtypes)

labelencoder = LabelEncoder()
data['location'] = labelencoder.fit_transform(data['location'])

#data = pd.get_dummies(data, columns=['location'])

x = data.drop('price', axis=1)
y = data['price']
print(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse  = mean_squared_error(y_test, y_pred)
print(mse)

# User input
sq_footage = float(input("enter sq_footage : "))
bedrooms = float(input("enter number of bedrooms : "))
bathrooms = float(input("enter number of bathrooms : "))
location = input("enter location (suburb, city, rural) : ")
''' input_lacation = [0,0,0,0]
if location == "city":
    input_lacation[0] = 1
elif location == "rural":
    input_lacation[1] = 1
elif location == "suburb":
    input_lacation[2] = 1
elif location == "urban":
    input_lacation[3] = 1 ''' 

#location = labelencoder.fit_transform([location])

user_input = {
    'square_footage' : [sq_footage],
    'num_bedrooms ' : [bedrooms],
    'num_bathrooms' : [bathrooms],
    'location' : [location]
}

''' 'location_city' : [input_lacation[0]],
    'location_rural' : [input_lacation[1]],
    'location_suburb' : [input_lacation[2]],
    'location_urban' : [input_lacation[3]] '''

#user_input['location'] = [user_input['location']]
user_input['location'] = labelencoder.transform(user_input['location'])
new_df = pd.DataFrame(user_input)


prediction = model.predict(new_df)
print(prediction)