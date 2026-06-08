import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv(r"C:\Users\LENOVO\Documents\machine learning\Scikit Learn\square_footage,num_bedrooms,num_bat [MConverter.eu].csv")

# Encode the location column
labelencoder = LabelEncoder()
# Fit the labelencoder on the location column only
labelencoder.fit(data['location'])
# Transform the location column using the fitted labelencoder
data['location'] = labelencoder.transform(data['location'])

# One-hot encode the location column
# This step is not necessary if the location column has already been encoded
# data = pd.get_dummies(data, columns=['location'])

# Split the data into training and testing sets
# The test_size parameter specifies the proportion of the data to be used for testing
X_train, X_test, y_train, y_test = train_test_split(data.drop('price', axis=1), data['price'], test_size=0.2, random_state=42)

# Train the model
# The n_estimators parameter specifies the number of boosting stages to perform
model = GradientBoostingRegressor(n_estimators=100)
# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the mean squared error
# The squared parameter specifies whether the error should be squared or not
mse = mean_squared_error(y_test, y_pred, squared=False)

# Print the mean squared error
print("Mean squared error:", mse)

#user input
square_footage = float(input("enter square_footage : "))
num_bedrooms = float(input("enter num_bedrooms : "))
num_bathrooms = float(input("enter num_bathrooms : "))
location = input("enter location (city, rural, urban, suburb) : ")


# Make a prediction for a new data point
new_data = {'square_footage': [square_footage], 'num_bedrooms': [num_bedrooms], 'num_bathrooms': [num_bathrooms], 'location': [location]}
# Convert the location string to a list
new_data['location'] = [new_data['location']]
# Transform the location column using the fitted labelencoder
new_data['location'] = labelencoder.transform(new_data['location'])
# Convert the new data to a DataFrame
new_df = pd.DataFrame(new_data)
# Make a prediction for the new data
prediction = model.predict(new_df)

# Print the prediction
print("Prediction:", prediction)