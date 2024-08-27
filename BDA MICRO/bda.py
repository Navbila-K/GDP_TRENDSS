import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
file_path = 'Country_data.csv'
df = pd.read_csv(file_path)

# Display data sample
st.write("Data Sample:")
st.write(df.head())

# Select the features for prediction
features = ['Total in km2', 'Land in km2', 'Water in km2', 'Water %', 'HDI', '%HDI Growth', 'Internet Users', 'Population 2023']
target = 'IMF Forecast GDP(Nominal)'

# Split the data into input (X) and output (y)
X = df[features]
y = df[target]

# Handle any missing data if present
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create classification labels based on IMF Forecast GDP(Nominal)
bins = [0, 1000000, 2000000, max(df[target])]
labels = ['Low', 'Medium', 'High']
df['GDP_Class'] = pd.cut(df[target], bins=bins, labels=labels)

# Update target to classification target
target_class = 'GDP_Class'
y_class = df[target_class]

# Split the dataset into training and testing sets for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Fill missing values with the mean of the respective columns
X_train_class = X_train_class.fillna(X_train_class.mean())
X_test_class = X_test_class.fillna(X_test_class.mean())

# Drop rows with any missing values
X_train_class = X_train_class.dropna()
X_test_class = X_test_class.dropna()
y_train_class = y_train_class[X_train_class.index]  # Ensure y_train_class has the same index as X_train_class
y_test_class = y_test_class[X_test_class.index]  # Ensure y_test_class has the same index as X_test_class

# Check for NaNs in the training and testing sets
st.write("Checking for NaNs in datasets:")
st.write("X_train_class NaNs:", X_train_class.isnull().sum().sum())
st.write("X_test_class NaNs:", X_test_class.isnull().sum().sum())
st.write("y_train_class NaNs:", y_train_class.isnull().sum())
st.write("y_test_class NaNs:", y_test_class.isnull().sum())

# Drop rows where the target variable has NaNs
y_train_class = y_train_class.dropna()
y_test_class = y_test_class.dropna()

# Ensure that the feature set aligns with the remaining target data
X_train_class = X_train_class.loc[y_train_class.index]
X_test_class = X_test_class.loc[y_test_class.index]

# Check for NaNs again after alignment
st.write("After handling NaNs, checking again:")
st.write("X_train_class NaNs:", X_train_class.isnull().sum().sum())
st.write("X_test_class NaNs:", X_test_class.isnull().sum().sum())
st.write("y_train_class NaNs:", y_train_class.isnull().sum())
st.write("y_test_class NaNs:", y_test_class.isnull().sum())

# Create and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_class, y_train_class)

# Predict on the testing set
y_pred_class = rf_model.predict(X_test_class)

# Calculate accuracy
accuracy = accuracy_score(y_test_class, y_pred_class)
st.write(f"Accuracy: {accuracy}")

# Generate classification report
class_report = classification_report(y_test_class, y_pred_class)
st.write("Classification Report:")
st.text(class_report)

# User input for prediction
st.write("## Predict GDP Class")

# Create input fields for user to enter the values for each feature
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"Enter {feature}", value=float(X_train_class[feature].mean()))

# Convert user input to a DataFrame
user_input_df = pd.DataFrame(user_input, index=[0])

# Predict the GDP class based on user input
if st.button("Predict"):
    prediction = rf_model.predict(user_input_df)
    st.write(f"The predicted GDP Class is: **{prediction[0]}**")