import pandas as pd

 #Create the dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'CGPA': [3.8, 3.2, 3.5, 2.8, 3.9],
    'Marks': [85, 78, 82, 70, 90],
    'Percentage': [85, 78, 82, 70, 90],
    'Pass': [1, 1, 1, 0, 1]  # 1 for Pass, 0 for Fail based on arbitrary criteria
}

df = pd.DataFrame(data)
print(df)

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df[['cgpa', 'marks']]
y = df['percentage']

 #Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Initialize a linear Regression model
model = LinearRegression()
#Fit the model to the training data
model.fit(X_train, y_train)
print("Training set:")
print(X_train)
print(y_train)
print("Testing set:")
print(X_test)
print(y_test)