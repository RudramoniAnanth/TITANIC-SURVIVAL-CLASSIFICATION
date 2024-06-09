import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

data=pd.read_csv(r"C:\Users\rudra\Desktop\BHARAT INTERN\datasets\Titanic-Dataset.csv")
#GETTING THE DETAILS OF THE DATA
print(data.shape)        #GETTING THE SHAPE OF DATA
print(data.size)         #GETTING THE SIZE OF DATA

#GETTING THE ROWS AND COLUMN INDECES
print(data.index)
print(data.columns)

#GETTING THE TOP FIVE AND BOTTOM FIVE SAMPLES OF DATA
print(data.head())
print("`~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
print(data.tail())
#GETTING THE OVERVIEW AND THE ATTRIBUTES OF THE  SAMPLES OF DATA
print(data.info())
print(data.isnull().mean() * 100)

'''#GETTING THE DESCRIPTION AND THE ATTRIBUTES AND PRESENCE OF  NULL VALUES  OF THE  COLUMNS OF DATA 
AND DROPING THE COLUMNS WITH NULL VALUE PERCENTAGE MORE THAN 75%'''
for i in data.columns :
    print(i)
    print(data[i].describe())
    print("NULL VALUES PRESENCE: ",True in np.unique(pd.isnull(data[i])))
    print("NULL VALUE PERCENTAGE : ",data[i].isnull().mean() * 100)
    print("`~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
    

#APPLYING DATA IMPUTATION
for i in data.columns :
    if data[i].dtypes==object and data[i].isnull().mean() * 100!=0:
        data[i].fillna(data[i].value_counts().index[0],inplace=True)
        
    if  data[i].dtypes!=object:
        data[i].fillna(data[i].mean(),inplace=True)
        
# GETTING THE INFORMATION ABOUT THE DATA AFTER THE PRE PROCESSING
print(data.info())
print("`~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

#determining The dependent and the remaining as independent variables
X = data.drop('Survived', axis=1)
y = data['Survived']

#creating dummy variables of the data
# Convert categorical features to numerical
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

# Feature selection
selector = SelectKBest(f_classif, k=5)
X_new = selector.fit_transform(X, y)
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]
print("Selected features:", selected_feature_names)

# Split the data into train and test sets
X_train_new, X_test_new, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Create and evaluate models
models = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

model_accuracies = []
for model in models:
    model.fit(X_train_new, y_train)
    y_pred = model.predict(X_test_new)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies.append(accuracy)
    print(f"{type(model).__name__} Accuracy: {accuracy}")

# Select the best model based on the highest accuracy
best_model_index = model_accuracies.index(max(model_accuracies))
best_model = models[best_model_index]
print(f"\nThe best model is: {type(best_model).__name__}")

# Fit the best model on the selected features
best_model.fit(X_new, y)

# Prompt the user to enter the features
user_features = []
for feature in selected_feature_names:
    value = input(f"Enter the value for '{feature}': ")
    user_features.append(value)

# Make a prediction
prediction = best_model.predict([user_features])

# Print the prediction
if prediction[0] == 1:
    print("The person has survived.")
else:
    print("The person has not survived.")