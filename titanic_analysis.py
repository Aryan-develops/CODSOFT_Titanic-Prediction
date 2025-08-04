import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("C:/Users/Aryan/OneDrive/Desktop/Intership Projects/Data science/Titanic Prediction/CODSOFT_Titanic-Prediction/Data/train.csv")
print(train.head())
print(train.info())
print(train.describe())
print(train.isnull().sum())

# Visualize survival count
sns.countplot(data=train, x='Survived')
plt.show()

import pandas as pd

def preprocess(df):
    # Fill missing Age with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    # Fill missing Embarked with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # Fill missing Fare with median
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Convert Sex to numeric
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    # Convert Embarked to numeric
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Drop columns not used
    df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True, errors='ignore')

    return df

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# from src.data_preprocessing import preprocess
from feature_engineering import add_family_size

train = pd.read_csv('C:/Users/Aryan/OneDrive/Desktop/Intership Projects/Data science/Titanic Prediction/CODSOFT_Titanic-Prediction/Data/train.csv')
test = pd.read_csv('C:/Users/Aryan/OneDrive/Desktop/Intership Projects/Data science/Titanic Prediction/CODSOFT_Titanic-Prediction/Data/test.csv')
train = preprocess(train)
train = add_family_size(train)

X = train.drop('Survived', axis=1)
y = train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
# Retrain on all data
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X, y)

# Prepare test data
test = preprocess(test)
test = add_family_size(test)

# Predict
preds = final_model.predict(test)

# Prepare submission
submission = pd.DataFrame({
    'PassengerId': pd.read_csv('C:/Users/Aryan/OneDrive/Desktop/Intership Projects/Data science/Titanic Prediction/CODSOFT_Titanic-Prediction/Data/test.csv')['PassengerId'],
    'Survived': preds
})
submission.to_csv('C:/Users/Aryan/OneDrive/Desktop/Intership Projects/Data science/Titanic Prediction/CODSOFT_Titanic-Prediction/outputs/submission.csv', index=False)