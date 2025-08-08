#import libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

#accessing data 
data = pd.read_csv('diabetes (1).csv')
X=data[['Glucose','BloodPressure','BMI','DiabetesPedigreeFunction','Age']]
y=data['Outcome']

#train model and fit features inside model 
X_train,X_test,Y_train,Y_test=train_test_split(X,y, test_size=0.2, random_state=42)
model=RandomForestClassifier()
model.fit(X_train,Y_train)
 

#predicting model accuracy 
predict=model.predict(X_test)
accuracy=accuracy_score(Y_test,predict)
print(f'Accuracy of this model : {accuracy*100:.2f}')

#taking user input for diabetes checking
print('Enter given Details below : ')
Glucose = float(input("Enter Glucose level: "))
BloodPressure = float(input("Enter Blood Pressure: "))
BMI = float(input("Enter BMI: "))
DiabetesPedigreeFunction = float(input("Enter Diabetes Pedigree Function: "))
Age = float(input("Enter Age: "))

## creating a new data frame with user input which is useful for predicting diabetes 
input= pd.DataFrame([{
    'Glucose':Glucose,
    'BloodPressure':BloodPressure,
    'BMI':BMI,
    'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
    'Age':Age
}])

#predicting user input using model 
probability=model.predict(input)[0]
probability_percentage=model.predict_proba(input)[0][1]

if probability==1:
    print(f'Likely to have diabetes, with a probability of {probability_percentage*100:.2f} %')
else:
    print(f'Likely not to have diabetes with a probability of {probability_percentage*100:.2f} %')




joblib.dump(model, 'diabetes_model.joblib')