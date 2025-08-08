Diabetes Prediction with ML model

--Abstract--
This project develops a machine learning model to predict diabetes likelihood based on health parameters using Random Forest Classifier. The model aids early diagnosis with good accuracy.

--Table of Contents--
1. Abstract
2. Introduction
3. Existing Methods
4. Proposed Method (Architecture)
5. Methodology
6. Implementation
7. Results
8. Conclusion

--Introduction--
Diabetes is a metabolic disorder that causes elevated blood sugar levels over a prolonged period. If left undiagnosed or untreated, it may lead to serious complications such as heart disease, kidney failure, or blindness. Manual diagnosis is often time-consuming and requires lab testing. In this project, we utilize machine learning techniques to build a predictive model that can analyze patient data and predict the likelihood of diabetes with reasonable accuracy

--Existing Methods--
Several traditional and machine learning methods have been explored for diabetes prediction:
•	Logistic Regression
•	Decision Tree Classifier
•	Support Vector Machine (SVM)
•	Naive Bayes
•	K-Nearest Neighbors (KNN)
While effective to some extent, these models often face limitations in handling nonlinear relationships and feature interactions.

--Proposed Method (Architecture)--

A Random Forest Classifier is proposed. It combines multiple decision trees for more robust and accurate predictions.
--Methodology--
1.	Data Collection:
•	Dataset used: diabetes (1).csv
•	Features: Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age
•	Target: Outcome (0 = Non-Diabetic, 1 = Diabetic)
2.	Data Preprocessing:
•	Selected the 5 most relevant features.
•	Removed unnecessary columns.
3.	Splitting Dataset:
•	80% for training, 20% for testing.
•	Used train_test_split() from sklearn.
4.	Model Training:
•	Used RandomForestClassifier() with default parameters.
•	Fit the model using the training data.
5.	Evaluation:
•	Predicted on test data.
•	Calculated accuracy using accuracy_score.
6.	User Prediction:
•	Took input via CLI (command-line).
•	Created a DataFrame for prediction.
•	Used model.predict() and model.predict_proba() for prediction and probability

--Implementation--
Libraries: pandas, sklearn, joblib
Model saved with joblib
User input taken via command line
Model used to predict and return probability
We also created joblib file which is useful in API, and websites for data handling and predicting 

--Results--
Accuracy: ~77%
Takes 5 inputs (Glucose, BloodPressure, BMI, DPF, Age)
Returns binary prediction + probability
Useful for quick assessments

--Conclusion--
This project demonstrates the use of machine learning in health diagnostics. The Random Forest Classifier offers a high-performing and interpretable model for diabetes prediction. It can be expanded into a web or mobile application for real-time use by healthcare professionals and patients. Future improvements may include using additional features like insulin levels or physical activity and comparing performance with deep learning models.
