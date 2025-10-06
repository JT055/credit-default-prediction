#import kagglehub
#import pandas as pd
#import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler, LabelEncoder
#from scipy import stats
#from sklearn.ensemble import  RandomForestClassifier
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, mean_absolute_error, r2_score
#from sklearn.metrics import root_mean_squared_error
#import matplotlib.pyplot as plt
#import joblib
#import numpy as np

# Download latest version
#CreditData = kagglehub.dataset_download("")
#print("Path to dataset files:", CreditData)

# Retrieves the dataset from my device 
#CreditData = pd.read_csv("file path from device")

# Rename the column 
CreditData = CreditData.rename(columns={'default.payment.next.month': 'default_payment_next_month'}) 

# Check for missing values and fill them
for column in CreditData.columns:
    if CreditData[column].isnull().sum() > 0:
        CreditData[column].fillna(CreditData[column].mean(), inplace=True)
print("Missing Values after filling:\n", CreditData.isnull().sum())

# Check for outliers  
CreditData['Z_score'] = stats.zscore(CreditData['LIMIT_BAL'])
print(CreditData[CreditData['Z_score'] > 5])  # Show rows that have a higher Z_score than 5
for Credit_Data in CreditData[CreditData['Z_score'] > 5]['ID']:
    print("Outlier detected:", Credit_Data)
    CreditData = CreditData[CreditData['ID'] != Credit_Data]  # Remove the outlier

# Encode categorical variables 
CreditData['SEX'] = CreditData['SEX'].map({1: 'Male', 2: 'Female'})
CreditData['EDUCATION'] = CreditData['EDUCATION'].map({1: 'Graduate School', 2: 'University', 3: 'Secondary School', 4: 'Others', 5: 'Unknown', 6: 'Unknown'})
CreditData['MARRIAGE'] = CreditData['MARRIAGE'].map({1: 'Married', 2: 'Single', 3: 'Others'})

# Encode the categorical variables using LabelEncoder
label_encoder = LabelEncoder()
CreditData['SEX'] = label_encoder.fit_transform(CreditData['SEX'])
CreditData['EDUCATION'] = label_encoder.fit_transform(CreditData['EDUCATION'])
CreditData['MARRIAGE'] = label_encoder.fit_transform(CreditData['MARRIAGE'])

# Drop any irrelevant columns
CreditData = CreditData.drop(columns=['ID'])
print("Dataset preview after dropping columns:\n", CreditData.head())

# Scale features if necessary
scaler = StandardScaler()  # Initialize the scaler 
features_to_scale = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
CreditData[features_to_scale] = scaler.fit_transform(CreditData[features_to_scale])

# Split the data into training and testing sets
X = CreditData.drop(columns=['default_payment_next_month', 'Z_score'])
y = CreditData['default_payment_next_month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the classifier
classifier =  RandomForestClassifier(random_state=42, max_depth=5)  # Added max_depth to prevent overfitting

# Train the classifier
classifier.fit(X_train, y_train)

# Save the trained model to disk
joblib.dump(classifier, 'model.pkl')

# Load the trained model
model = joblib.load('model.pkl')

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy, 2))

# Print classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix with meaningful labels
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix for Default Payment Predictions')
plt.xlabel('Predicted Default Payment')
plt.ylabel('Actual Default Payment')
plt.show()

# ROC Curve for Decision Tree with updated labels for predicting if a person defaults
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Predicting No Default When Default)')
plt.ylabel('True Positive Rate (Predicting Default When Default)')
plt.title('Receiver Operating Characteristic (ROC) Curve for Predicting Credit Default')
plt.legend(loc="lower right")
plt.show()

# Print all the predictions vs the actual values with the features, sorted by the difference between the actual and predicted values
results = X_test.copy()
results['Actual Default Payment'] = y_test.values
results['Predicted Default Payment'] = y_pred

results['Difference'] = abs(results['Actual Default Payment'] - results['Predicted Default Payment'])
results = results.sort_values(by='Difference', ascending=False)
print("Predictions vs. Actual Default Payment :\n", results)

# Using the trained model, predict the default payment of a creditor with the following stats:
new_CreditData = {
    'SEX': 1,  # Male
    'EDUCATION': 3,  # High school
    'MARRIAGE': 0,  # Unknown marital status
    'LIMIT_BAL': 6000,  # Credit limit
    'AGE': 40,  # Age in years
    'BILL_AMT1': 500,  # Bill statement amount in September 2005
    'BILL_AMT2': 450,  # Bill statement amount in August 2005
    'BILL_AMT3': 400,  # Bill statement amount in July 2005
    'BILL_AMT4': 350,  # Bill statement amount in June 2005
    'BILL_AMT5': 300,  # Bill statement amount in May 2005
    'BILL_AMT6': 250,  # Bill statement amount in April 2005
    'PAY_AMT1': 100,  # Previous payment amount in September 2005
    'PAY_AMT2': 100,  # Previous payment amount in August 2005
    'PAY_AMT3': 100,  # Previous payment amount in July 2005
    'PAY_AMT4': 100,  # Previous payment amount in June 2005
    'PAY_AMT5': 100,  # Previous payment amount in May 2005
    'PAY_AMT6': 100,  # Previous payment amount in April 2005
    'PAY_0': 674,  # Repayment status in September 2005
    'PAY_2': 455,  # Repayment status in August 2005
    'PAY_3': 933,  # Repayment status in July 2005
    'PAY_4': 12394,  # Repayment status in June 2005
    'PAY_5': 4005,  # Repayment status in May 2005
    'PAY_6': 494   # Repayment status in April 2005
}

# DataFrame from the new credit data 
new_credit_df = pd.DataFrame([new_CreditData])

# Ensure the order of columns matches the training data exactly.
new_credit_df = new_credit_df[X.columns]

# Scale the features of the new credit data using the same scaler fitted on training data.
new_credit_df[features_to_scale] = scaler.transform(new_credit_df[features_to_scale])

new_credit_default_payment_next_month = model.predict(new_credit_df)
print("Predicted default payment:", new_credit_default_payment_next_month[0])

# Visualize feature importance using a bar plot for DecisionTreeClassifier
feature_importance = model.feature_importances_

plt.figure(figsize=(10, 5))
plt.bar(X.columns, feature_importance)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()

# Print the feature importance of each column in the dataset 
print("Feature Importance:")
for i in range(len(X.columns)):
    print(X.columns[i], ":", feature_importance[i])

# An error plot showing the residuals of the model
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Default Payments")
plt.ylabel("Residuals")
plt.title("Error Plot")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# calucation of mean absolute error, root mean squared error and r2 score
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# display the calucations 
print("mean absolute error:", mae)
print("root mean squared error", rmse)
print("r2 score", r2)