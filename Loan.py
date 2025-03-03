
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE  # Importing SMOTE for oversampling

# Load dataset
filePath = "C:/Users/Adiseshan Ramanan/Documents/AdiDevs/DS&ML/LoanPrediction/Loan_default.csv"
loanData = pd.read_csv(filePath)

# Display basic information
print(loanData.head())
print(loanData.info())

# Drop missing values
loanData = loanData.dropna()

# Selecting relevant columns
loanData = loanData[['Default', 'LoanAmount', 'Income', 'DTIRatio', 'CreditScore']]
loanData['Default'] = loanData['Default'].astype('category')

# Plot Loan Amount Distribution
plt.figure(figsize=(8, 5))
sns.histplot(loanData['LoanAmount'], bins=30, kde=True, color='steelblue')
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')
plt.ylabel('Count')
plt.show()

# Splitting data into train and test sets
X = loanData[['CreditScore', 'LoanAmount', 'Income', 'DTIRatio']]
y = loanData['Default']

# Check class distribution
print("Class Distribution Before Resampling:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# Oversampling the minority class using SMOTE
smote = SMOTE(random_state=123)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Check class distribution after resampling
print("Class Distribution After Resampling:\n", y_train.value_counts())

# Standardizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model with balanced class weights
model = LogisticRegression(class_weight='balanced', random_state=123)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluating model performance
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))  # Avoid warning
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Save cleaned data
cleaned_file_path = "C:/Users/Adiseshan Ramanan/Documents/AdiDevs/DS&ML/LoanPrediction/cleanedData.csv"
loanData.to_csv(cleaned_file_path, index=False)
