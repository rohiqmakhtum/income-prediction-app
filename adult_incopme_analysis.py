import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# Load data
data = pd.read_excel('D:\\GAWE BECIK\\Data\\adult.xlsx')

# Feature and target selection
X = data[['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'hours-per-week']]
y = data['income']

# Numerical and categorical features
numerical_features = ['age', 'hours-per-week']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate models
results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"\n{name} Performance:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    results[name] = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': cm
    }

# Cross-validation
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    cv_results = cross_validate(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"\n{name} Cross-Validation:")
    print("Mean Accuracy:", cv_results['test_score'].mean())

# Hyperparameter tuning with GridSearchCV
param_grids = {
    'Decision Tree': {
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'poly', 'rbf']
    },
    'Random Forest': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30]
    }
}

for name, model in models.items():
    param_grid = param_grids[name]
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"\n{name} Best Parameters:")
    print(grid_search.best_params_)

# Plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

for name, result in results.items():
    plot_confusion_matrix(result['confusion_matrix'], f'{name} Confusion Matrix')

# Save the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', SVC(C=1, kernel='rbf'))])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'ahmad_rohiqim_svm_model.pkl')

# Streamlit app
st.title('Income Prediction App')

# Input fields
age = st.number_input('Age', min_value=18, max_value=100, value=30)
hours_per_week = st.number_input('Hours Per Week', min_value=1, max_value=100, value=40)
workclass = st.selectbox('Workclass', ['Private', 'Self-Employed', 'Government', 'Unknown'])
education = st.selectbox('Education', ['Bachelors', 'Masters', 'PhD', 'HS-grad', 'Some-college'])
marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Never-married', 'Divorced', 'Widowed'])
occupation = st.selectbox('Occupation', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial'])
relationship = st.selectbox('Relationship', ['Husband', 'Not-in-family', 'Other-relative', 'Own-child'])
race = st.selectbox('Race', ['White', 'Black', 'Asian-Pac-Islander', 'Other'])
gender = st.selectbox('Gender', ['Male', 'Female'])

# Create dataframe for prediction
input_data = pd.DataFrame({
    'age': [age],
    'hours-per-week': [hours_per_week],
    'workclass': [workclass],
    'education': [education],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender]
})

# Load model and predict
model = joblib.load('ahmad_rohiqim_svm_model.pkl')
prediction = model.predict(input_data)
st.write(f'Predicted Income: {"<=50K" if prediction[0] == "<=50K" else ">50K"}')
