import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris, fetch_california_housing
import base64

# Set page config
st.set_page_config(
    page_title="Machine Learning App",
    page_icon="https://i.imgur.com/C6lAamP.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

#function to load a bg image
def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://c.wallhere.com/photos/91/50/mountain_top_black_dark_nature_monochrome_landscape_mountains-1296041.jpg!d");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Call the function to set background image
set_bg_hack_url()

# Custom CSS styles with base64-encoded background image
st.markdown(
    f"""
    <style>
    .sidebar .sidebar-content {{
        background-image: linear-gradient(#D5DBDB, #F2F3F4);
    }}
    .st-cc {{
        color: #566573;
    }}
    .st-cq {{
        color: #566573;
    }}
    .st-c8 {{
        color: #FFCDD2;
    }}
    .st-top-right {{
    position: fixed;
    top: 60px;
    right: 10px;
    font-size: 16px;
    color: #FF0000; /* Red text color */
    background-color: rgba(255, 255, 255, 0.8); /* White background color with some transparency */
    padding: 5px 10px; /* Add padding to the text */
    border-radius: 5px; /* Add border radius for rounded corners */
    font-family: Arial, sans-serif;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Function to build regression model
def build_regression_model(df, algorithm, hyperparameters):
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(y.name)

    model = algorithm(**hyperparameters)
    model.fit(X_train, y_train)

    # Model performance
    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    y_pred_train = model.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(y_train, y_pred_train))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(y_train, y_pred_train))

    st.markdown('**2.2. Test set**')
    y_pred_test = model.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(y_test, y_pred_test))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(y_test, y_pred_test))

    # Model parameters
    st.subheader('3. Model Parameters')
    st.write(model.get_params())

    return model


# Function to build classification model
def build_classification_model(df, algorithm, hyperparameters):
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(y.name)

    model = algorithm(**hyperparameters)
    model.fit(X_train, y_train)

    # Model performance
    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    y_pred_train = model.predict(X_train)
    st.write('Accuracy Score:')
    st.info(accuracy_score(y_train, y_pred_train))

    st.markdown('**2.2. Test set**')
    y_pred_test = model.predict(X_test)
    st.write('Accuracy Score:')
    st.info(accuracy_score(y_test, y_pred_test))

    # Model parameters
    st.subheader('3. Model Parameters')
    st.write(model.get_params())

    return model

# The Machine Learning App
st.write(
    """
    # Machine Learning App
    Select the problem type, dataset, model, and hyperparameters.
    """
)

# Sidebar - Select problem type
problem_type = st.sidebar.selectbox("Select Problem Type", ["Regression", "Classification"])

# Sidebar - Select dataset or upload CSV file
if problem_type == "Classification":
    st.sidebar.write("### Select Dataset or Upload CSV File for Classification")
    dataset_option = st.sidebar.selectbox("Select Dataset", ["Iris", "Upload CSV File"])
elif problem_type == "Regression":
    st.sidebar.write("### Select Dataset or Upload CSV File for Regression")
    dataset_option = st.sidebar.selectbox("Select Dataset", ["California Housing", "Upload CSV File"])

# Load dataset
df = None
if dataset_option == "Iris":
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target
elif dataset_option == "California Housing":
    housing = fetch_california_housing()
    df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    df["target"] = housing.target
elif dataset_option == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# Sidebar - Select model and hyperparameters
st.sidebar.write("### Select Model and Hyperparameters")

if problem_type == "Regression":
    regression_algorithm = st.sidebar.selectbox(
        "Select Regression Algorithm",
        ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest Regression", "Gradient Boosting Regression"]
    )
    if regression_algorithm == "Linear Regression":
        hyperparameters = {}
        algorithm = LinearRegression
    elif regression_algorithm == "Ridge Regression":
        alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.5, 0.01)
        hyperparameters = {"alpha": alpha}
        algorithm = Ridge
    elif regression_algorithm == "Lasso Regression":
        alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.5, 0.01)
        hyperparameters = {"alpha": alpha}
        algorithm = Lasso
    elif regression_algorithm == "Random Forest Regression":
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 1000, 100)
        max_features = st.sidebar.selectbox("Max Features", ["sqrt", "log2"])
        hyperparameters = {"n_estimators": n_estimators, "max_features": max_features}
        algorithm = RandomForestRegressor
    elif regression_algorithm == "Gradient Boosting Regression":
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 1000, 100)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        hyperparameters = {"n_estimators": n_estimators, "learning_rate": learning_rate}
        algorithm = GradientBoostingRegressor

elif problem_type == "Classification":
    classification_algorithm = st.sidebar.selectbox(
        "Select Classification Algorithm",
        ["Logistic Regression", "Support Vector Classifier", "Random Forest Classifier", "Gradient Boosting Classifier"]
    )
    if classification_algorithm == "Logistic Regression":
        hyperparameters = {"max_iter": 1000}
        algorithm = LogisticRegression
    elif classification_algorithm == "Support Vector Classifier":
        kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        hyperparameters = {"kernel": kernel}
        algorithm = SVC
    elif classification_algorithm == "Random Forest Classifier":
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 1000, 100)
        max_features = st.sidebar.selectbox("Max Features", ["sqrt", "log2"])
        hyperparameters = {"n_estimators": n_estimators, "max_features": max_features}
        algorithm = RandomForestClassifier
    elif classification_algorithm == "Gradient Boosting Classifier":
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 1000, 100)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        hyperparameters = {"n_estimators": n_estimators, "learning_rate": learning_rate}
        algorithm = GradientBoostingClassifier

# Main panel
st.write("## Results")

if df is not None:
    st.write("### Data Preview")
    st.write("First 20 rows of the dataset:")
    st.write(df.head(20))  # Displaying the first 20 rows of the dataset

    if problem_type == "Regression":
        model = build_regression_model(df, algorithm, hyperparameters)
        st.write("### Regression Model")
        st.write("Model:", algorithm.__name__)
    elif problem_type == "Classification":
        model = build_classification_model(df, algorithm, hyperparameters)
        st.write("### Classification Model")
        st.write("Model:", algorithm.__name__)
else:
    st.info("Please upload a CSV file or select an example dataset.")

# Text in the top-right corner
st.markdown('<p class="st-top-right">Created by - RAJDEEP CHAKRAVORTY</p>', unsafe_allow_html=True)
