# Machine Learning App

This is a Streamlit web application for machine learning tasks. It allows users to select problem types (regression or classification), datasets, machine learning algorithms, and hyperparameters to build and evaluate models.

## Preview
This Project has been deployed at https://app-upload-csv-to-build-a-ml-model.streamlit.app/

(https://www.dropbox.com/scl/fi/xww5yy9ef6narl2lphf7i/Streamlit-MLalgoapp-demo.gif?rlkey=yvql0jler0yd4doibhyak9tue&st=nukc6gs8&dl=0)

## Features

- Supports regression and classification tasks.
- Allows users to upload their own CSV files or select example datasets (e.g., Iris, California Housing).
- Offers various machine learning algorithms including Linear Regression, Ridge Regression, Lasso Regression, Logistic Regression, Support Vector Classifier, Random Forest Classifier, Gradient Boosting Classifier, Random Forest Regression, and Gradient Boosting Regression.
- Provides model evaluation metrics such as accuracy score, coefficient of determination (R^2), and mean squared error (MSE).
- Customizable background image and sidebar logo.
- Compatible with dark mode for optimal viewing experience.

## Installation

To run this application locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone <https://github.com/RajDeep-Chakravorty/STREAMLIT-UPLOAD-CSV-TO-BUILD-A-ML-MODEL>
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Upon running the Streamlit app, you will be presented with a sidebar where you can select the problem type (regression or classification), dataset, machine learning algorithm, and hyperparameters.

2. Choose the desired options and click on the "Run" button to build and evaluate the model.

3. View the results including data preview, model performance metrics, and model parameters in the main panel.

4. Optionally, upload your own CSV file by selecting the "Upload CSV File" option in the sidebar.

5. Enjoy exploring different machine learning models and datasets!

## Credits

- This application was created by [RAJDEEP CHAKRAVORTY](https://github.com/RajDeep-Chakravorty)
