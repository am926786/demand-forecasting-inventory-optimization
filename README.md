# demand-forecasting-inventory-optimization
This project focuses on predictive inventory management using machine learning techniques. It employs data analysis and modeling to forecast inventory needs effectively, ensuring optimal stock levels.

# Project Structure
The repository is organized as follows:

CAPSTONE/
│
├── app.py                          # Main application script
├── eda.ipynb                       # Exploratory Data Analysis notebook
├── model.ipynb                     # Model training and evaluation notebook
├── predicted_inventory_management.xls # Predicted results
├── rfplot.png                      # Random Forest model visualization
├── xgbplot.png                     # XGBoost model visualization
├── rf_model.pkl                    # Random Forest saved model
├── seasonal_xgboost_model.joblib   # XGBoost seasonal model
├── xgboost_best_model.pkl          # XGBoost saved model
├── test.py                         # Script for testing the model
├── Walmart.csv                     # Dataset used for analysis and modeling
├── .gradio/                        # Gradio UI components (if any)

# Key Features
Exploratory Data Analysis (EDA): Insights and trends visualized using eda.ipynb.
Predictive Modeling: Utilizes Random Forest and XGBoost for inventory predictions.
Visualization: Includes graphical representations of model performance.
Interactive UI: Potential use of Gradio for a user-friendly interface.

# Requirements
Install the required Python packages using the following command:

pip install -r requirements.txt

# Usage
Data Analysis: Open eda.ipynb in a Jupyter Notebook to explore and understand the data.

Model Training: Use model.ipynb to train the models on the provided dataset (Walmart.csv).

Predictions: Run app.py to generate predictions and interact with the application.

Testing: Execute test.py to test the deployed models.

# Results
Model Performance: 

Random Forest and XGBoost models were evaluated for accuracy and efficiency.

Visualizations (rfplot.png and xgbplot.png) demonstrate their performance.

Output:

Predicted results are saved in predicted_inventory_management.xls.

Dataset

Name: Walmart Dataset

Description: Contains historical data for inventory analysis and forecasting.

File: Walmart.csv

# Getting Started

Clone this repository:

git clone https://github.com/your-username/your-repo-name.git

cd your-repo-name

Run the application:

python app.py
