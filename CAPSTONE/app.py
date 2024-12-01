import gradio as gr
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the pre-trained XGBoost model
model = joblib.load("C:\\Users\\Lenovo\\Downloads\\xgboost_best_model.pkl")

# Define the required feature columns for the model (including missing ones)
feature_columns = [
    'Store', 'Date', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'Quarter', 'DayOfMonth', 'WeekOfYear', 'Year', 'Month', 'Day', 'Weekday'
]

# Initial inventory level (you can change this if necessary)
initial_inventory_level = 15000000

# Function to generate input data for prediction
def generate_input_data(start_date, num_days):
    date_rng = pd.date_range(start=start_date, periods=num_days, freq='D')
    dummy_data = pd.DataFrame(date_rng, columns=['Date'])
    dummy_data['Store'] = 1  # Assume Store 1, or adjust as needed
    dummy_data['Holiday_Flag'] = 0  # Assuming no holiday, modify based on your data
    dummy_data['Temperature'] = 15 + 10 * np.random.rand(num_days)
    dummy_data['Fuel_Price'] = 3 + np.random.rand(num_days)
    dummy_data['CPI'] = 100 + np.random.rand(num_days)  # Placeholder, adjust as needed
    dummy_data['Unemployment'] = 5 + np.random.rand(num_days)  # Placeholder, adjust as needed
    dummy_data['Quarter'] = dummy_data['Date'].dt.quarter
    dummy_data['DayOfMonth'] = dummy_data['Date'].dt.day
    dummy_data['WeekOfYear'] = dummy_data['Date'].dt.isocalendar().week
    dummy_data['Year'] = dummy_data['Date'].dt.year
    dummy_data['Month'] = dummy_data['Date'].dt.month
    dummy_data['Day'] = dummy_data['Date'].dt.day
    dummy_data['Weekday'] = dummy_data['Date'].dt.weekday

    X_input = dummy_data[feature_columns]
    return dummy_data, X_input

# Function to predict demand and handle inventory
def predict_demand_gui(start_date, num_days):
    try:
        # Convert num_days to an integer
        num_days = int(num_days)

        # Generate dummy data based on user input
        dummy_data, X_input = generate_input_data(start_date, num_days)

        # Predict demand using the pre-trained model
        predictions = model.predict(X_input)
        dummy_data['Predicted Demand'] = predictions

        # Inventory management logic
        inventory_levels = [initial_inventory_level]
        restock_threshold = 1000000
        restock_details = []
        restocked = False

        # Loop through predictions to update inventory
        for i, prediction in enumerate(predictions):
            new_inventory = inventory_levels[-1] - prediction
            if new_inventory < restock_threshold and not restocked:
                restock_needed = restock_threshold - new_inventory
                new_inventory += restock_needed
                restock_details.append((dummy_data['Date'][i], restock_needed))
                restocked = True

            inventory_levels.append(new_inventory)

        # Add the calculated inventory levels to the dataframe
        dummy_data['Inventory Level'] = inventory_levels[1:]

        # Create an interactive plot using Plotly
        fig = px.line(dummy_data, x='Date', y=['Predicted Demand', 'Inventory Level'],
                      labels={'value': 'Count', 'Date': 'Date'},
                      title='Predicted Demand and Inventory Levels',
                      line_shape='linear', markers=True)

        # Add restock threshold line to the plot
        fig.add_hline(y=restock_threshold, line_dash="dash", line_color="red", 
                      annotation_text="Restock Threshold", annotation_position="bottom right")

        # Return result as a DataFrame and the interactive plot
        result = dummy_data[['Date', 'Predicted Demand', 'Inventory Level']]
        return result, fig

    except Exception as e:
        return str(e), None


# Evaluation Metrics for Random Forest and XGBoost
rf_metrics_text = """
**Random Forest Regressor - Testing Results:**
- MAE: 60210.69
- RMSE: 126998.28
- MAPE: 5.13%
"""

xgb_metrics_text = """
**XGBoost Regressor - Testing Results:**
- MAE: 50377.07
- RMSE: 95767.41
- MAPE: 4.91%
"""

# Assuming the plots for both models are saved as images (update paths as necessary)
rf_plot_image_path = "C:\\Users\\Lenovo\\Downloads\\rfplot.png"
xgb_plot_image_path = "C:\\Users\\Lenovo\\Downloads\\xgbplot.png"

# Gradio Interface
with gr.Blocks(css=""" 
    .gradio-container {
        background-color: #f0f4f8;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .gr-button-primary {
        background-color: #007bff;
        border-color: #007bff;
        color: white;
    }
    .gr-button-primary:hover {
        background-color: #0056b3;
        border-color: #0056b3;
    }
    .gr-markdown {
        font-size: 18px;
        line-height: 1.5;
    }
    .gr-textbox input, .gr-slider input {
        border-radius: 8px;
        border: 1px solid #ccc;
    }
    .gr-dataframe table {
        font-size: 14px;
        color: #333;
        border-collapse: collapse;
    }
    .gr-slider {
        margin-top: 10px;
    }
""") as interface:

    # Tabs for Navigation
    with gr.Tabs():
        # First Tab - Demand Prediction with Inventory Management
        with gr.TabItem("Demand Prediction"):
            gr.Markdown("# 📈 **Demand Prediction with Inventory Management**")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧑‍🏫 Prediction Table")
                    output_table = gr.Dataframe(headers=["Date", "Predicted Demand", "Inventory Level"], show_label=False)

                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Interactive Prediction Plot")
                    output_image = gr.Plot(label="Demand and Inventory Plot")

            gr.Markdown("## ✨ Input Parameters")
            with gr.Row():
                start_date = gr.Textbox(value=str(datetime.now().date()), label="Start Date (YYYY-MM-DD)", interactive=True)
                num_days = gr.Textbox(value="7", label="Number of Days", interactive=True)  # Textbox for number of days

            predict_button = gr.Button("🔮 Predict Demand & Inventory", variant="primary")
            predict_button.click(
                predict_demand_gui, 
                inputs=[start_date, num_days],
                outputs=[output_table, output_image]
            )

        # Second Tab - Model Testing
        with gr.TabItem("Model Testing"):
            gr.Markdown("# 🚀 **Model Testing - XGBoost & Random Forest**")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### XGBoost Model Results")
                    xgboost_plot = gr.Image(value=xgb_plot_image_path, label="XGBoost Evaluation Plot")  # Use Image instead of Plot
                    # Use Markdown to display larger text without issue
                    gr.Markdown(xgb_metrics_text, label="XGBoost Evaluation Metrics")

                with gr.Column(scale=1):
                    gr.Markdown("### Random Forest Model Results")
                    rf_plot = gr.Image(value=rf_plot_image_path, label="Random Forest Evaluation Plot")  # Use Image instead of Plot
                    # Use Markdown to display larger text without issue
                    gr.Markdown(rf_metrics_text, label="Random Forest Evaluation Metrics")

# Launch the Gradio app
interface.launch(share=False, debug=True)
