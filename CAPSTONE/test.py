import gradio as gr

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

# Gradio interface
with gr.Blocks() as demo:
    # Second page - Model Testing
    with gr.Tab("Model Testing"):
        gr.Markdown("# 🚀 **Model Testing - XGBoost & Random Forest**")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### XGBoost Model Results")
                xgboost_plot = gr.Image(value=xgb_plot_image_path, label="XGBoost Evaluation Plot")  # Use Image instead of Plot
                xgboost_metrics = gr.Textbox(value=xgb_metrics_text, label="XGBoost Evaluation Metrics", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### Random Forest Model Results")
                rf_plot = gr.Image(value=rf_plot_image_path, label="Random Forest Evaluation Plot")  # Use Image instead of Plot
                rf_metrics = gr.Textbox(value=rf_metrics_text, label="Random Forest Evaluation Metrics", interactive=False)

# Launch the interface
demo.launch()
