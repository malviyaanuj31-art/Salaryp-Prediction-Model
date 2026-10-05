"""
app.py
------
Flask web application for the Salary Prediction project.

Routes:
  GET  /                  – serve the main page
  POST /upload            – upload & parse CSV, return dataset info
  POST /train             – train the model, return metrics + chart data
  POST /predict           – predict salary for given years of experience
  GET  /sample-dataset    – download the sample Salary_Data.csv
"""

import os
import io
import json

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from flask import Flask, jsonify, render_template, request, send_file

from model.model import SalaryPredictor

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit

# A single global predictor instance – sufficient for a demo/internship app
predictor = SalaryPredictor()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def build_scatter_chart(predictor_obj):
    """
    Build a Plotly figure with:
      - Scatter: actual data points
      - Line:    regression line
    Returns JSON string for embedding in the page.
    """
    df = predictor_obj.data
    x_line, y_line = predictor_obj.get_regression_line_points()

    fig = go.Figure()

    # --- Actual data points ---
    fig.add_trace(go.Scatter(
        x=df["YearsExperience"].tolist(),
        y=df["Salary"].tolist(),
        mode="markers",
        name="Actual Data",
        marker=dict(color="#3b82f6", size=10, line=dict(width=1, color="#1d4ed8")),
        hovertemplate="<b>Experience:</b> %{x} yrs<br><b>Salary:</b> ₹%{y:,.0f}<extra></extra>",
    ))

    # --- Regression line ---
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="Regression Line",
        line=dict(color="#ef4444", width=2.5, dash="solid"),
        hovertemplate="<b>Experience:</b> %{x:.1f} yrs<br><b>Predicted:</b> ₹%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title="Years of Experience vs Salary",
        xaxis_title="Years of Experience",
        yaxis_title="Salary (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Segoe UI, system-ui, sans-serif", size=13),
        hovermode="closest",
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def build_actual_vs_predicted_chart(predictor_obj):
    """
    Build a Plotly figure comparing actual vs predicted salaries.
    """
    actual, predicted = predictor_obj.get_actual_vs_predicted()

    # Perfect-prediction reference line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))

    fig = go.Figure()

    # Perfect line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="#6b7280", width=1.5, dash="dash"),
        hoverinfo="skip",
    ))

    # Actual vs predicted scatter
    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode="markers",
        name="Predicted vs Actual",
        marker=dict(color="#8b5cf6", size=10, line=dict(width=1, color="#6d28d9")),
        hovertemplate="<b>Actual:</b> ₹%{x:,.0f}<br><b>Predicted:</b> ₹%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title="Actual Salary vs Predicted Salary",
        xaxis_title="Actual Salary (₹)",
        yaxis_title="Predicted Salary (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Segoe UI, system-ui, sans-serif", size=13),
        hovermode="closest",
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept a CSV file upload, validate it, and return dataset info.
    """
    global predictor

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected. Please choose a CSV file."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"success": False, "error": "Invalid file type. Please upload a CSV file (.csv)."}), 400

    try:
        content = file.read().decode("utf-8")
        df_raw = pd.read_csv(io.StringIO(content))
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not parse CSV: {str(exc)}"}), 400

    # Create a fresh predictor so previous state is cleared
    predictor = SalaryPredictor()

    try:
        _, df_clean = predictor.load_and_validate(df_raw)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    # Build preview (first 10 rows)
    preview = df_clean.head(10).to_dict(orient="records")
    # Round for display
    preview = [
        {"YearsExperience": r["YearsExperience"], "Salary": int(r["Salary"])}
        for r in preview
    ]

    stats = df_clean.describe().round(2).to_dict()

    return jsonify({
        "success"      : True,
        "num_records"  : len(df_clean),
        "num_features" : 2,
        "preview"      : preview,
        "stats"        : stats,
        "columns"      : ["YearsExperience", "Salary"],
    })


@app.route("/train", methods=["POST"])
def train():
    """
    Train the Linear Regression model and return metrics + chart JSON.
    """
    if predictor.data is None:
        return jsonify({"success": False, "error": "No dataset loaded. Please upload a CSV file first."}), 400

    try:
        predictor.train(test_size=0.2, random_state=42)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Training failed: {str(exc)}"}), 500

    m = predictor.metrics

    scatter_json   = build_scatter_chart(predictor)
    avp_json       = build_actual_vs_predicted_chart(predictor)

    equation = (
        f"Salary = {m['intercept']:,.2f} + {m['coefficient']:,.2f} × YearsExperience"
    )

    return jsonify({
        "success"       : True,
        "metrics"       : m,
        "equation"      : equation,
        "scatter_chart" : json.loads(scatter_json),
        "avp_chart"     : json.loads(avp_json),
        "train_size"    : len(predictor.X_train),
        "test_size"     : len(predictor.X_test),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept years_experience in JSON body and return a salary prediction.
    """
    if not predictor.is_trained:
        return jsonify({"success": False, "error": "Model not trained yet. Please train the model first."}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"success": False, "error": "Invalid JSON body."}), 400

    years_raw = data.get("years_experience", "")

    if years_raw == "" or years_raw is None:
        return jsonify({"success": False, "error": "Please enter the number of years of experience."}), 400

    try:
        years = float(years_raw)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Years of experience must be a valid number."}), 400

    try:
        salary = predictor.predict(years)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    return jsonify({
        "success"          : True,
        "years_experience" : years,
        "predicted_salary" : salary,
    })


@app.route("/sample-dataset")
def sample_dataset():
    """Allow user to download the bundled sample CSV."""
    sample_path = os.path.join(os.path.dirname(__file__), "dataset", "Salary_Data.csv")
    if not os.path.exists(sample_path):
        return jsonify({"error": "Sample dataset not found."}), 404
    return send_file(sample_path, as_attachment=True, download_name="Salary_Data.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
