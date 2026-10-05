"""
model/model.py
--------------
Contains the SalaryPredictor class that encapsulates the complete
Linear Regression machine learning pipeline:
  1. Data loading & validation
  2. Data cleaning
  3. Train/Test split
  4. Model training (scikit-learn LinearRegression)
  5. Model evaluation (MAE, MSE, RMSE, R²)
  6. Salary prediction for new input
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SalaryPredictor:
    """
    Encapsulates the complete Linear Regression pipeline for salary prediction.
    """

    def __init__(self):
        # The trained scikit-learn LinearRegression model
        self.model = None
        # Processed DataFrame
        self.data = None
        # Train/test arrays
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # Evaluation metrics dict
        self.metrics = {}
        # Flag – True once the model has been trained
        self.is_trained = False

    # ------------------------------------------------------------------
    # Step 1 & 2  –  Load and validate the CSV dataset
    # ------------------------------------------------------------------
    def load_and_validate(self, filepath_or_df):
        """
        Load the CSV from a file path (string) or accept a pre-parsed DataFrame.
        Validates that required columns exist and data is clean.

        Returns (True, cleaned_df) on success, or raises ValueError with a
        user-friendly message on failure.
        """
        # Accept either a file path or a DataFrame directly
        if isinstance(filepath_or_df, pd.DataFrame):
            df = filepath_or_df.copy()
        else:
            df = pd.read_csv(filepath_or_df)

        # --- Column validation ---
        required = {"YearsExperience", "Salary"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {', '.join(missing)}. "
                "Please upload a file with 'YearsExperience' and 'Salary' columns."
            )

        # Keep only the two relevant columns
        df = df[["YearsExperience", "Salary"]].copy()

        # --- Type coercion ---
        df["YearsExperience"] = pd.to_numeric(df["YearsExperience"], errors="coerce")
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

        # --- Drop rows with any NaN ---
        before = len(df)
        df.dropna(inplace=True)
        after = len(df)
        if after < before:
            print(f"[Info] Dropped {before - after} row(s) with missing/non-numeric values.")

        # --- Empty dataset check ---
        if df.empty:
            raise ValueError("The uploaded dataset is empty after cleaning. Please upload a valid CSV.")

        # --- Negative years check ---
        if (df["YearsExperience"] < 0).any():
            raise ValueError(
                "Dataset contains negative values in 'YearsExperience'. "
                "Please fix the data before uploading."
            )

        # --- Minimum rows for split ---
        if len(df) < 5:
            raise ValueError(
                f"Dataset has only {len(df)} valid row(s). "
                "At least 5 rows are required to train and test the model."
            )

        self.data = df
        return True, df

    # ------------------------------------------------------------------
    # Step 3–6  –  Split data and train the model
    # ------------------------------------------------------------------
    def train(self, test_size=0.2, random_state=42):
        """
        Splits the cleaned data into training/testing sets and trains a
        scikit-learn LinearRegression model.

        Parameters
        ----------
        test_size    : float  – fraction of data used for testing (default 0.2 = 20 %)
        random_state : int    – seed for reproducibility
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call load_and_validate() first.")

        X = self.data[["YearsExperience"]].values   # 2-D array expected by sklearn
        y = self.data["Salary"].values

        # Train / Test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train the Linear Regression model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        self.is_trained = True
        self._compute_metrics()
        return self

    # ------------------------------------------------------------------
    # Step 8  –  Evaluation metrics
    # ------------------------------------------------------------------
    def _compute_metrics(self):
        """Compute MAE, MSE, RMSE, and R² on the test set."""
        y_pred_test = self.model.predict(self.X_test)

        mae  = mean_absolute_error(self.y_test, y_pred_test)
        mse  = mean_squared_error(self.y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2   = r2_score(self.y_test, y_pred_test)

        self.metrics = {
            "coefficient" : round(float(self.model.coef_[0]), 4),
            "intercept"   : round(float(self.model.intercept_), 4),
            "mae"         : round(mae,  2),
            "mse"         : round(mse,  2),
            "rmse"        : round(rmse, 2),
            "r2"          : round(r2,   4),
        }

    # ------------------------------------------------------------------
    # Step 10  –  Predict salary for new input
    # ------------------------------------------------------------------
    def predict(self, years_experience):
        """
        Predict the salary for a given number of years of experience.

        Parameters
        ----------
        years_experience : float – years of professional experience

        Returns
        -------
        float – predicted salary
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Please train the model first.")

        years_experience = float(years_experience)

        if years_experience < 0:
            raise ValueError("Years of experience cannot be negative.")

        prediction = self.model.predict([[years_experience]])[0]
        return round(float(prediction), 2)

    # ------------------------------------------------------------------
    # Helpers  –  data for visualisation
    # ------------------------------------------------------------------
    def get_regression_line_points(self, n_points=100):
        """
        Return evenly spaced (x, y_pred) pairs spanning the data range –
        used to draw the regression line on charts.
        """
        if not self.is_trained:
            return [], []

        x_min = float(self.data["YearsExperience"].min())
        x_max = float(self.data["YearsExperience"].max())
        x_line = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
        y_line = self.model.predict(x_line)
        return x_line.flatten().tolist(), y_line.tolist()

    def get_actual_vs_predicted(self):
        """
        Return lists of actual and predicted salaries for the full dataset –
        used in the Actual vs Predicted scatter plot.
        """
        if not self.is_trained:
            return [], []

        y_all_pred = self.model.predict(self.data[["YearsExperience"]].values)
        return self.data["Salary"].tolist(), y_all_pred.tolist()

    def get_data_as_dict(self):
        """Return the cleaned dataset as a list of dicts (JSON-serialisable)."""
        if self.data is None:
            return []
        return self.data.to_dict(orient="records")
