# Salary Prediction Using Linear Regression

> **IBM Applied AI & Machine Learning Internship Project**

---

## 1. Project Title
**Salary Prediction Using Machine Learning** – A beginner-friendly web application that predicts a person's salary based on their years of professional experience using a Linear Regression model.

---

## 2. Project Objective
Build an end-to-end machine learning web application that demonstrates the complete ML pipeline:

```
CSV Dataset → Data Preprocessing → Train/Test Split → Linear Regression → Evaluation → Visualization → Salary Prediction
```

---

## 3. Problem Statement
Given historical data of employees' years of experience and their corresponding salaries, train a Linear Regression model that can predict the expected salary for any new number of years of experience.

---

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| Flask | Web framework (backend) |
| scikit-learn | Linear Regression model |
| Pandas | Data loading & manipulation |
| NumPy | Numerical computations |
| Plotly | Interactive visualizations |
| HTML / CSS / JS | Frontend web interface |

---

## 5. Dataset Description

The dataset is a CSV file with two columns:

| Column | Description |
|---|---|
| `YearsExperience` | Number of years of professional experience (float) |
| `Salary` | Annual salary in INR (integer) |

A sample dataset (`dataset/Salary_Data.csv`) with 30 records is included. You can also upload your own CSV file from the web interface.

**Example rows:**
```
YearsExperience,Salary
1.1,39343
3.0,60150
7.1,98273
10.5,121872
```

---

## 6. Machine Learning Workflow

1. **Load** the CSV dataset using Pandas
2. **Validate** required columns (`YearsExperience`, `Salary`)
3. **Clean** the data – handle missing values, non-numeric entries, negative values
4. **Define** `YearsExperience` as `X` (feature) and `Salary` as `y` (target)
5. **Split** the dataset: 80% training, 20% testing (using `train_test_split`)
6. **Train** a `LinearRegression` model from scikit-learn
7. **Predict** salaries on the test set
8. **Evaluate** the model using MAE, MSE, RMSE, and R² Score
9. **Visualize** the regression line and Actual vs Predicted chart
10. **Predict** salary for any new input from the web interface

---

## 7. Linear Regression Explanation

Linear Regression finds the best-fit straight line through the data:

```
Salary = Intercept + Coefficient × YearsExperience
```

- **Coefficient (slope)**: How much the salary increases per additional year of experience.
- **Intercept**: The predicted salary at zero years of experience.
- **R² Score**: Proportion of salary variance explained by the model (0 to 1, higher is better).

The model minimizes the sum of squared differences between actual and predicted salaries.

---

## 8. Project Structure

```
salary-prediction/
│
├── app.py                        # Flask application (routes & API)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── dataset/
│   └── Salary_Data.csv           # Sample dataset (30 records)
│
├── model/
│   ├── __init__.py               # Package init
│   └── model.py                  # SalaryPredictor class (ML pipeline)
│
├── templates/
│   └── index.html                # Main web page (Jinja2 template)
│
├── static/
│   ├── style.css                 # Responsive stylesheet
│   └── script.js                 # Frontend JavaScript
│
└── notebooks/
    └── salary_prediction.ipynb   # Jupyter Notebook walkthrough
```

---

## 9. Installation Instructions

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Steps

```bash
# 1. Navigate to the project folder
cd salary-prediction

# 2. (Recommended) Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 10. How to Run the Project

```bash
# Make sure you are in the salary-prediction directory
# and the virtual environment is activated

python app.py
```

Then open your browser and go to: **http://localhost:5000**

### Using the Application

1. **Upload Dataset** – Click "Upload & Preview Dataset" and select a CSV file, or use the sample dataset provided.
2. **Train Model** – Click "Train Model" to run the Linear Regression pipeline.
3. **View Metrics** – See R², MAE, MSE, RMSE, coefficient, and intercept.
4. **Explore Charts** – Interact with the scatter plot and Actual vs Predicted chart.
5. **Predict Salary** – Enter years of experience and click "Predict Salary".

---

## 11. Example Prediction

After training the model on the sample dataset:

| Input | Output |
|---|---|
| 5.5 years | ₹77,000 (approx.) |
| 3.0 years | ₹58,000 (approx.) |
| 10.0 years | ₹1,17,000 (approx.) |

> ⚠️ These are **model estimates**, not guaranteed salaries.

---

## 12. Model Evaluation Metrics

| Metric | Description | Ideal Value |
|---|---|---|
| **R² Score** | % of variance explained | Close to 1.0 |
| **MAE** (Mean Absolute Error) | Average absolute prediction error | Lower is better |
| **MSE** (Mean Squared Error) | Average squared prediction error | Lower is better |
| **RMSE** (Root Mean Squared Error) | Same unit as Salary | Lower is better |

The sample dataset typically produces an **R² ≈ 0.96**, meaning the model explains ~96% of salary variance.

---

## 13. Screenshots

> *(Add screenshots of your running application here)*

- Screenshot 1: Dashboard with uploaded dataset
- Screenshot 2: Model training results and metrics
- Screenshot 3: Interactive scatter plot with regression line
- Screenshot 4: Salary prediction result

---

## 14. Future Improvements

- Add more features (education level, city, industry) → Multiple Linear Regression
- Compare with other models (Decision Tree, Random Forest, SVR)
- Add cross-validation for more robust evaluation
- Export the trained model using `joblib` for reuse
- Deploy to a cloud platform (IBM Cloud, Heroku, Render)
- Add a history of past predictions

---

## 15. Conclusion

This project demonstrates a complete, beginner-friendly machine learning pipeline using **Linear Regression** to predict salaries from years of experience. It covers every stage from raw data upload to interactive visualization and salary prediction — making it an ideal demonstration of applied ML concepts for an internship evaluation.

---

*IBM Applied AI & Machine Learning Internship Project | Built with Python, Flask, scikit-learn, and Plotly*
