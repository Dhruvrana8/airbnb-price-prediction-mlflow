# 🏠 Airbnb Price Prediction with MLflow

A machine learning pipeline designed to predict Airbnb listing prices based on features like location, room type, and reviews. This project utilizes **Scikit-Learn** for modeling and **MLflow** for end-to-end machine learning lifecycle management, including experiment tracking, model logging, and performance comparison.

## 📊 Project Overview

The goal of this project is to build a regression model that accurately estimates the price of an Airbnb rental. The workflow includes:

1.  **Data Ingestion:** Loading data from AWS S3 (or local source).
2.  **Exploratory Data Analysis (EDA):** Visualizing price distributions and correlations.
3.  **Preprocessing:**
    - Handling missing values.
    - Capping outliers (99th percentile).
    - Log-transforming skewed features (`number_of_reviews`, `minimum_nights`).
    - One-Hot Encoding categorical variables (`neighbourhood`, `room_type`).
4.  **Modeling:** Training Linear Regression, Random Forest, and Gradient Boosting models.
5.  **MLflow Tracking:** Logging hyperparameters, evaluation metrics (RMSE, MAE, R2), residual plots, and model artifacts.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn
- **Experiment Tracking:** MLflow
- **Cloud Storage:** AWS Boto3

---

## 📸 MLflow Dashboard & Results

This project uses MLflow to track every run. Below are insights from the experiment logs.

### 1. Experiment Dashboard

Overview of the different runs (Linear Regression, Random Forest, Gradient Boosting) with their respective metrics.
![MLflow Dashboard](/assets/images/Ml%20Flow%20Main%20Screen.png)

### 2. Model Artifacts & Comparisons

Detailed view of the logged parameters and custom artifacts (such as Actual vs. Predicted plots and Feature Importance) saved for every run.
![MLflow Models](/assets/images/AirBnb%20Price%20Prediction%20Model.png)

### 3. Best Performing Model

Programmatic selection of the best model based on the lowest RMSE.
![Best Model](/assets/images/Random%20Forest%20Model.png)

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Dhruvrana8/airbnb-price-prediction-mlflow
cd airbnb-price-prediction
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

_Ensure `mlflow`, `scikit-learn`, `pandas`, `boto3`, and `python-dotenv` are in your requirements._

### 4. Environment Variables (.env)

Create a `.env` file in the root directory if you are loading data from S3:

```
aws_access_key_id=YOUR_ACCESS_KEY
aws_secret_access_key=YOUR_SECRET_KEY
region=us-east-1
bucket_name=your-bucket-name
object_key=your-data-file.csv
```

---

## 🏃‍♂️ Usage

1.  **Run the Notebook:**
    Open `index.ipynb` in Jupyter or VS Code and execute the cells to preprocess data and train models.

2.  **Launch MLflow UI:**
    To view the dashboard shown in the screenshots above, run the following command in your terminal:

    ```bash
    mlflow ui
    ```

3.  **Access Dashboard:**
    Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 📈 Model Performance

Based on the validation set, the models performed as follows:

| Model             | RMSE      | MAE       | R2 Score  |
| ----------------- | --------- | --------- | --------- |
| **Random Forest** | **78.82** | **46.63** | **0.475** |
| Gradient Boosting | 80.04     | 47.79     | 0.459     |
| Linear Regression | 84.44     | 51.33     | 0.397     |

**Conclusion:** The Random Forest model outperformed the others, capturing the non-linear relationships in the housing data more effectively.

---

## 📂 Directory Structure

```
├── assets
│   └── images
│       ├── AirBnb Price Prediction Model.png
│       ├── Ml Flow Main Screen.png
│       └── Random Forest Model.png
├── mlruns/                  # MLflow local tracking logs
├── notebook
│   └── index.ipynb          # Main Python Code
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .env                     # AWS Credentials (Not committed)
```
