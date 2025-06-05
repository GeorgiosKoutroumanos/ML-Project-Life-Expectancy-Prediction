
# Life Expectancy Prediction using Machine Learning


## Introduction
This Jupyter Notebook analyzes a dataset on life expectancy across countries and applies various machine learning models to predict life expectancy based on multiple socio-economic and health-related factors.

The goal is to build a robust regression model that accurately predicts life expectancy based on input features, and to explore which factors contribute the most to variations in life expectancy globally.

## Data

The dataset includes information on the following columns:

- Country
- Year
- Status: Developed or Developing
- Life expectancy: In years
- Adult Mortality: Probability of dying between 15 and 60 years per 1000 population
- Infant deaths
- Alcohol: Recorded per capita (15+) alcohol consumption (liters of pure alcohol)
- Percentage expenditure: Health expenditure as a % of GDP per capita
- Hepatitis B: Immunization coverage among 1-year-olds (%)
- Measles: Reported cases per 1000 population
- BMI: Average Body Mass Index of the population

Data source: Life Expectancy Data.csv (https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who/data?select=Life+Expectancy+Data.csv)

**Libraries Used**:
- pandas, numpy – Data manipulation
- matplotlib, seaborn – Visualization
- scikit-learn – Model training and evaluation

**Models Implemented**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Bagging Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor

Each model's performance is assessed using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Root Mean Squared Error (RMSE)

**Features Engineering & Scaling**
The notebook includes:
- Feature selection
- Scaling using StandardScaler and RobustScaler
- Cross-validation
- Grid Search for hyperparameter tuning

**How to Run**

Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

Run each cell in the notebook sequentially to:

1) Load and preprocess the data
2) Train multiple models
3) Evaluate and compare their performance
