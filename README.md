
# Life Expectancy Prediction using Machine Learning


## Introduction
This Jupyter Notebook analyzes a dataset on life expectancy across countries and applies various machine learning models to predict life expectancy based on multiple socio-economic and health-related factors.

The goal is to build a robust regression model that accurately predicts life expectancy based on input features, and to explore which factors contribute the most to variations in life expectancy.

## Data

The dataset includes the following columns:

- Country
- Year
- Status: Developed or Developing status
- Life expectancy: Life Expectancy in age
- Adult Mortality: Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)
- infant deaths: Number of Infant Deaths per 1000 population
- Alcohol: Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)
- percentage expenditure: Expenditure on health as a percentage of Gross Domestic Product per capita(%)
- Hepatitis B: Hepatitis B (HepB) immunization coverage among 1-year-olds (%)
- Measles: Measles - number of reported cases per 1000 population
- BMI: Average Body Mass Index of entire population

Data source: Life Expectancy Data.csv (https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who/data?select=Life+Expectancy+Data.csv)

**Libraries Used**:

📦 Data Manipulation & Analysis
- pandas – DataFrames and data operations
- numpy – Numerical computations

📊 Data Visualization
- matplotlib.pyplot – Plotting and graphs
- seaborn – Statistical data visualization

🧹 Preprocessing
- re – Regular expressions for string/text cleaning
- StandardScaler, RobustScaler from sklearn.preprocessing – Feature scaling

📈 Model Selection & Evaluation
- train_test_split, cross_val_score, learning_curve, GridSearchCV from sklearn.model_selection
- mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error from sklearn.metrics

🤖 Machine Learning Models
- LinearRegression from sklearn.linear_model
- DecisionTreeRegressor from sklearn.tree
- RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor from sklearn.ensemble

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
