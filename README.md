# Data Science Vision - Employee Attrition Analysis

A comprehensive data science project analyzing employee attrition patterns using exploratory data analysis, classification, regression, and clustering techniques.

## 📋 Project Overview

This project analyzes employee data to predict attrition, estimate monthly income, and segment employees into meaningful clusters. The analysis provides actionable insights for HR departments to reduce employee turnover and optimize workforce management.

### Dataset

**Source**: Employee Dataset (employee.csv)  
**Size**: 2,940 employees  
**Features**: 35 columns including demographics, work metrics, and satisfaction indicators

## 🎯 Key Objectives

1. **Exploratory Data Analysis (EDA)**: Understand patterns in employee demographics and work characteristics
2. **Classification**: Predict employee attrition using machine learning models
3. **Regression**: Estimate monthly income based on employee features
4. **Clustering**: Segment employees based on job level and income

## 📊 Key Findings

### EDA Insights

#### Univariate Analysis - Numerical Features

**Age Distribution**
- Mean age: 36.9 years (range: 18-60 years)
- Distribution is slightly left-skewed, indicating a productive workforce
- 75% of employees are above 30 years old

**Monthly Income**
- Highest outlier percentage: 7.76% (228 employees)
- Wide salary range reflects diverse job roles and seniority levels
- High earners often in strategic or technical positions

**Environment & Performance**
- 75% of employees report "High" job satisfaction
- Performance ratings: Only "Excellent" (3) and "Outstanding" (4) categories
- Positive correlation between environment satisfaction and performance

#### Univariate Analysis - Categorical Features

**Department Distribution**
- Research & Development: 65.4% (dominant department)
- Sales: 30.3%
- Human Resources: 4.3%

**Business Travel**
- Travel Rarely: 71.0%
- Travel Frequently: 18.8%
- Non-Travel: 10.2%
- Minimal travel requirements indicate primarily on-site work

**Work-Life Balance**
- 71.7% of employees do NOT work overtime
- Indicates good work-life balance policies

#### Multivariate Analysis - Attrition Patterns

**Key Attrition Drivers:**

1. **Job Role Impact**
   - Sales roles: 39.8% attrition (highest risk)
   - Technical roles: 23.9% attrition
   - Correlation with high-pressure targets and performance demands

2. **Distance from Home**
   - Longer commute distances correlate with higher attrition
   - Physical proximity to workplace affects retention

3. **Overtime Effect**
   - 30.5% of overtime workers experience attrition
   - Work-life imbalance is a significant factor

4. **Business Travel**
   - 25% of frequent travelers leave the company
   - Travel demands contribute to turnover

5. **Experience & Compensation**
   - **Formula**: Attrition Rate ∝ Distance / (Age + Income + Experience)
   - Higher compensation and experience reduce attrition likelihood

### Statistical Tests

#### Independent t-test (TotalWorkingYears vs Attrition)
- **Result**: Significant difference found (p < 0.05)
- **Conclusion**: Work experience significantly impacts attrition decisions
- Employees who leave have different experience profiles than those who stay

#### One-Way ANOVA (Age across Departments)
- **Result**: No significant difference (p ≥ 0.05)
- **Conclusion**: Age distribution is consistent across all departments
- No department-specific age bias exists

## 🤖 Machine Learning Models

### 1. Classification - Attrition Prediction

#### Models Evaluated
- Logistic Regression
- Decision Tree Classifier
- XGBoost Classifier

#### Best Model: **XGBoost Classifier**

**Hyperparameters:**
```python
{
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 200
}
```

**Performance Metrics:**

| Metric | Train | Test |
|--------|-------|------|
| **Accuracy** | 100.0% | **96.9%** |
| **Precision (Attrition)** | 1.00 | **1.00** |
| **Recall (Attrition)** | 1.00 | **0.81** |
| **F1-Score (Attrition)** | 1.00 | **0.90** |

**Classification Report (Test Set):**
```
              precision    recall  f1-score   support
           0       0.96      1.00      0.98       493
           1       1.00      0.81      0.90        95
    accuracy                           0.97       588
```

**Why XGBoost is Best:**
- Highest test accuracy (96.9%)
- Perfect precision for attrition detection (no false positives)
- Balanced F1-score (0.90) for minority class
- Better generalization than Decision Tree
- Minimizes costly false positives for HR interventions

#### Model Comparison

| Model | Test Accuracy | Attrition F1-Score | Notes |
|-------|--------------|-------------------|-------|
| Logistic Regression | 89.5% | 0.58 | Baseline, underfits |
| Decision Tree | 95.9% | 0.87 | Overfits (100% train) |
| **XGBoost** | **96.9%** | **0.90** | **Best balance** |

### 2. Regression - Monthly Income Prediction

#### Models Evaluated
- Polynomial Regression (degree 3)
- Random Forest Regressor
- XGBoost Regressor

#### Best Model: **XGBoost Regressor**

**Hyperparameters:**
```python
{
    'colsample_bytree': 1.0,
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 200,
    'subsample': 0.8
}
```

**Performance Metrics:**

| Metric | Train | Test |
|--------|-------|------|
| **R² Score** | 0.9965 | **0.9855** |
| **RMSE** | 281.22 | **545.19** |
| **MAE** | 201.22 | **377.28** |
| **MAPE** | 0.0471 | **0.0938** |

**Why XGBoost is Best:**
- Highest R² (98.55%) - explains variance best
- Lowest RMSE (545) - smallest prediction error
- Lower MAE than Random Forest
- Controlled overfitting (reasonable train-test gap)
- Best balance of accuracy and generalization

#### Model Comparison

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
| Polynomial (degree 3) | 0.9826 | 597.99 | 198.63 |
| Random Forest | 0.9834 | 582.89 | 413.88 |
| **XGBoost** | **0.9855** | **545.19** | **377.28** |

### 3. Clustering - Employee Segmentation

#### Methodology
- **Algorithm**: K-Means Clustering
- **Features**: JobLevel, MonthlyIncome (standardized)
- **Optimal K**: 4 clusters

#### Cluster Selection Process

**Elbow Method**: Identified elbow at k=4  
**Silhouette Score Analysis**:
- k=2: 0.6680
- k=3: 0.6353
- **k=4: 0.7275** ← Optimal
- k=5: 0.7404

Selected k=4 for better interpretability while maintaining high cohesion.

#### Cluster Profiles

| Cluster | Avg Job Level | Avg Monthly Income | Interpretation |
|---------|---------------|-------------------|----------------|
| **0** | 2.00 | $5,378 | Mid-level employees |
| **1** | 4.41 | $17,113 | **Senior executives** |
| **2** | 1.00 | $2,787 | Entry-level staff |
| **3** | 2.96 | $9,893 | Upper mid-level |

**Silhouette Score**: 0.727 (good cluster separation)

## 🛠️ Technologies & Libraries

### Core Libraries
```python
pandas==2.2.2
numpy==2.0.2
scikit-learn==1.6.1
xgboost==3.1.2
```

### Visualization
```python
matplotlib==3.10.0
seaborn==0.13.2
plotly==5.24.1
```

### Statistical Analysis
```python
scipy==1.16.3
```

## 📁 Project Structure

```
Data_Science_Vision/
│
├── coding test.ipynb          # Main analysis notebook
├── employee.csv               # Dataset
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## 🚀 Installation & Usage

### Prerequisites
- Python 3.12+
- Jupyter Notebook or Google Colab

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Jasman123/Data_Science_Vision.git
cd Data_Science_Vision
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

5. **Open** `coding test.ipynb` and run all cells

## 📈 Results Summary

### Classification (Attrition Prediction)
✅ **96.9% accuracy** with XGBoost  
✅ **Perfect precision** (1.00) for detecting attrition  
✅ Successfully identifies 81% of employees who will leave

### Regression (Income Prediction)
✅ **98.55% variance explained** (R²)  
✅ **$545 average error** (RMSE)  
✅ **9.4% MAPE** - highly accurate predictions

### Clustering (Employee Segmentation)
✅ **4 distinct employee segments** identified  
✅ **0.727 silhouette score** - well-separated clusters  
✅ Clear segmentation from entry-level to executives

## 💡 Business Insights & Recommendations

### 1. Reduce Sales Attrition (39.8% rate)
- Implement performance support programs
- Review target-setting policies
- Enhance compensation packages
- Provide clearer career progression paths

### 2. Address Overtime Impact (30.5% attrition)
- Enforce work-life balance policies
- Monitor overtime hours systematically
- Consider workload redistribution
- Implement flexible working arrangements

### 3. Distance Management
- Offer remote/hybrid work options
- Provide transportation benefits
- Consider satellite offices
- Relocate assistance programs

### 4. Retention Strategy Focus
- Prioritize employees with:
  - Lower job levels (higher risk)
  - Shorter tenure (vulnerable period)
  - Technical roles (23.9% attrition)
  - Frequent travel requirements

### 5. Compensation Optimization
- Use regression model to ensure competitive salaries
- Address outliers in pay structure
- Regular market benchmarking
- Performance-based incentives

## 🔬 Methodology

### Data Preprocessing
1. Removed unnecessary columns (Unnamed: 0, EmployeeNumber, Over18)
2. Handled categorical variables with One-Hot Encoding
3. Standardized numerical features using StandardScaler
4. Train-test split: 80-20 ratio with stratification

### Feature Engineering
- Created interaction features for models
- Polynomial features (degree 3) for regression
- Scaled features for distance-based algorithms

### Model Selection
- Used GridSearchCV with 5-fold cross-validation
- Stratified K-Fold for classification
- Regular K-Fold for regression
- Optimized for accuracy (classification) and RMSE (regression)

## 📊 Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score

### Regression
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

### Clustering
- Elbow Method
- Silhouette Score
- Within-Cluster Sum of Squares (WCSS)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Jasman**
- GitHub: [@Jasman123](https://github.com/Jasman123)

## 🙏 Acknowledgments

- Dataset provided for educational purposes
- Inspired by real-world HR analytics challenges
- Built with industry-standard data science practices

---

⭐ **If you found this project helpful, please consider giving it a star!**

## 📞 Contact

For questions or feedback, please open an issue in the GitHub repository.

**Last Updated**: February 2026
