# Predicting Student Academic Performance Using Data Analytics

> A comprehensive machine learning project analyzing 5,000+ student records to predict academic outcomes using regression, classification, clustering, and anomaly detection techniques.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

This project applies data science and machine learning techniques to analyze the academic performance of university students and identify the behavioral and academic factors that most significantly impact their final results.

The dataset includes **5,000 student records** with **23 features** covering academic metrics, behavioral factors, and demographic variables. After thorough cleaning, **4,018 valid records** were used for analysis.

Four analytical approaches were implemented and compared:

| Analysis Type | Models Used | Best Result |
|--------------|-------------|-------------|
| **Regression** | Simple/Multiple Linear, Polynomial, Non-Linear | R² = 0.0026 (Multiple Non-Linear) |
| **Classification** | Decision Tree, Random Forest, Logistic Regression, Gradient Boosting | Accuracy = 26% (Logistic Regression) |
| **Clustering** | K-Means, Agglomerative Hierarchical | Optimal k=2 (Elbow Method) |
| **Anomaly Detection** | Z-Score, IQR/Boxplot, Isolation Forest, Local Outlier Factor | Meaningful outlier identification |

---

## Business Problem

Educational institutions face significant challenges in accurately predicting student outcomes due to the complex interplay of behavioral, psychological, and academic factors. This project addresses:

- **Early intervention** - Identify at-risk students before they fail
- **Pattern discovery** - Uncover hidden behavioral clusters among students
- **Resource allocation** - Help institutions target support where it's most needed
- **Data-driven policies** - Enable evidence-based academic decision-making

### Research Questions

1. To what extent do stress levels, sleep duration, and study hours affect a student's final score?
2. Can a student's grade (A-F) be accurately predicted using academic and behavioral features?
3. Are there observable clusters or behavior patterns among students based on performance indicators?
4. Are there anomalies in the dataset that may reflect unusual student behavior?

---

## Dataset Description

**Source:** [Students Grading Dataset - Kaggle](https://www.kaggle.com/datasets/omerckn/students-grading-dataset)

| Property | Value |
|----------|-------|
| Original Records | 5,000 |
| After Cleaning | 4,018 |
| Features | 23 |
| Numerical Variables | 12 |
| Categorical Variables | 7 |

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| `Student_ID` | Identifier | Unique student identifier |
| `Final_Score` | Numerical | Final exam performance |
| `Midterm_Score` | Numerical | Midterm exam score |
| `Attendance` | Numerical | Attendance percentage |
| `Projects_Score` | Numerical | Project performance score |
| `Assignments_Avg` | Numerical | Average assignment score |
| `Study_Hours_per_Week` | Numerical | Weekly study hours |
| `Sleep_Hours_per_Night` | Numerical | Nightly sleep duration |
| `Stress_Level` | Numerical | Self-reported stress (1-10) |
| `Participation_Score` | Numerical | Class participation score |
| `Gender` | Categorical | Student gender |
| `Department` | Categorical | Academic department |
| `Grade` | Categorical | Letter grade (A-F) |
| `Parent_Education_Level` | Categorical | Highest parental education |
| `Family_Income_Level` | Categorical | Family income bracket |
| `Internet_Access_at_Home` | Categorical | Home internet availability |

---

## Data Cleaning & Preprocessing

The raw dataset required significant preprocessing to ensure quality:

1. **Missing Values** - `Parent_Education_Level` filled with "No Education"; rows with missing critical columns (`Attendance`, `Assignments_Avg`) removed
2. **Text Cleaning** - Custom function to remove special characters and excess spaces
3. **Gender Correction** - Labels validated and corrected based on name-gender matching against predefined name lists
4. **Grade Correction** - Letter grades recalculated from `Total_Score`; over **3,200 incorrect entries** detected and corrected
5. **Duplicate Removal** - Duplicate records identified and removed

**Result:** Dataset refined from 5,000 to 4,018 valid records.

---

## Methodology

### 1. Regression Analysis
Explored the relationship between continuous features and Final Score:

- **Simple Linear Regression** - Stress Level vs. Final Score
- **Multiple Linear Regression** - Study Hours, Sleep Hours, Stress Level, Attendance vs. Final Score
- **Simple Polynomial Regression** (degree=2) - Study Hours vs. Final Score
- **Multiple Non-Linear Regression** - Age, Attendance, Study Hours vs. Final Score

### 2. Classification Analysis
Predicted Corrected Grade (A-F) using academic and personal features:

- **Logistic Regression** - Best accuracy (26%), strong bias toward F grade
- **Random Forest** - Moderate performance with ensemble learning
- **Decision Tree** - Least effective, suffered from underfitting
- **Gradient Boosting** - Most balanced F1 score across classes

### 3. Clustering Analysis
Unsupervised learning to discover student behavior patterns:

- **K-Means Clustering** - Tested with k=2, 7, 9 (Elbow Method optimal: k=2)
- **Agglomerative Hierarchical Clustering** - Dendrogram analysis
- **PCA** applied for dimensionality reduction and visualization

### 4. Anomaly Detection
Identified students with unusual behavior patterns:

- **Z-Score** (Statistical)
- **IQR & Boxplot** (Statistical)
- **Isolation Forest** (Unsupervised ML)
- **Local Outlier Factor** (Unsupervised ML)

### Training Configuration
- **Split:** 80% Training / 20% Testing
- **Validation:** 5-Fold Cross-Validation on all models
- **Evaluation:** R², Adjusted R², RMSE, Accuracy, Precision, Recall, F1-Score, Silhouette Score

---

## Key Findings

1. **Weak predictive power for regression** - All regression models produced R² values near zero, indicating that stress level, study hours, and sleep alone do not linearly predict final scores. The best model (Multiple Non-Linear) achieved R² = 0.0026.

2. **Classification struggles with class imbalance** - The "F" grade was overrepresented, causing models to bias toward predicting failure. Gradient Boosting showed the most balanced performance across all grade categories.

3. **Clustering reveals meaningful student segments** - K-Means with k=2 identified two distinct student groups based on behavioral and academic similarities, suggesting a fundamental split in student engagement patterns.

4. **Anomaly detection uncovered outlier behaviors** - Isolation Forest and LOF successfully identified students with unusual combinations (e.g., high stress with low performance), which could signal students needing support.

5. **Data quality matters enormously** - Over 3,200 grade entries were incorrectly labeled in the original dataset, demonstrating the critical importance of thorough preprocessing.

6. **Academic performance is multi-factorial** - Simple feature relationships failed to capture the complex interplay of factors affecting student outcomes, suggesting the need for more sophisticated feature engineering.

---

## Model Comparison

### Regression Models

| Model | R² | Adjusted R² | RMSE |
|-------|-----|-------------|------|
| Simple Linear | -0.0001 | -0.0002 | 17.28 |
| Multiple Linear | -0.007 | -0.012 | 17.336 |
| Simple Non-Linear | -0.0003 | -0.0028 | 17.183 |
| **Multiple Non-Linear** | **0.0026** | **-0.0008** | **17.087** |

### Classification Models

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | **0.26** | 0.18 | 0.20 | 0.09 |
| Random Forest | 0.22 | 0.20 | 0.20 | 0.19 |
| Decision Tree | 0.21 | 0.21 | 0.21 | **0.21** |
| **Gradient Boosting** | 0.23 | 0.20 | 0.20 | 0.18 |

---

## Recommendations

1. **Apply class balancing** - Use SMOTE or undersampling to address the F-grade overrepresentation
2. **Advanced feature engineering** - Create interaction features, ratios, and composite scores
3. **Explore advanced models** - Try XGBoost, LightGBM, or neural networks for classification
4. **Increase dataset size** - Larger datasets may reveal patterns not visible at 4K records
5. **Feature transformation** - Apply logarithmic or polynomial transformations to enhance relationships
6. **Hyperparameter tuning** - Use GridSearchCV or RandomizedSearchCV for optimal parameters
7. **Domain-specific features** - Incorporate temporal data (semester progression) and engagement metrics

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core programming language |
| Pandas & NumPy | Data manipulation and preprocessing |
| Matplotlib & Seaborn | Data visualization |
| scikit-learn | ML models, evaluation, and pipelines |
| SciPy | Hierarchical clustering & statistical analysis |
| Google Colab | Development environment |

---

## Project Structure

```
student-performance-prediction/
│
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
│
├── notebooks/
│   └── Student_Performance_Analysis.ipynb # Main analysis notebook
│
├── data/
│   └── README.md                          # Dataset download instructions
│
├── reports/
│   └── Student_Performance_Report.pdf     # Final project report
│
└── images/
    ├── regression_plots.png
    ├── confusion_matrices.png
    ├── clustering_results.png
    └── anomaly_detection.png
```

---

## How to Run

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O9JtR55L-NAfZNJxrSV4hVX1N0ycCRWa?usp=sharing)

1. Open the notebook in Google Colab
2. Upload the dataset when prompted
3. Run all cells sequentially

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/Rolw4-53872/student-performance-prediction.git
cd student-performance-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/Student_Performance_Analysis.ipynb
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset: [Students Grading Dataset](https://www.kaggle.com/datasets/omerckn/students-grading-dataset) on Kaggle
- Built as part of a Data Analysis course project
