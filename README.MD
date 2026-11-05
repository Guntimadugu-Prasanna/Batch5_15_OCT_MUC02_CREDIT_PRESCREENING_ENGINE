# MUC02 – Credit Pre-Screening Engine

### AI-Powered Credit Risk Analysis & Loan Application Pre-Screening

**Project:** Calibo AI Academy – Mini Use Case 02 (MUC02)  
**Domain:** Banking, Financial Services & Machine Learning  
**Algorithm:** Logistic Regression  
**Language:** Python

---

## 1. Project Overview

The Credit Pre-Screening Engine is a machine learning-based solution designed to help banks analyse personal loan applications and identify applications that are highly likely to be declined based on historical credit decisions.

Banks traditionally depend on manual credit assessment, where analysts review applicant information such as income, credit score, existing EMIs, employment type and repayment history.

This project uses historical loan application data to understand credit approval patterns, engineer financial features, train a Logistic Regression classification model and evaluate different probability thresholds.

The objective is to support credit analysts by identifying applications that may be routed away from the standard manual-review process while satisfying a defined wrong-decline constraint.

## 2. Business Problem

A bank processes approximately 3,000 personal loan applications every month.

- Each application requires approximately 25 minutes of manual review.
- A significant proportion of applications are historically declined.
- Manual assessment consumes considerable analyst time.
- The bank wants to improve operational efficiency without unnecessarily rejecting applicants who would historically have been approved.

**Business objective:** Develop a credit pre-screening model that identifies high-probability historical declines while keeping the historically approved proportion within the auto-decline group at or below 5%.

## 3. Project Objectives

- Analyse historical loan application data.
- Identify factors influencing historical credit approval decisions.
- Clean missing, inconsistent and unusual data.
- Engineer financial features such as FOIR and Loan-to-Income Ratio.
- Build a Logistic Regression classification model.
- Evaluate model performance using classification metrics.
- Identify an appropriate probability threshold.
- Estimate potential manual-review capacity improvements.
- Examine gender-based differences and possible proxy variables.
- Provide data-driven recommendations for credit policy analysts.

## 4. Dataset Information

The project uses a historical credit application dataset containing:

| Attribute | Description |
|---|---|
| Total records | 12,000 |
| Total columns | 19 |
| Target variable | Decision |
| Target classes | Approved / Declined |
| Problem type | Binary Classification |

### Important Features

- Monthly Income
- Employment Type
- Existing EMI
- Credit Score
- Credit History Months
- Number of Active Loans
- Delinquencies in the Last 24 Months
- Loan Purpose
- Loan Amount
- Loan Tenure
- Proposed EMI
- Gender
- Historical Credit Decision

The target variable is encoded as:

- `1` – Declined
- `0` – Approved

## 5. Data Cleaning & Preprocessing

The dataset required several data-quality decisions.

| Data issue | Treatment |
|---|---|
| 441 missing income values | Imputed using employment-type median income |
| 1,039 missing credit scores | Retained and identified as New-to-Credit applicants |
| 12 unusually high monthly incomes | Corrected by dividing by 12 based on the assignment's unit assumption |

### New-to-Credit Identification

Applicants with missing credit scores and zero credit-history months were identified as New-to-Credit (NTC) applicants.

Instead of removing these records, an additional feature was created:

`is_ntc`

This preserves useful information about applicants without an established credit history.

## 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand historical approval patterns.

Key analyses:

- Approval rate by credit-score band.
- Approval rate by employment type.
- Approval rate by FOIR category.
- Income and affordability patterns.
- Historical approval differences across applicant groups.

### Key Findings

- Applicants with credit scores below 650 had very low historical approval rates.
- Approval rates increased substantially in higher credit-score bands.
- New-to-Credit applicants had a historical approval rate of approximately 53.9%.
- Salaried applicants had higher historical approval rates than Gig / Contract workers.
- Higher FOIR was associated with lower historical approval rates.

## 7. Feature Engineering

New features were created to improve the representation of applicant affordability.

### 7.1 FOIR – Fixed Obligation to Income Ratio

FOIR measures the proportion of monthly income committed to existing and proposed EMIs.

Formula:

FOIR = (Existing EMI + Proposed EMI) / Monthly Income

### 7.2 Loan-to-Income Ratio

Measures the requested loan amount relative to annual income.

Formula:

Loan-to-Income = Loan Amount / (Monthly Income × 12)

### 7.3 New-to-Credit Flag

Identifies applicants without an established credit history.

- `1` – New-to-Credit
- `0` – Existing credit history

### FOIR Analysis Findings

| FOIR Band | Historical Approval Rate |
|---|---:|
| Below 30% | 65.4% |
| 30–40% | 59.0% |
| 40–50% | 47.4% |
| 50–55% | 31.7% |
| Above 55% | 15.5% |

Higher FOIR values were associated with lower historical approval rates.

## 8. Machine Learning Model

### Logistic Regression

Logistic Regression was selected because the problem involves binary classification and the model provides interpretable coefficients.

**Model configuration:**

- Algorithm: Logistic Regression
- Training/Test Split: 80/20
- Training Records: 9,600
- Testing Records: 2,400
- Target: Historical Credit Decision

### Preprocessing Techniques

- SimpleImputer – Handles missing values.
- StandardScaler – Standardises numerical features.
- OneHotEncoder – Converts categorical variables into numerical representations.
- ColumnTransformer – Applies different preprocessing to numerical and categorical columns.
- Pipeline – Combines preprocessing and model training into one workflow.

## 9. Model Evaluation

The model was evaluated using the test dataset.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 79.08% |
| ROC-AUC | 0.8674 |
| Precision (Declined) | 79.1% |
| Recall (Declined) | 75.5% |

### Confusion Matrix

| Actual / Predicted | Approved | Declined |
|---|---:|---:|
| Approved | 1,046 | 225 |
| Declined | 277 | 852 |

The confusion matrix helps identify correct predictions, missed declines and historically approved applicants incorrectly predicted as declined.

## 10. Probability Threshold Selection

A probability threshold determines when an application enters the proposed auto-decline group.

Different thresholds were evaluated against the assignment's maximum 5% wrong-decline constraint.

| Threshold | Auto-Decline Rate | Wrong-Decline Rate |
|---|---:|---:|
| 0.50 | 44.9% | 20.9% |
| 0.60 | 37.6% | 16.8% |
| 0.70 | 30.8% | 12.0% |
| 0.80 | 23.4% | 7.1% |
| 0.85 | 19.7% | 5.9% |
| **0.90** | **15.2%** | **4.7%** |
| 0.95 | 10.0% | 2.1% |

### Selected Threshold: 0.90

The 0.90 threshold was selected because it was the lowest tested threshold satisfying the assignment's 5% wrong-decline constraint.

This demonstrates the importance of balancing prediction performance with business requirements.

## 11. Business Impact Analysis

Based on the assignment assumptions:

- Monthly applications: 3,000
- Manual review time: 25 minutes
- Selected threshold: 0.90
- Applications potentially routed away from manual review: 456 per month

Estimated analyst capacity potentially freed:

456 × 25 / 60 = **190 hours per month**

This represents potential operational capacity improvement, not guaranteed cost savings.

## 12. Fairness Analysis

Fairness analysis was conducted to understand differences in historical and model outcomes across gender.

| Metric | Female | Male |
|---|---:|---:|
| Historical Approval Rate | 50.4% | 54.1% |
| Median Monthly Income | ₹45,000 | ₹55,000 |
| Gig / Contract Share | 21.4% | 11.6% |
| Auto-Decline Rate at 0.90 | 15.60% | 15.03% |

Gender was excluded from model inputs, but income and employment type may act as proxy variables.

Similar auto-decline rates do not independently establish fairness. Continuous monitoring and additional subgroup analysis are required.

## 13. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Scikit-learn | Machine learning and model evaluation |
| Jupyter Notebook | Development and experimentation |
| Git & GitHub | Version control and project management |

## 14. Project Workflow

1. Business Problem Understanding – CBIM Problem Canvas.
2. Data Loading and Inspection.
3. Data Cleaning and Preprocessing.
4. Exploratory Data Analysis.
5. Feature Engineering.
6. Logistic Regression Model Development.
7. Model Evaluation.
8. Probability Threshold Comparison.
9. Fairness Assessment.
10. Business Recommendations and Credit Policy Briefing.

## 15. Key Business Recommendations

- Consider 0.90 as the candidate threshold for the assignment's proposed pre-screening policy.
- Retain human review for New-to-Credit applicants.
- Monitor credit-score, affordability and repayment-history patterns.
- Investigate potential proxy-related fairness issues.
- Validate future deployment against actual repayment and default outcomes.
- Maintain human oversight and obtain appropriate policy and compliance approvals.

## 16. Project Limitations

The model learns historical credit approval and rejection decisions rather than actual future repayment behaviour.

Therefore:

- Historical approval does not guarantee future repayment.
- Historical rejection does not establish future default.
- Model performance may differ across applicant groups.
- The threshold requires independent validation before operational use.
- Production deployment requires repayment/default data, fairness assessment, governance and human oversight.

## 17. Learning Outcomes

Through this project, I gained practical experience in:

- Business problem formulation.
- Data cleaning and data-quality assessment.
- Exploratory Data Analysis.
- Feature engineering.
- Supervised machine learning.
- Logistic Regression.
- Classification evaluation metrics.
- Confusion matrix interpretation.
- Probability threshold optimisation.
- Business impact estimation.
- Fairness analysis and responsible AI.
- Communicating analytical findings to business stakeholders.

---

## Conclusion

The MUC02 Credit Pre-Screening Engine demonstrates how data analytics and machine learning can support banking operations.

By analysing historical credit decisions, engineering affordability-related features and applying Logistic Regression, the project identifies patterns in past lending decisions and evaluates a conservative pre-screening threshold.

The project highlights an important principle of applied machine learning: a successful solution must combine technical performance, business value, fairness, explainability and responsible decision-making.

**Project developed as part of Calibo AI Academy – Path 2, Mini Use Case 02.**
