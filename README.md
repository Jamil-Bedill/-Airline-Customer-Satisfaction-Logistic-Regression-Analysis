# Airline Customer Satisfaction – Logistic Regression Analysis 

## Overview
This project demonstrates **Binomial Logistic Regression** using a real-world airline passenger dataset. The analysis explores how *inflight entertainment quality* affects *customer satisfaction*.  
It forms part of the **Google Advanced Data Analytics** programme and showcases end-to-end skills in **data cleaning**, **encoding**, **modelling**, and **evaluation**.

## Data Information

**Dataset:** `Invistico_Airline.csv`  
**Source:** Provided through Coursera (Google Advanced Data Analytics – Regression Module)  
**Rows:** 129,487  
**Columns:** 22  

### Data Dictionary
| Column | Type | Description |
|:-------|:------|:------------|
| `satisfaction` | object | Passenger satisfaction (satisfied / dissatisfied) |
| `Customer Type` | object | Loyal vs disloyal customer |
| `Type of Travel` | object | Business or Personal travel |
| `Class` | object | Travel class (Eco, Business, Eco Plus) |
| `Inflight entertainment` | int | Rating of inflight entertainment service |
| Other columns | int / float | Ratings for comfort, food, gate location, service, and delay times |

## Objectives
- Import and explore a large dataset of airline passenger feedback.  
- Clean and prepare data for a logistic regression model.  
- Encode categorical variables and handle missing values.  
- Build and evaluate a **binomial logistic regression model** using scikit-learn.  
- Interpret and visualise results to extract actionable insights.


## Step 1: Imports

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
```
## Step 2: Data Preparation
**Load and Explore Data**
```
df_original = pd.read_csv("Invistico_Airline.csv")
df_original.head(10)
```
<img width="1012" height="466" alt="image" src="https://github.com/user-attachments/assets/26bfc733-8a33-4f29-8031-506a51d6f533" />

The dataset has many columns and thus cannot be fit on the screen. I took its screenshot from the visible window. 
Let us find out the shape of the data.

```
df_original.shape
```

(129880, 22)
The dataset contains 129,487 rows and 22 columns.

```
print(df_original.dtypes)
print('------\nUnque Values')
for col in ['satisfaction', 'Customer Type', 'Type of Travel']:
    print(f'The unique value of {col}:{df_original[col].unique()}')
```

<img width="636" height="490" alt="image" src="https://github.com/user-attachments/assets/8c09a354-b9f0-4c56-a1de-6c06322cad0c" />

It is better to understand the number and percentage of satisfied and unsatisfied customers before the model

```
print('The number of value counts of satisfaction\n----')
print(df_original['satisfaction'].value_counts())
print('Percentage of value counts of staisfaction\n----')
print(df_original['satisfaction'].value_counts(normalize = True)*100)
```

<img width="425" height="184" alt="image" src="https://github.com/user-attachments/assets/4fc18ab5-2314-4506-a660-63a6722598bd" />

We can see that 54.75% are satisfied and 45.26% are not satisfied.

 **Missing Values**
Let's check if the dataset has missing values.

```
df_original.isnull().sum()
```
<img width="461" height="403" alt="image" src="https://github.com/user-attachments/assets/5315a41b-9ec5-4174-83c2-b9546fde34cd" />

Though the columns used in the dataset do not have missing values, and only one column(Arrival Delay in Minutes) has  missing values, we still remove the rows with the missing values. 

```
df_original.dropna(axis = 0, inplace = True)
df_original = df_original.reset_index(drop = True)
```
**Encoding and Cleaning**

Convert data types and encode categorical variables. For the model, we need to change Inflight entertainment to float and satisfaction column numeric using OneHotEncoder from sklearn.preprocessing. 

```
df_original['Inflight entertainment'] = df_original['Inflight entertainment'].astype(float)
df_original['satisfaction'] = OneHotEncoder(drop='first', sparse=False).fit_transform(df_original[['satisfaction']])
```
**Splitting Data**
Now we split the data into train and test. We leave 70% for training and 30% for splitting. We create X data frame and y.

```
X = df_original[['Inflight entertainment']]
y = df_original['satisfaction']
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
```
##Step 3: Model Building
**Fitting Logistic Regression**

Now that the data is ready, let's fit the model

```
clf = LogisticRegression().fit(X_train, y_train)
```
Let us find out the coefficient and intercept

```
print('The coeffecient is {} and the interept is {}'.format(clf.coef_,clf.intercept_))
```
The coeffecient is [[1.00724567]] and the interept is [-3.23515113]

Coefficient = 1.0072:
For each one-unit increase in inflight-entertainment rating,
the log-odds of a customer being satisfied increase by 1.0072.
Converting to odds → 
```
e^{1.0072} \approx 2.74$
```
This means the odds of satisfaction are 2.7 times higher for every additional point in entertainment score.
Intercept = -3.2351:
When inflight-entertainment = 0, the log-odds of satisfaction are -3.2351,
corresponding to a base probability of roughly 3.7 %.

This means the odds of satisfaction are 2.7 times higher for every additional point in entertainment score.

Intercept = -3.2351:
When inflight-entertainment = 0, the log-odds of satisfaction are -3.2351,
corresponding to a base probability of roughly 3.7 %.

**Visualising the Model**

```
sns.regplot(x=df_original['Inflight entertainment'],
            y=df_original['satisfaction'],
            logistic=True, ci=None)
plt.title("Inflight Entertainment vs Satisfaction")
plt.show()
```

<img width="530" height="289" alt="image" src="https://github.com/user-attachments/assets/91b347bc-c840-4b10-92e4-02e1239abbd5" />

The graph shows that the higher the entertainment, the higher the satisfaction.

## Step 4: Evaluation and Results

**Predictions**

```
y_pred = clf.predict(X_test)
probs = clf.predict_proba(X_test)
```

<img width="254" height="152" alt="image" src="https://github.com/user-attachments/assets/545662e8-f974-4499-8b22-ce4d545d1168" />

**Model Performance Metrics**

```
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}")
print(f"Recall: {metrics.recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {metrics.f1_score(y_test, y_pred):.4f}")
```

<img width="283" height="80" alt="image" src="https://github.com/user-attachments/assets/0f33b2bf-059a-4aaa-986c-4b69c73b27e8" />

**Produce a confusion matrix**
Let us draw a confusion matrix and see the type of errors the model produced on the test data. 

```

cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot(values_format='d')
```

<img width="373" height="265" alt="image" src="https://github.com/user-attachments/assets/92cf4123-8012-4996-9969-19aaa49772c7" />

```
### Confusion Matrix Summary

| Metric | Description | Count |
|---------|--------------|-------|
| **True Positive (TP)** | Model correctly predicted class **1** when the actual value was **1**. | 17,423 |
| **True Negative (TN)** | Model correctly predicted class **0** when the actual value was **0**. | 13,714 |
| **False Positive (FP)** | Model incorrectly predicted class **1** when the actual value was **0**. | 3,925 |
| **False Negative (FN)** | Model incorrectly predicted class **0** when the actual value was **1**. | 3,785 |
```
**Interpretation:**  
The model performs well, with a high number of correct predictions (TP and TN) compared to incorrect ones (FP and FN). This indicates strong predictive accuracy for both classes, though there’s slight room for improvement in reducing misclassifications.

## Summary
Customers who rated in-flight entertainment highly were significantly more likely to report overall satisfaction. Enhancing this service area could therefore contribute to improved customer satisfaction levels.
The model achieved an accuracy of 80.2%, representing a substantial improvement over the baseline customer satisfaction rate of 54.7%.
These results highlight the potential value of further model development. Expanding the analysis by incorporating additional independent variables may yield even better predictive performance. Beyond prediction, such modelling can provide valuable insights into the key factors that drive customer satisfaction, enabling the airline to make more informed and strategic decisions.


