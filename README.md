# Credit-Card-Fraud-Detection-with-Random-Forest-Classifier
This project implementations a fraud detection system for credit card transactions using a Random Forest Classifier.

## Overview
The goal of this project is indentify fraudulent transactions while minimizing false positives (non-fraudulent transactions flagged as fraud) and false negatives (missed fraudulent transations). The project includes bahavorial analysis, fearture engineering, model training, cost-sensitive threshold optimization, and model interpretatbility using SHAP values.

The whole dataset used is https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset/ containning 1,296,675 transactions with 24 columns from 2019-01-01 to 2020-06-21, including transaction details, cardholder information, and fraud lables. The dataset is highly imbalanced with approximately 0.58% of transactions labled as fraudulent (is_fruad = 1). I use only 924,850 stransaction 2019-01-01 to 2019-12-31 to analysis month and season trends.

## Objective
- Detect fraudulent transactions with high recall (>80%) to minimize missed fraud cases.

- Optimize the decision threshold to balance precision and recall, minimizing costs (assume false negative cost = average amount fruadulent transaction + investigation management, false postive cost = investigation team and management).

- Analyze feature importance and SHAP values to understand key drivers of fraud predictions.

- Provide insights for stakeholders using visualizations (when, where, who and how credit-card transaction behavior)

## Dataset
- trans_date_trans_time is the data and time of the transactions
- cc_num is the credit card number 
- merchant is the merchant name or store name where the transaction occurred
- category is the category of the transaction (e.g., groceries, electronics)
- amt is the transaction amount
- first and last are the first and last names of the cardholder
- gender is the gender of the cardholder
- street is the street address of the cardholder
- city is the city of the cardholder
- state is the state of the cardholder
- zip is the zip code of the cardholder
- lat and long are the latitude and longitude of the merchant location (geographic coordinates of the transaction)
- city_pop is the population of the city where the transaction occurred
- job is the job title of the cardholder
- dob is the date of birth of the cardholder
- trans_num is the transaction number
- unix_time is the Unix timestamp of the transaction
- merch_lat and merch_long are the latitude and longitude of the merchant location
- is_fraud indicates whether the transaction is fraudulent (1) or not (0)
- merch_zipcode is the zip code of the merchant location
- 
### Key statistics data for credit card strasacion in 2019
- Total stransaction: 924,850
- Fraudlendent transaction: 5,220 (0.56%)
- Non-fraudulent transaction: 919,630 (99.46%)

## Methodology

### Feature Engineering

The following features were engineered to capture behavioral patterns:
- hour: Hour of the transaction, extracted from trans_date_trans_time.
- distance: Haversine distance between cardholder (lat, long) and merchant (merch_lat, merch_long).
- age: Cardholder's age, derived from dob.
- category_fraud_risk: Average fraud rate per transaction category.
- trans_freq_hour: Number of transactions per card in a 1-hour window.
- amt_category_fraud_risk: Transaction amount multiplied by category_fraud_risk.
- time_difference: Time difference between consecutive transactions for the same card.
- category_merchant_risk: Product of category_fraud_risk and merchant_fraud_risk.
- spending_category: Categorical bins of transaction amount (Very Low, Low, Medium, High Medium, High).
- population_category: Categorical bins of city population.

### Data Preprocessing
- Scaling: Numerical features standardized using StandardScaler.
- Train-Test Split: 80% training, 20% testing with stratification to maintain fraud ratio.
- Imbalance Handling: Used class_weight='balanced' in Random Forest to address the 0.56% fraud rate.

### Model Training
- Model: Random Forest Classifier (max_depth=20, min_samples_leaf=10, n_estimators=100, class_weight='balanced').
- Cross-Validation: 5-fold stratified cross-validation with F1-score metric.
- Threshold Optimization: Cost-sensitive threshold tuning with false negative cost = $530 + $10 and false positive cost = $10.

### Feature Importance: Analyzed using Random Forest's built-in feature importances.

SHAP Values: Computed using the SHAP library to interpret feature contributions to fraud predictions, visualized with a summary plot.

## Result

- Precision-Recall vs Threshold

<img width="846" height="545" alt="Image" src="https://github.com/user-attachments/assets/7890786c-8cc9-48eb-ba5e-60fab749569f" />

```
| Classification Report for Threshold 0.9620:           |
|-------------------------------------------------------|
|               precision    recall  f1-score   support |
|                                                       |
|            0       1.00      1.00      1.00    183926 |
|            1       0.88      0.57      0.69      1044 |
|                                                       |
|     accuracy                           1.00    184970 |
|    macro avg       0.94      0.78      0.84    184970 |
| weighted avg       1.00      1.00      1.00    184970 |
|                                                       |
| Confusion Matrix for Threshold 0.9620:                |
| [[183845     81]                                      |
|  [   454    590]]                                     |
|-------------------------------------------------------|
| Classification Report for Threshold 0.9475:           |
|               precision    recall  f1-score   support |
|                                                       |
|            0       1.00      1.00      1.00    183926 |
|            1       0.82      0.60      0.70      1044 |
|                                                       |
|     accuracy                           1.00    184970 |
|    macro avg       0.91      0.80      0.85    184970 |
| weighted avg       1.00      1.00      1.00    184970 |
|                                                       |
| Confusion Matrix for Threshold 0.9475:                |
| [[183790    136]                                      |
|  [   413    631]]                                     |
|-------------------------------------------------------|
| Classification Report for Threshold 0.9154:           |
|               precision    recall  f1-score   support |
|                                                       |
|            0       1.00      1.00      1.00    183926 |
|            1       0.68      0.66      0.67      1044 |
|                                                       |
|     accuracy                           1.00    184970 |
|    macro avg       0.84      0.83      0.83    184970 |
| weighted avg       1.00      1.00      1.00    184970 |
|                                                       |
| Confusion Matrix for Threshold 0.9154:                |
| [[183598    328]                                      |
|  [   350    694]]                                     |
|-------------------------------------------------------|

```
- Optimal Threshold for Cost Minimization: 0.4925


<img width="833" height="545" alt="Image" src="https://github.com/user-attachments/assets/c472f0a3-f33e-4504-aa01-67b623c26bf2" />

```
| Random Forest Training Classification Report with Cost Threshold: |
|-------------------------------------------------------------------|
|               precision    recall  f1-score   support             |
|                                                                   |
|            0       1.00      0.98      0.99    735704             |
|            1       0.19      0.93      0.32      4176             |
|                                                                   |
|     accuracy                           0.98    739880             |
|    macro avg       0.60      0.96      0.65    739880             |
| weighted avg       1.00      0.98      0.98    739880             |
|                                                                   |
|                                                                   |
| Random Forest Training Confusion Matrix with Cost Threshold:      |
| [[719180  16524]                                                  |
|  [   282   3894]]                                                 |
|-------------------------------------------------------------------|
| Random Forest Test Classification Report:                         |
|               precision    recall  f1-score   support             |
|                                                                   |
|            0       1.00      0.98      0.99    183926             |
|            1       0.18      0.93      0.31      1044             |
|                                                                   |
|     accuracy                           0.98    184970             |
|    macro avg       0.59      0.95      0.65    184970             |
| weighted avg       0.99      0.98      0.98    184970             |
|                                                                   |
|                                                                   |
| Random Forest Test Confusion Matrix:                              |
| [[179636   4290]                                                  |
|  [    74    970]]                                                 |
|-------------------------------------------------------------------|
```

- Total cost without Model: $2,403,264.57
  
- Total cost on Test Data with Model: $82,877.34
  
- Cost savings by using the Model: $2,320,387.24

- Percentage cost savings by using the Model: 96.55%
## Feature importance


<img width="1816" height="1238" alt="Image" src="https://github.com/user-attachments/assets/97e7fef9-3c54-4128-b6aa-acb5ecc1e278" />


<img width="787" height="940" alt="Image" src="https://github.com/user-attachments/assets/2b3ed15f-9232-4db7-b803-83527fc4c444" />

## Key findings
1. Hourly trends indicate peak transaction times from 12 PM to 11 PM, with higher average transaction amounts during late night hours (10 PM to 2 AM). Fraudulent activities are more likely to occur during late-night hours (10 PM to 3 AM).
2. Daily trends show the highest transaction counts on Mondays and Sundays, with average amounts peaking on Thursdays. Fraudulent activities are more prevalent on Fridays, Tuesdays, and Wednesdays.
3. Monthly trends reveal a significant increase in transactions during December, likely due to holiday shopping. Fraudulent activity peaks in February and is lowest in July.
4. Merchant analysis identifies gas transportation and grocery stores as top merchants by transaction count, while shopping net has the highest fraudulent transaction rate (1.73%).
5. Geographic analysis shows that cities like Hubbell and Byesville have 100% fraudulent transaction rates, while high transaction count cities tend to have lower fraud rates.
6. The states with high transaction counts, such as CA and TX, exhibit lower average transaction amounts and fraudulent rates. In contrast, states with low transaction counts, such as DE and RI, exhibit high average transaction amounts and fraudulent rates.
7. Population analysis indicates that rural areas have the highest transaction counts, while metropolitan areas have higher average transaction amounts. Fraudulent rates are highest in mid-sized cities (50,000-100,000 population).
8. Demographic analysis indicates that the 25-34 age group has the highest transaction count, while fraudulent activities are more common among younger (under 18) and older (55 and older) age groups.
9. Female cardholders exhibit higher transaction counts, while male cardholders exhibit a higher fraudulent transaction rate (0.6%) compared to female cardholders (0.5%).
10. Job title analysis reveals that film/video editors have the highest transaction counts, while information officers and ship brokers exhibit 100% fraudulent transaction rates.
11. Spending behavior analysis shows that most transactions fall within the 50-100 USD range, while high-value transactions (over 500 USD) have the highest fraudulent rates (22.8%).
1. The Random Forest Classifier demonstrates strong recall (0.93) but low precision (0.19), indicating effective fraud detection but a high false positive rate. Stratified k-fold cross-validation confirms consistent performance across folds.
2. Optimal decision thresholds are identified to balance precision and recall, with a threshold of 0.9620 minimizing overall costs associated with false negatives and false positives.
3. Cost analysis suggests significant savings (96.55%) by implementing the model with the optimal threshold, reducing financial losses from fraud and investigation costs.
4. SHAP and feature importance analyses highlight key features influencing fraud predictions, aiding interpretability and model transparency.

## Future Work
1. Explore advanced ensemble methods or deep learning techniques to improve fraud detection performance.
2. Investigate the impact of external factors (e.g., economic indicators, social media trends) on fraudulent activities.
3. Develop real-time fraud detection systems for proactive monitoring and response.
4. Conduct user studies to understand the implications of fraud detection systems on user experience and privacy.

### Contact

Thanh Xuyen, Nguyen

LinkedIn: [xuyen-thanh-nguyen-0518](https://www.linkedin.com/in/xuyen-thanh-nguyen-0518/)

Email: thanhxuyen.nguyen@outlook.com
