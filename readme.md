# Customer Segmentation and Classification Project

This project combines **K-Means Clustering** to segment mall customers and **Boosting Algorithms** to build models that classify new customers into these established segments.

---

## 1. Environment Setup and Data Loading

We import all necessary libraries and load the `Mall_Customers.csv` dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

df=pd.read_csv('Mall_Customers.csv')
```

## 2.K-means clustering

- The analysis focuses on 'Annual Income (k$)' and 'Spending Score (1-100)'(refer **main1.ipynb** for it)
-  Determining Optimal K (Elbow Method)The Inertia is calculated for $k=1$ to $10$.
 -  We look for the "elbow" point where the improvement significantly diminishes.
  - The final model with $k=5$ is fit, cluster labels are added to the DataFrame, and the quality is measured using the Silhouette Score.

  ## 3.Boosting algorithms

 Gradient Boosting Classifier (GB)
```Python

gb=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
gb.fit(x_train,y_train)
y_pred_gb=gb.predict(x_test)
accuracy_gb=accuracy_score(y_test,y_pred_gb)
print(f'Gradient Boosting Classifier Accuracy: {accuracy_gb:.2f}')
```
XGBoost Classifier
```Python

xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax', num_class=len(y_boost.unique()),
    n_estimators=100, learning_rate=0.1, max_depth=3,
    use_label_encoder=False, eval_metric='mlogloss', random_state=42
)
xgb_clf.fit(x_train, y_train)
y_pred_xgb = xgb_clf.predict(x_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'XGBoost Classifier Accuracy: {accuracy_xgb:.2f}')
```
CatBoost Classifier
``` Python

cat_clf = CatBoostClassifier(
    iterations=100, learning_rate=0.1, loss_function='MultiClass', 
    random_state=42, verbose=0 
)
cat_clf.fit(x_train, y_train)
y_pred_cat = cat_clf.predict(x_test).flatten()
accuracy_cat = accuracy_score(y_test, y_pred_cat)
print(f'CatBoost Classifier Accuracy: {accuracy_cat:.2f}')
```

## 4. Summary of Results
- K-Means Silhouette Score[ k=5: 0.553931997444648]
- Gradient Boosting (GB)Classification Accuracy[0.97]
- XGBoostClassification Accuracy[0.95]
- CatBoostClassification Accuracy[0.95]