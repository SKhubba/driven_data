from statsmodels.imputation import mice
import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns

# import submission function
from data_munge import make_country_sub, impute

# import the raw training data for reference
from data_munge import a_i_train, a_h_train, b_i_train, b_h_train, c_i_train, c_h_train

# import the preprocessed test data
from data_munge import a_i_test, a_h_test, b_i_test, b_h_test, c_i_test, c_h_test

# import the preprocessed data
from data_munge import aX_h_train, ay_h_train, bX_h_train, by_h_train, cX_h_train, cy_h_train
from data_munge import aX_i_train, ay_i_train, bX_i_train, by_i_train, cX_i_train, cy_i_train

bX_h_train = impute(bX_h_train)
b_h_test_imputed = impute(b_h_test)
aX_i_train = impute(aX_i_train)
a_i_test_imputed = impute(a_i_test)
bX_i_train = impute(bX_i_train)
b_i_test_imputed = impute(b_i_test)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

# Model A
model1 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=0)
scores1 = cross_val_score(model1, aX_i_train, ay_i_train, scoring='neg_log_loss')

model1.fit(aX_i_train, ay_i_train)
results = model1.predict(aX_i_train)
confusion_matrix(ay_i_train, results)

a_preds = model1.predict_proba(aX_i_train)
a_preds = model1.predict_proba(a_i_test_imputed)

a_sub = make_country_sub(a_preds, a_i_test_imputed, 'A')

a_sub.reset_index(inplace=True)

sns.boxplot(y="poor", data=a_sub)

# Model B
scores1 = cross_val_score(model1, bX_i_train, by_i_train, scoring='neg_log_loss')

model1.fit(bX_i_train, by_i_train)
results = model1.predict(bX_i_train)
confusion_matrix(by_i_train, results)

b_preds = model1.predict_proba(bX_i_train)
b_preds = model1.predict_proba(b_i_test_imputed)

b_sub = make_country_sub(b_preds, b_i_test_imputed, 'B')

b_sub.reset_index(inplace=True)

b_sub = b_sub.groupby(['id', 'country'], as_index=False)['poor'].mean().reset_index()

sns.boxplot(y="poor", data=b_sub)

# Model C
scores1 = cross_val_score(model1, cX_i_train, cy_i_train, scoring='neg_log_loss')

model1.fit(cX_i_train, cy_i_train)
results = model1.predict(cX_i_train)
confusion_matrix(cy_i_train, results)

c_preds = model1.predict_proba(cX_i_train)
c_preds = model1.predict_proba(c_i_test)

c_sub = make_country_sub(c_preds, c_i_test, 'A')

c_sub.reset_index(inplace=True)

c_sub = c_sub.groupby(['id', 'country'], as_index=False)['poor'].mean().reset_index()

sns.boxplot(y="poor", data=c_sub)

submission = pd.concat([a_sub, b_sub, c_sub])
submission.to_csv('submissions/svm_3.csv')










model2 = svm.SVC(probability=True, cache_size=500, class_weight={1:12})
scores2 = cross_val_score(model2, bX_h_train, by_h_train, scoring='neg_log_loss')

model3 = svm.SVC(probability=True, cache_size=500, class_weight={1:5.5})
scores3 = cross_val_score(model3, cX_h_train, cy_h_train, scoring='neg_log_loss')

model2.fit(bX_h_train, by_h_train)
results = model2.predict(bX_h_train)
confusion_matrix(by_h_train, results)

model3.fit(cX_h_train, cy_h_train)
results = model3.predict(cX_h_train)
confusion_matrix(cy_h_train, results)

a_preds = model1.predict_proba(a_h_test)
b_preds = model2.predict_proba(b_h_test_imputed)
c_preds = model3.predict_proba(c_h_test)

a_sub = make_country_sub(a_preds, a_h_test, 'A')
b_sub = make_country_sub(b_preds, b_h_test_imputed, 'B')
c_sub = make_country_sub(c_preds, c_h_test, 'C')

submission = pd.concat([a_sub, b_sub, c_sub])
submission.to_csv('submissions/svm_3.csv')

import seaborn as sns
sns.boxplot(x="country", y="poor", data=submission)