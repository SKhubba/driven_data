from statsmodels.imputation import mice
import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd

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

from sklearn import svm
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

model1 = svm.SVC(probability=True, cache_size=500, class_weight={1:1.3})
scores1 = cross_val_score(model, aX_h_train, ay_h_train, scoring='neg_log_loss')

model2 = svm.SVC(probability=True, cache_size=500, class_weight={1:12})
scores2 = cross_val_score(model, bX_h_train, by_h_train, scoring='neg_log_loss')

model3 = svm.SVC(probability=True, cache_size=500, class_weight={1:5.5})
scores3 = cross_val_score(model, cX_h_train, cy_h_train, scoring='neg_log_loss')


model1.fit(aX_h_train, ay_h_train)
results = model1.predict(aX_h_train)
confusion_matrix(ay_h_train, results)

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