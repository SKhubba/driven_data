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
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Model A
model1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)

model1_cccv=CalibratedClassifierCV(model1, cv=3)

model1_cccv.fit(aX_i_train, ay_i_train)

model2_cccv=CalibratedClassifierCV(model1, cv=3)

model2_cccv.fit(bX_i_train, by_i_train)

model3_cccv=CalibratedClassifierCV(model1, cv=3)

model3_cccv.fit(cX_i_train, cy_i_train)

a_preds = model1_cccv.predict_proba(a_i_test_imputed)
b_preds = model2_cccv.predict_proba(b_i_test_imputed)
c_preds = model3_cccv.predict_proba(c_i_test)

a_sub = make_country_sub(a_preds, a_i_test_imputed, 'A')
b_sub = make_country_sub(b_preds, b_i_test_imputed, 'B')
c_sub = make_country_sub(c_preds, c_i_test, 'C')

def avg_predictions(sub_df):
    sub_df.reset_index(inplace=True)
    return sub_df.groupby(['id', 'country'], as_index=False)['poor'].mean()

a_sub, b_sub, c_sub = avg_predictions(a_sub), avg_predictions(b_sub), avg_predictions(c_sub)

submission = pd.concat([a_sub, b_sub, c_sub])

submission_final = pd.read_csv('submissions/submission_format.csv')

submission_final = pd.merge(submission_final, submission, on='id')[['id', 'country_x', 'poor_y']]
submission_final.columns = ['id', 'country', 'poor']
submission_final.to_csv('submissions/RF_individual_calibrated.csv', index=False)

sns.boxplot(x="country", y="poor", data=submission_final)