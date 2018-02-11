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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


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

a_sub = a_sub.groupby(['id', 'country'], as_index=False)['poor'].mean()





def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()








##### OTHER MODELS

# Model B
scores2 = cross_val_score(model2, bX_i_train, by_i_train, scoring='neg_log_loss')
model2 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=0)
model2.fit(bX_i_train, by_i_train)
results = model2.predict(bX_i_train)
confusion_matrix(by_i_train, results)

b_preds = model2.predict_proba(bX_i_train)
b_preds = model2.predict_proba(b_i_test_imputed)

b_sub = make_country_sub(b_preds, b_i_test_imputed, 'B')

b_sub.reset_index(inplace=True)

b_sub = b_sub.groupby(['id', 'country'], as_index=False)['poor'].mean()

sns.boxplot(y="poor", data=b_sub)

# Model C
scores3 = cross_val_score(model3, cX_i_train, cy_i_train, scoring='neg_log_loss')

model3 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=0)
model3.fit(cX_i_train, cy_i_train)
results = model3.predict(cX_i_train)
confusion_matrix(cy_i_train, results)

c_preds = model3.predict_proba(cX_i_train)
c_preds = model3.predict_proba(c_i_test)

c_sub = make_country_sub(c_preds, c_i_test, 'C')

c_sub.reset_index(inplace=True)

c_sub = c_sub.groupby(['id', 'country'], as_index=False)['poor'].mean()

sns.boxplot(y="poor", data=c_sub)

submission = pd.concat([a_sub, b_sub, c_sub])

submission_final = pd.read_csv('submissions/submission_format.csv')

submission_final = pd.merge(submission_final, submission, on='id')[['id', 'country_x', 'poor_y']]
submission_final.columns = ['id', 'country', 'poor']
submission_final.to_csv('submissions/RF_individual_1.csv', index=False)

