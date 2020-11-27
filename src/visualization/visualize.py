import matplotlib.pyplot as plt
from sklearn.metrics import *
import numpy as np

# Function for Plotting ROC-Curves
def plot_roc_curve(y_test,y_pred_proba_1, y_pred_proba_2, y_pred_proba_3, model_1, model_2, model_3):
    fpr1, tpr1, _ = roc_curve(y_test, y_pred_proba_1)
    auc1 = roc_auc_score(y_test, y_pred_proba_1)

    fpr2, tpr2, _ = roc_curve(y_test, y_pred_proba_2)
    auc2 = roc_auc_score(y_test, y_pred_proba_2)

    fpr3, tpr3, _ = roc_curve(y_test, y_pred_proba_3)
    auc3 = roc_auc_score(y_test, y_pred_proba_3)

    plt.figure(1,figsize=(12,8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr1, tpr1, label=f'{model_1} AUC={round(auc1,3)}')
    plt.plot(fpr2, tpr2, label=f'{model_2} AUC={round(auc2,3)}')
    plt.plot(fpr3, tpr3, label=f'{model_3} AUC={round(auc3,3)}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    return


def plot_feature_importance_chart(model, X):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(list(X.columns))[sorted_idx])
    plt.title('Feature Importance')
    fig.tight_layout
    plt.show()

    return

