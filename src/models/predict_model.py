from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


def predict_model(X_train, y_train, X_test, model, trees = 100):
    if model == LogisticRegression:
        model = model(max_iter=1000)
    else:
        model = model(n_estimators = trees)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]

    return y_pred, y_pred_proba, model


def print_metrics(y_test, y_pred, model_name = 'Model'):
    print(model_name, "- Accuracy Score: ", accuracy_score(y_test, y_pred))
    print(model_name, "- Precision Score: ", precision_score(y_test, y_pred))
    print(model_name, "- Recall Score: ", recall_score(y_test, y_pred))
    print(model_name, "- F1 Score: ", f1_score(y_test, y_pred, average = 'weighted'))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("TP:", tp, "  ", "FN:", fn, "  ", "TN:", tn, "  ", "FP:", fp)

    return






