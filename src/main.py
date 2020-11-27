from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *

from src.data.data import *
from src.models.predict_model import *
from src.visualization.visualize import *

def _get_project_root() -> Path:
    return Path(__file__).parent.parent


root_dir = str(_get_project_root())
training_data = '/data/churn_train.csv'
test_data = '/data/churn_test.csv'


if __name__ == "__main__":
    # Load data
    df = load_data(root_dir, training_data)
    df_final = run_data_pipeline(df)
    X, y = create_variables(df_final)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 11)

    # Create models and predict data
    y_pred_lr, y_pred_proba_lr, lr_model = predict_model(X_train, y_train, X_test, LogisticRegression)
    y_pred_rf, y_pred_proba_rf, rf_model = predict_model(X_train, y_train, X_test, RandomForestClassifier)
    y_pred_gb, y_pred_proba_gb, gb_model = predict_model(X_train, y_train, X_test, GradientBoostingClassifier)

    # Visualize results
    print_metrics(y_test, y_pred_lr, 'Logistic Regression')
    print_metrics(y_test, y_pred_rf, 'Random Forest Classifier')
    print_metrics(y_test,y_pred_gb, 'Gradient Boosting Classifier')
    plot_roc_curve(y_test,y_pred_proba_lr, y_pred_proba_rf, y_pred_proba_gb,'LR','RF','GB')

    # Plot feature importance for gradient boosting classifier model
    plot_feature_importance_chart(gb_model, X_train)
