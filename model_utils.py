"""Model definitions, cross-validated model selection, cost-based thresholds, calibration and subgroup checks."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from data_utils import ENGINEERED_NUMERIC, FEATURES, NUMERIC, TARGET, add_features

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

MODEL_NAMES = ['Logistic regression', 'Random forest', 'Gradient boosting']

# Which columns each feature set feeds to the model. 'No gender' tests whether Gender earns its place.
FEATURE_SETS = {
    'Raw': {'engineered': False, 'categorical': ['Geography', 'Gender']},
    'Engineered': {'engineered': True, 'categorical': ['Geography', 'Gender']},
    'Raw, no gender': {'engineered': False, 'categorical': ['Geography']},
    'Engineered, no gender': {'engineered': True, 'categorical': ['Geography']},
}
WITH_GENDER = ['Raw', 'Engineered']

# Illustrative campaign assumptions (GBP). The app lets the user change them.
DEFAULT_ECONOMICS = {'cost_per_contact': 5.0, 'value_per_save': 100.0, 'save_rate': 0.20}


def make_model(model_name, feature_set='Engineered'):
    """Build a full pipeline: feature engineering -> preprocessing -> estimator.
    Everything is fitted inside the pipeline, so cross-validation never sees the held-out fold."""
    spec = FEATURE_SETS[feature_set]
    numeric = NUMERIC + (ENGINEERED_NUMERIC if spec['engineered'] else [])
    numeric_pipeline = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical_pipeline = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                     ('encode', OneHotEncoder(handle_unknown='ignore'))])
    # remainder='drop' means any column not listed (e.g. Gender in the no-gender set) is never seen by the model.
    prep = ColumnTransformer([('numeric', numeric_pipeline, numeric),
                              ('categorical', categorical_pipeline, spec['categorical'])],
                             remainder='drop', verbose_feature_names_out=False)
    features = FunctionTransformer(add_features) if spec['engineered'] else 'passthrough'

    # No class weighting: re-weighting inflates predicted probabilities, and this project handles the
    # class imbalance by choosing the decision threshold from campaign costs instead.
    if model_name == 'Logistic regression':
        estimator = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    elif model_name == 'Random forest':
        estimator = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == 'Gradient boosting':
        estimator = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                                   l2_regularization=1.0, early_stopping=False,
                                                   random_state=RANDOM_STATE)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return Pipeline([('features', features), ('preprocess', prep), ('model', estimator)])


def split_data(df):
    """One stratified 80/20 split. The test set is set aside and used exactly once, at the end."""
    return train_test_split(df[FEATURES], df[TARGET], test_size=TEST_SIZE, stratify=df[TARGET],
                            random_state=RANDOM_STATE)


def cross_validate_configs(X, y, configs):
    """5-fold stratified CV on the training data for each (model, feature set) pair.
    Returns a summary table and the out-of-fold probabilities for every config."""
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))
    rows, oof = [], {}
    for model_name, feature_set in configs:
        probs = np.zeros(len(y))
        roc, pr, brier = [], [], []
        for train_idx, val_idx in folds:
            model = make_model(model_name, feature_set)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            p = model.predict_proba(X.iloc[val_idx])[:, 1]
            probs[val_idx] = p
            roc.append(roc_auc_score(y.iloc[val_idx], p))
            pr.append(average_precision_score(y.iloc[val_idx], p))
            brier.append(brier_score_loss(y.iloc[val_idx], p))
        oof[(model_name, feature_set)] = probs
        rows.append({'Model': model_name, 'Features': feature_set,
                     'PR-AUC': np.mean(pr), 'PR-AUC sd': np.std(pr, ddof=1),
                     'ROC-AUC': np.mean(roc), 'ROC-AUC sd': np.std(roc, ddof=1),
                     'Brier': np.mean(brier)})
    return pd.DataFrame(rows), oof


def net_benefit(y_true, prob, threshold, cost_per_contact, value_per_save, save_rate):
    """Illustrative campaign profit: contact everyone at or above the threshold. Each contacted customer
    costs `cost_per_contact`; each contacted customer who would have churned is saved with probability
    `save_rate`, worth `value_per_save`. Customers who would not have churned are pure cost."""
    contacted = prob >= threshold
    churners_reached = int(np.asarray(y_true)[contacted].sum())
    return churners_reached * save_rate * value_per_save - int(contacted.sum()) * cost_per_contact


def threshold_curve(y_true, prob, economics, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.981, 0.01), 2)
    values = [net_benefit(y_true, prob, t, **economics) for t in thresholds]
    return pd.DataFrame({'threshold': thresholds, 'net_benefit': values})


def choose_threshold(y_true, prob, economics):
    """Pick the threshold that maximises net benefit. Called on out-of-fold training predictions, never
    on the test set, so the reported test result is not tuned to the test data."""
    curve = threshold_curve(y_true, prob, economics)
    return float(curve.loc[curve['net_benefit'].idxmax(), 'threshold']), curve


def breakeven_threshold(economics):
    """With calibrated probabilities, contacting pays off when p * save_rate * value > cost."""
    return economics['cost_per_contact'] / (economics['save_rate'] * economics['value_per_save'])


def classification_metrics(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {'ROC-AUC': roc_auc_score(y_true, prob), 'PR-AUC': average_precision_score(y_true, prob),
            'Brier': brier_score_loss(y_true, prob),
            'Precision': precision_score(y_true, pred, zero_division=0),
            'Recall': recall_score(y_true, pred, zero_division=0),
            'F1': f1_score(y_true, pred, zero_division=0),
            'Contact rate': pred.mean(), 'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)}


def calibration_table(y_true, prob, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy='quantile')
    return pd.DataFrame({'mean_predicted': mean_pred, 'observed_rate': frac_pos})


def subgroup_report(X, y_true, prob, threshold, column):
    """Performance and treatment by group: are some groups ranked worse, or contacted far more often?"""
    rows = []
    y_true = np.asarray(y_true)
    for group in sorted(X[column].unique()):
        m = (X[column] == group).to_numpy()
        yg, pg = y_true[m], prob[m]
        pred = pg >= threshold
        rows.append({'Group': f'{column}: {group}', 'Customers': int(m.sum()),
                     'Actual churn rate': yg.mean(), 'Mean predicted risk': pg.mean(),
                     'ROC-AUC': roc_auc_score(yg, pg), 'Contact rate': pred.mean(),
                     'Recall': pred[yg == 1].mean(), 'False positive rate': pred[yg == 0].mean()})
    return pd.DataFrame(rows)


def with_age_band(X):
    return X.assign(**{'Age band': pd.cut(X['Age'], bins=[17, 30, 40, 50, 60, 120],
                                          labels=['18-30', '31-40', '41-50', '51-60', '61+']).astype(str)})


def run_experiment(df, economics=None):
    """Full workflow shared by train.py and the app:
    1. split off a test set; 2. cross-validate every model/feature-set pair on the training data;
    3. pick the winner by mean CV PR-AUC; 4. decide on Gender (drop it unless it clearly helps);
    5. choose the threshold on out-of-fold predictions; 6. refit on all training data; 7. score the test set once."""
    economics = economics or DEFAULT_ECONOMICS
    X_train, X_test, y_train, y_test = split_data(df)
    configs = [(m, f) for f in FEATURE_SETS for m in MODEL_NAMES]
    cv_table, oof = cross_validate_configs(X_train, y_train, configs)

    with_gender = cv_table[cv_table['Features'].isin(WITH_GENDER)]
    best = with_gender.loc[with_gender['PR-AUC'].idxmax()]
    twin = f"{best['Features']}, no gender"
    no_gender = cv_table[(cv_table['Model'] == best['Model']) & (cv_table['Features'] == twin)].iloc[0]
    # Pre-set rule: keep Gender only if it improves CV PR-AUC by more than one standard error across folds.
    std_error = best['PR-AUC sd'] / np.sqrt(N_FOLDS)
    gender_gain = best['PR-AUC'] - no_gender['PR-AUC']
    drop_gender = gender_gain <= std_error
    final_features = twin if drop_gender else best['Features']
    final_name = best['Model']

    threshold, oof_curve = choose_threshold(y_train, oof[(final_name, final_features)], economics)

    final_model = make_model(final_name, final_features).fit(X_train, y_train)
    test_prob = final_model.predict_proba(X_test)[:, 1]

    # Comparison on the test set at the same threshold: the simplest baseline and a 0.5 default.
    baseline = make_model('Logistic regression', 'Raw').fit(X_train, y_train)
    baseline_prob = baseline.predict_proba(X_test)[:, 1]

    test_curve = threshold_curve(y_test, test_prob, economics)
    strategies = {
        'Contact nobody': 0.0,
        'Contact everyone': net_benefit(y_test, test_prob, 0.0, **economics),
        'Final model, threshold 0.50': net_benefit(y_test, test_prob, 0.5, **economics),
        f'Final model, chosen threshold {threshold:.2f}': net_benefit(y_test, test_prob, threshold, **economics),
        f'Baseline logistic regression, threshold {threshold:.2f}': net_benefit(y_test, baseline_prob, threshold, **economics),
    }

    return {
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'cv_table': cv_table, 'oof': oof, 'economics': economics,
        'final_name': final_name, 'final_features': final_features, 'final_model': final_model,
        'gender_decision': {'model': final_name, 'with_gender': best['Features'], 'without_gender': twin,
                            'gain_from_gender': float(gender_gain), 'std_error': float(std_error),
                            'dropped': bool(drop_gender)},
        'threshold': threshold, 'breakeven_threshold': breakeven_threshold(economics), 'oof_curve': oof_curve,
        'test_prob': test_prob, 'baseline_prob': baseline_prob, 'test_curve': test_curve,
        'test_metrics': classification_metrics(y_test, test_prob, threshold),
        'test_metrics_at_05': classification_metrics(y_test, test_prob, 0.5),
        'baseline_test_metrics': classification_metrics(y_test, baseline_prob, threshold),
        'strategies': strategies,
        'calibration': calibration_table(y_test, test_prob),
        'subgroups': pd.concat([subgroup_report(with_age_band(X_test), y_test, test_prob, threshold, c)
                                for c in ['Gender', 'Geography', 'Age band']], ignore_index=True),
    }


def shap_explanation(model, X):
    """SHAP values for the fitted pipeline, in terms of the model's input columns (after preprocessing).
    Tree models use the exact TreeExplainer; the linear model uses LinearExplainer."""
    import shap
    transformed = model[:-1].transform(X)
    names = list(model.named_steps['preprocess'].get_feature_names_out())
    estimator = model.named_steps['model']
    if isinstance(estimator, LogisticRegression):
        explainer = shap.LinearExplainer(estimator, transformed)
        values = explainer.shap_values(transformed)
        base = explainer.expected_value
    else:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        base = explainer.expected_value
        if isinstance(values, list):  # random forest returns one array per class
            values, base = values[1], base[1]
        elif values.ndim == 3:
            values, base = values[:, :, 1], np.ravel(base)[-1]
    return pd.DataFrame(values, columns=names, index=X.index), float(np.ravel(base)[0]), names


# ---- Saved results for the app -------------------------------------------------------------------
# train.py saves everything the app needs to this file, which is committed to git. The app loads it
# instead of retraining, so it opens in seconds. The scikit-learn version is stored alongside because a
# model saved under one version is not guaranteed to load under another.
from pathlib import Path  # noqa: E402

BUNDLE_PATH = Path(__file__).resolve().parent / 'artifacts' / 'results.joblib'


def save_results(results, path=BUNDLE_PATH):
    import joblib
    import sklearn
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'sklearn_version': sklearn.__version__, 'results': results}, path, compress=3)


def load_results(path=BUNDLE_PATH):
    """Return saved results, or None if the file is missing or was saved with a different scikit-learn."""
    import joblib
    import sklearn
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
    except Exception:
        return None
    if bundle.get('sklearn_version') != sklearn.__version__:
        return None
    return bundle['results']
