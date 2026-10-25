"""Run `python train.py` to reproduce every number and figure in the README.

Writes:
  models/churn_model.joblib   final fitted pipeline (not committed to git; rebuild with this script)
  reports/results.json        cross-validation table, test metrics, threshold, subgroup checks
  reports/figures/*.png       charts used in the README
  artifacts/results.joblib    everything the Streamlit app shows (committed, so the app never retrains)
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from data_utils import TARGET, load_data  # noqa: E402
from model_utils import BUNDLE_PATH, N_FOLDS, run_experiment, save_results, shap_explanation  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'reports' / 'figures'

# Chart colours (colour-blind-safe order) and neutral ink/grid tones.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'text.color': INK, 'axes.titlecolor': INK, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'figure.dpi': 150,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches='tight')
    plt.close(fig)


def pct(ax, axis='y'):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(fmt)


def eda_figures(df):
    grp = df.groupby('NumOfProducts')[TARGET].agg(['mean', 'size'])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(grp.index.astype(str), grp['mean'], color=BLUE, width=0.6)
    for x, (rate, n) in enumerate(zip(grp['mean'], grp['size'])):
        ax.text(x, rate + 0.02, f'{rate:.0%}\n(n={n:,})', ha='center', va='bottom', fontsize=9, color=INK_2)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    pct(ax)
    ax.grid(axis='x', visible=False)
    ax.set_xlabel('Number of products held')
    ax.set_ylabel('Churn rate')
    ax.set_title('Churn is lowest with 2 products and very high with 3 or more')
    save(fig, 'eda_products.png')

    bands = pd.cut(df['Age'], bins=[17, 30, 40, 50, 60, 120], labels=['18–30', '31–40', '41–50', '51–60', '61+'])
    grp = df.groupby([bands, 'IsActiveMember'], observed=True)[TARGET].mean().unstack()
    x = np.arange(len(grp))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(x - 0.2, grp[0], width=0.38, color=ORANGE, label='Inactive member')
    ax.bar(x + 0.2, grp[1], width=0.38, color=BLUE, label='Active member')
    ax.set_xticks(x, grp.index)
    pct(ax)
    ax.grid(axis='x', visible=False)
    ax.set_xlabel('Age band')
    ax.set_ylabel('Churn rate')
    ax.legend(loc='upper left')
    ax.set_title('Over-40s who are inactive churn far more than active customers of the same age')
    save(fig, 'eda_age_activity.png')


def cv_figure(cv):
    data = cv[cv['Features'].isin(['Raw', 'Engineered'])]
    models = ['Logistic regression', 'Random forest', 'Gradient boosting']
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, (feat, colour) in enumerate([('Raw', ORANGE), ('Engineered', BLUE)]):
        rows = data[data['Features'] == feat].set_index('Model').loc[models]
        y = np.arange(len(models)) + (0.12 if i else -0.12)
        ax.errorbar(rows['PR-AUC'], y, xerr=rows['PR-AUC sd'], fmt='o', color=colour, markersize=7,
                    capsize=3, linewidth=1.5, label=f'{feat} features')
        for val, yy in zip(rows['PR-AUC'], y):
            ax.text(val, yy + (0.3 if i else -0.18), f'{val:.3f}', ha='center', fontsize=8, color=INK_2)
    ax.set_yticks(np.arange(len(models)), models)
    ax.set_ylim(-0.6, len(models) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel(f'PR-AUC, mean ± sd across {N_FOLDS} CV folds (training data only)')
    ax.legend(loc='lower left', bbox_to_anchor=(0.22, 0.02), fontsize=9)
    ax.set_title('Feature engineering lifts the linear model; trees learn the patterns anyway')
    save(fig, 'cv_comparison.png')


def roc_pr_figure(r):
    y = r['y_test']
    curves = [(r['test_prob'], f"{r['final_name']} (final)", BLUE), (r['baseline_prob'], 'Logistic regression, raw (baseline)', ORANGE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    for prob, label, colour in curves:
        fpr, tpr, _ = roc_curve(y, prob)
        a.plot(fpr, tpr, color=colour, linewidth=2, label=label)
        prec, rec, _ = precision_recall_curve(y, prob)
        b.plot(rec, prec, color=colour, linewidth=2, label=label)
    a.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle='--', label='Random guessing')
    b.axhline(y.mean(), color=INK_2, linewidth=1, linestyle='--', label=f'Base rate ({y.mean():.0%})')
    a.set(xlabel='False positive rate', ylabel='True positive rate (recall)', title='ROC curve, test set')
    b.set(xlabel='Recall', ylabel='Precision', title='Precision–recall curve, test set', ylim=(0, 1.02))
    a.legend(loc='lower right', fontsize=8)
    b.legend(loc='lower left', bbox_to_anchor=(0.0, 0.22), fontsize=8)
    save(fig, 'roc_pr.png')


def calibration_figure(r):
    cal = r['calibration']
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle='--', label='Perfect calibration')
    ax.plot(cal['mean_predicted'], cal['observed_rate'], color=BLUE, linewidth=2, marker='o', markersize=6,
            label=f"{r['final_name']} (test set, deciles)")
    pct(ax)
    pct(ax, 'x')
    ax.set(xlabel='Mean predicted churn risk', ylabel='Observed churn rate', xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(f"Predicted risk is well calibrated (Brier {r['test_metrics']['Brier']:.3f})")
    save(fig, 'calibration.png')


def threshold_figure(r):
    n_train, n_test = len(r['y_train']), len(r['y_test'])
    oof, test = r['oof_curve'], r['test_curve']
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(oof['threshold'], oof['net_benefit'] / n_train * 1000, color=BLUE, linewidth=2,
            label='Training data (out-of-fold) – used to choose')
    ax.plot(test['threshold'], test['net_benefit'] / n_test * 1000, color=ORANGE, linewidth=2,
            label='Test set – checked once afterwards')
    ax.axvline(r['threshold'], color=INK_2, linewidth=1, linestyle='--')
    ax.text(r['threshold'] + 0.01, -350, f"chosen threshold {r['threshold']:.2f}", color=INK_2, fontsize=9)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel('Contact customers with predicted risk at or above')
    ax.set_ylabel('Net benefit per 1,000 customers (£)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.62, 0.12), fontsize=8)
    e = r['economics']
    ax.set_title(f"Campaign value by threshold (£{e['cost_per_contact']:.0f}/contact, "
                 f"{e['save_rate']:.0%} save rate, £{e['value_per_save']:.0f}/save)", fontsize=11)
    save(fig, 'threshold_profit.png')


def shap_figure(r):
    values, _, _ = shap_explanation(r['final_model'], r['X_test'])
    imp = values.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Mean |SHAP value| (log-odds), test set')
    ax.set_title('What drives the model’s predictions')
    save(fig, 'shap_importance.png')
    return imp.sort_values(ascending=False)


def to_jsonable(r, shap_imp):
    tm = {k: float(v) for k, v in r['test_metrics'].items()}
    return {
        'final_model': r['final_name'], 'final_features': r['final_features'],
        'train_size': len(r['y_train']), 'test_size': len(r['y_test']),
        'economics': r['economics'], 'chosen_threshold': r['threshold'],
        'breakeven_threshold': r['breakeven_threshold'],
        'gender_decision': r['gender_decision'],
        'cross_validation': r['cv_table'].round(4).to_dict(orient='records'),
        'test_metrics_at_chosen_threshold': tm,
        'test_metrics_at_0.5': {k: float(v) for k, v in r['test_metrics_at_05'].items()},
        'baseline_test_metrics_at_chosen_threshold': {k: float(v) for k, v in r['baseline_test_metrics'].items()},
        'test_net_benefit_by_strategy': {k: float(v) for k, v in r['strategies'].items()},
        'subgroups_test_set': r['subgroups'].round(4).to_dict(orient='records'),
        'mean_abs_shap_test_set': shap_imp.round(4).to_dict(),
    }


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / 'models').mkdir(exist_ok=True)
    df = load_data()
    print('Running cross-validation (this takes about 30 seconds)...')
    r = run_experiment(df)
    eda_figures(df)
    cv_figure(r['cv_table'])
    roc_pr_figure(r)
    calibration_figure(r)
    threshold_figure(r)
    shap_imp = shap_figure(r)

    joblib.dump(r['final_model'], ROOT / 'models' / 'churn_model.joblib')
    save_results(r)
    results = to_jsonable(r, shap_imp)
    (ROOT / 'reports' / 'results.json').write_text(json.dumps(results, indent=2))

    pd.set_option('display.width', 200)
    print('\nCross-validation (training data):')
    print(r['cv_table'].round(3).to_string(index=False))
    print(f"\nFinal model: {r['final_name']} with {r['final_features']} features")
    print(f"Gender decision: {r['gender_decision']}")
    print(f"Chosen threshold: {r['threshold']:.2f} (break-even if calibrated: {r['breakeven_threshold']:.2f})")
    print('\nTest set (used once):')
    print({k: round(float(v), 3) for k, v in r['test_metrics'].items()})
    print('\nTest-set net benefit by strategy (£):')
    for k, v in r['strategies'].items():
        print(f'  {k}: {v:,.0f}')
    print('\nSubgroups (test set):')
    print(r['subgroups'].round(3).to_string(index=False))
    print(f'\nSaved models/churn_model.joblib, reports/results.json, reports/figures/ and '
          f'{BUNDLE_PATH.relative_to(ROOT)} (loaded by the app; commit it)')


if __name__ == '__main__':
    main()
