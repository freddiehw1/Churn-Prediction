import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from data_utils import TARGET, load_data
from model_utils import (N_FOLDS, breakeven_threshold, choose_threshold, classification_metrics, net_benefit,
                         load_results, run_experiment, shap_explanation, threshold_curve)

BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#8a8984'

st.set_page_config(page_title='Bank Churn Prediction', page_icon='📊', layout='wide')
st.title('Customer churn prediction')
st.caption('Portfolio project on a public 10,000-customer bank dataset · Cross-validated model selection · '
           'Cost-based targeting · Fairness checks')


@st.cache_data(show_spinner='Loading dataset...')
def get_data():
    return load_data()


@st.cache_resource(show_spinner='Loading saved model and results...')
def get_results(_df):
    saved = load_results()
    if saved is not None:
        return saved
    # Fallback: no saved file (or a scikit-learn version mismatch), so rebuild from scratch.
    with st.spinner(f'No saved results found. Running {N_FOLDS}-fold cross-validation (about 30 seconds)...'):
        return run_experiment(_df)


try:
    df = get_data()
except Exception as exc:
    st.error(f'Dataset unavailable: {exc}')
    st.stop()

r = get_results(df)
model = r['final_model']
uses_gender = 'no gender' not in r['final_features']


def style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text='',
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig


overview, selection, test_tab, fairness, explain, strategy = st.tabs(
    ['Overview', 'Model selection', 'Test results', 'Fairness', 'Explain a prediction', 'Retention strategy'])

with overview:
    a, b, c, d = st.columns(4)
    a.metric('Customers', f'{len(df):,}')
    b.metric('Churn rate', f'{df[TARGET].mean():.1%}')
    c.metric('Final model', r['final_name'])
    d.metric('Test PR-AUC', f"{r['test_metrics']['PR-AUC']:.3f}",
             help='Area under the precision-recall curve. Random guessing scores the churn rate (about 0.20).')
    left, right = st.columns(2)
    with left:
        grp = df.groupby('NumOfProducts', as_index=False)[TARGET].agg(['mean', 'size'])
        fig = px.bar(grp, x='NumOfProducts', y='mean', text=grp['mean'].map('{:.0%}'.format),
                     hover_data={'size': ':,'}, color_discrete_sequence=[BLUE],
                     labels={'mean': 'Churn rate', 'NumOfProducts': 'Products held', 'size': 'Customers'},
                     title='Churn by number of products')
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(style(fig), width='stretch')
    with right:
        bands = pd.cut(df['Age'], bins=[17, 30, 40, 50, 60, 120], labels=['18–30', '31–40', '41–50', '51–60', '61+'])
        grp = (df.assign(band=bands, active=df['IsActiveMember'].map({1: 'Active member', 0: 'Inactive member'}))
               .groupby(['band', 'active'], observed=True)[TARGET].mean().reset_index())
        fig = px.bar(grp, x='band', y=TARGET, color='active', barmode='group',
                     color_discrete_map={'Inactive member': ORANGE, 'Active member': BLUE},
                     labels={TARGET: 'Churn rate', 'band': 'Age band'}, title='Churn by age and activity')
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(style(fig), width='stretch')
    st.markdown(
        '**Key patterns.** Churn is not monotonic in product count: 2 products is the safest group, 3 or more is '
        'the riskiest. Churn peaks for customers in their 50s, and inactive customers over 40 churn far more than '
        'active ones of the same age. German customers churn at about twice the rate of French and Spanish ones. '
        'These are associations in a teaching dataset, not proven causes.')

with selection:
    st.subheader(f'{N_FOLDS}-fold cross-validation on the training data')
    st.write('Every model and feature set was compared using cross-validation on the 8,000 training customers only. '
             'The 2,000 test customers were not touched until the final model was chosen. The winner is the '
             'highest mean PR-AUC, which suits an imbalanced problem better than accuracy or ROC-AUC.')
    cv = r['cv_table'].copy()
    st.dataframe(cv.style.format({c: '{:.3f}' for c in cv.columns if c not in ('Model', 'Features')})
                 .highlight_max(subset=['PR-AUC', 'ROC-AUC'], color='#cde2fb')
                 .highlight_min(subset=['Brier'], color='#cde2fb'), width='stretch', hide_index=True)
    plot = cv[cv['Features'].isin(['Raw', 'Engineered'])]
    fig = px.scatter(plot, x='PR-AUC', y='Model', color='Features', error_x='PR-AUC sd',
                     color_discrete_map={'Raw': ORANGE, 'Engineered': BLUE},
                     title='Feature engineering lifts logistic regression; tree models already capture the patterns')
    fig.update_traces(marker_size=11)
    st.plotly_chart(style(fig, 320), width='stretch')
    g = r['gender_decision']
    st.info(f"**Gender decision.** Adding Gender to {g['model']} improves CV PR-AUC by only "
            f"{g['gain_from_gender']:.3f}, less than one standard error across folds ({g['std_error']:.3f}). "
            f"Under the rule set before running the comparison, Gender is "
            f"{'**dropped**' if g['dropped'] else '**kept**'}. "
            f"Final model: **{r['final_name']}** with **{r['final_features']}** features.")

with test_tab:
    tm, t05 = r['test_metrics'], r['test_metrics_at_05']
    st.subheader('Final evaluation on the held-out test set (used once)')
    a, b, c, d = st.columns(4)
    a.metric('ROC-AUC', f"{tm['ROC-AUC']:.3f}")
    b.metric('PR-AUC', f"{tm['PR-AUC']:.3f}")
    c.metric('Brier score', f"{tm['Brier']:.3f}", help='Mean squared error of the probabilities; lower is better.')
    d.metric('Baseline PR-AUC', f"{r['baseline_test_metrics']['PR-AUC']:.3f}",
             help='Plain logistic regression on raw features.')
    left, right = st.columns(2)
    y = r['y_test']
    with left:
        fig = go.Figure()
        for prob, name, col in [(r['test_prob'], f"{r['final_name']} (final)", BLUE),
                                (r['baseline_prob'], 'Logistic regression (baseline)', ORANGE)]:
            p, rc, _ = precision_recall_curve(y, prob)
            fig.add_trace(go.Scatter(x=rc, y=p, name=name, line=dict(color=col, width=2)))
        fig.add_hline(y=y.mean(), line_dash='dash', line_color=GREY, annotation_text='Base rate')
        fig.update_layout(title='Precision-recall curve', xaxis_title='Recall', yaxis_title='Precision')
        st.plotly_chart(style(fig), width='stretch')
    with right:
        cal = r['calibration']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Perfect calibration', line=dict(color=GREY, dash='dash')))
        fig.add_trace(go.Scatter(x=cal['mean_predicted'], y=cal['observed_rate'], name='Final model (deciles)',
                                 mode='lines+markers', line=dict(color=BLUE, width=2)))
        fig.update_layout(title='Calibration: does a 30% prediction mean a 30% churn rate?',
                          xaxis_title='Mean predicted risk', yaxis_title='Observed churn rate',
                          xaxis_tickformat='.0%', yaxis_tickformat='.0%')
        st.plotly_chart(style(fig), width='stretch')
    fpr, tpr, _ = roc_curve(y, r['test_prob'])
    with st.expander('ROC curve'):
        fig = px.area(x=fpr, y=tpr, labels={'x': 'False positive rate', 'y': 'True positive rate'},
                      color_discrete_sequence=[BLUE])
        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color=GREY))
        st.plotly_chart(style(fig, 320), width='stretch')
    comp = pd.DataFrame({f"Chosen threshold ({r['threshold']:.2f})": tm, 'Default threshold (0.50)': t05}).T
    st.write('**Threshold choice changes the trade-off, not the ranking quality:**')
    st.dataframe(comp[['Precision', 'Recall', 'F1', 'Contact rate', 'TP', 'FP', 'FN']]
                 .style.format({'Precision': '{:.1%}', 'Recall': '{:.1%}', 'F1': '{:.3f}', 'Contact rate': '{:.1%}',
                                'TP': '{:.0f}', 'FP': '{:.0f}', 'FN': '{:.0f}'}), width='stretch')

with fairness:
    st.subheader('Does the model treat groups differently? (test set)')
    st.write('Gender, age and nationality-linked attributes are sensitive. The table checks whether the model ranks '
             'each group equally well (ROC-AUC) and whether it reaches churners (recall) or wrongly flags loyal '
             'customers (false positive rate) at similar rates.')
    sub = r['subgroups']
    pct_cols = ['Actual churn rate', 'Mean predicted risk', 'Contact rate', 'Recall', 'False positive rate']
    st.dataframe(sub.style.format({**{c: '{:.1%}' for c in pct_cols}, 'ROC-AUC': '{:.3f}'}),
                 width='stretch', hide_index=True)
    fig = px.bar(sub.melt(id_vars='Group', value_vars=['Recall', 'False positive rate']), x='Group', y='value',
                 color='variable', barmode='group', color_discrete_map={'Recall': BLUE, 'False positive rate': ORANGE},
                 labels={'value': 'Rate', 'variable': ''}, title='Recall and false positive rate by group')
    fig.update_yaxes(tickformat='.0%')
    st.plotly_chart(style(fig, 380), width='stretch')
    st.markdown(
        '- **Gender:** with Gender removed as an input, recall and false positive rates are almost identical for '
        'men and women. Average predicted risk for women is still a little below their actual churn rate, so the '
        'gap in churn is not fully explained by the other features.\n'
        '- **Geography:** German customers are contacted about twice as often and have a higher false positive '
        'rate. Their churn rate really is higher, but a campaign should check whether flagged German customers are '
        'getting fair offers rather than more pressure.\n'
        '- **Age:** the model is weakest for under-40s (lower ROC-AUC and recall). Young churners are rare and harder '
        'to tell apart, so a separate approach may be needed for them.\n'
        '- Age is a protected characteristic in the UK. It is kept here because it is a strong predictor for a '
        'low-stakes retention offer, but any real use would need a compliance review.')

with explain:
    st.subheader('Score a customer and see why')
    l, rcol = st.columns(2)
    with l:
        creditscore = st.slider('Credit score', 350, 850, 650)
        geography = st.selectbox('Country', sorted(df['Geography'].unique()))
        gender = st.selectbox('Gender', sorted(df['Gender'].unique()), disabled=not uses_gender,
                              help=None if uses_gender else 'Not used by the final model (see Model selection).')
        age = st.slider('Age', 18, 92, 45)
        tenure = st.slider('Tenure (years)', 0, 10, 5)
    with rcol:
        balance = st.number_input('Account balance', min_value=0.0, value=100000.0, step=5000.0)
        products = st.slider('Number of products', 1, 4, 1)
        card = st.selectbox('Has credit card', ['Yes', 'No'])
        active = st.selectbox('Active member', ['No', 'Yes'])
        salary = st.number_input('Estimated salary', min_value=0.0, value=75000.0, step=5000.0)
    record = pd.DataFrame([{'CreditScore': creditscore, 'Geography': geography, 'Gender': gender, 'Age': age,
                            'Tenure': tenure, 'Balance': balance, 'NumOfProducts': products,
                            'HasCrCard': int(card == 'Yes'), 'IsActiveMember': int(active == 'Yes'),
                            'EstimatedSalary': salary}])
    prob = float(model.predict_proba(record)[0, 1])
    a, b = st.columns(2)
    a.metric('Predicted churn probability', f'{prob:.1%}')
    b.metric('Campaign decision', 'Contact' if prob >= r['threshold'] else 'Do not contact',
             help=f"Contact if risk is at least {r['threshold']:.2f} (chosen in the Retention strategy tab).")
    values, base, _ = shap_explanation(model, record)
    contrib = values.iloc[0].sort_values(key=np.abs, ascending=False).head(8)[::-1]
    fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation='h',
                           marker_color=[ORANGE if v > 0 else BLUE for v in contrib.values]))
    fig.update_layout(title='SHAP contributions (log-odds): orange raises risk, blue lowers it', xaxis_title='Effect on log-odds')
    st.plotly_chart(style(fig, 380), width='stretch')
    st.caption('SHAP values show how this model reached its score, not what would cause the customer to leave.')

with strategy:
    st.subheader('Choosing who to contact')
    st.write('A retention offer costs money. The threshold is chosen to **maximise campaign value** using '
             'out-of-fold predictions on the training data, then checked on the test set. All economics are '
             'illustrative assumptions.')
    c1, c2, c3 = st.columns(3)
    econ = {'cost_per_contact': c1.number_input('Cost per contact (£)', 0.5, 100.0, r['economics']['cost_per_contact'], 0.5),
            'value_per_save': c2.number_input('Value of keeping a customer (£)', 1.0, 5000.0, r['economics']['value_per_save'], 10.0),
            'save_rate': c3.slider('Share of contacted churners who stay', 0.05, 1.0, r['economics']['save_rate'], 0.05)}
    key = (r['final_name'], r['final_features'])
    thr, oof_curve = choose_threshold(r['y_train'], r['oof'][key], econ)
    test_curve = threshold_curve(r['y_test'], r['test_prob'], econ)
    n_tr, n_te = len(r['y_train']), len(r['y_test'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=oof_curve['threshold'], y=oof_curve['net_benefit'] / n_tr * 1000,
                             name='Training data (out-of-fold): used to choose', line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=test_curve['threshold'], y=test_curve['net_benefit'] / n_te * 1000,
                             name='Test set: checked afterwards', line=dict(color=ORANGE, width=2)))
    fig.add_vline(x=thr, line_dash='dash', line_color=GREY, annotation_text=f'chosen {thr:.2f}')
    fig.update_layout(title='Net benefit per 1,000 customers', xaxis_title='Contact if predicted risk is at least',
                      yaxis_title='£ per 1,000 customers')
    st.plotly_chart(style(fig, 380), width='stretch')
    m = classification_metrics(r['y_test'], r['test_prob'], thr)
    a, b, c, d = st.columns(4)
    a.metric('Chosen threshold', f'{thr:.2f}', help=f'Theoretical break-even for a calibrated model: '
             f'cost ÷ (save rate × value) = {breakeven_threshold(econ):.2f}')
    b.metric('Customers contacted (test)', f"{m['TP'] + m['FP']:,}")
    c.metric('Churners reached', f"{m['Recall']:.0%}")
    d.metric('Test net benefit', f"£{net_benefit(r['y_test'], r['test_prob'], thr, **econ):,.0f}")
    comp = pd.DataFrame({'Strategy': ['Contact nobody', 'Contact everyone', 'Model, threshold 0.50',
                                      f'Model, chosen threshold {thr:.2f}', f'Baseline logistic regression, {thr:.2f}'],
                         'Test net benefit (£)': [0, net_benefit(r['y_test'], r['test_prob'], 0.0, **econ),
                                                  net_benefit(r['y_test'], r['test_prob'], 0.5, **econ),
                                                  net_benefit(r['y_test'], r['test_prob'], thr, **econ),
                                                  net_benefit(r['y_test'], r['baseline_prob'], thr, **econ)]})
    st.dataframe(comp.style.format({'Test net benefit (£)': '£{:,.0f}'}), width='stretch', hide_index=True)
    st.caption('Assumes the offer only changes the behaviour of customers who would otherwise churn, at a fixed '
               'save rate. Real uplift would need a controlled experiment (A/B test).')

st.divider()
st.caption('Data: public "Bank Churn Modelling" teaching dataset (YBI Foundation GitHub). '
           'Not production-ready. See README for methods and limitations.')
