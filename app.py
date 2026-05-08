# ============================================================
# HYDRAPHARM — CHURN PREDICTION SYSTEM
# Streamlit Application — v3.0 (English Dataset)
# Master's Thesis — Pharmaceutical Sector Churn Prediction
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Hydrapharm — Churn Prediction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #F8F9FB; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 4px solid #2C3E50;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .metric-card.churn { border-left-color: #E74C3C; }
    .metric-card.safe  { border-left-color: #27AE60; }
    .metric-card.warn  { border-left-color: #F39C12; }
    .metric-card.blue  { border-left-color: #2980B9; }
    .metric-card h3 {
        font-size: 12px; color: #7F8C8D; margin: 0 0 4px 0;
        font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .metric-card h2 {
        font-size: 26px; font-weight: 700; color: #2C3E50; margin: 0;
        font-family: 'IBM Plex Mono', monospace;
    }

    .prediction-churn {
        background: linear-gradient(135deg, #FDEDEC, #FADBD8);
        border: 2px solid #E74C3C; border-radius: 16px;
        padding: 28px; text-align: center; margin: 16px 0;
    }
    .prediction-safe {
        background: linear-gradient(135deg, #EAFAF1, #D5F5E3);
        border: 2px solid #27AE60; border-radius: 16px;
        padding: 28px; text-align: center; margin: 16px 0;
    }
    .prediction-churn h1, .prediction-safe h1 { font-size: 44px; margin: 0; }
    .prediction-churn h2, .prediction-safe h2 { font-size: 20px; margin: 8px 0 4px 0; font-weight: 700; }
    .prediction-churn p,  .prediction-safe p  { font-size: 13px; color: #555; margin: 0; }

    .section-title {
        font-size: 16px; font-weight: 700; color: #2C3E50;
        border-bottom: 2px solid #ECF0F1;
        padding-bottom: 8px; margin: 20px 0 14px 0;
    }
    .stButton>button {
        background-color: #2C3E50; color: white; border: none;
        border-radius: 8px; padding: 12px 32px; font-size: 15px;
        font-weight: 600; width: 100%;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stButton>button:hover { background-color: #1A252F; }

    .sidebar-info {
        background: #EBF5FB; border-radius: 8px;
        padding: 12px 16px; font-size: 13px; color: #2C3E50; margin-top: 16px;
    }
    .risk-critical { background:#FADBD8; color:#E74C3C; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }
    .risk-high     { background:#FDEBD0; color:#E67E22; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }
    .risk-medium   { background:#FEF9E7; color:#F39C12; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }
    .risk-low      { background:#D5F5E3; color:#27AE60; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }

    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ── COLORS ────────────────────────────────────────────────────
COLORS = {
    'primary'  : '#2C3E50',
    'churn'    : '#E74C3C',
    'no_churn' : '#2980B9',
    'accent'   : '#27AE60',
    'gold'     : '#F39C12',
    'light'    : '#ECF0F1'
}

plt.rcParams.update({
    'figure.facecolor' : 'white',
    'axes.facecolor'   : '#FAFAFA',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.family'      : 'DejaVu Sans',
    'axes.grid'        : True,
    'grid.color'       : '#E0E0E0',
    'grid.linewidth'   : 0.6,
})

# ── Load data & model ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('hydrapharm_final_dataset_english.csv')
    df['Region']  = df['Region'].fillna('Unknown')
    df['Segment'] = df['Segment'].fillna('Unknown')
    return df

@st.cache_resource
def load_model():
    model  = joblib.load('gradient_boosting_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('feature_names.json') as f:
        features = json.load(f)
    return model, scaler, features

try:
    df = load_data()
    model, scaler, feature_names = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading files: {e}")
    st.info("Make sure these files are in the same folder as app.py:\n"
            "- hydrapharm_final_dataset_english.csv\n"
            "- gradient_boosting_model.pkl\n"
            "- scaler.pkl\n"
            "- feature_names.json")
    model_loaded = False
    st.stop()

# ── Pre-computed KPIs ─────────────────────────────────────────
total             = len(df)
churners          = int(df['Churn'].sum())
retained          = total - churners
churn_rate        = churners / total * 100
avg_rev_churner   = df[df['Churn']==1]['Total_Revenue_DA'].mean()
avg_rev_retained  = df[df['Churn']==0]['Total_Revenue_DA'].mean()
avg_recency_churn = df[df['Churn']==1]['Recency_Days'].mean()

# ── Risk tier helper ──────────────────────────────────────────
def get_risk_tier(proba):
    if proba >= 0.75: return "Critical", "risk-critical", "🔴"
    if proba >= 0.50: return "High",     "risk-high",     "🟠"
    if proba >= 0.25: return "Medium",   "risk-medium",   "🟡"
    return                   "Low",      "risk-low",      "🟢"

# ── Build model input ─────────────────────────────────────────
def build_input(inputs: dict) -> pd.DataFrame:
    """Align user inputs with feature_names and return ready DataFrame."""
    input_df = pd.DataFrame([inputs])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[feature_names].fillna(0)

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💊 Hydrapharm")
    st.markdown("**Churn Prediction System**")
    st.markdown("*Pharmaceutical Sector — Algeria*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔮 Churn Predictor", "📋 Client Lookup",
         "📈 SHAP Analysis", "⬇️ Export"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"""
    <div class="sidebar-info">
        <b>📦 Dataset Overview</b><br>
        {total:,} clients &nbsp;|&nbsp; {df['Region'].nunique()} regions<br>
        {df['Client_Type'].nunique()} client types<br><br>
        <b>🔴 Churn Rate</b><br>
        {churn_rate:.1f}% &nbsp;({churners:,} churners)<br><br>
        <b>🤖 Active Model</b><br>
        Gradient Boosting<br>
        AUC-ROC: 0.9965 &nbsp;|&nbsp; F1: 0.9545
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("# 📊 Churn Dashboard — Hydrapharm")
    st.markdown("Real-time overview of customer churn across the pharmaceutical portfolio.")
    st.markdown("---")

    # ── KPI Row ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card blue"><h3>Total Clients</h3><h2>{total:,}</h2></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card churn"><h3>Churners</h3><h2>{churners:,}</h2></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card safe"><h3>Retained</h3><h2>{retained:,}</h2></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card warn"><h3>Churn Rate</h3><h2>{churn_rate:.1f}%</h2></div>',
                    unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><h3>Avg Recency (Churners)</h3>'
                    f'<h2>{avg_recency_churn:.0f}d</h2></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Churn by Client Type + Region ──────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Churn Rate by Client Type</div>',
                    unsafe_allow_html=True)
        churn_ct = (df.groupby('Client_Type')['Churn']
                    .agg(Churn_Rate='mean', Total='count')
                    .reset_index())
        churn_ct['Churn_Rate'] *= 100
        churn_ct = churn_ct.sort_values('Churn_Rate', ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        norm   = plt.Normalize(churn_ct['Churn_Rate'].min(), churn_ct['Churn_Rate'].max())
        colors = [plt.cm.RdYlGn_r(norm(v)) for v in churn_ct['Churn_Rate']]
        bars   = ax.barh(churn_ct['Client_Type'], churn_ct['Churn_Rate'],
                         color=colors, edgecolor='white', height=0.55)
        for bar, val, tot in zip(bars, churn_ct['Churn_Rate'], churn_ct['Total']):
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%  (n={tot:,})', va='center', fontsize=9,
                    fontweight='bold', color=COLORS['primary'])
        ax.set_xlabel('Churn Rate (%)', fontsize=10)
        ax.set_xlim(0, churn_ct['Churn_Rate'].max() + 14)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown('<div class="section-title">Churn Rate by Region</div>',
                    unsafe_allow_html=True)
        churn_reg = (df[df['Region'] != 'Unknown']
                     .groupby('Region')['Churn']
                     .agg(Churn_Rate='mean', Total='count')
                     .reset_index())
        churn_reg['Churn_Rate'] *= 100
        churn_reg = churn_reg.sort_values('Churn_Rate', ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        norm   = plt.Normalize(churn_reg['Churn_Rate'].min(), churn_reg['Churn_Rate'].max())
        colors = [plt.cm.RdYlGn_r(norm(v)) for v in churn_reg['Churn_Rate']]
        bars   = ax.barh(churn_reg['Region'], churn_reg['Churn_Rate'],
                         color=colors, edgecolor='white', height=0.55)
        for bar, val, tot in zip(bars, churn_reg['Churn_Rate'], churn_reg['Total']):
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%  (n={tot:,})', va='center', fontsize=9,
                    fontweight='bold', color=COLORS['primary'])
        ax.set_xlabel('Churn Rate (%)', fontsize=10)
        ax.set_xlim(0, churn_reg['Churn_Rate'].max() + 14)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Row 2: Recency Distribution + RFM Score ───────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Recency Distribution by Churn Status</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[df['Churn']==0]['Recency_Days'], bins=50,
                alpha=0.65, color=COLORS['no_churn'], label='Retained', edgecolor='white')
        ax.hist(df[df['Churn']==1]['Recency_Days'], bins=50,
                alpha=0.65, color=COLORS['churn'],    label='Churner',  edgecolor='white')
        ax.axvline(90, color=COLORS['gold'], linestyle='--', linewidth=1.8,
                   label='Churn threshold (90 days)')
        ax.set_xlabel('Days Since Last Order', fontsize=10)
        ax.set_ylabel('Number of Clients',     fontsize=10)
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_d:
        st.markdown('<div class="section-title">RFM Score Distribution by Churn Status</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        rfm_churn = df[df['Churn']==1]['RFM_Score'].value_counts().sort_index()
        rfm_ret   = df[df['Churn']==0]['RFM_Score'].value_counts().sort_index()
        x = np.arange(3, 16)
        w = 0.4
        ax.bar(x - w/2, [rfm_ret.get(i, 0)   for i in x], w,
               color=COLORS['no_churn'], label='Retained', edgecolor='white', alpha=0.85)
        ax.bar(x + w/2, [rfm_churn.get(i, 0) for i in x], w,
               color=COLORS['churn'],    label='Churner',  edgecolor='white', alpha=0.85)
        ax.set_xlabel('RFM Score', fontsize=10)
        ax.set_ylabel('Number of Clients', fontsize=10)
        ax.set_xticks(x)
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Row 3: Churn by Segment + Convention Status ───────────
    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<div class="section-title">Churn Rate by Segment</div>',
                    unsafe_allow_html=True)
        churn_seg = (df[df['Segment'] != 'Unknown']
                     .groupby('Segment')['Churn']
                     .agg(Churn_Rate='mean', Total='count')
                     .reset_index())
        churn_seg['Churn_Rate'] *= 100
        churn_seg = churn_seg.sort_values('Churn_Rate', ascending=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        palette = [COLORS['churn'], '#E67E22', COLORS['gold'],
                   COLORS['no_churn'], COLORS['accent'], '#8E44AD', '#16A085']
        bars = ax.bar(churn_seg['Segment'], churn_seg['Churn_Rate'],
                      color=palette[:len(churn_seg)], edgecolor='white', width=0.5)
        for bar, val in zip(bars, churn_seg['Churn_Rate']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f'{val:.1f}%', ha='center', fontsize=9,
                    fontweight='bold', color=COLORS['primary'])
        ax.set_ylabel('Churn Rate (%)', fontsize=10)
        ax.set_ylim(0, churn_seg['Churn_Rate'].max() + 8)
        plt.xticks(rotation=20, ha='right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_f:
        st.markdown('<div class="section-title">Churn by Convention Status</div>',
                    unsafe_allow_html=True)
        conv_map   = {0: 'Non-Conventional', 1: 'Conventional'}
        churn_conv = (df.groupby('Convention_Status')['Churn']
                      .agg(Churn_Rate='mean', Total='count', Churners='sum')
                      .reset_index())
        churn_conv['Churn_Rate'] *= 100
        churn_conv['Label']       = churn_conv['Convention_Status'].map(conv_map)

        fig, axes = plt.subplots(1, 2, figsize=(7, 4))
        axes[0].bar(churn_conv['Label'], churn_conv['Churn_Rate'],
                    color=[COLORS['churn'], COLORS['accent']],
                    edgecolor='white', width=0.45)
        for i, (val, tot) in enumerate(zip(churn_conv['Churn_Rate'], churn_conv['Total'])):
            axes[0].text(i, val + 0.5, f'{val:.1f}%\n(n={tot:,})',
                         ha='center', fontsize=9, fontweight='bold')
        axes[0].set_ylabel('Churn Rate (%)', fontsize=10)
        axes[0].set_ylim(0, churn_conv['Churn_Rate'].max() + 12)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        non_ch = churn_conv['Total'] - churn_conv['Churners']
        axes[1].bar(churn_conv['Label'], non_ch,
                    color=COLORS['no_churn'], label='Retained', edgecolor='white', width=0.45)
        axes[1].bar(churn_conv['Label'], churn_conv['Churners'],
                    bottom=non_ch, color=COLORS['churn'],
                    label='Churners', edgecolor='white', width=0.45)
        axes[1].set_ylabel('Clients', fontsize=10)
        axes[1].legend(fontsize=8)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Row 4: Payment Method + Revenue comparison ────────────
    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown('<div class="section-title">Churn Rate by Payment Method</div>',
                    unsafe_allow_html=True)
        churn_pm = (df.groupby('Payment_Method')['Churn']
                    .agg(Churn_Rate='mean', Total='count')
                    .reset_index())
        churn_pm['Churn_Rate'] *= 100
        churn_pm = churn_pm.sort_values('Churn_Rate', ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        colors_pm = [COLORS['churn'], '#E67E22', COLORS['gold'],
                     COLORS['no_churn'], COLORS['accent']]
        bars = ax.barh(churn_pm['Payment_Method'], churn_pm['Churn_Rate'],
                       color=colors_pm[:len(churn_pm)], edgecolor='white', height=0.5)
        for bar, val, tot in zip(bars, churn_pm['Churn_Rate'], churn_pm['Total']):
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%  (n={tot:,})', va='center', fontsize=9,
                    fontweight='bold', color=COLORS['primary'])
        ax.set_xlabel('Churn Rate (%)', fontsize=10)
        ax.set_xlim(0, churn_pm['Churn_Rate'].max() + 14)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_h:
        st.markdown('<div class="section-title">Average Revenue — Churners vs Retained</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = ['Retained', 'Churners']
        values = [avg_rev_retained / 1e6, avg_rev_churner / 1e6]
        bars   = ax.bar(labels, values,
                        color=[COLORS['no_churn'], COLORS['churn']],
                        edgecolor='white', width=0.45)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}M DA', ha='center', fontsize=11,
                    fontweight='bold', color=COLORS['primary'])
        ax.set_ylabel('Average Revenue (Millions DA)', fontsize=10)
        ax.set_ylim(0, max(values) * 1.2)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════
# PAGE 2 — CHURN PREDICTOR
# ════════════════════════════════════════════════════════════
elif page == "🔮 Churn Predictor":
    st.markdown("# 🔮 Individual Churn Predictor")
    st.markdown("Enter client characteristics to predict churn probability using Gradient Boosting.")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="section-title">Client Profile</div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["📦 RFM & Orders", "💳 Payments & Blocks", "🏷️ Profile"])

        with tab1:
            recency_days      = st.number_input("Recency Days",                    0,    1000,  30)
            frequency_total   = st.number_input("Frequency Total (orders)",        0,    5000,  50)
            total_revenue     = st.number_input("Total Revenue (DA)",              0.0,  2e9,   5000000.0,  step=100000.0)
            avg_rev_per_trans = st.number_input("Avg Revenue / Transaction (DA)",  0.0,  1e7,   100000.0,   step=10000.0)
            regularity_rate   = st.slider("Regularity Rate",                       0.0,  1.0,   0.8,  0.01)
            avg_days_between  = st.number_input("Avg Days Between Orders",         0.0,  500.0, 10.0)
            rfm_score         = st.slider("RFM Score",                             3,    15,    9)
            revenue_evolution = st.number_input("Revenue Evolution (%)",          -100.0, 500.0, 5.0)

        with tab2:
            nb_blockages      = st.number_input("Number of Blockages",             0,    50,    0)
            pre_litigation    = st.selectbox("Pre-Litigation",                     [0, 1],
                                             format_func=lambda x: "Yes" if x==1 else "No")
            nb_late_payments  = st.number_input("Number of Late Payments",         0,    100,   0)
            avg_pay_delay     = st.number_input("Avg Payment Delay (days)",        0.0,  500.0, 30.0)
            outstanding_bal   = st.number_input("Outstanding Balance (DA)",        0.0,  1e9,   0.0,  step=10000.0)
            nb_cancelled      = st.number_input("Nb Cancelled / Returned Orders",  0,    100,   0)
            discount_rate     = st.slider("Discount Rate",                         0.0,  1.0,   0.0,  0.001)

        with tab3:
            client_type     = st.selectbox("Client Type", [
                'Pharmacy', 'Private Clinic', 'Wholesaler',
                'Public Hospital', 'Group Affiliate'
            ])
            region          = st.selectbox("Region", [
                'ALGIERS', 'CENTER', 'EAST', 'EAST_HP',
                'WEST', 'WEST_HP', 'SOUTH'
            ])
            segment         = st.selectbox("Segment", [
                'Core', 'Partner', 'Negotiator', 'GHP Affiliate',
                'CL-PARA', 'Hospital'
            ])
            payment_method  = st.selectbox("Payment Method", [
                'Cheque', 'Bill of Exchange', 'Bank Transfer', 'Cash', 'Unknown'
            ])
            conv_status     = st.selectbox("Convention Status", [0, 1],
                                           format_func=lambda x: "Conventional" if x==1
                                                                  else "Non-Conventional")
            seniority       = st.number_input("Client Seniority (months)",  0.0, 300.0, 24.0)
            payment_terms   = st.number_input("Payment Terms (days)",       0,   360,   90)
            credit_limit    = st.number_input("Credit Limit (DA)",          0.0, 1e9,   0.0, step=100000.0)

        predict_btn = st.button("🔮 Predict Churn")

    with col_result:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            raw = {
                'Frequency_Total'              : frequency_total,
                'Total_Revenue_DA'             : total_revenue,
                'Avg_Revenue_per_Transaction'  : avg_rev_per_trans,
                'Regularity_Rate'              : regularity_rate,
                'Avg_Days_Between_Orders'      : avg_days_between,
                'RFM_Score'                    : rfm_score,
                'Revenue_Evolution_Pct'        : revenue_evolution,
                'Nb_Blockages'                 : nb_blockages,
                'Pre_Litigation'               : pre_litigation,
                'Nb_Late_Payments'             : nb_late_payments,
                'Avg_Payment_Delay_Days'       : avg_pay_delay,
                'Outstanding_Balance_DA'       : outstanding_bal,
                'Nb_Cancelled_Returned_Orders' : nb_cancelled,
                'Discount_Rate'                : discount_rate,
                'Convention_Status_Enc'        : conv_status,
                'Pre_Litigation_Enc'           : pre_litigation,
                'Client_Seniority_Months'      : seniority,
                'Payment_Terms_Days'           : payment_terms,
                'Credit_Limit'                 : credit_limit,
            }

            # One-hot encode Client_Type
            for ct in ['Pharmacy','Private Clinic','Wholesaler','Public Hospital','Group Affiliate']:
                raw[f'Client_Type_{ct}'] = 1 if client_type == ct else 0

            # One-hot encode Payment_Method
            for pm in ['Cheque','Bill of Exchange','Bank Transfer','Cash','Unknown']:
                raw[f'Payment_Method_{pm}'] = 1 if payment_method == pm else 0

            input_df = build_input(raw)

            try:
                proba      = model.predict_proba(input_df)[0, 1]
                prediction = int(proba >= 0.5)
            except Exception:
                input_sc   = scaler.transform(input_df)
                proba      = model.predict_proba(input_sc)[0, 1]
                prediction = int(proba >= 0.5)

            tier, badge_class, emoji = get_risk_tier(proba)

            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-churn">
                    <h1>⚠️</h1>
                    <h2 style="color:#E74C3C;">CHURN RISK DETECTED</h2>
                    <p>This client is predicted to churn</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-safe">
                    <h1>✅</h1>
                    <h2 style="color:#27AE60;">CLIENT RETAINED</h2>
                    <p>This client is predicted to stay</p>
                </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-card {"churn" if prediction==1 else "safe"}">'
                            f'<h3>Churn Probability</h3>'
                            f'<h2>{proba*100:.1f}%</h2></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card warn">'
                            f'<h3>Risk Tier</h3>'
                            f'<h2>{emoji} {tier}</h2></div>', unsafe_allow_html=True)

            # Probability gauge
            st.markdown("**Churn Probability Gauge**")
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.barh([''], [1],     color='#ECF0F1', height=0.5)
            bar_color = COLORS['churn'] if proba >= 0.5 else COLORS['accent']
            ax.barh([''], [proba], color=bar_color, height=0.5)
            ax.set_xlim(0, 1)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.tick_params(left=False)
            ax.text(min(proba + 0.02, 0.92), 0, f'{proba*100:.1f}%',
                    va='center', fontweight='bold', color=bar_color, fontsize=12)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Recommendations
            st.markdown('<div class="section-title">💡 Action Recommendations</div>',
                        unsafe_allow_html=True)
            if prediction == 1:
                recs = []
                if recency_days > 90:
                    recs.append("📞 **Urgent contact** — Client inactive for over 3 months")
                if nb_blockages > 2:
                    recs.append("🚫 **Resolve blockages** — Multiple account blockages detected")
                if avg_pay_delay > 60:
                    recs.append("💳 **Payment follow-up** — Significant payment delays observed")
                if nb_late_payments > 3:
                    recs.append("📋 **Credit review** — Repeated late payment history")
                if revenue_evolution < -20:
                    recs.append("📉 **Revenue declining** — Schedule a commercial visit immediately")
                if rfm_score < 6:
                    recs.append("🎯 **Re-engagement campaign** — Very low RFM score")
                if outstanding_bal > 0:
                    recs.append("💰 **Outstanding balance** — Follow up on unpaid invoices")
                if pre_litigation == 1:
                    recs.append("⚖️ **Pre-litigation alert** — Legal team involvement may be required")
                if not recs:
                    recs.append("📋 **Preventive visit** — Assign sales rep for immediate follow-up")
                for r in recs:
                    st.markdown(f"- {r}")
            else:
                if proba > 0.25:
                    st.warning("⚠️ Client is stable but shows moderate risk signals. Monitor closely.")
                else:
                    st.success("✅ Client is stable and loyal. Continue regular relationship management.")
        else:
            st.info("👈 Fill in the form across the three tabs and click **Predict Churn**.")

# ════════════════════════════════════════════════════════════
# PAGE 3 — CLIENT LOOKUP
# ════════════════════════════════════════════════════════════
elif page == "📋 Client Lookup":
    st.markdown("# 📋 Client Lookup")
    st.markdown("Search and explore individual client profiles from the dataset.")
    st.markdown("---")

    col_search, col_filter = st.columns([2, 1])
    with col_search:
        search = st.text_input("🔍 Search by Client Code")
    with col_filter:
        filter_churn = st.selectbox("Filter by Status",
                                    ["All", "Churners only", "Retained only"])

    df_display = df.copy()
    if filter_churn == "Churners only":
        df_display = df_display[df_display['Churn'] == 1]
    elif filter_churn == "Retained only":
        df_display = df_display[df_display['Churn'] == 0]

    if search:
        client = df[df['Client_Code'].astype(str).str.upper() == search.strip().upper()]
        if len(client) == 0:
            st.error(f"Client '{search}' not found in the dataset.")
        else:
            row = client.iloc[0]
            churn_label = "🔴 CHURNER" if row['Churn'] == 1 else "🟢 RETAINED"

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card {"churn" if row["Churn"]==1 else "safe"}">'
                            f'<h3>Churn Status</h3><h2>{churn_label}</h2></div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card blue"><h3>Client Type</h3>'
                            f'<h2 style="font-size:14px;">{row["Client_Type"]}</h2></div>',
                            unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><h3>Region</h3>'
                            f'<h2>{row["Region"]}</h2></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card warn"><h3>RFM Score</h3>'
                            f'<h2>{int(row["RFM_Score"])}/15</h2></div>', unsafe_allow_html=True)

            st.markdown("---")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown('<div class="section-title">📦 Order Behavior</div>',
                            unsafe_allow_html=True)
                order_data = {
                    'Metric': ['Recency Days', 'Frequency Total', 'Regularity Rate',
                               'Avg Days Between Orders', 'Nb Distinct Products',
                               'Nb Cancelled Orders', 'RFM Score'],
                    'Value' : [f"{row['Recency_Days']:.0f} days",
                               f"{row['Frequency_Total']:.0f}",
                               f"{row['Regularity_Rate']:.3f}",
                               f"{row['Avg_Days_Between_Orders']:.1f} days",
                               f"{row['Nb_Distinct_Products']:.0f}",
                               f"{row['Nb_Cancelled_Returned_Orders']:.0f}",
                               f"{int(row['RFM_Score'])}/15"]
                }
                st.dataframe(pd.DataFrame(order_data), use_container_width=True, hide_index=True)

            with col_b:
                st.markdown('<div class="section-title">💰 Financial Profile</div>',
                            unsafe_allow_html=True)
                fin_data = {
                    'Metric': ['Total Revenue (DA)', 'Avg Revenue / Transaction',
                               'Revenue Evolution', 'Avg Payment Delay',
                               'Nb Late Payments', 'Outstanding Balance (DA)',
                               'Nb Blockages'],
                    'Value' : [f"{row['Total_Revenue_DA']:,.0f}",
                               f"{row['Avg_Revenue_per_Transaction']:,.0f}",
                               f"{row['Revenue_Evolution_Pct']:.1f}%",
                               f"{row['Avg_Payment_Delay_Days']:.1f} days",
                               f"{row['Nb_Late_Payments']:.0f}",
                               f"{row['Outstanding_Balance_DA']:,.0f}",
                               f"{row['Nb_Blockages']:.0f}"]
                }
                st.dataframe(pd.DataFrame(fin_data), use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">📋 Full Client Record</div>',
                        unsafe_allow_html=True)
            st.dataframe(client.T.rename(columns={client.index[0]: 'Value'}),
                         use_container_width=True)
    else:
        st.markdown('<div class="section-title">📊 Client Database</div>',
                    unsafe_allow_html=True)
        st.caption(f"Showing {len(df_display):,} clients")
        display_cols = ['Client_Code', 'Entity', 'Client_Type', 'Region', 'Segment',
                        'Recency_Days', 'Frequency_Total', 'Total_Revenue_DA',
                        'RFM_Score', 'Nb_Blockages', 'Payment_Method', 'Churn']
        st.dataframe(df_display[display_cols], use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 — SHAP ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "📈 SHAP Analysis":
    st.markdown("# 📈 SHAP Feature Importance")
    st.markdown("Explainability analysis of the Gradient Boosting model.")
    st.markdown("---")

    st.info("⏳ Computing SHAP values using TreeExplainer — optimized for Gradient Boosting...")

    try:
        df_model = df.copy()

        # Encode categoricals
        df_model['Convention_Status_Enc'] = df_model['Convention_Status'].astype(int)
        df_model['Pre_Litigation_Enc']    = df_model['Pre_Litigation'].astype(int)
        df_model = pd.get_dummies(df_model, columns=['Client_Type', 'Payment_Method'])

        drop_cols = [
            'Client_Code', 'Entity', 'Region', 'Segment',
            'Integration_Date', 'Last_Order_Date',
            'Convention_Status', 'Pre_Litigation',
            'Revenue_Period_N_DA', 'Revenue_Period_N1_DA',
            'Total_Credit_Notes_DA', 'Avg_Revenue_per_Transaction',
            'Total_Quantity', 'Total_Paid', 'Nb_Payments_Total',
            'R_Score', 'F_Score', 'M_Score',
            'Wilaya_Code', 'Payment_Terms_Days', 'Credit_Limit',
            'Frequency_6months', 'Recency_Days', 'Churn'
        ]
        df_model.drop(columns=[c for c in drop_cols if c in df_model.columns], inplace=True)

        for col in feature_names:
            if col not in df_model.columns:
                df_model[col] = 0
        X_all = df_model[feature_names].fillna(0)

        # TreeExplainer
        explainer  = shap.TreeExplainer(model)
        sample_idx = np.random.choice(len(X_all), size=min(300, len(X_all)), replace=False)
        X_sample   = X_all.iloc[sample_idx]
        shap_vals  = explainer.shap_values(X_sample)

        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        else:
            sv = shap_vals

        mean_abs    = np.abs(sv).mean(axis=0)
        mean_signed = sv.mean(axis=0)
        shap_df = pd.DataFrame({
            'Feature'   : feature_names,
            'Mean_SHAP' : mean_abs,
            'Direction' : ['↑ Increases Churn' if v > 0 else '↓ Decreases Churn'
                           for v in mean_signed]
        }).sort_values('Mean_SHAP', ascending=False).reset_index(drop=True)
        shap_df.index += 1

        t1, t2, t3 = st.tabs(["📊 Feature Importance", "🐝 Beeswarm Plot", "📋 SHAP Table"])

        with t1:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown('<div class="section-title">Top 15 Features — Mean |SHAP|</div>',
                            unsafe_allow_html=True)
                top15      = shap_df.head(15).sort_values('Mean_SHAP', ascending=True)
                colors_bar = ['#E74C3C' if '↑' in d else '#2980B9' for d in top15['Direction']]

                fig, ax = plt.subplots(figsize=(8, 7))
                bars = ax.barh(top15['Feature'], top15['Mean_SHAP'],
                               color=colors_bar, edgecolor='white', linewidth=1.2, height=0.6)
                for bar, val in zip(bars, top15['Mean_SHAP']):
                    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                            f'{val:.4f}', va='center', fontsize=8,
                            fontweight='bold', color=COLORS['primary'])
                red_p  = mpatches.Patch(color='#E74C3C', label='↑ Increases Churn Risk')
                blue_p = mpatches.Patch(color='#2980B9', label='↓ Decreases Churn Risk')
                ax.legend(handles=[red_p, blue_p], fontsize=9, loc='lower right')
                ax.set_xlabel('Mean |SHAP Value|', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            with col2:
                st.markdown('<div class="section-title">Top Features Summary</div>',
                            unsafe_allow_html=True)
                st.dataframe(shap_df[['Feature','Mean_SHAP','Direction']].head(15).round(5),
                             use_container_width=True)

        with t2:
            st.markdown('<div class="section-title">SHAP Beeswarm Plot — Top 15 Features</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(sv, X_sample, max_display=15, plot_type='dot',
                              show=False, alpha=0.6)
            plt.title('SHAP Summary — Gradient Boosting — Hydrapharm',
                      fontsize=13, fontweight='bold', color=COLORS['primary'])
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with t3:
            st.markdown('<div class="section-title">Complete SHAP Feature Importance Table</div>',
                        unsafe_allow_html=True)
            st.dataframe(shap_df.round(6), use_container_width=True)

    except Exception as e:
        st.error(f"SHAP computation error: {e}")
        st.info("Make sure the model file is a trained Gradient Boosting (sklearn) object.")

# ════════════════════════════════════════════════════════════
# PAGE 5 — EXPORT
# ════════════════════════════════════════════════════════════
elif page == "⬇️ Export":
    st.markdown("# ⬇️ Export Churn Scores")
    st.markdown("Score all clients and download the at-risk list.")
    st.markdown("---")

    st.markdown('<div class="section-title">Risk Tier Definitions</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card safe"><h3>Low Risk</h3><h2>0 – 25%</h2></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card warn"><h3>Medium Risk</h3><h2>25 – 50%</h2></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card churn"><h3>High Risk</h3><h2>50 – 75%</h2></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card churn"><h3>Critical</h3><h2>75 – 100%</h2></div>',
                    unsafe_allow_html=True)

    if st.button("🚀 Score All Clients & Generate Export"):
        with st.spinner(f"Scoring all {total:,} clients..."):
            try:
                df_model = df.copy()
                df_model['Convention_Status_Enc'] = df_model['Convention_Status'].astype(int)
                df_model['Pre_Litigation_Enc']    = df_model['Pre_Litigation'].astype(int)
                df_model = pd.get_dummies(df_model, columns=['Client_Type', 'Payment_Method'])

                drop_cols = [
                    'Client_Code', 'Entity', 'Region', 'Segment',
                    'Integration_Date', 'Last_Order_Date',
                    'Convention_Status', 'Pre_Litigation',
                    'Revenue_Period_N_DA', 'Revenue_Period_N1_DA',
                    'Total_Credit_Notes_DA', 'Avg_Revenue_per_Transaction',
                    'Total_Quantity', 'Total_Paid', 'Nb_Payments_Total',
                    'R_Score', 'F_Score', 'M_Score',
                    'Wilaya_Code', 'Payment_Terms_Days', 'Credit_Limit',
                    'Frequency_6months', 'Recency_Days', 'Churn'
                ]
                df_model.drop(columns=[c for c in drop_cols if c in df_model.columns],
                              inplace=True)
                for col in feature_names:
                    if col not in df_model.columns:
                        df_model[col] = 0
                X_all  = df_model[feature_names].fillna(0)
                probas = model.predict_proba(X_all)[:, 1]

                df_scored = df[['Client_Code', 'Entity', 'Client_Type', 'Region',
                                'Segment', 'Recency_Days', 'Frequency_Total',
                                'Total_Revenue_DA', 'RFM_Score', 'Nb_Blockages',
                                'Avg_Payment_Delay_Days', 'Outstanding_Balance_DA',
                                'Payment_Method', 'Churn']].copy()
                df_scored['Churn_Probability'] = probas.round(4)
                df_scored['Risk_Tier'] = pd.cut(
                    df_scored['Churn_Probability'],
                    bins=[0, 0.25, 0.50, 0.75, 1.0],
                    labels=['Low', 'Medium', 'High', 'Critical']
                )
                df_scored = df_scored.sort_values('Churn_Probability', ascending=False)

                tier_counts = df_scored['Risk_Tier'].value_counts()
                st.success(f"✅ {len(df_scored):,} clients scored successfully!")
                st.markdown("---")

                cc1, cc2, cc3, cc4 = st.columns(4)
                for col_w, tier, card_class in zip(
                    [cc1, cc2, cc3, cc4],
                    ['Critical', 'High', 'Medium', 'Low'],
                    ['churn', 'churn', 'warn', 'safe']
                ):
                    cnt = tier_counts.get(tier, 0)
                    with col_w:
                        st.markdown(f'<div class="metric-card {card_class}">'
                                    f'<h3>{tier}</h3><h2>{cnt:,}</h2></div>',
                                    unsafe_allow_html=True)

                st.markdown('<div class="section-title">Top 20 Highest Risk Clients</div>',
                            unsafe_allow_html=True)
                st.dataframe(df_scored.head(20), use_container_width=True, hide_index=True)

                csv = df_scored.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Full Scored Dataset (CSV)",
                    data=csv,
                    file_name="hydrapharm_churn_scores.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Scoring error: {e}")
    else:
        st.info("👆 Click the button above to score all clients and enable the download.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#BDC3C7; font-size:12px;'>"
    "💊 Hydrapharm Churn Prediction System &nbsp;|&nbsp; "
    "Master's Thesis — Machine Learning in Pharmaceutical Sector &nbsp;|&nbsp; Algeria 2026"
    "</p>",
    unsafe_allow_html=True
)
