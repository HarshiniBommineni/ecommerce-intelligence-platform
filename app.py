import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="E-Commerce Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');
* { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
.main .block-container { padding: 2rem 3rem; max-width: 1400px; }



.dash-header { border-bottom: 1px solid #1e1e2e; padding-bottom: 1.5rem; margin-bottom: 2rem; }
.dash-title { font-size: 2rem; font-weight: 600; color: #ffffff; letter-spacing: -0.5px; margin: 0; }
.dash-subtitle { font-size: 0.85rem; color: #6b6b8a; margin-top: 4px; font-family: 'DM Mono', monospace; letter-spacing: 0.5px; }

.kpi-card { background: #111118; border: 1px solid #1e1e2e; border-radius: 12px; padding: 1.25rem 1.5rem; position: relative; overflow: hidden; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.kpi-label { font-size: 0.75rem; color: #6b6b8a; text-transform: uppercase; letter-spacing: 1px; font-weight: 500; margin-bottom: 8px; }
.kpi-value { font-size: 1.75rem; font-weight: 600; color: #ffffff; line-height: 1; }
.kpi-sub { font-size: 0.75rem; color: #4ade80; margin-top: 6px; }

.section-header { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; color: #6b6b8a; font-weight: 500; margin-bottom: 1rem; margin-top: 2rem; }

.insight-box { background: #0d1117; border: 1px solid #1e1e2e; border-left: 3px solid #6366f1; border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 0.75rem; }
.insight-title { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #6366f1; font-weight: 600; margin-bottom: 6px; }
.insight-text { font-size: 13px; color: #c8c8e0; line-height: 1.6; }

.rec-box { background: #0d1117; border: 1px solid #1e1e2e; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 0.75rem; }
.rec-priority-high { border-left: 3px solid #f87171; }
.rec-priority-med { border-left: 3px solid #fbbf24; }
.rec-priority-low { border-left: 3px solid #4ade80; }
.rec-tag { font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 6px; }
.rec-title { font-size: 14px; font-weight: 600; color: #ffffff; margin-bottom: 4px; }
.rec-text { font-size: 13px; color: #9999bb; line-height: 1.6; }
.rec-impact { font-size: 11px; margin-top: 8px; padding: 3px 10px; border-radius: 99px; display: inline-block; }

.forecast-box { background: #0d1117; border: 1px solid #1e1e2e; border-left: 3px solid #4ade80; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.5rem; }

.badge-champion { background: #14532d; color: #4ade80; padding: 2px 10px; border-radius: 99px; font-size: 12px; font-weight: 500; }
.badge-vip { background: #422006; color: #fbbf24; padding: 2px 10px; border-radius: 99px; font-size: 12px; font-weight: 500; }
.badge-regular { background: #1e1b4b; color: #818cf8; padding: 2px 10px; border-radius: 99px; font-size: 12px; font-weight: 500; }
.badge-lost { background: #450a0a; color: #f87171; padding: 2px 10px; border-radius: 99px; font-size: 12px; font-weight: 500; }

.tech-badge { display: inline-block; background: #111118; border: 1px solid #1e1e2e; border-radius: 6px; padding: 4px 10px; font-size: 11px; font-family: 'DM Mono', monospace; color: #6b6b8a; margin: 3px; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stMetric"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/data.csv', encoding='ISO-8859-1')
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['Hour'] = df['InvoiceDate'].dt.hour
    return df

@st.cache_data
def load_segments():
    return pd.read_csv('data/customer_segments.csv')

def build_forecast(df, months_ahead=6):
    from sklearn.linear_model import LinearRegression
    monthly = df.groupby('Month')['TotalPrice'].sum().reset_index()
    monthly['MonthNum'] = range(len(monthly))
    X = monthly[['MonthNum']].values
    y = monthly['TotalPrice'].values
    model = LinearRegression()
    model.fit(X, y)
    last_period = pd.Period(monthly['Month'].iloc[-1], 'M')
    future_months = [str(last_period + i) for i in range(1, months_ahead + 1)]
    future_nums = np.array([[len(monthly) + i] for i in range(1, months_ahead + 1)])
    predictions = np.maximum(model.predict(future_nums), 0)
    forecast_df = pd.DataFrame({
        'Month': future_months,
        'TotalPrice': predictions,
        'Type': 'Forecast'
    })
    historical_df = pd.DataFrame({
        'Month': monthly['Month'],
        'TotalPrice': monthly['TotalPrice'],
        'Type': 'Historical'
    })
    return pd.concat([historical_df, forecast_df], ignore_index=True), predictions

@st.cache_data
def build_churn_model(df):
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()
    def churn_risk(row):
        if row['Recency'] > 180 or (row['Frequency'] == 1 and row['Recency'] > 90):
            return 'High Risk'
        elif row['Recency'] > 90 or row['Frequency'] <= 2:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    rfm['ChurnRisk'] = rfm.apply(churn_risk, axis=1)
    return rfm

df_full = load_data()
segments = load_segments()
churn_df = build_churn_model(df_full)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-size:13px;font-weight:600;color:#ffffff;
    margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid #1e1e2e;'>
    📊 Dashboard Controls
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#6b6b8a;font-weight:500;margin-bottom:8px;">Data Filters</div>', unsafe_allow_html=True)

    countries = ['All Countries'] + sorted(df_full['Country'].unique().tolist())
    selected_country = st.selectbox("🌍 Country", countries)

    months_list = sorted(df_full['Month'].unique().tolist())
    date_range = st.select_slider(
        "📅 Date Range",
        options=months_list,
        value=(months_list[0], months_list[-1])
    )

    segments_list = ['All Segments'] + sorted(segments['Segment'].unique().tolist())
    selected_segment = st.selectbox("👥 Customer Segment", segments_list)

    st.markdown('<div style="margin-top:1.5rem;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#6b6b8a;font-weight:500;margin-bottom:8px;">Forecast Settings</div>', unsafe_allow_html=True)

    forecast_months = st.slider(
        "📈 Months to Forecast",
        min_value=1, max_value=12, value=6, step=1,
        help="Drag to predict further into the future"
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:12px;color:#6b6b8a;line-height:2;'>
    <span style='color:#6366f1;font-weight:600;font-size:11px;
    text-transform:uppercase;letter-spacing:1px;'>Active Filters</span><br>
    <span style='color:#9999bb;'>🌍</span> {selected_country}<br>
    <span style='color:#9999bb;'>📅</span> {date_range[0]} → {date_range[1]}<br>
    <span style='color:#9999bb;'>👥</span> {selected_segment}<br>
    <span style='color:#9999bb;'>📈</span> {forecast_months} month forecast
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#6b6b8a;font-weight:500;margin-bottom:8px;">Export Data</div>', unsafe_allow_html=True)

    csv_buffer = io.StringIO()
    segments.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Segments CSV",
        data=csv_buffer.getvalue(),
        file_name="customer_segments.csv",
        mime="text/csv",
        use_container_width=True
    )

    churn_buffer = io.StringIO()
    churn_df.to_csv(churn_buffer, index=False)
    st.download_button(
        label="⬇️ Download Churn Report",
        data=churn_buffer.getvalue(),
        file_name="churn_report.csv",
        mime="text/csv",
        use_container_width=True
    )

# ── Apply Filters ─────────────────────────────────────────
df = df_full.copy()
if selected_country != 'All Countries':
    df = df[df['Country'] == selected_country]
df = df[(df['Month'] >= date_range[0]) & (df['Month'] <= date_range[1])]
if selected_segment != 'All Segments':
    filtered_ids = segments[segments['Segment'] == selected_segment]['CustomerID'].values
    df = df[df['CustomerID'].isin(filtered_ids)]

forecast_data, predictions = build_forecast(df_full, months_ahead=forecast_months)

PLOT_THEME = dict(
    paper_bgcolor='#111118', plot_bgcolor='#111118',
    font=dict(family='DM Sans', color='#9999bb', size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor='#1e1e2e', linecolor='#1e1e2e'),
    yaxis=dict(gridcolor='#1e1e2e', linecolor='#1e1e2e'),
)

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div class="dash-title">📊 E-Commerce Intelligence Platform</div>
  <div class="dash-subtitle">UK ONLINE RETAIL · 2010–2011 · 397,884 TRANSACTIONS · ML-POWERED ANALYTICS</div>
</div>
""", unsafe_allow_html=True)

# ── Computed Values ───────────────────────────────────────
total_rev = df['TotalPrice'].sum()
total_orders = df['InvoiceNo'].nunique()
total_customers = df['CustomerID'].nunique()
monthly_rev = df.groupby('Month')['TotalPrice'].sum()
peak_month = monthly_rev.idxmax() if len(monthly_rev) > 0 else 'N/A'
first_rev = monthly_rev.iloc[0] if len(monthly_rev) > 0 else 1
peak_rev = monthly_rev.max() if len(monthly_rev) > 0 else 0
pct_change = ((peak_rev - first_rev) / first_rev * 100) if first_rev > 0 else 0
uk_rev = df_full[df_full['Country'] == 'United Kingdom']['TotalPrice'].sum()
uk_pct = (uk_rev / df_full['TotalPrice'].sum() * 100)
vip_avg = segments[segments['Segment'] == 'VIP']['Monetary'].mean() if 'VIP' in segments['Segment'].values else 0
regular_avg = segments[segments['Segment'] == 'Regular']['Monetary'].mean() if 'Regular' in segments['Segment'].values else 1
vip_multiplier = vip_avg / regular_avg if regular_avg > 0 else 0
lost_count = len(segments[segments['Segment'] == 'Lost'])
lost_pct = lost_count / len(segments) * 100
forecast_total = predictions.sum()
forecast_avg = predictions.mean()
high_risk_count = len(churn_df[churn_df['ChurnRisk'] == 'High Risk'])
med_risk_count = len(churn_df[churn_df['ChurnRisk'] == 'Medium Risk'])
low_risk_count = len(churn_df[churn_df['ChurnRisk'] == 'Low Risk'])
total_c = len(churn_df)

# ── AI Executive Summary ──────────────────────────────────
st.markdown('<div class="section-header">AI Executive Summary</div>', unsafe_allow_html=True)
i1, i2 = st.columns(2)
with i1:
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">📈 Revenue Growth</div>
        <div class="insight-text">Revenue grew <strong style='color:#4ade80;'>{pct_change:.0f}%</strong>
        from first to peak month ({peak_month}), driven by seasonal Q4 purchasing.
        ML model forecasts <strong style='color:#4ade80;'>£{forecast_avg/1000:.0f}k avg/month</strong>
        over the next {forecast_months} months.</div>
    </div>
    <div class="insight-box">
        <div class="insight-title">🌍 Geographic Concentration Risk</div>
        <div class="insight-text">United Kingdom contributes
        <strong style='color:#4ade80;'>{uk_pct:.0f}%</strong>
        of total revenue across 38 countries. Heavy geographic concentration
        represents business risk — international expansion strongly recommended.</div>
    </div>
    """, unsafe_allow_html=True)
with i2:
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">💎 VIP Revenue Concentration</div>
        <div class="insight-text">VIP customers spend
        <strong style='color:#fbbf24;'>{vip_multiplier:.0f}x more</strong>
        than regular customers. Only 13 VIP accounts — losing even one
        represents significant revenue risk. Immediate retention focus required.</div>
    </div>
    <div class="insight-box">
        <div class="insight-title">⚠️ Churn Risk Alert</div>
        <div class="insight-text">
        <strong style='color:#f87171;'>{lost_pct:.1f}% of customers</strong>
        ({lost_count:,} total) are classified as lost. Churn model identifies
        <strong style='color:#f87171;'>{high_risk_count:,} high-risk</strong>
        customers needing immediate win-back campaigns.</div>
    </div>
    """, unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────
st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Revenue</div><div class="kpi-value">£{total_rev/1e6:.2f}M</div><div class="kpi-sub">↑ Peak Nov 2011</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Orders</div><div class="kpi-value">{total_orders:,}</div><div class="kpi-sub">↑ Avg 1,500/month</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Unique Customers</div><div class="kpi-value">{total_customers:,}</div><div class="kpi-sub">↑ 38 countries</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">High Churn Risk</div><div class="kpi-value" style="color:#f87171;">{high_risk_count:,}</div><div class="kpi-sub" style="color:#f87171;">↓ Needs attention</div></div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

# ── Revenue Trend + Segments ──────────────────────────────
st.markdown('<div class="section-header">Revenue Performance</div>', unsafe_allow_html=True)
c1, c2 = st.columns([3, 2])
with c1:
    monthly_chart = df.groupby('Month')['TotalPrice'].sum().reset_index()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=monthly_chart['Month'], y=monthly_chart['TotalPrice'],
        mode='lines+markers', name='Historical',
        line=dict(color='#6366f1', width=2.5),
        marker=dict(color='#8b5cf6', size=6),
        fill='tozeroy', fillcolor='rgba(99,102,241,0.08)',
        hovertemplate='<b>%{x}</b><br>£%{y:,.0f}<extra></extra>'
    ))
    fig1.update_layout(
        title=dict(text='Monthly Revenue Trend', font=dict(size=13, color='#e8e8f0')),
        **PLOT_THEME, yaxis_tickprefix='£', yaxis_tickformat=',.0f', height=280
    )
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    seg_counts = segments['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    colors = {'Regular': '#6366f1', 'Lost': '#f87171', 'Champions': '#4ade80', 'VIP': '#fbbf24'}
    fig2 = go.Figure(go.Pie(
        labels=seg_counts['Segment'], values=seg_counts['Count'],
        hole=0.65,
        marker_colors=[colors.get(s, '#888') for s in seg_counts['Segment']],
        textinfo='label+percent',
        textfont=dict(size=11, color='#e8e8f0'),
        hovertemplate='<b>%{label}</b><br>%{value} customers<extra></extra>'
    ))
    fig2.update_layout(
        title=dict(text='Customer Segments', font=dict(size=13, color='#e8e8f0')),
        **PLOT_THEME, showlegend=False, height=280
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Revenue Forecast ──────────────────────────────────────
st.markdown(f'<div class="section-header">ML Revenue Forecast — Next {forecast_months} Months</div>', unsafe_allow_html=True)
fig_forecast = go.Figure()
hist = forecast_data[forecast_data['Type'] == 'Historical']
fore = forecast_data[forecast_data['Type'] == 'Forecast']
fig_forecast.add_trace(go.Scatter(
    x=hist['Month'], y=hist['TotalPrice'],
    mode='lines+markers', name='Historical Revenue',
    line=dict(color='#6366f1', width=2.5), marker=dict(size=5),
    hovertemplate='<b>%{x}</b><br>£%{y:,.0f}<extra></extra>'
))
fig_forecast.add_trace(go.Scatter(
    x=fore['Month'], y=fore['TotalPrice'],
    mode='lines+markers', name='Forecasted Revenue',
    line=dict(color='#4ade80', width=2.5, dash='dash'),
    marker=dict(color='#4ade80', size=8, symbol='diamond'),
    hovertemplate='<b>%{x} (Forecast)</b><br>£%{y:,.0f}<extra></extra>'
))
if len(fore) > 0:
    fig_forecast.add_vrect(
        x0=fore['Month'].iloc[0], x1=fore['Month'].iloc[-1],
        fillcolor='rgba(74,222,128,0.05)', line_width=0,
        annotation_text=f"{forecast_months}-Month Forecast Zone",
        annotation_position="top left",
        annotation_font_color="#4ade80", annotation_font_size=11
    )
fig_forecast.update_layout(
    title=dict(text=f'Revenue Forecast — Linear Regression Model ({forecast_months} months ahead)',
               font=dict(size=13, color='#e8e8f0')),
    **PLOT_THEME, yaxis_tickprefix='£', yaxis_tickformat=',.0f', height=350,
    legend=dict(bgcolor='#111118', bordercolor='#1e1e2e', font=dict(color='#9999bb'))
)
st.plotly_chart(fig_forecast, use_container_width=True)

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown(f'<div class="forecast-box"><div style="font-size:11px;color:#4ade80;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">{forecast_months}-Month Forecast</div><div style="font-size:1.5rem;font-weight:600;color:#fff;">£{forecast_total/1e6:.2f}M</div><div style="font-size:12px;color:#6b6b8a;margin-top:4px;">Total predicted revenue</div></div>', unsafe_allow_html=True)
with f2:
    st.markdown(f'<div class="forecast-box"><div style="font-size:11px;color:#4ade80;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Monthly Average</div><div style="font-size:1.5rem;font-weight:600;color:#fff;">£{forecast_avg/1000:.0f}k</div><div style="font-size:12px;color:#6b6b8a;margin-top:4px;">Predicted per month</div></div>', unsafe_allow_html=True)
with f3:
    trend = "↑ Upward" if len(predictions) > 1 and predictions[-1] > predictions[0] else "↓ Downward"
    trend_color = "#4ade80" if "↑" in trend else "#f87171"
    st.markdown(f'<div class="forecast-box"><div style="font-size:11px;color:#4ade80;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Trend Direction</div><div style="font-size:1.5rem;font-weight:600;color:{trend_color};">{trend}</div><div style="font-size:12px;color:#6b6b8a;margin-top:4px;">Based on regression model</div></div>', unsafe_allow_html=True)

# ── Churn Prediction ──────────────────────────────────────
st.markdown('<div class="section-header">Customer Churn Prediction Model</div>', unsafe_allow_html=True)
ch1, ch2 = st.columns([2, 3])
with ch1:
    churn_counts = churn_df['ChurnRisk'].value_counts().reset_index()
    churn_counts.columns = ['Risk', 'Count']
    churn_colors_map = {'High Risk': '#f87171', 'Medium Risk': '#fbbf24', 'Low Risk': '#4ade80'}
    fig_churn = go.Figure(go.Bar(
        x=churn_counts['Count'], y=churn_counts['Risk'], orientation='h',
        marker_color=[churn_colors_map.get(r, '#888') for r in churn_counts['Risk']],
        hovertemplate='<b>%{y}</b><br>%{x} customers<extra></extra>'
    ))
    fig_churn.update_layout(
        title=dict(text='Churn Risk Distribution', font=dict(size=13, color='#e8e8f0')),
        **PLOT_THEME, height=280
    )
    st.plotly_chart(fig_churn, use_container_width=True)

with ch2:
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f'<div class="kpi-card" style="border-left:3px solid #f87171;"><div class="kpi-label">High Risk</div><div class="kpi-value" style="color:#f87171;">{high_risk_count:,}</div><div class="kpi-sub" style="color:#f87171;">{high_risk_count/total_c*100:.1f}%</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="kpi-card" style="border-left:3px solid #fbbf24;"><div class="kpi-label">Medium Risk</div><div class="kpi-value" style="color:#fbbf24;">{med_risk_count:,}</div><div class="kpi-sub" style="color:#fbbf24;">{med_risk_count/total_c*100:.1f}%</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="kpi-card" style="border-left:3px solid #4ade80;"><div class="kpi-label">Low Risk</div><div class="kpi-value" style="color:#4ade80;">{low_risk_count:,}</div><div class="kpi-sub" style="color:#4ade80;">{low_risk_count/total_c*100:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#6b6b8a;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Search Customer Churn Risk</div>', unsafe_allow_html=True)
    s1, s2 = st.columns([2, 1])
    with s1:
        search_id = st.text_input("Customer ID", placeholder="e.g. 12347", label_visibility="collapsed")
    with s2:
        risk_filter = st.selectbox("Risk", ['All', 'High Risk', 'Medium Risk', 'Low Risk'], label_visibility="collapsed")
    display_df = churn_df.copy()
    if search_id:
        display_df = display_df[display_df['CustomerID'].astype(str).str.contains(search_id)]
    if risk_filter != 'All':
        display_df = display_df[display_df['ChurnRisk'] == risk_filter]
    display_df = display_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'ChurnRisk']].head(10)
    display_df['Monetary'] = display_df['Monetary'].apply(lambda x: f'£{x:,.0f}')
    display_df['Recency'] = display_df['Recency'].apply(lambda x: f'{x} days')
    st.dataframe(
        display_df.rename(columns={
            'CustomerID': 'Customer ID', 'Recency': 'Last Purchase',
            'Frequency': 'Orders', 'Monetary': 'Total Spend', 'ChurnRisk': 'Churn Risk'
        }),
        use_container_width=True, height=200, hide_index=True
    )

# ── Geographic + Products ─────────────────────────────────
st.markdown('<div class="section-header">Geographic & Product Intelligence</div>', unsafe_allow_html=True)
d1, d2 = st.columns(2)
with d1:
    country_data = df.groupby('Country')['TotalPrice'].sum().reset_index()
    country_data = country_data.sort_values('TotalPrice', ascending=True).tail(10)
    fig3 = go.Figure(go.Bar(
        x=country_data['TotalPrice'], y=country_data['Country'], orientation='h',
        marker=dict(color=country_data['TotalPrice'],
                    colorscale=[[0, '#1e1b4b'], [1, '#6366f1']], showscale=False),
        hovertemplate='<b>%{y}</b><br>£%{x:,.0f}<extra></extra>'
    ))
    fig3.update_layout(
        title=dict(text='Top 10 Countries by Revenue', font=dict(size=13, color='#e8e8f0')),
        **PLOT_THEME, height=300
    )
    st.plotly_chart(fig3, use_container_width=True)

with d2:
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=True).tail(10).reset_index()
    fig4 = go.Figure(go.Bar(
        x=top_products['Quantity'], y=top_products['Description'], orientation='h',
        marker=dict(color=top_products['Quantity'],
                    colorscale=[[0, '#14532d'], [1, '#4ade80']], showscale=False),
        hovertemplate='<b>%{y}</b><br>%{x:,} units<extra></extra>'
    ))
    fig4.update_layout(
        title=dict(text='Top 10 Best-Selling Products', font=dict(size=13, color='#e8e8f0')),
        **PLOT_THEME, height=300
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Behaviour Patterns ────────────────────────────────────
st.markdown('<div class="section-header">Customer Behaviour Patterns</div>', unsafe_allow_html=True)
b1, b2 = st.columns(2)
with b1:
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_rev = df.groupby('DayOfWeek')['TotalPrice'].sum().reindex(day_order).reset_index()
    day_rev.columns = ['Day', 'Revenue']
    fig5 = go.Figure(go.Bar(
        x=day_rev['Day'], y=day_rev['Revenue'],
        marker=dict(color=day_rev['Revenue'],
                    colorscale=[[0, '#1e1b4b'], [1, '#8b5cf6']], showscale=False),
        hovertemplate='<b>%{x}</b><br>£%{y:,.0f}<extra></extra>'
    ))
    fig5.update_layout(
        title=dict(text='Revenue by Day of Week', font=dict(size=13, color='#e8e8f0')),
        **PLOT_THEME, yaxis_tickprefix='£', yaxis_tickformat=',.0f', height=280
    )
    st.plotly_chart(fig5, use_container_width=True)

with b2:
    hour_rev = df.groupby('Hour')['TotalPrice'].sum().reset_index()
    fig6 = go.Figure(go.Scatter(
        x=hour_rev['Hour'], y=hour_rev['TotalPrice'],
        mode='lines+markers',
        line=dict(color='#f59e0b', width=2),
        marker=dict(color='#f59e0b', size=5),
        fill='tozeroy', fillcolor='rgba(245,158,11,0.08)',
        hovertemplate='<b>%{x}:00</b><br>£%{y:,.0f}<extra></extra>'
    ))
    fig6.update_layout(
        title=dict(text='Revenue by Hour of Day', font=dict(size=13, color='#e8e8f0')),
        paper_bgcolor='#111118', plot_bgcolor='#111118',
        font=dict(family='DM Sans', color='#9999bb', size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(gridcolor='#1e1e2e', linecolor='#1e1e2e',
                   tickprefix='£', tickformat=',.0f'),
        xaxis=dict(tickmode='linear', tick0=0, dtick=2,
                   gridcolor='#1e1e2e', linecolor='#1e1e2e'),
        height=280
    )
    st.plotly_chart(fig6, use_container_width=True)

# ── Segment Table ─────────────────────────────────────────
st.markdown('<div class="section-header">Customer Segment Breakdown</div>', unsafe_allow_html=True)
summary = segments.groupby('Segment').agg(
    Customers=('CustomerID', 'count'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
).round(1).reset_index()
badge_map = {
    'Champions': '<span class="badge-champion">Champions</span>',
    'VIP': '<span class="badge-vip">VIP</span>',
    'Regular': '<span class="badge-regular">Regular</span>',
    'Lost': '<span class="badge-lost">Lost</span>'
}
table_html = """<table style="width:100%;border-collapse:collapse;font-size:13px;">
<thead><tr style="border-bottom:1px solid #1e1e2e;color:#6b6b8a;font-size:11px;text-transform:uppercase;letter-spacing:1px;">
  <th style="padding:10px;text-align:left;">Segment</th>
  <th style="padding:10px;text-align:right;">Customers</th>
  <th style="padding:10px;text-align:right;">Avg Recency</th>
  <th style="padding:10px;text-align:right;">Avg Frequency</th>
  <th style="padding:10px;text-align:right;">Avg Spend</th>
</tr></thead><tbody>"""
for _, row in summary.iterrows():
    table_html += f"""<tr style="border-bottom:1px solid #1e1e2e;">
  <td style="padding:12px 10px;">{badge_map.get(row['Segment'], row['Segment'])}</td>
  <td style="padding:12px 10px;text-align:right;color:#e8e8f0;font-weight:500;">{int(row['Customers']):,}</td>
  <td style="padding:12px 10px;text-align:right;color:#9999bb;">{row['Avg_Recency']} days</td>
  <td style="padding:12px 10px;text-align:right;color:#9999bb;">{row['Avg_Frequency']} orders</td>
  <td style="padding:12px 10px;text-align:right;color:#4ade80;font-weight:500;">£{row['Avg_Monetary']:,.1f}</td>
</tr>"""
table_html += "</tbody></table>"
st.markdown(table_html, unsafe_allow_html=True)

# ── Business Recommendations ──────────────────────────────
st.markdown('<div class="section-header">Business Recommendations</div>', unsafe_allow_html=True)
rec1, rec2, rec3 = st.columns(3)
with rec1:
    st.markdown(f"""
    <div class="rec-box rec-priority-high">
        <div class="rec-tag" style="color:#f87171;">🔴 High Priority</div>
        <div class="rec-title">Launch VIP Retention Program</div>
        <div class="rec-text">Only 13 VIP customers generate {vip_multiplier:.0f}x average revenue.
        Implement dedicated account managers, early product access, and
        personalised outreach to protect this revenue concentration.</div>
        <span class="rec-impact" style="background:#450a0a;color:#f87171;">Impact: Revenue Protection</span>
    </div>""", unsafe_allow_html=True)
with rec2:
    st.markdown(f"""
    <div class="rec-box rec-priority-high">
        <div class="rec-tag" style="color:#f87171;">🔴 High Priority</div>
        <div class="rec-title">Win-Back Campaign for {high_risk_count:,} At-Risk Customers</div>
        <div class="rec-text">Churn model identifies {high_risk_count:,} high-risk customers
        with no purchases in 150+ days. Deploy targeted email campaign with
        10-15% discount incentive to re-engage before permanent loss.</div>
        <span class="rec-impact" style="background:#450a0a;color:#f87171;">Impact: Customer Recovery</span>
    </div>""", unsafe_allow_html=True)
with rec3:
    st.markdown(f"""
    <div class="rec-box rec-priority-med">
        <div class="rec-tag" style="color:#fbbf24;">🟡 Medium Priority</div>
        <div class="rec-title">Expand International Markets</div>
        <div class="rec-text">UK drives {uk_pct:.0f}% of revenue from 38 countries.
        Netherlands and Germany show strong growth potential. Localised
        marketing in top 5 international markets could reduce geographic risk.</div>
        <span class="rec-impact" style="background:#422006;color:#fbbf24;">Impact: Revenue Diversification</span>
    </div>""", unsafe_allow_html=True)

rec4, rec5, rec6 = st.columns(3)
with rec4:
    st.markdown("""
    <div class="rec-box rec-priority-med">
        <div class="rec-tag" style="color:#fbbf24;">🟡 Medium Priority</div>
        <div class="rec-title">Optimise Thursday Marketing</div>
        <div class="rec-text">Thursday generates peak weekly revenue. Schedule
        email campaigns, promotions, and new product launches on
        Tuesday–Thursday to maximise customer engagement during
        peak purchasing windows.</div>
        <span class="rec-impact" style="background:#422006;color:#fbbf24;">Impact: Revenue Optimisation</span>
    </div>""", unsafe_allow_html=True)
with rec5:
    st.markdown("""
    <div class="rec-box rec-priority-low">
        <div class="rec-tag" style="color:#4ade80;">🟢 Growth Opportunity</div>
        <div class="rec-title">Convert Regular → Champion Customers</div>
        <div class="rec-text">3,054 Regular customers represent the largest segment.
        A loyalty programme with purchase milestones could move even 5%
        into the Champions tier — generating significant incremental revenue.</div>
        <span class="rec-impact" style="background:#14532d;color:#4ade80;">Impact: Customer Upgrade</span>
    </div>""", unsafe_allow_html=True)
with rec6:
    st.markdown("""
    <div class="rec-box rec-priority-low">
        <div class="rec-tag" style="color:#4ade80;">🟢 Growth Opportunity</div>
        <div class="rec-title">Weekend Revenue Activation</div>
        <div class="rec-text">Saturday and Sunday show significantly lower revenue
        than weekdays, suggesting untapped consumer potential. Weekend-exclusive
        promotions could unlock a new revenue stream.</div>
        <span class="rec-impact" style="background:#14532d;color:#4ade80;">Impact: New Revenue Stream</span>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid #1e1e2e;">
    <div style="font-size:11px;color:#6b6b8a;font-family:'DM Mono',monospace;
    margin-bottom:12px;text-transform:uppercase;letter-spacing:1px;">Tech Stack</div>
    <div>
        <span class="tech-badge">Python 3.11</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">Pandas</span>
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">Linear Regression</span>
        <span class="tech-badge">K-Means Clustering</span>
        <span class="tech-badge">RFM Analysis</span>
        <span class="tech-badge">Churn Prediction</span>
    </div>
    <div style="margin-top:1rem;font-size:11px;color:#3d3d5c;
    font-family:'DM Mono',monospace;display:flex;justify-content:space-between;">
        <span>E-COMMERCE INTELLIGENCE PLATFORM</span>
        <span>BUILT WITH ❤️ USING PYTHON & STREAMLIT</span>
    </div>
</div>""", unsafe_allow_html=True)