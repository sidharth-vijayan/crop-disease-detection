# =============================================================================
# INCREMENT 4 — Streamlit Frontend Dashboard
# File    : dashboard.py
# Run     : streamlit run dashboard.py
# Requires: api.py running on http://localhost:8000
# =============================================================================
# pip install streamlit requests plotly pillow pandas numpy
# =============================================================================

import io, base64, json
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title = "Crop Disease Early Warning System",
    page_icon  = "🌿",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

API_URL = "http://localhost:8000"

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1a5c1a;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #4CAF50;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fff8; border: 1px solid #c8e6c9;
        border-radius: 10px; padding: 1rem;
        text-align: center;
    }
    .risk-high    { color: #d32f2f; font-weight: 700; font-size: 1.2rem; }
    .risk-moderate{ color: #f57c00; font-weight: 700; font-size: 1.2rem; }
    .risk-low     { color: #388e3c; font-weight: 700; font-size: 1.2rem; }
    .disease-tag  {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 0.85rem;
        font-weight: 600; margin: 2px;
    }
    .tag-high     { background: #ffebee; color: #c62828; }
    .tag-moderate { background: #fff3e0; color: #e65100; }
    .tag-low      { background: #e8f5e9; color: #2e7d32; }
    .intervention-box {
        background: #e3f2fd; border-left: 4px solid #1976d2;
        border-radius: 6px; padding: 1rem; margin: 0.5rem 0;
    }
    .stProgress > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False


def get_config():
    try:
        r = requests.get(f"{API_URL}/config", timeout=5)
        return r.json()
    except:
        return None


def risk_color(score):
    if score > 0.6:   return "#d32f2f"
    elif score > 0.3: return "#f57c00"
    else:             return "#388e3c"


def risk_label(score):
    if score > 0.6:   return "HIGH"
    elif score > 0.3: return "MODERATE"
    else:             return "LOW"


def b64_to_image(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def plot_7day_forecast(forecast_data):
    """Plotly chart for 7-day disease risk forecast."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    # dates can be top-level or inside each disease dict
    top_level_dates = forecast_data.get('dates', [])

    top_level_dates = forecast_data.get('dates', [])

    for i, disease_f in enumerate(forecast_data['top5'][:5]):
        name  = disease_f['disease'].replace('___', ' — ')
        daily = disease_f['daily']
        dates = (disease_f.get('dates')
                 or top_level_dates
                 or [f"Day {j+1}" for j in range(len(daily))])
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=dates, y=daily,
            name=name, mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=8),
            fill='tozeroy', fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.08)'),
            hovertemplate='%{y:.1%}<extra>' + name + '</extra>',
        ))

    fig.add_hline(y=0.6, line_dash='dash', line_color='red',
                  annotation_text='High risk threshold',
                  annotation_position='bottom right')
    fig.add_hline(y=0.3, line_dash='dot', line_color='orange',
                  annotation_text='Moderate risk threshold',
                  annotation_position='bottom right')

    fig.update_layout(
        title      = '7-Day Disease Risk Forecast',
        xaxis_title= 'Date',
        yaxis_title= 'Risk Score',
        yaxis      = dict(range=[0, 1], tickformat='.0%'),
        height     = 400,
        legend     = dict(orientation='h', yanchor='bottom',
                          y=1.02, xanchor='right', x=1),
        hovermode  = 'x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def plot_fusion_risks(top5_diseases):
    """Horizontal bar chart for fusion risk assessment."""
    diseases = [d['disease'].replace('___', '\n') for d in top5_diseases]
    probs    = [d['probability'] for d in top5_diseases]
    errors   = [d['uncertainty'] for d in top5_diseases]
    colors   = [risk_color(p) for p in probs]

    fig = go.Figure(go.Bar(
        x           = probs[::-1],
        y           = diseases[::-1],
        orientation = 'h',
        marker_color= colors[::-1],
        error_x     = dict(type='data', array=errors[::-1],
                           color='gray', thickness=1.5),
        text        = [f"{p:.1%}" for p in probs[::-1]],
        textposition= 'outside',
        hovertemplate='%{y}: %{x:.1%} ± %{error_x.array:.2f}<extra></extra>',
    ))

    fig.add_vline(x=0.6, line_dash='dash', line_color='red')
    fig.add_vline(x=0.3, line_dash='dot',  line_color='orange')

    fig.update_layout(
        title       = 'Fusion Risk Assessment (with uncertainty)',
        xaxis_title = 'Disease Probability',
        xaxis       = dict(range=[0, 1.1], tickformat='.0%'),
        height      = 320,
        plot_bgcolor= 'rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def plot_yield_impact(yield_data):
    """Gauge + bar for yield impact."""
    loss_pct = yield_data['loss_pct']

    fig = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = loss_pct * 100,
        title  = {'text': "Estimated Yield Loss (%)"},
        delta  = {'reference': 0},
        gauge  = {
            'axis'    : {'range': [0, 60]},
            'bar'     : {'color': risk_color(loss_pct)},
            'steps'   : [
                {'range': [0,  15], 'color': '#e8f5e9'},
                {'range': [15, 35], 'color': '#fff3e0'},
                {'range': [35, 60], 'color': '#ffebee'},
            ],
            'threshold': {
                'line' : {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 35,
            },
        },
        number = {'suffix': '%', 'font': {'size': 36}},
    ))
    fig.update_layout(height=250,
                      paper_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_financial(yield_data):
    """Stacked bar for financial impact vs treatment cost."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Financial Loss', x=['Impact'],
        y=[yield_data['financial_loss']],
        marker_color='#ef5350',
        text=[f"₹{yield_data['financial_loss']:,.0f}"],
        textposition='auto',
    ))
    fig.add_trace(go.Bar(
        name='Treatment Cost', x=['Impact'],
        y=[yield_data['treatment_cost']],
        marker_color='#42a5f5',
        text=[f"₹{yield_data['treatment_cost']:,.0f}"],
        textposition='auto',
    ))
    fig.add_trace(go.Bar(
        name='Value Saved', x=['Impact'],
        y=[yield_data['saved_value']],
        marker_color='#66bb6a',
        text=[f"₹{yield_data['saved_value']:,.0f}"],
        textposition='auto',
    ))
    fig.update_layout(
        title        = 'Financial Analysis',
        yaxis_title  = 'Amount (₹)',
        barmode      = 'group',
        height       = 280,
        plot_bgcolor = 'rgba(0,0,0,0)',
        paper_bgcolor= 'rgba(0,0,0,0)',
        legend       = dict(orientation='h'),
    )
    return fig


def plot_historical(historical_data):
    """Time series of historical disease risk."""
    dates  = pd.to_datetime(historical_data['dates'])
    scores = historical_data['scores']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=scores,
        fill='tozeroy', mode='lines',
        line=dict(color='#4CAF50', width=1.5),
        fillcolor='rgba(76,175,80,0.15)',
        name='Daily risk score',
    ))
    # 30-day rolling mean
    rolling = pd.Series(scores).rolling(30).mean()
    fig.add_trace(go.Scatter(
        x=dates, y=rolling,
        mode='lines', line=dict(color='#ff7043', width=2, dash='dot'),
        name='30-day average',
    ))
    fig.add_hline(y=0.6, line_dash='dash', line_color='red',
                  annotation_text='High risk')
    fig.add_hline(y=0.3, line_dash='dot', line_color='orange',
                  annotation_text='Moderate risk')

    disease_name = historical_data['disease'].replace('___', ' — ')
    fig.update_layout(
        title       = f'Historical Risk — {disease_name}',
        xaxis_title = 'Date',
        yaxis_title = 'DCWS Risk Score',
        yaxis       = dict(range=[0, 1]),
        height      = 350,
        plot_bgcolor= 'rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode   = 'x unified',
    )
    return fig


def plot_crop_comparison(compare_data):
    """Bar chart comparing risk across all crops."""
    crops  = list(compare_data['crops'].keys())
    risks  = [compare_data['crops'][c]['max_risk']   for c in crops]
    levels = [compare_data['crops'][c]['risk_level'] for c in crops]
    colors = [risk_color(r) for r in risks]
    labels = [compare_data['crops'][c]['top_disease'].split('___')[-1]
              for c in crops]

    fig = go.Figure(go.Bar(
        x           = crops,
        y           = risks,
        marker_color= colors,
        text        = [f"{r:.0%}" for r in risks],
        textposition= 'outside',
        customdata  = list(zip(labels, levels)),
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Risk: %{y:.1%}<br>'
            'Top disease: %{customdata[0]}<br>'
            'Level: %{customdata[1]}<extra></extra>'
        ),
    ))
    fig.add_hline(y=0.6, line_dash='dash', line_color='red')
    fig.add_hline(y=0.3, line_dash='dot',  line_color='orange')
    fig.update_layout(
        title       = 'Multi-Crop Risk Comparison (current weather)',
        xaxis_title = 'Crop',
        yaxis_title = 'Maximum Disease Risk',
        yaxis       = dict(range=[0, 1.1], tickformat='.0%'),
        height      = 380,
        plot_bgcolor= 'rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-rain.png", width=80)
    st.markdown("### 🌿 Early Warning System")
    st.markdown("---")

    # API status
    api_ok = check_api()
    if api_ok:
        st.success("API connected ✓")
    else:
        st.error("API offline — run: `uvicorn api:app --reload`")

    st.markdown("---")
    st.markdown("### 📍 Location")

    location_mode = st.selectbox(
        "Mode", ["Preset", "Custom GPS"],
        help="Choose a preset location or enter GPS coordinates"
    )
    PRESETS = {
        "Pune, Maharashtra"    : (18.5204, 73.8567),
        "Nashik, Maharashtra"  : (20.0059, 73.7898),
        "Nagpur, Maharashtra"  : (21.1458, 79.0882),
        "Delhi"                : (28.6139, 77.2090),
        "Bangalore, Karnataka" : (12.9716, 77.5946),
        "Hyderabad, Telangana" : (17.3850, 78.4867),
        "Kolkata, West Bengal" : (22.5726, 88.3639),
        "Ludhiana, Punjab"     : (30.9010, 75.8573),
    }
    if location_mode == "Preset":
        loc_name = st.selectbox("Location", list(PRESETS.keys()))
        lat, lon = PRESETS[loc_name]
    else:
        lat = st.number_input("Latitude",  value=18.5204, format="%.4f")
        lon = st.number_input("Longitude", value=73.8567, format="%.4f")

    st.markdown("---")
    st.markdown("### 🌱 Crop Details")

    cfg = get_config()
    crop_types    = cfg['crop_types']    if cfg else ["Tomato"]
    growth_stages = cfg['growth_stages'] if cfg else ["fruiting"]

    crop_type     = st.selectbox("Crop type",     crop_types)
    growth_stage  = st.selectbox("Growth stage",  growth_stages, index=3)
    dsp           = st.slider("Days since planting",  10, 180, 75)
    dth           = st.slider("Days to harvest",       5,  90, 20)
    area_ha       = st.number_input("Farm area (hectares)", 0.1, 100.0, 1.0)
    market_price  = st.number_input("Market price (₹/kg)",  5.0, 200.0, 25.0)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    mc_passes     = st.slider("MC Dropout passes", 10, 100, 30,
                               help="More passes = better uncertainty estimate, slower")


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

st.markdown(
    '<div class="main-header">🌿 Crop Disease Early Warning System</div>',
    unsafe_allow_html=True
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Analyze Leaf",
    "📈 7-Day Forecast",
    "🌾 Multi-Crop Compare",
    "📊 Historical Trends",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Analyze Leaf
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Upload a leaf image for full disease analysis")

    col_upload, col_result = st.columns([1, 2], gap="large")

    with col_upload:
        uploaded = st.file_uploader(
            "Choose leaf image",
            type=["jpg", "jpeg", "png"],
            help="Clear, well-lit photo of a single leaf works best"
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", use_column_width=True)

        analyze_btn = st.button(
            "🔍 Run Full Analysis",
            disabled=(not uploaded or not api_ok),
            type="primary",
            use_container_width=True,
        )

    with col_result:
        if analyze_btn and uploaded:
            with st.spinner("Running CNN + LSTM + Fusion analysis..."):
                uploaded.seek(0)
                resp = requests.post(
                    f"{API_URL}/analyze",
                    files={"file": (uploaded.name,
                                    uploaded.read(),
                                    uploaded.type)},
                    params={
                        "lat"                 : lat,
                        "lon"                 : lon,
                        "crop_type"           : crop_type,
                        "growth_stage"        : growth_stage,
                        "days_since_planting" : dsp,
                        "days_to_harvest"     : dth,
                        "area_ha"             : area_ha,
                        "market_price_per_kg" : market_price,
                        "n_mc_passes"         : mc_passes,
                    },
                    timeout=120,
                )

            if resp.status_code == 200:
                result = resp.json()
                st.session_state['last_result'] = result

                # ── CNN diagnosis ──
                cnn  = result['cnn']
                conf = cnn['confidence']
                st.markdown("##### 🔬 CNN Diagnosis")
                c1, c2 = st.columns(2)
                with c1:
                    disease_display = cnn['detected'].replace('___', ' — ')
                    st.metric("Detected", disease_display)
                    st.metric("Confidence", f"{conf:.1%}")
                    if conf < 0.45:
                        st.warning("Low confidence — retake in better lighting")
                with c2:
                    if 'gradcam_b64' in cnn:
                        gc_img = b64_to_image(cnn['gradcam_b64'])
                        st.image(gc_img,
                                 caption="Grad-CAM — red = infection focus",
                                 use_column_width=True)

                st.markdown("---")

                # ── Fusion risk ──
                fusion = result['fusion']
                st.markdown("##### 🧠 Fusion Risk Assessment")
                st.plotly_chart(
                    plot_fusion_risks(fusion['top5_diseases']),
                    use_container_width=True
                )

                top_risk  = fusion['risk_score']
                top_unc   = fusion['uncertainty']
                top_dis   = fusion['top_disease'].replace('___', ' — ')
                risk_cls  = 'risk-high' if top_risk > 0.6 \
                            else 'risk-moderate' if top_risk > 0.3 \
                            else 'risk-low'
                st.markdown(
                    f"**Top disease:** {top_dis} — "
                    f"<span class='{risk_cls}'>{top_risk:.1%} "
                    f"± {top_unc:.2f} [{risk_label(top_risk)}]</span>",
                    unsafe_allow_html=True
                )

                st.markdown("---")

                # ── Yield impact ──
                st.markdown("##### 📉 Yield Impact")
                yc1, yc2 = st.columns(2)
                with yc1:
                    st.plotly_chart(
                        plot_yield_impact(result['yield']),
                        use_container_width=True
                    )
                with yc2:
                    st.plotly_chart(
                        plot_financial(result['yield']),
                        use_container_width=True
                    )
                roi = result['yield']['roi']
                st.info(f"💰 ROI of treating: **{roi:.1f}x** — "
                        f"every ₹1 spent on treatment saves ₹{roi:.1f}")

                st.markdown("---")

                # ── Intervention ──
                intvn = result['intervention']
                st.markdown("##### 💊 Intervention Recommendation")
                urgency   = intvn['urgency']
                urg_color = "#d32f2f" if "TODAY" in urgency \
                            else "#f57c00" if "MONITOR" in urgency \
                            else "#388e3c"
                st.markdown(
                    f"<div class='intervention-box'>"
                    f"<b style='color:{urg_color};font-size:1.1rem'>"
                    f"⚠ {urgency}</b><br><br>"
                    f"<b>Fungicide:</b> {intvn['fungicide']}<br>"
                    f"<b>Frequency:</b> {intvn['frequency']}<br>"
                    f"<b>Timing:</b>    {intvn['timing']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # ── 7-day forecast ──
                st.markdown("---")
                st.markdown("##### 🌤 7-Day Risk Forecast")
                st.plotly_chart(
                    plot_7day_forecast(result['forecast']),
                    use_container_width=True
                )

            else:
                st.error(f"API error {resp.status_code}: {resp.text}")

        elif not uploaded:
            st.info("👈 Upload a leaf image to begin analysis")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — 7-Day Forecast
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 7-Day Disease Risk Forecast")
    st.markdown(f"Location: **{lat:.4f}, {lon:.4f}**")

    if st.button("🔄 Refresh Forecast", disabled=not api_ok,
                 type="primary"):
        with st.spinner("Fetching forecast..."):
            r = requests.get(f"{API_URL}/forecast",
                             params={"lat": lat, "lon": lon, "top_n": 10},
                             timeout=30)
        if r.status_code == 200:
            st.session_state['forecast'] = r.json()

    if 'forecast' in st.session_state:
        fdata = st.session_state['forecast']

        # Summary cards
        st.markdown("##### Top risk diseases this week")
        cols = st.columns(5)
        for i, d in enumerate(fdata['diseases'][:5]):
            with cols[i]:
                color = risk_color(d['peak_risk'])
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div style='font-size:0.75rem;color:#555'>"
                    f"{d['disease'].split('___')[-1]}</div>"
                    f"<div style='font-size:1.4rem;font-weight:700;"
                    f"color:{color}'>{d['peak_risk']:.0%}</div>"
                    f"<div style='font-size:0.7rem;color:{color}'>"
                    f"{d['level']}</div></div>",
                    unsafe_allow_html=True
                )

        st.markdown("")
        st.plotly_chart(
            plot_7day_forecast({'top5': fdata['diseases'][:5], 'dates': fdata.get('dates', [])}),
            use_container_width=True
        )

        # Detailed table
        st.markdown("##### Full risk table")
        rows = []
        for d in fdata['diseases']:
            rows.append({
                'Disease'  : d['disease'].replace('___', ' — '),
                'Peak Risk': f"{d['peak_risk']:.1%}",
                'Level'    : d['level'],
                'Day 1'    : f"{d['daily'][0]:.1%}",
                'Day 3'    : f"{d['daily'][2]:.1%}",
                'Day 7'    : f"{d['daily'][6]:.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
    else:
        st.info("Click 'Refresh Forecast' to load the 7-day outlook")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Multi-Crop Comparison
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Multi-Crop Risk Comparison")
    st.markdown(
        "Compares disease risk across all 14 crop types "
        "for current weather conditions at your location."
    )

    c1, c2 = st.columns(2)
    with c1:
        comp_stage = st.selectbox("Growth stage",
                                   growth_stages, index=3,
                                   key="comp_stage")
    with c2:
        comp_dth = st.slider("Days to harvest", 5, 90, 30, key="comp_dth")

    if st.button("🌾 Compare All Crops", disabled=not api_ok,
                 type="primary"):
        with st.spinner("Comparing across all crops..."):
            r = requests.get(
                f"{API_URL}/compare",
                params={"lat": lat, "lon": lon,
                        "growth_stage": comp_stage,
                        "days_to_harvest": comp_dth},
                timeout=60
            )
        if r.status_code == 200:
            st.session_state['comparison'] = r.json()

    if 'comparison' in st.session_state:
        cdata = st.session_state['comparison']
        st.plotly_chart(
            plot_crop_comparison(cdata),
            use_container_width=True
        )

        # Summary table
        rows = []
        for crop, info in sorted(cdata['crops'].items(),
                                  key=lambda x: -x[1]['max_risk']):
            rows.append({
                'Crop'        : crop,
                'Max Risk'    : f"{info['max_risk']:.1%}",
                'Risk Level'  : info['risk_level'],
                'Top Disease' : info['top_disease'].split('___')[-1],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
    else:
        st.info("Click 'Compare All Crops' to see risk across all crop types")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Historical Trends
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Historical Disease Risk Trends")

    cfg2 = get_config()
    disease_options = cfg2['disease_classes'] if cfg2 else []

    c1, c2 = st.columns(2)
    with c1:
        selected_disease = st.selectbox(
            "Select disease",
            [d.replace('___', ' — ') for d in disease_options],
        )
        actual_disease = disease_options[
            [d.replace('___', ' — ') for d in disease_options
             ].index(selected_disease)
        ] if disease_options else None

    with c2:
        days_back = st.select_slider(
            "History window",
            options=[90, 180, 365, 730],
            value=365,
            format_func=lambda x: f"{x} days"
        )

    if st.button("📊 Load Historical Data", disabled=not api_ok,
                 type="primary"):
        if actual_disease:
            with st.spinner("Loading historical data..."):
                r = requests.get(
                    f"{API_URL}/historical",
                    params={"disease": actual_disease,
                            "days_back": days_back},
                    timeout=30
                )
            if r.status_code == 200:
                st.session_state['historical'] = r.json()
            else:
                st.error(f"Error: {r.text}")

    if 'historical' in st.session_state:
        hdata = st.session_state['historical']

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Risk",    f"{hdata['mean']:.1%}")
        col2.metric("Peak Risk",    f"{hdata['max']:.1%}")
        col3.metric("Days Analyzed", f"{len(hdata['scores']):,}")

        st.plotly_chart(
            plot_historical(hdata),
            use_container_width=True
        )

        # Monthly average heatmap
        dates  = pd.to_datetime(hdata['dates'])
        scores = hdata['scores']
        df_hist = pd.DataFrame({'date': dates, 'risk': scores})
        df_hist['month'] = df_hist['date'].dt.strftime('%b')
        df_hist['year']  = df_hist['date'].dt.year
        pivot = df_hist.pivot_table(
            values='risk', index='year', columns='month',
            aggfunc='mean'
        )
        month_order = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        pivot = pivot.reindex(columns=[m for m in month_order
                                        if m in pivot.columns])

        fig_hm = px.imshow(
            pivot, color_continuous_scale='RdYlGn_r',
            title=f'Monthly Average Risk — {selected_disease}',
            labels=dict(color='Risk Score'),
            aspect='auto',
            zmin=0, zmax=1,
        )
        fig_hm.update_layout(height=300,
                              paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Select a disease and click 'Load Historical Data'")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem'>"
    "Crop Disease Early Warning System — "
    "EfficientNet-B4 + BiLSTM + Cross-Attention Fusion | "
    "Built with PyTorch · FastAPI · Streamlit"
    "</div>",
    unsafe_allow_html=True
)
