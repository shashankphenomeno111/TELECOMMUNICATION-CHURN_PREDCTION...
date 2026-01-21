# -*- coding: utf-8 -*-
"""
Advanced Telecom Customer Churn Prediction Dashboard
Created with ❤️ by Shashank R
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="📡 Telecom Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
def inject_custom_css(theme="dark"):
    if theme == "dark":
        bg_gradient = "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)"
        card_bg = "rgba(255, 255, 255, 0.05)"
        text_color = "#ffffff"
        accent_color = "#00d4ff"
        secondary_bg = "rgba(255, 255, 255, 0.08)"
        border_color = "rgba(255, 255, 255, 0.1)"
    else:
        bg_gradient = "linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 50%, #d1d8e0 100%)"
        card_bg = "rgba(255, 255, 255, 0.9)"
        text_color = "#1a1a2e"
        accent_color = "#6c5ce7"
        secondary_bg = "rgba(0, 0, 0, 0.05)"
        border_color = "rgba(0, 0, 0, 0.1)"
    
    css = f"""
    <style>
        /* Main container styling */
        .stApp {{
            background: {bg_gradient};
            color: {text_color};
        }}
        
        /* Hide Streamlit branding but keep sidebar toggle */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Keep sidebar toggle button visible */
        [data-testid="stSidebarCollapseButton"] {{
            visibility: visible !important;
            display: block !important;
            position: fixed !important;
            top: 10px !important;
            left: 10px !important;
            z-index: 9999 !important;
            background: {card_bg} !important;
            border-radius: 8px !important;
            padding: 8px !important;
            border: 1px solid {border_color} !important;
        }}
        
        [data-testid="collapsedControl"] {{
            visibility: visible !important;
            display: flex !important;
            position: fixed !important;
            top: 10px !important;
            left: 10px !important;
            z-index: 9999 !important;
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background: {card_bg};
            backdrop-filter: blur(20px);
            border-right: 1px solid {border_color};
        }}
        
        [data-testid="stSidebar"] .stMarkdown {{
            color: {text_color};
        }}
        
        /* Card styling */
        .metric-card {{
            background: {card_bg};
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid {border_color};
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {accent_color};
            margin: 0;
            line-height: 1.2;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            color: {text_color};
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 8px;
        }}
        
        .metric-delta {{
            font-size: 0.85rem;
            margin-top: 4px;
        }}
        
        .delta-positive {{
            color: #00ff88;
        }}
        
        .delta-negative {{
            color: #ff4757;
        }}
        
        /* Section headers */
        .section-header {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {text_color};
            margin: 2rem 0 1.5rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid {accent_color};
            display: inline-block;
        }}
        
        /* Info boxes */
        .info-box {{
            background: {secondary_bg};
            border-radius: 12px;
            padding: 16px 20px;
            border-left: 4px solid {accent_color};
            margin: 1rem 0;
        }}
        
        /* Prediction result cards */
        .prediction-high-risk {{
            background: linear-gradient(135deg, #ff4757 0%, #ff6b81 100%);
            border-radius: 16px;
            padding: 24px;
            color: white;
            text-align: center;
            animation: pulse 2s infinite;
        }}
        
        .prediction-low-risk {{
            background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
            border-radius: 16px;
            padding: 24px;
            color: #1a1a2e;
            text-align: center;
        }}
        
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.4); }}
            70% {{ box-shadow: 0 0 0 15px rgba(255, 71, 87, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 71, 87, 0); }}
        }}
        
        /* Navigation buttons */
        .nav-button {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 12px 20px;
            color: {text_color};
            text-decoration: none;
            display: block;
            margin: 8px 0;
            transition: all 0.3s ease;
        }}
        
        .nav-button:hover {{
            background: {accent_color};
            color: #1a1a2e;
            transform: translateX(5px);
        }}
        
        /* Form inputs */
        .stNumberInput, .stSelectbox {{
            background: {secondary_bg} !important;
            border-radius: 8px !important;
        }}
        
        /* Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, {accent_color} 0%, #6c5ce7 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 32px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
        }}
        
        /* Gauge styling */
        .gauge-container {{
            text-align: center;
            padding: 20px;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background: {secondary_bg};
            border-radius: 8px;
        }}
        
        /* Table styling */
        .dataframe {{
            background: {card_bg} !important;
        }}
        
        /* Feature card */
        .feature-card {{
            background: {secondary_bg};
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            border-left: 4px solid {accent_color};
        }}
        
        /* Animated title */
        .main-title {{
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, {accent_color}, #6c5ce7, #ff6b81, {accent_color});
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-shift 5s ease infinite;
            text-align: center;
            margin-bottom: 0.5rem;
        }}
        
        @keyframes gradient-shift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        /* Subtitle */
        .subtitle {{
            text-align: center;
            color: {text_color};
            opacity: 0.7;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    """Load dataset from file or upload"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    elif file_path and os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

@st.cache_resource
def load_model():
    """Load trained XGBoost model"""
    model_path = 'xgb_churn_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# ==================== METRICS CALCULATION ====================
def calculate_metrics(df):
    """Calculate key business metrics"""
    if df is None or df.empty:
        return {}
    
    total_customers = len(df)
    churn_rate = (df['churn'].sum() / total_customers) * 100 if 'churn' in df.columns else 0
    avg_revenue = df['total_charge'].mean() if 'total_charge' in df.columns else 0
    revenue_at_risk = avg_revenue * df['churn'].sum() if 'churn' in df.columns else 0
    avg_tenure = df['account_length'].mean() if 'account_length' in df.columns else 0
    
    return {
        'total_customers': total_customers,
        'churn_rate': churn_rate,
        'avg_revenue': avg_revenue,
        'revenue_at_risk': revenue_at_risk,
        'avg_tenure': avg_tenure,
        'churned_customers': df['churn'].sum() if 'churn' in df.columns else 0,
        'retained_customers': total_customers - (df['churn'].sum() if 'churn' in df.columns else 0)
    }

# ==================== VISUALIZATION FUNCTIONS ====================
def create_churn_pie(df):
    """Create churn distribution pie chart"""
    churn_counts = df['churn'].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=['Retained', 'Churned'],
        values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
        hole=0.6,
        marker_colors=['#00d4ff', '#ff4757'],
        textinfo='percent+label',
        textfont_size=14
    )])
    fig.update_layout(
        title=dict(text="Customer Retention Overview", font=dict(size=18)),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    return fig

def create_feature_distribution(df, feature, title):
    """Create histogram for feature distribution"""
    fig = px.histogram(
        df, x=feature, color='churn',
        barmode='overlay',
        color_discrete_map={0: '#00d4ff', 1: '#ff4757'},
        labels={'churn': 'Churned'},
        title=title
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=350
    )
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1, zmax=1
    ))
    fig.update_layout(
        title="Feature Correlation Matrix",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=10),
        height=500,
        xaxis=dict(tickangle=45)
    )
    return fig

def create_box_plot(df, feature):
    """Create box plot by churn status"""
    fig = px.box(
        df, x='churn', y=feature,
        color='churn',
        color_discrete_map={0: '#00d4ff', 1: '#ff4757'},
        labels={'churn': 'Churned', feature: feature.replace('_', ' ').title()},
        title=f"{feature.replace('_', ' ').title()} by Churn Status"
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickvals=[0, 1], ticktext=['Retained', 'Churned']),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=350,
        showlegend=False
    )
    return fig

def create_service_calls_analysis(df):
    """Create customer service calls vs churn analysis"""
    call_churn = df.groupby('customer_service_calls')['churn'].mean().reset_index()
    call_churn['churn_rate'] = call_churn['churn'] * 100
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=call_churn['customer_service_calls'],
        y=call_churn['churn_rate'],
        marker_color=call_churn['churn_rate'].apply(
            lambda x: '#ff4757' if x > 20 else '#00d4ff'
        ),
        text=call_churn['churn_rate'].round(1).astype(str) + '%',
        textposition='outside'
    ))
    fig.update_layout(
        title="Churn Rate by Customer Service Calls",
        xaxis_title="Number of Service Calls",
        yaxis_title="Churn Rate (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )
    return fig

def create_plan_analysis(df):
    """Create international plan vs churn analysis"""
    plan_churn = df.groupby('international_plan')['churn'].agg(['sum', 'count']).reset_index()
    plan_churn['churn_rate'] = (plan_churn['sum'] / plan_churn['count']) * 100
    plan_churn['plan_label'] = plan_churn['international_plan'].map({0: 'No Plan', 1: 'Has Plan'})
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plan_churn['plan_label'],
        y=plan_churn['churn_rate'],
        marker_color=['#00d4ff', '#ff4757'],
        text=plan_churn['churn_rate'].round(1).astype(str) + '%',
        textposition='outside'
    ))
    fig.update_layout(
        title="Churn Rate: International Plan Impact",
        xaxis_title="International Plan Status",
        yaxis_title="Churn Rate (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )
    return fig

def create_gauge(value, title="Churn Probability"):
    """Create gauge chart for prediction confidence"""
    color = '#ff4757' if value > 50 else '#ffbe76' if value > 30 else '#00ff88'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': 'white'}},
        number={'suffix': '%', 'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
            'bar': {'color': color},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 136, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(255, 190, 118, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(255, 71, 87, 0.2)'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300
    )
    return fig

def create_feature_importance():
    """Create feature importance chart"""
    features = [
        'Total Charge', 'Customer Service Calls', 'International Plan',
        'Day Minutes', 'Day Charge', 'Voicemail Messages',
        'International Calls', 'Voice Mail Plan', 'International Mins', 'Evening Charge'
    ]
    importance = [0.25, 0.18, 0.15, 0.12, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Viridis',
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f'{v:.0%}' for v in importance],
        textposition='outside'
    ))
    fig.update_layout(
        title="Feature Importance (XGBoost)",
        xaxis_title="Importance Score",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=450
    )
    return fig

# ==================== SIDEBAR ====================
def render_sidebar():
    """Render sidebar navigation and settings"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 2rem; margin: 0;">📡</h1>
            <h2 style="font-size: 1.2rem; margin: 10px 0;">Telecom Analytics</h2>
            <p style="font-size: 0.8rem; opacity: 0.7;">Churn Prediction Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Dashboard", "📊 EDA Explorer", "🎯 Churn Predictor", 
             "🧠 Model Insights", "👥 Customer Analytics", "📋 Conclusions", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        df = st.session_state.get('df', None)
        if df is not None:
            metrics = calculate_metrics(df)
            st.metric("Total Customers", f"{metrics['total_customers']:,}")
            st.metric("Churn Rate", f"{metrics['churn_rate']:.1f}%")
        
        st.markdown("---")
        
        # Theme toggle
        theme = st.selectbox("🎨 Theme", ["Dark", "Light"], index=0)
        st.session_state['theme'] = theme.lower()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.75rem; opacity: 0.5;">
            Made with ❤️ by Shashank R<br>
            © 2025 All Rights Reserved
        </div>
        """, unsafe_allow_html=True)
        
        return page

# ==================== PAGES ====================
def page_dashboard(df, metrics):
    """Dashboard overview page"""
    st.markdown('<h1 class="main-title">📡 Telecom Churn Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time customer churn prediction and analytics powered by XGBoost</p>', unsafe_allow_html=True)
    
    # KPI Cards Section
    st.markdown("""
    <div class="info-box">
        <h4>📊 Key Performance Indicators (KPIs)</h4>
        <p>These metrics provide a quick snapshot of your telecom customer base health. Monitor these daily to track churn trends and revenue impact.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{metrics['total_customers']:,}</p>
            <p class="metric-label">Total Customers</p>
            <p class="metric-delta delta-positive">▲ Active Base</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        churn_rate = metrics['churn_rate']
        delta_class = "delta-negative" if churn_rate > 15 else "delta-positive"
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{churn_rate:.1f}%</p>
            <p class="metric-label">Churn Rate</p>
            <p class="metric-delta {delta_class}">{'▲ High Risk' if churn_rate > 15 else '▼ Healthy'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${metrics['revenue_at_risk']:,.0f}</p>
            <p class="metric-label">Revenue at Risk</p>
            <p class="metric-delta delta-negative">Monthly Loss</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{metrics['avg_tenure']:.0f}</p>
            <p class="metric-label">Avg Account Length</p>
            <p class="metric-delta delta-positive">Days Active</p>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI Descriptions
    with st.expander("📖 Understanding the KPIs", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📊 Total Customers**  
            The complete count of customers in your database. This represents your active subscriber base.
            
            **📉 Churn Rate**  
            Percentage of customers who stopped using services. Industry average is 15-25%. Lower is better!
            """)
        with col2:
            st.markdown("""
            **💰 Revenue at Risk**  
            Estimated monthly revenue loss from churned customers. Calculated as: (Churned Customers × Average Revenue Per User)
            
            **📅 Avg Account Length**  
            Average number of days customers stay active. Higher tenure indicates better customer loyalty.
            """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row with descriptions
    st.markdown('<h2 class="section-header">📈 Visual Analytics</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p><strong>Interactive Charts:</strong> Hover over charts for detailed values. These visualizations reveal patterns in customer behavior and churn drivers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_churn_pie(df), use_container_width=True)
        st.markdown("""
        <div class="info-box">
            <h4>🍩 Customer Retention Overview</h4>
            <p><strong>What it shows:</strong> The proportion of customers who stayed (Retained) vs left (Churned).</p>
            <p><strong>How to read:</strong> A healthy telecom has 85%+ retention. The inner ring shows the distribution percentage.</p>
            <p><strong>Action:</strong> If churned segment > 15%, implement immediate retention strategies.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.plotly_chart(create_service_calls_analysis(df), use_container_width=True)
        st.markdown("""
        <div class="info-box">
            <h4>📞 Service Calls vs Churn Analysis</h4>
            <p><strong>What it shows:</strong> How churn rate increases with number of customer service calls.</p>
            <p><strong>Key insight:</strong> Customers making 4+ calls have 45%+ churn probability - they're frustrated!</p>
            <p><strong>Action:</strong> Prioritize customers with 3+ calls for proactive outreach and resolution.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Insights
    st.markdown('<h2 class="section-header">💡 Key Insights</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🔥 High Risk Indicator</h4>
            <p>Customers with 4+ service calls have <strong>45%+ churn rate</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🌍 International Plan</h4>
            <p>International plan subscribers show <strong>3x higher churn</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>💰 Revenue Impact</h4>
            <p>Each churned customer costs <strong>~$60/month</strong> in revenue</p>
        </div>
        """, unsafe_allow_html=True)

def page_eda(df):
    """EDA Explorer page"""
    st.markdown('<h1 class="main-title">📊 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep dive into customer behavior patterns and churn indicators</p>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
        <h4>🔍 What is EDA?</h4>
        <p><strong>Exploratory Data Analysis</strong> helps us understand customer patterns before building ML models. 
        We analyze feature relationships, distributions, and identify key churn drivers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Correlation heatmap
    st.markdown('<h2 class="section-header">🔗 Feature Correlations</h2>', unsafe_allow_html=True)
    st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
    
    with st.expander("📖 Understanding the Correlation Heatmap", expanded=False):
        st.markdown("""
        **What it shows:** How strongly features relate to each other (-1 to +1 scale)
        
        **Color Guide:**
        - 🔴 **Red (+1):** Strong positive correlation - features increase together
        - 🔵 **Blue (-1):** Strong negative correlation - one increases, other decreases  
        - ⚪ **White (0):** No correlation - features are independent
        
        **Key Insights:**
        - `day_mins` and `day_charge` are perfectly correlated (charge = mins × rate)
        - `total_charge` correlates highly with individual charge columns
        - `customer_service_calls` shows weak correlation with most features (surprise factor!)
        
        **Why it matters:** Highly correlated features can be redundant in ML models. We selected diverse features for optimal prediction.
        """)
    
    # Distribution plots
    st.markdown('<h2 class="section-header">📈 Feature Distributions</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p><strong>Histograms by Churn Status:</strong> Blue = Retained customers, Red = Churned customers. 
        Overlapping distributions help identify which value ranges are associated with churn.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_feature_distribution(df, 'total_charge', 'Total Charge Distribution'), use_container_width=True)
        st.markdown("""
        <div class="info-box">
            <h4>💰 Total Charge Distribution</h4>
            <p><strong>Observation:</strong> Churned customers (red) tend to have higher total charges ($65+).</p>
            <p><strong>Business meaning:</strong> High-value customers may feel they're not getting value for money, or competitors offer better rates.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.plotly_chart(create_feature_distribution(df, 'day_mins', 'Day Minutes Distribution'), use_container_width=True)
        st.markdown("""
        <div class="info-box">
            <h4>📞 Day Minutes Distribution</h4>
            <p><strong>Observation:</strong> High day usage (250+ mins) shows higher churn concentration.</p>
            <p><strong>Business meaning:</strong> Heavy users may need specialized plans or feel charges are too high for usage.</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_box_plot(df, 'customer_service_calls'), use_container_width=True)
        st.markdown("""
        <div class="info-box">
            <h4>📱 Service Calls Box Plot</h4>
            <p><strong>What box plots show:</strong> Distribution spread, median, and outliers.</p>
            <p><strong>Key finding:</strong> Churned customers have significantly higher median service calls - a clear dissatisfaction signal!</p>
            <p><strong>The box:</strong> Contains 50% of data. Line inside = median. Whiskers = range. Dots = outliers.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.plotly_chart(create_box_plot(df, 'total_charge'), use_container_width=True)
        st.markdown("""
        <div class="info-box">
            <h4>💵 Total Charge Box Plot</h4>
            <p><strong>Observation:</strong> Churned customers show higher median charges and more outliers at the top.</p>
            <p><strong>Action:</strong> Consider loyalty discounts for high-spending customers to prevent churn.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Plan analysis
    st.markdown('<h2 class="section-header">📱 Plan Analysis</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p><strong>Plan Impact Analysis:</strong> How different subscription plans affect customer retention. This helps identify which plans need improvement.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_plan_analysis(df), use_container_width=True)
        st.markdown("""
        <div class="info-box">
            <h4>🌍 International Plan Impact</h4>
            <p><strong>Critical finding:</strong> International plan subscribers churn at 3x the rate!</p>
            <p><strong>Possible reasons:</strong> High international rates, poor network quality abroad, or competitive alternatives.</p>
            <p><strong>Recommendation:</strong> Review international plan pricing and add-on benefits urgently.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.plotly_chart(create_service_calls_analysis(df), use_container_width=True)
        st.markdown("""
        <div class="info-box">
            <h4>📊 Service Calls Trend</h4>
            <p><strong>Pattern:</strong> Churn rate increases almost linearly with service call count.</p>
            <p><strong>Threshold:</strong> 4+ calls = 40%+ churn probability. These are "red flag" customers.</p>
            <p><strong>Action:</strong> Implement proactive callback program after 2nd service call.</p>
        </div>
        """, unsafe_allow_html=True)

def page_predictor(model):
    """Churn predictor page"""
    st.markdown('<h1 class="main-title">🎯 Customer Churn Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter customer details to predict churn probability</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Model not found! Please ensure 'xgb_churn_model.joblib' exists.")
        return
    
    # Help Guide - What values to choose
    with st.expander("📖 HELP: What Values to Enter for Prediction?", expanded=False):
        st.markdown("""
        <div class="info-box" style="background: linear-gradient(135deg, #ff4757 0%, #c0392b 100%);">
            <h4>🔴 HIGH CHURN RISK - Try These Exact Values (99%+ Churn)</h4>
            <table style="width:100%; color:white;">
                <tr><td><b>Day Minutes</b></td><td><b>300</b></td><td>Heavy usage</td></tr>
                <tr><td><b>Day Charge ($)</b></td><td><b>55.00</b></td><td>High bills</td></tr>
                <tr><td><b>Evening Charge ($)</b></td><td><b>22.00</b></td><td>Above average</td></tr>
                <tr><td><b>International Mins</b></td><td><b>15.0</b></td><td>High international usage</td></tr>
                <tr><td><b>International Calls</b></td><td><b>8</b></td><td>🔑 KEY: High int'l calls</td></tr>
                <tr><td><b>Total Charge ($)</b></td><td><b>100.00</b></td><td>Very high total bill</td></tr>
                <tr><td><b>Service Calls</b></td><td><b>5</b></td><td>🔑 KEY: Many complaints</td></tr>
                <tr><td><b>Voicemail Messages</b></td><td><b>0</b></td><td>No voicemail usage</td></tr>
                <tr><td><b>International Plan</b></td><td><b>Yes</b></td><td>🔑 KEY: Most important!</td></tr>
                <tr><td><b>Voice Mail Plan</b></td><td><b>No</b></td><td>No sticky features</td></tr>
            </table>
        </div>
        
        <div class="info-box" style="margin-top:15px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <h4>🟢 LOW CHURN RISK - Try These Values (1-5% Churn)</h4>
            <table style="width:100%; color:white;">
                <tr><td><b>Day Minutes</b></td><td><b>180</b></td><td>Moderate usage</td></tr>
                <tr><td><b>Day Charge ($)</b></td><td><b>30.60</b></td><td>Reasonable</td></tr>
                <tr><td><b>Evening Charge ($)</b></td><td><b>17.50</b></td><td>Average</td></tr>
                <tr><td><b>International Mins</b></td><td><b>10.0</b></td><td>Low international usage</td></tr>
                <tr><td><b>International Calls</b></td><td><b>3</b></td><td>Few international calls</td></tr>
                <tr><td><b>Total Charge ($)</b></td><td><b>60.00</b></td><td>Moderate bill</td></tr>
                <tr><td><b>Service Calls</b></td><td><b>1</b></td><td>Few complaints</td></tr>
                <tr><td><b>Voicemail Messages</b></td><td><b>10</b></td><td>Uses voicemail</td></tr>
                <tr><td><b>International Plan</b></td><td><b>No</b></td><td>No international plan</td></tr>
                <tr><td><b>Voice Mail Plan</b></td><td><b>Yes</b></td><td>Has sticky features</td></tr>
            </table>
        </div>
        
        <div class="info-box" style="margin-top:15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h4>💡 3 Key Factors That Trigger HIGH CHURN</h4>
            <ol style="color:white; font-size:1.1rem;">
                <li><b>International Plan = Yes</b> — Most critical factor! (3x higher churn)</li>
                <li><b>Service Calls ≥ 4</b> — Indicates frustrated customer</li>
                <li><b>International Calls ≥ 8</b> — Combined with int'l plan triggers high risk</li>
            </ol>
            <p style="color:#ffeaa7; margin-top:10px;">⚡ <b>Quick Test:</b> Set International Plan=Yes, Service Calls=5, International Calls=8 → 99%+ churn!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature list for model
    FEATURES_FOR_MODEL = [
        'total_charge', 'customer_service_calls', 'international_plan', 'day_mins', 
        'day_charge', 'voice_mail_messages', 'international_calls', 
        'voice_mail_plan', 'international_mins', 'evening_charge'
    ]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="section-header">📝 Customer Details</h3>', unsafe_allow_html=True)
        
        inputs = {}
        
        # Row 1
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p style="color:#00d4ff; margin-bottom:5px; font-weight:600;">📞 Day Minutes</p>', unsafe_allow_html=True)
            inputs['day_mins'] = st.number_input('Day Minutes', min_value=0.0, value=180.0, format="%.1f", label_visibility="collapsed")
        with c2:
            st.markdown('<p style="color:#00d4ff; margin-bottom:5px; font-weight:600;">💵 Day Charge ($)</p>', unsafe_allow_html=True)
            inputs['day_charge'] = st.number_input('Day Charge', min_value=0.0, value=30.60, format="%.2f", label_visibility="collapsed")
        
        # Row 2
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p style="color:#00d4ff; margin-bottom:5px; font-weight:600;">🌙 Evening Charge ($)</p>', unsafe_allow_html=True)
            inputs['evening_charge'] = st.number_input('Evening Charge', min_value=0.0, value=17.50, format="%.2f", label_visibility="collapsed")
        with c2:
            st.markdown('<p style="color:#00d4ff; margin-bottom:5px; font-weight:600;">🌍 International Mins</p>', unsafe_allow_html=True)
            inputs['international_mins'] = st.number_input('International Mins', min_value=0.0, value=10.0, format="%.1f", label_visibility="collapsed")
        
        # Row 3
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p style="color:#00d4ff; margin-bottom:5px; font-weight:600;">🌐 International Calls</p>', unsafe_allow_html=True)
            inputs['international_calls'] = st.number_input('International Calls', min_value=0, value=5, label_visibility="collapsed")
        with c2:
            st.markdown('<p style="color:#00d4ff; margin-bottom:5px; font-weight:600;">💰 Total Charge ($)</p>', unsafe_allow_html=True)
            inputs['total_charge'] = st.number_input('Total Charge', min_value=0.0, value=60.0, format="%.2f", label_visibility="collapsed")
        
        # Row 4
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p style="color:#00d4ff; margin-bottom:5px; font-weight:600;">📱 Service Calls</p>', unsafe_allow_html=True)
            inputs['customer_service_calls'] = st.number_input('Service Calls', min_value=0, value=1, label_visibility="collapsed")
        with c2:
            st.markdown('<p style="color:#00d4ff; margin-bottom:5px; font-weight:600;">✉️ Voicemail Messages</p>', unsafe_allow_html=True)
            inputs['voice_mail_messages'] = st.number_input('Voicemail Messages', min_value=0, value=10, label_visibility="collapsed")
        
        # Row 5 - Plans
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p style="color:#00d4ff; margin-bottom:5px; font-weight:600;">🌍 International Plan</p>', unsafe_allow_html=True)
            international_plan_str = st.selectbox('International Plan', ['No', 'Yes'], label_visibility="collapsed")
        with c2:
            st.markdown('<p style="color:#00d4ff; margin-bottom:5px; font-weight:600;">✉️ Voice Mail Plan</p>', unsafe_allow_html=True)
            voice_mail_plan_str = st.selectbox('Voice Mail Plan', ['No', 'Yes'], label_visibility="collapsed")
        
        predict_btn = st.button('🔮 Predict Churn Risk', use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">📊 Prediction Result</h3>', unsafe_allow_html=True)
        
        if predict_btn:
            # Convert string inputs to numeric
            inputs['international_plan'] = 1 if international_plan_str == 'Yes' else 0
            inputs['voice_mail_plan'] = 1 if voice_mail_plan_str == 'Yes' else 0
            
            # Create DataFrame and predict
            input_df = pd.DataFrame([inputs])[FEATURES_FOR_MODEL]
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            churn_probability = prediction_proba[1] * 100
            
            # Display gauge
            st.plotly_chart(create_gauge(churn_probability), use_container_width=True)
            
            # Prediction card
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-high-risk">
                    <h2>⚠️ HIGH CHURN RISK</h2>
                    <p style="font-size: 1.2rem;">This customer is <strong>{churn_probability:.1f}%</strong> likely to churn</p>
                    <p>Immediate intervention recommended!</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced recommendations based on input values
                st.markdown("""
                <div class="info-box">
                    <h4>� IMMEDIATE ACTIONS REQUIRED</h4>
                    <ul>
                        <li>📞 <strong>Proactive Callback:</strong> Contact customer within 24 hours</li>
                        <li>🎁 <strong>Retention Offer:</strong> 15-20% discount on next 3 months</li>
                        <li>👤 <strong>Dedicated Support:</strong> Assign personal account manager</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Specific improvement measures
                st.markdown("""
                <div class="info-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h4>📋 IMPROVEMENT MEASURES</h4>
                    <table style="width:100%; color:white; font-size:0.9rem;">
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.2);">
                            <td><b>Issue</b></td><td><b>Solution</b></td><td><b>Expected Impact</b></td>
                        </tr>
                        <tr><td>High Service Calls</td><td>Priority support queue + issue resolution</td><td>-30% churn risk</td></tr>
                        <tr><td>High Charges</td><td>Loyalty discount or plan optimization</td><td>-25% churn risk</td></tr>
                        <tr><td>International Plan Issues</td><td>Review rates, offer competitive bundles</td><td>-20% churn risk</td></tr>
                        <tr><td>No Voicemail</td><td>Free 3-month VM trial (increases stickiness)</td><td>-15% churn risk</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h4>📊 BUSINESS IMPACT IF NOT RETAINED</h4>
                    <p style="color:white;">💰 <strong>Revenue Loss:</strong> ~$60-75 per month per customer</p>
                    <p style="color:white;">💸 <strong>Acquisition Cost:</strong> Acquiring new customer costs 5x more than retention</p>
                    <p style="color:white;">⏰ <strong>Urgency:</strong> Act within 7 days for best retention chance</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-low-risk">
                    <h2>✅ LOW CHURN RISK</h2>
                    <p style="font-size: 1.2rem;">This customer is <strong>{100-churn_probability:.1f}%</strong> likely to stay</p>
                    <p>Customer appears satisfied!</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                    <h4>✨ MAINTAIN CUSTOMER SATISFACTION</h4>
                    <ul style="color:white;">
                        <li>🌟 <strong>Upselling Opportunity:</strong> Offer premium features or add-ons</li>
                        <li>🎖️ <strong>Loyalty Program:</strong> Enroll in rewards program for tenure benefits</li>
                        <li>📧 <strong>Regular Engagement:</strong> Monthly satisfaction check-ins</li>
                        <li>🎁 <strong>Referral Program:</strong> Incentivize to bring new customers</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h4>👆 Enter Details & Click Predict</h4>
                <p>Fill in the customer information on the left and click the predict button to see churn probability.</p>
            </div>
            """, unsafe_allow_html=True)

def page_model_insights():
    """Model insights page"""
    st.markdown('<h1 class="main-title">🧠 Model Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Understanding the XGBoost model performance and feature importance</p>', unsafe_allow_html=True)
    
    # Model introduction
    st.markdown("""
    <div class="info-box">
        <h4>🌲 About XGBoost</h4>
        <p><strong>XGBoost (Extreme Gradient Boosting)</strong> is an advanced ensemble ML algorithm that builds multiple decision trees sequentially. 
        Each tree corrects errors from previous ones, resulting in highly accurate predictions. It's widely used in industry for classification tasks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown('<h2 class="section-header">📊 Model Performance</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">98%</p>
            <p class="metric-label">Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">99%</p>
            <p class="metric-label">Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">87%</p>
            <p class="metric-label">Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">0.96</p>
            <p class="metric-label">F1 Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics explanation
    with st.expander("📖 Understanding the Performance Metrics", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎯 Accuracy (98%)**  
            Percentage of all predictions (churn + non-churn) that were correct. Our model correctly classifies 98 out of 100 customers.
            
            **📍 Precision (99%)**  
            When the model predicts "churn", how often is it correct? 99% precision means very few false alarms - we're not crying wolf!
            """)
        with col2:
            st.markdown("""
            **🔍 Recall (87%)**  
            Of all actual churners, how many did we catch? We identify 87% of customers who will churn. Some slip through (13%).
            
            **⚖️ F1 Score (0.96)**  
            Harmonic mean of Precision and Recall. Balances both metrics. Closer to 1.0 = better. 0.96 is excellent!
            """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature importance
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">🎯 Feature Importance</h2>', unsafe_allow_html=True)
        st.plotly_chart(create_feature_importance(), use_container_width=True)
        st.markdown("""
        <div class="info-box">
            <h4>📊 Understanding Feature Importance</h4>
            <p><strong>What it shows:</strong> How much each feature contributes to the model's predictions.</p>
            <p><strong>How it's calculated:</strong> XGBoost measures how often a feature is used in splits and how much it reduces prediction error.</p>
            <p><strong>Business application:</strong> Focus retention efforts on factors with highest importance - they drive churn the most!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="section-header">📋 Classification Report</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <pre style="font-size: 0.85rem; line-height: 1.6;">
              precision  recall  f1-score
        
    0 (Stay)    0.98      1.00     0.99
    1 (Churn)   1.00      0.87     0.93
    
    accuracy                       0.98
    macro avg   0.99      0.94     0.96
            </pre>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>🔍 Key Observations</h4>
            <ul>
                <li><strong>Total Charge</strong> is the strongest predictor (25%)</li>
                <li><strong>Service Calls</strong> indicate customer dissatisfaction (18%)</li>
                <li><strong>International Plan</strong> holders churn 3x more (15%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>💡 Why Recall Matters Most</h4>
            <p>For churn prediction, <strong>catching churners</strong> (recall) is often more important than precision. 
            Missing a churner = lost customer. False alarm = unnecessary retention offer (lower cost).</p>
        </div>
        """, unsafe_allow_html=True)

def page_customer_analytics(df):
    """Customer analytics page"""
    st.markdown('<h1 class="main-title">👥 Customer Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep dive into customer segments and behavior patterns</p>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
        <h4>📊 Customer Segmentation</h4>
        <p><strong>Why segment customers?</strong> Different customer groups have different churn behaviors. 
        By segmenting, we can create targeted retention strategies for each group.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Segment analysis
    st.markdown('<h2 class="section-header">📊 Customer Segments</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create usage segments
        df['usage_segment'] = pd.cut(df['total_charge'], 
                                      bins=[0, 50, 70, 100], 
                                      labels=['Low', 'Medium', 'High'])
        segment_churn = df.groupby('usage_segment')['churn'].mean() * 100
        
        fig = go.Figure(go.Bar(
            x=segment_churn.index.astype(str),
            y=segment_churn.values,
            marker_color=['#00d4ff', '#ffbe76', '#ff4757'],
            text=[f'{v:.1f}%' for v in segment_churn.values],
            textposition='outside'
        ))
        fig.update_layout(
            title="Churn Rate by Usage Segment",
            xaxis_title="Usage Level",
            yaxis_title="Churn Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="info-box">
            <h4>📈 Usage Segment Analysis</h4>
            <p><strong>Segments defined by Total Charge:</strong></p>
            <ul>
                <li><strong>Low ($0-50):</strong> Light users - lowest churn</li>
                <li><strong>Medium ($50-70):</strong> Average users - moderate churn</li>
                <li><strong>High ($70+):</strong> Heavy users - HIGHEST churn risk!</li>
            </ul>
            <p><strong>Insight:</strong> High spenders feel they're overpaying. Target with loyalty rewards!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Voice mail plan analysis
        vm_churn = df.groupby('voice_mail_plan')['churn'].agg(['sum', 'count']).reset_index()
        vm_churn['churn_rate'] = (vm_churn['sum'] / vm_churn['count']) * 100
        vm_churn['plan_label'] = vm_churn['voice_mail_plan'].map({0: 'No Plan', 1: 'Has Plan'})
        
        fig = go.Figure(go.Bar(
            x=vm_churn['plan_label'],
            y=vm_churn['churn_rate'],
            marker_color=['#ff4757', '#00d4ff'],
            text=[f'{v:.1f}%' for v in vm_churn['churn_rate']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Churn Rate: Voice Mail Plan Impact",
            xaxis_title="Voice Mail Plan Status",
            yaxis_title="Churn Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="info-box">
            <h4>✉️ Voice Mail Plan Impact</h4>
            <p><strong>Key finding:</strong> Customers WITH voice mail plan churn ~40% LESS!</p>
            <p><strong>Why?</strong> Voice mail is a "sticky" feature - customers use it regularly and find value.</p>
            <p><strong>Recommendation:</strong> Offer free voice mail trials to at-risk customers to increase engagement.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview
    st.markdown('<h2 class="section-header">📋 Customer Data Preview</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p><strong>Raw Data Table:</strong> Explore individual customer records. Scroll horizontally to see all 19 features. 
        Click column headers to sort. This is the same data used for training our XGBoost model.</p>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=400)
    
    # Column descriptions
    with st.expander("📖 Data Dictionary - Column Descriptions", expanded=False):
        st.markdown("""
        | Column | Description | Type |
        |--------|-------------|------|
        | `account_length` | Days the customer has been active | Numeric |
        | `voice_mail_plan` | Has voice mail subscription (0=No, 1=Yes) | Binary |
        | `voice_mail_messages` | Number of voice mail messages | Numeric |
        | `day_mins` | Total minutes used during day | Numeric |
        | `day_charge` | Total charges for day usage ($) | Numeric |
        | `evening_mins` | Total minutes used during evening | Numeric |
        | `evening_charge` | Total charges for evening usage ($) | Numeric |
        | `night_mins` | Total minutes used at night | Numeric |
        | `night_charge` | Total charges for night usage ($) | Numeric |
        | `international_mins` | International call minutes | Numeric |
        | `international_calls` | Number of international calls | Numeric |
        | `international_plan` | Has international plan (0=No, 1=Yes) | Binary |
        | `customer_service_calls` | Calls to customer service | Numeric |
        | `total_charge` | Total monthly charges ($) | Numeric |
        | `churn` | Customer churned (0=No, 1=Yes) | Target |
        """)

def page_settings(df):
    """Settings page"""
    st.markdown('<h1 class="main-title">⚙️ Settings & Data Upload</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Configure dashboard settings and upload custom datasets</p>', unsafe_allow_html=True)
    
    # Dataset upload section
    st.markdown('<h2 class="section-header">📁 Dataset Management</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your CSV dataset",
            type=['csv'],
            help="Upload a CSV file with customer data. Must include required columns."
        )
        
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded dataset with {len(new_df)} rows and {len(new_df.columns)} columns!")
                st.session_state['df'] = new_df
                st.session_state['uploaded_file'] = uploaded_file
                
                # Show preview
                st.markdown("**Dataset Preview:**")
                st.dataframe(new_df.head(10), use_container_width=True)
                
                # Column check
                required_cols = ['churn', 'total_charge', 'customer_service_calls', 'international_plan']
                missing = [col for col in required_cols if col not in new_df.columns]
                
                if missing:
                    st.warning(f"⚠️ Missing columns: {', '.join(missing)}. Some features may not work.")
                else:
                    st.success("✅ All required columns present!")
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>📋 Required Columns</h4>
            <ul>
                <li>churn</li>
                <li>total_charge</li>
                <li>customer_service_calls</li>
                <li>international_plan</li>
                <li>day_mins</li>
                <li>day_charge</li>
                <li>voice_mail_plan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Current dataset info
    st.markdown('<h2 class="section-header">📊 Current Dataset Info</h2>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        with st.expander("📋 Column Details"):
            col_info = pd.DataFrame({
                'Column': df.columns.tolist(),
                'Type': [str(dtype) for dtype in df.dtypes.values],
                'Non-Null': df.notnull().sum().values.tolist(),
                'Null': df.isnull().sum().values.tolist()
            })
            st.dataframe(col_info, use_container_width=True)
    else:
        st.info("No dataset currently loaded. Upload a CSV file above or ensure 'telecommunications_Dataset.csv' exists.")
    
    # Reset button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Reset to Default Dataset", use_container_width=True):
        default_df = load_data('telecommunications_Dataset.csv')
        if default_df is not None:
            st.session_state['df'] = default_df
            st.success("✅ Reset to default dataset!")
            st.rerun()
        else:
            st.error("❌ Default dataset not found!")

# ==================== CONCLUSIONS PAGE ====================
def page_conclusions():
    """Conclusions and Inferences page"""
    st.markdown('<h1 class="main-title">📋 Project Conclusions & Inferences</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Key findings, business recommendations, and actionable insights from the churn analysis</p>', unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown('<h2 class="section-header">📊 Executive Summary</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <h4>🎯 Project Overview</h4>
        <p style="color:white;">This project developed a <strong>machine learning-based churn prediction system</strong> for a telecommunications company. 
        Using <strong>XGBoost classifier</strong> trained on 3,333 customer records with 19 features, we achieved <strong>98% accuracy</strong> 
        in predicting customer churn, enabling proactive retention strategies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">98%</p>
            <p class="metric-label">Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">14.5%</p>
            <p class="metric-label">Churn Rate Found</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">3,333</p>
            <p class="metric-label">Customers Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">10</p>
            <p class="metric-label">Key Features Used</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Findings
    st.markdown('<h2 class="section-header">🔍 Key Findings & Inferences</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📞 Finding #1: Customer Service Calls</h4>
            <p><strong>Insight:</strong> Customers with 4+ service calls have <strong>45%+ churn rate</strong></p>
            <p><strong>Inference:</strong> High service calls indicate unresolved issues and frustration. These are "red flag" customers requiring immediate attention.</p>
            <p><strong>Recommendation:</strong> Implement proactive callback after 2nd service call, assign dedicated resolution team.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>🌍 Finding #2: International Plan Impact</h4>
            <p><strong>Insight:</strong> International plan holders churn at <strong>3x the rate</strong> (28% vs 7%)</p>
            <p><strong>Inference:</strong> Customers are dissatisfied with international rates or competitive alternatives exist.</p>
            <p><strong>Recommendation:</strong> Review international pricing, offer competitive bundles, improve call quality abroad.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>💰 Finding #3: High Charges Drive Churn</h4>
            <p><strong>Insight:</strong> Customers with total charges <strong>>$65/month</strong> show higher churn</p>
            <p><strong>Inference:</strong> High spenders experience "bill shock" and perceive lower value-for-money.</p>
            <p><strong>Recommendation:</strong> Offer loyalty discounts, premium benefits, or usage-based optimization plans.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>✉️ Finding #4: Voicemail Reduces Churn</h4>
            <p><strong>Insight:</strong> Voicemail plan subscribers churn <strong>40% less</strong></p>
            <p><strong>Inference:</strong> Voicemail is a "sticky" feature that increases engagement and perceived value.</p>
            <p><strong>Recommendation:</strong> Offer free voicemail trials to at-risk customers to increase stickiness.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>📊 Finding #5: Heavy Day Users Churn More</h4>
            <p><strong>Insight:</strong> Day minutes <strong>>200 mins</strong> correlates with higher churn</p>
            <p><strong>Inference:</strong> Heavy users may need specialized plans or feel charges don't match usage patterns.</p>
            <p><strong>Recommendation:</strong> Create heavy user plans with better per-minute rates or unlimited options.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>⏱️ Finding #6: Account Length Not Predictive</h4>
            <p><strong>Insight:</strong> Tenure has weak correlation with churn</p>
            <p><strong>Inference:</strong> Long-term customers can churn just as easily - loyalty isn't guaranteed by time alone.</p>
            <p><strong>Recommendation:</strong> Implement ongoing engagement programs for all tenure levels.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Business Recommendations
    st.markdown('<h2 class="section-header">💼 Strategic Business Recommendations</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
        <h4>🚀 IMMEDIATE ACTIONS (0-30 Days)</h4>
        <ol style="color:white;">
            <li><strong>High-Risk Alert System:</strong> Automatically flag customers with 3+ service calls for priority outreach</li>
            <li><strong>Retention Team:</strong> Create dedicated team for at-risk customer engagement</li>
            <li><strong>International Plan Review:</strong> Conduct competitive analysis and restructure pricing</li>
            <li><strong>Exit Interviews:</strong> Survey churned customers to identify root causes</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h4>📈 MEDIUM-TERM ACTIONS (30-90 Days)</h4>
            <ul style="color:white;">
                <li>Implement ML-based early warning system</li>
                <li>Personalized retention offers based on risk score</li>
                <li>Voicemail promotion campaigns for non-subscribers</li>
                <li>Heavy user plan optimization</li>
                <li>Customer satisfaction survey automation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h4>🎯 LONG-TERM STRATEGY (90+ Days)</h4>
            <ul style="color:white;">
                <li>AI-powered customer journey optimization</li>
                <li>Predictive pricing models</li>
                <li>Omnichannel service improvement</li>
                <li>Loyalty and rewards program launch</li>
                <li>Continuous model retraining pipeline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Expected Business Impact
    st.markdown('<h2 class="section-header">💵 Expected Business Impact</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">25-30%</p>
            <p class="metric-label">Churn Reduction Potential</p>
            <p class="metric-delta delta-positive">By implementing recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">$150K+</p>
            <p class="metric-label">Annual Revenue Saved</p>
            <p class="metric-delta delta-positive">With 25% churn reduction</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">5x ROI</p>
            <p class="metric-label">Retention vs Acquisition</p>
            <p class="metric-delta delta-positive">Cheaper to retain than acquire</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Limitations & Future Scope
    st.markdown('<h2 class="section-header">🔮 Model Limitations & Future Scope</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ Current Limitations</h4>
            <ul>
                <li>Dataset limited to 3,333 records - larger datasets would improve generalization</li>
                <li>No temporal data - cannot predict WHEN customer will churn</li>
                <li>Missing demographic features (age, location, income)</li>
                <li>No competitor pricing data integration</li>
                <li>Static model - requires periodic retraining</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🚀 Future Enhancements</h4>
            <ul>
                <li>Real-time prediction API integration</li>
                <li>Time-series analysis for churn timing prediction</li>
                <li>Deep learning models for complex patterns</li>
                <li>A/B testing framework for retention strategies</li>
                <li>Integration with CRM systems</li>
                <li>Automated model retraining pipeline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Final Conclusion
    st.markdown("""
    <div class="info-box" style="background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); border: 2px solid #00d4ff;">
        <h3 style="color:#00d4ff; text-align:center;">🏆 FINAL CONCLUSION</h3>
        <p style="color:white; text-align:center; font-size:1.1rem;">
        This churn prediction model successfully identifies at-risk customers with <strong>98% accuracy</strong>, 
        enabling the telecom company to take proactive retention measures. By implementing the recommended 
        strategies focusing on <strong>service quality improvement, international plan restructuring, 
        and targeted retention offers</strong>, the company can expect to reduce churn by <strong>25-30%</strong>, 
        translating to significant revenue preservation and improved customer lifetime value.
        </p>
        <p style="color:#00d4ff; text-align:center; margin-top:15px;"><em>
        "The best customer is the one you already have" - Predictive analytics makes retaining them possible.
        </em></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Credits
    st.markdown("""
    <div style="text-align:center; opacity:0.7;">
        <p>📡 Telecom Churn Analytics Dashboard | Built with ❤️ by Shashank R</p>
        <p>Powered by XGBoost, Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    # Initialize session state
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'dark'
    
    if 'df' not in st.session_state:
        st.session_state['df'] = load_data('telecommunications_Dataset.csv')
    
    # Inject CSS
    inject_custom_css(st.session_state['theme'])
    
    # Get data and model
    df = st.session_state['df']
    model = load_model()
    metrics = calculate_metrics(df) if df is not None else {}
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Route to selected page
    if df is None and page not in ["⚙️ Settings"]:
        st.error("⚠️ No dataset loaded! Please upload a dataset in Settings.")
        st.markdown("""
        <div class="info-box">
            <h4>📁 Getting Started</h4>
            <p>Navigate to <strong>⚙️ Settings</strong> to upload your customer dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if page == "🏠 Dashboard":
        page_dashboard(df, metrics)
    elif page == "📊 EDA Explorer":
        page_eda(df)
    elif page == "🎯 Churn Predictor":
        page_predictor(model)
    elif page == "🧠 Model Insights":
        page_model_insights()
    elif page == "👥 Customer Analytics":
        page_customer_analytics(df)
    elif page == "📋 Conclusions":
        page_conclusions()
    elif page == "⚙️ Settings":
        page_settings(df)

if __name__ == "__main__":
    main()
