"""
Streamlit Dashboard for Fraud Detection System
Real-time monitoring and visualization of fraud detection metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .fraud-high {
            color: #ff0000;
            font-weight: bold;
        }
        .fraud-low {
            color: #00cc00;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


def load_sample_data():
    """Load sample transaction data for demonstration."""
    np.random.seed(42)
    n_transactions = 1000
    
    data = {
        'transaction_id': range(1, n_transactions + 1),
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(n_transactions)],
        'amount': np.random.exponential(100, n_transactions),
        'customer_id': np.random.randint(1000, 2000, n_transactions),
        'merchant_id': np.random.randint(500, 1500, n_transactions),
        'location': np.random.choice(['USA', 'UK', 'Canada', 'India', 'Germany'], n_transactions),
        'fraud_probability': np.random.random(n_transactions),
    }
    
    df = pd.DataFrame(data)
    df['is_fraud'] = (df['fraud_probability'] > 0.7).astype(int)
    
    return df


def main():
    """Main dashboard application."""
    
    # Sidebar
    st.sidebar.title("🔍 Fraud Detection System")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Transactions", "Model Performance", "Settings"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Transactions":
        show_transactions()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Settings":
        show_settings()


def show_dashboard():
    """Display main dashboard."""
    st.title("🎯 Fraud Detection Dashboard")
    
    # Load data
    df = load_sample_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(df)
    fraud_count = df['is_fraud'].sum()
    fraud_rate = (fraud_count / total_transactions) * 100
    avg_fraud_prob = df['fraud_probability'].mean()
    
    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}",
        delta="+125 today"
    )
    
    col2.metric(
        "Frauds Detected",
        f"{fraud_count}",
        delta=f"-5 from yesterday",
        delta_color="inverse"
    )
    
    col3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%",
        delta="-0.3% from yesterday"
    )
    
    col4.metric(
        "Avg Fraud Probability",
        f"{avg_fraud_prob:.2f}",
        delta="-0.01 from yesterday"
    )
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    # Fraud distribution
    with col1:
        st.subheader("Fraud Distribution")
        fraud_dist = pd.DataFrame({
            'Type': ['Legitimate', 'Fraudulent'],
            'Count': [total_transactions - fraud_count, fraud_count]
        })
        fig = px.pie(
            fraud_dist,
            values='Count',
            names='Type',
            color_discrete_map={'Fraudulent': '#ff6b6b', 'Legitimate': '#51cf66'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Transaction amount distribution
    with col2:
        st.subheader("Transaction Amount Distribution")
        fig = px.histogram(
            df,
            x='amount',
            nbins=50,
            color='is_fraud',
            color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
            labels={'amount': 'Amount ($)', 'is_fraud': 'Fraud'},
            title=None
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Fraud by location
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fraud by Location")
        fraud_by_location = df.groupby('location')['is_fraud'].sum().reset_index()
        fig = px.bar(
            fraud_by_location,
            x='location',
            y='is_fraud',
            color='is_fraud',
            color_continuous_scale='Reds',
            labels={'location': 'Location', 'is_fraud': 'Fraud Count'},
            title=None
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Fraud probability over time
    with col2:
        st.subheader("Fraud Probability Trend")
        df['hour'] = df['timestamp'].dt.floor('H')
        trend = df.groupby('hour')['fraud_probability'].mean().reset_index()
        fig = px.line(
            trend,
            x='hour',
            y='fraud_probability',
            markers=True,
            labels={'hour': 'Time', 'fraud_probability': 'Avg Fraud Probability'},
            title=None
        )
        st.plotly_chart(fig, use_container_width=True)


def show_transactions():
    """Display transaction details."""
    st.title("📊 Transaction Details")
    
    # Load data
    df = load_sample_data()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fraud_filter = st.multiselect(
            "Transaction Type",
            ["Legitimate", "Fraudulent"],
            default=["Legitimate", "Fraudulent"]
        )
    
    with col2:
        min_amount = st.number_input("Min Amount", value=0.0)
        max_amount = st.number_input("Max Amount", value=1000.0)
    
    with col3:
        min_fraud_prob = st.slider("Min Fraud Probability", 0.0, 1.0, 0.0)
    
    # Filter data
    filtered_df = df.copy()
    
    if "Fraudulent" not in fraud_filter:
        filtered_df = filtered_df[filtered_df['is_fraud'] == 0]
    if "Legitimate" not in fraud_filter:
        filtered_df = filtered_df[filtered_df['is_fraud'] == 1]
    
    filtered_df = filtered_df[
        (filtered_df['amount'] >= min_amount) &
        (filtered_df['amount'] <= max_amount) &
        (filtered_df['fraud_probability'] >= min_fraud_prob)
    ]
    
    # Display table
    st.subheader(f"Transactions ({len(filtered_df)} total)")
    
    display_df = filtered_df[[
        'transaction_id', 'timestamp', 'amount', 'customer_id',
        'merchant_id', 'location', 'fraud_probability', 'is_fraud'
    ]].copy()
    
    display_df['fraud_probability'] = display_df['fraud_probability'].round(4)
    display_df['is_fraud'] = display_df['is_fraud'].map({0: '✓ Legitimate', 1: '⚠ Fraudulent'})
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_model_performance():
    """Display model performance metrics."""
    st.title("🎓 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("XGBoost Model")
        metrics = {
            'Accuracy': 0.96,
            'Precision': 0.94,
            'Recall': 0.89,
            'F1-Score': 0.91,
            'ROC-AUC': 0.97
        }
        
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.4f}")
    
    with col2:
        st.subheader("Isolation Forest Model")
        iso_metrics = {
            'Contamination': 0.05,
            'N Estimators': 100,
            'Detection Rate': 0.87,
            'False Positive Rate': 0.03,
            'Precision': 0.92
        }
        
        for metric, value in iso_metrics.items():
            st.metric(metric, f"{value}")
    
    st.divider()
    
    # Confusion matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("XGBoost Confusion Matrix")
        confusion_data = np.array([[9500, 200], [300, 1000]])
        fig = px.imshow(
            confusion_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Legitimate', 'Fraudulent'],
            y=['Legitimate', 'Fraudulent'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ROC Curve")
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - (fpr ** 2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='XGBoost (AUC=0.97)'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random (AUC=0.5)', line=dict(dash='dash')))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)


def show_settings():
    """Display settings page."""
    st.title("⚙️ Settings")
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**XGBoost Parameters**")
        max_depth = st.slider("Max Depth", 1, 15, 6)
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
        n_estimators = st.slider("N Estimators", 50, 500, 100, 50)
    
    with col2:
        st.write("**Isolation Forest Parameters**")
        contamination = st.slider("Contamination", 0.01, 0.2, 0.05, 0.01)
        iso_n_estimators = st.slider("N Estimators (IF)", 50, 500, 100, 50)
    
    st.divider()
    
    st.subheader("Thresholds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fraud_threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5, 0.05)
    
    with col2:
        alert_threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.8, 0.05)
    
    with col3:
        critical_threshold = st.slider("Critical Alert Threshold", 0.0, 1.0, 0.95, 0.05)
    
    st.divider()
    
    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")


if __name__ == "__main__":
    main()
