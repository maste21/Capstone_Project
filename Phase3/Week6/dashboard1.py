import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyodbc
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(
    page_title="My Energy Dashboard", 
    layout="wide",
    page_icon="🏠"
)

# Simple, clean styling
st.markdown("""
<style>
    .big-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .status-good {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .status-alert {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_energy_data():
    """Load your energy data directly from SQL Server"""
    try:
        conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=ge-prd.database.windows.net;"
            "DATABASE=GreenEnergy_DBP;"
            "UID=Nalinpgdde@chndsrnvsgmail.onmicrosoft.com;"
            "PWD=Neilapple7#;"
            "Authentication=ActiveDirectoryPassword;"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
        )
        conn = pyodbc.connect(conn_str)
        query = "SELECT * FROM [dbo].[vw_daily_consumption_summary]"
        df = pd.read_sql(query, conn)
        
        df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d')  # Assuming ISO format
        df = df.set_index('day')
        df = df.sort_index()

        conn.close()
        return df
    except Exception as e:
        st.error(f"⚠️ Problem loading your data: {e}")
        return None

def simple_forecast(data, days_ahead=30):
    """Create a simple forecast that anyone can understand"""
    try:
        # Use a simple ARIMA model
        model = ARIMA(data, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Predict the future
        future_values = fitted_model.forecast(steps=days_ahead)
        
        # Create dates for the forecast
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
        
        return future_values, future_dates
    except:
        return None, None

# Load the data
df = load_energy_data()
if df is None:
    st.stop()

# 🏠 WELCOME HEADER
st.title("🏠 My Energy Dashboard")
st.markdown("**Understanding your energy usage made simple**")
st.markdown("---")

# 📅 DATE PICKER (Simple version)
st.sidebar.header("📅 Choose Your Time Period")
st.sidebar.markdown("*Select the dates you want to look at*")

min_date = df.index.min().to_pydatetime().date()
max_date = df.index.max().to_pydatetime().date()

# Simple date inputs
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)

# Filter the data
filtered_df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

# 🔔 SIMPLE ALERTS
st.sidebar.header("🔔 Set Your Alerts")
st.sidebar.markdown("*Get notified when things are unusual*")

high_usage_alert = st.sidebar.number_input(
    "Alert me when daily usage goes above (kWh):", 
    min_value=0.0, 
    value=2.5, 
    step=0.1,
    help="You'll see a warning if your energy use goes above this amount"
)

# Check for alerts
max_usage = filtered_df['peak_kwh'].max()
avg_usage = filtered_df['daily_avg_kwh'].mean()

# 📊 YOUR ENERGY AT A GLANCE
st.header("📊 Your Energy at a Glance")
st.markdown("*Here's what's happening with your energy usage*")

# Big, easy-to-read numbers
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Average Daily Use**")
    st.markdown(f'<div class="big-number">{avg_usage:.1f}</div>', unsafe_allow_html=True)
    st.markdown("kWh per day")

with col2:
    st.markdown("**Highest Single Day**")
    st.markdown(f'<div class="big-number">{max_usage:.1f}</div>', unsafe_allow_html=True)
    st.markdown("kWh peak usage")

with col3:
    total_solar = filtered_df['total_solar'].sum()
    st.markdown("**Solar Power Generated**")
    st.markdown(f'<div class="big-number">{total_solar/1000:.1f}</div>', unsafe_allow_html=True)
    st.markdown("MWh total")

with col4:
    days_selected = len(filtered_df)
    st.markdown("**Days Analyzed**")
    st.markdown(f'<div class="big-number">{days_selected}</div>', unsafe_allow_html=True)
    st.markdown("days of data")

# 🚨 STATUS CHECK
st.header("🚨 How Are You Doing?")

if max_usage > high_usage_alert:
    st.markdown(f'''
    <div class="status-alert">
        <strong>⚠️ High Usage Alert!</strong><br>
        Your peak usage ({max_usage:.1f} kWh) went above your alert level ({high_usage_alert:.1f} kWh).<br>
        This happened on {filtered_df.loc[filtered_df['peak_kwh'].idxmax()].name.strftime('%B %d, %Y')}.
    </div>
    ''', unsafe_allow_html=True)
elif avg_usage > 2.0:
    st.markdown(f'''
    <div class="status-warning">
        <strong>📊 Normal Usage</strong><br>
        Your average daily usage is {avg_usage:.1f} kWh. This is within normal range.
    </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown(f'''
    <div class="status-good">
        <strong>✅ Great Job!</strong><br>
        Your average daily usage is {avg_usage:.1f} kWh. You're doing well with energy conservation!
    </div>
    ''', unsafe_allow_html=True)

# 📈 SIMPLE CHARTS
st.header("📈 Your Energy Patterns")
st.markdown("*Visual look at your energy usage over time*")

# Chart 1: Daily usage
st.subheader("Daily Energy Usage")
st.markdown("*This shows how much energy you use each day*")

chart_data = filtered_df[['daily_avg_kwh']].rename(columns={'daily_avg_kwh': 'Daily Usage (kWh)'})
st.line_chart(chart_data)

# Chart 2: Solar generation
if filtered_df['total_solar'].sum() > 0:
    st.subheader("Solar Power You Generated")
    st.markdown("*This shows how much solar power you produced*")
    
    solar_data = filtered_df[['total_solar']].rename(columns={'total_solar': 'Solar Generation (kWh)'})
    st.area_chart(solar_data)

# 📅 MONTHLY SUMMARY
if len(filtered_df) > 30:
    st.header("📅 Monthly Summary")
    st.markdown("*How you're doing month by month*")
    
    monthly_data = filtered_df.resample('M').agg({
        'daily_avg_kwh': 'mean',
        'peak_kwh': 'max',
        'total_solar': 'sum'
    }).round(1)
    
    monthly_data.columns = ['Avg Daily Use (kWh)', 'Peak Day (kWh)', 'Solar Generated (kWh)']
    
    st.dataframe(monthly_data, use_container_width=True)

# 🔮 SIMPLE FORECAST
st.header("🔮 What to Expect Next Month")
st.markdown("*Based on your past usage, here's what we predict*")

forecast_days = 30
forecast_values, forecast_dates = simple_forecast(filtered_df['daily_avg_kwh'], forecast_days)

if forecast_values is not None:
    # Show forecast summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Forecast Summary**")
        st.markdown(f"• **Expected daily average:** {forecast_values.mean():.1f} kWh")
        st.markdown(f"• **Highest expected day:** {forecast_values.max():.1f} kWh")
        st.markdown(f"• **Lowest expected day:** {forecast_values.min():.1f} kWh")
        
        # Simple recommendation
        if forecast_values.mean() > avg_usage * 1.1:
            st.markdown("📈 **Trend:** Your usage might increase next month")
        elif forecast_values.mean() < avg_usage * 0.9:
            st.markdown("📉 **Trend:** Your usage might decrease next month")
        else:
            st.markdown("📊 **Trend:** Your usage should stay about the same")
    
    with col2:
        # Simple forecast chart
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Show last 30 days of actual data
        recent_data = filtered_df['daily_avg_kwh'].tail(30)
        ax.plot(recent_data.index, recent_data.values, 'b-', linewidth=2, label='Your Recent Usage')
        
        # Show forecast
        ax.plot(forecast_dates, forecast_values, 'r--', linewidth=2, label='Predicted Usage')
        
        ax.set_title('Next Month Forecast', fontsize=14)
        ax.set_ylabel('Daily Usage (kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# 💡 SIMPLE TIPS
st.header("💡 Tips to Save Energy")
st.markdown("*Based on your usage patterns*")

tips = []

# Peak usage tip
if max_usage > avg_usage * 1.5:
    tips.append("🔥 **Watch peak days**: Your highest usage day was much higher than average. Try to identify what caused the spike.")

# Solar tip
if filtered_df['total_solar'].sum() > 0:
    solar_ratio = filtered_df['total_solar'].sum() / filtered_df['daily_avg_kwh'].sum()
    if solar_ratio < 0.5:
        tips.append("☀️ **Use more solar**: You're not using all your solar power. Consider shifting energy use to sunny hours.")

# Consistency tip
usage_variation = filtered_df['daily_avg_kwh'].std()
if usage_variation > avg_usage * 0.3:
    tips.append("📊 **Consistent usage**: Your daily usage varies quite a bit. Try to maintain more consistent habits.")

# General tips
tips.extend([
    "🌙 **Night savings**: Use major appliances during off-peak hours if possible",
    "🔧 **Regular maintenance**: Keep appliances clean and well-maintained for efficiency",
    "💰 **Track progress**: Check your dashboard regularly to stay aware of your usage"
])

for tip in tips:
    st.markdown(f"• {tip}")

# 📋 SIMPLE DATA VIEW
with st.expander("📋 See Your Raw Data"):
    st.markdown("*Click here if you want to see the actual numbers*")
    
    # Simplify column names
    display_df = filtered_df.copy()
    display_df.columns = ['Daily Average (kWh)', 'Peak Usage (kWh)', 'Solar Generated (kWh)']
    
    st.dataframe(display_df)

# 📥 DOWNLOAD YOUR DATA
st.header("📥 Download Your Data")
st.markdown("*Save your energy data to use elsewhere*")

col1, col2 = st.columns(2)

with col1:
    # Simple CSV download
    csv_data = filtered_df.to_csv()
    st.download_button(
        label="💾 Download My Energy Data",
        data=csv_data,
        file_name=f'my_energy_data_{start_date}_to_{end_date}.csv',
        mime='text/csv'
    )

with col2:
    if forecast_values is not None:
        # Forecast download
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted Usage (kWh)': forecast_values.round(1)
        })
        forecast_csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="🔮 Download Forecast Data",
            data=forecast_csv,
            file_name=f'energy_forecast_{forecast_days}_days.csv',
            mime='text/csv'
        )

# 🏠 FOOTER
st.markdown("---")
st.markdown("### 🏠 Your Energy Dashboard")
st.markdown(f"**Data period:** {start_date} to {end_date} • **Last updated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
st.markdown("*This dashboard helps you understand and manage your energy usage in simple terms.*")