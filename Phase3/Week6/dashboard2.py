import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
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
    .main-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
        text-align: center;
    }
    
    .big-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        margin: 10px 0;
    }
    
    .small-text {
        color: #666;
        font-size: 0.9rem;
    }
    
    .excellent {
        background: linear-gradient(135deg, #4CAF50, #8BC34A);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .good {
        background: linear-gradient(135deg, #2196F3, #03DAC6);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .warning {
        background: linear-gradient(135deg, #FF9800, #FFC107);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .alert {
        background: linear-gradient(135deg, #F44336, #E91E63);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .tip-box {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .cost-box {
        background: #e3f2fd;
        border: 2px solid #2196F3;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_sample_data():
    """Create sample energy data for demonstration."""
    # Generate 60 days of sample data
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    
    # Create realistic energy patterns
    np.random.seed(42)  # For reproducible data
    
    data = []
    for i, date in enumerate(dates):
        # Base consumption with seasonal variation
        base_usage = 2.5 + 0.5 * np.sin(i * 0.1)  # Seasonal variation
        daily_usage = base_usage + np.random.normal(0, 0.3)  # Random variation
        daily_usage = max(0.5, daily_usage)  # Minimum usage
        
        # Peak usage is usually higher
        peak_usage = daily_usage * (1.2 + np.random.uniform(0, 0.5))
        
        # Solar generation (weather dependent)
        solar_potential = 3.0 + 0.8 * np.sin(i * 0.08)
        weather_factor = np.random.uniform(0.3, 1.0)  # Cloud cover etc.
        solar_generated = max(0, solar_potential * weather_factor)
        
        data.append({
            'day': date,
            'daily_usage': round(daily_usage, 2),
            'peak_usage': round(peak_usage, 2),
            'solar_generated': round(solar_generated, 2)
        })
    
    df = pd.DataFrame(data)
    return df

@st.cache_data
def load_energy_data():
    """Load energy data with simple error handling."""
    try:
        # Try to load the actual file first
        for delimiter in ['\t', ',', ';']:
            try:
                df = pd.read_csv('energy_features01.csv', delimiter=delimiter)
                if len(df.columns) > 1:
                    break
            except:
                continue
        
        # Check if we loaded actual data
        if 'df' not in locals() or len(df.columns) <= 1:
            raise FileNotFoundError("Could not load CSV file")
        
        # Check if we have the right columns
        if 'day' not in df.columns:
            raise ValueError("CSV file needs a 'day' column with dates")
            
        # Convert dates
        df['day'] = pd.to_datetime(df['day'])
        
        # Make sure we have the energy columns
        energy_cols = ['avg_consumption_kwh', 'peak_consumption_kwh', 'solar_output_kwh']
        for col in energy_cols:
            if col not in df.columns:
                df[col] = 0  # Add missing columns as zeros
        
        # Group by day and get daily totals
        daily_df = df.groupby('day').agg({
            'avg_consumption_kwh': 'mean',
            'peak_consumption_kwh': 'max',
            'solar_output_kwh': 'sum'
        }).reset_index()
        
        # Rename for clarity
        daily_df.columns = ['day', 'daily_usage', 'peak_usage', 'solar_generated']
        
    except (FileNotFoundError, ValueError) as e:
        # If we can't load the file, create sample data
        st.info("📝 Using sample data for demonstration. To use your own data, upload 'energy_features01.csv' to the same directory.")
        daily_df = create_sample_data()
    
    # Calculate money saved and net usage
    daily_df['money_saved'] = daily_df['solar_generated'] * 0.12  # 12 cents per kWh
    daily_df['net_usage'] = daily_df['daily_usage'] - daily_df['solar_generated']
    daily_df['net_usage'] = daily_df['net_usage'].clip(lower=0)  # Can't be negative
    
    daily_df = daily_df.set_index('day').sort_index()
    
    return daily_df

def simple_forecast(data, days=30):
    """Simple forecast that's easy to understand."""
    try:
        # Use last 30 days average if we have enough data
        if len(data) >= 30:
            recent_average = data.tail(30).mean()
        else:
            recent_average = data.mean()
        
        # Add some seasonal variation (higher in summer/winter)
        today = datetime.now()
        future_dates = [today + timedelta(days=i) for i in range(1, days + 1)]
        
        forecast = []
        for date in future_dates:
            # Simple seasonal adjustment
            month = date.month
            if month in [12, 1, 2, 6, 7, 8]:  # Winter and summer months
                seasonal_factor = 1.1  # 10% higher usage
            else:
                seasonal_factor = 0.95  # 5% lower usage
            
            # Add small random variation
            daily_forecast = recent_average * seasonal_factor * np.random.uniform(0.9, 1.1)
            forecast.append(daily_forecast)
        
        return forecast, future_dates
        
    except Exception as e:
        st.error(f"❌ Forecast error: {e}")
        return None, None

# Load data
df = load_energy_data()
if df is None or len(df) == 0:
    st.error("❌ No data available. Please check your data file.")
    st.stop()

# Header
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="color: #2E8B57;">🏠 My Energy Dashboard</h1>
    <p style="color: #666; font-size: 1.1rem;">Simple insights about your home energy use</p>
</div>
""", unsafe_allow_html=True)

# Simple sidebar
with st.sidebar:
    st.header("📅 Choose Dates")
    
    # Date picker
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Ensure start_date <= end_date
    if start_date > end_date:
        st.error("Start date must be before end date")
        st.stop()
    
    st.header("⚠️ Set Alerts")
    usage_alert = st.number_input("Alert when daily usage exceeds:", value=3.0, step=0.5, min_value=0.1)
    
    st.header("💰 Energy Cost")
    cost_per_kwh = st.number_input("Cost per kWh ($):", value=0.12, step=0.01, min_value=0.01)

# Filter data
filtered_df = df.loc[start_date:end_date]

if len(filtered_df) == 0:
    st.error("❌ No data available for the selected date range.")
    st.stop()

# Main metrics
st.header("📊 Your Energy Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_daily = filtered_df['daily_usage'].mean()
    st.markdown(f"""
    <div class="main-card">
        <h4>Daily Average</h4>
        <div class="big-number">{avg_daily:.1f}</div>
        <div class="small-text">kWh per day</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    monthly_cost = avg_daily * 30 * cost_per_kwh
    st.markdown(f"""
    <div class="main-card">
        <h4>Monthly Cost</h4>
        <div class="big-number">${monthly_cost:.0f}</div>
        <div class="small-text">estimated</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_solar = filtered_df['solar_generated'].sum()
    st.markdown(f"""
    <div class="main-card">
        <h4>Solar Generated</h4>
        <div class="big-number">{total_solar:.0f}</div>
        <div class="small-text">kWh total</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    money_saved = filtered_df['money_saved'].sum()
    st.markdown(f"""
    <div class="main-card">
        <h4>Money Saved</h4>
        <div class="big-number">${money_saved:.0f}</div>
        <div class="small-text">from solar</div>
    </div>
    """, unsafe_allow_html=True)

# Status check
st.header("🚦 How Are You Doing?")

max_usage = filtered_df['peak_usage'].max()
recent_usage = filtered_df['daily_usage'].tail(7).mean()

if max_usage > usage_alert:
    st.markdown(f"""
    <div class="alert">
        <h4>🚨 High Usage Alert!</h4>
        <p>Your peak usage was {max_usage:.1f} kWh, which is above your alert level of {usage_alert:.1f} kWh.</p>
        <p><strong>What to do:</strong> Check what appliances were running on high-usage days.</p>
    </div>
    """, unsafe_allow_html=True)
elif recent_usage > avg_daily * 1.1:
    st.markdown(f"""
    <div class="warning">
        <h4>📈 Usage Increasing</h4>
        <p>Your recent usage ({recent_usage:.1f} kWh/day) is higher than your average.</p>
        <p><strong>Tip:</strong> Review your energy habits this week.</p>
    </div>
    """, unsafe_allow_html=True)
elif recent_usage < avg_daily * 0.9:
    st.markdown(f"""
    <div class="excellent">
        <h4>🎉 Great Job!</h4>
        <p>Your recent usage ({recent_usage:.1f} kWh/day) is lower than your average!</p>
        <p><strong>Keep it up!</strong> You're saving money and energy.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="good">
        <h4>✅ Normal Usage</h4>
        <p>Your usage is steady at {recent_usage:.1f} kWh per day.</p>
        <p><strong>Good work!</strong> You're maintaining consistent energy habits.</p>
    </div>
    """, unsafe_allow_html=True)

# Simple charts
st.header("📈 Your Energy Over Time")

# Chart 1: Daily usage
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=filtered_df.index,
    y=filtered_df['daily_usage'],
    mode='lines+markers',
    name='Daily Usage',
    line=dict(color='#2E8B57', width=3)
))

fig1.update_layout(
    title="Daily Energy Usage",
    xaxis_title="Date",
    yaxis_title="Energy Used (kWh)",
    showlegend=False,
    height=400
)

st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Solar vs Usage (if solar data exists)
if filtered_df['solar_generated'].sum() > 0:
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df['daily_usage'],
        mode='lines',
        name='Energy Used',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig2.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df['solar_generated'],
        mode='lines',
        name='Solar Generated',
        line=dict(color='#FFA500', width=2),
        fill='tozeroy'
    ))
    
    fig2.update_layout(
        title="Energy Used vs Solar Generated",
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# Simple forecast
st.header("🔮 What to Expect Next Month")

forecast_values, forecast_dates = simple_forecast(filtered_df['daily_usage'])

if forecast_values is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        forecast_avg = np.mean(forecast_values)
        forecast_cost = forecast_avg * 30 * cost_per_kwh
        
        st.markdown(f"""
        <div class="cost-box">
            <h4>Next Month Prediction</h4>
            <p><strong>Daily Average:</strong> {forecast_avg:.1f} kWh</p>
            <p><strong>Monthly Cost:</strong> ${forecast_cost:.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple trend
        if forecast_avg > avg_daily * 1.1:
            st.markdown("📈 **Trend:** Usage may increase")
        elif forecast_avg < avg_daily * 0.9:
            st.markdown("📉 **Trend:** Usage may decrease")
        else:
            st.markdown("📊 **Trend:** Usage should stay similar")
    
    with col2:
        # Forecast chart
        fig3 = go.Figure()
        
        # Recent data
        recent_data = filtered_df['daily_usage'].tail(30)
        fig3.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data.values,
            mode='lines',
            name='Recent Usage',
            line=dict(color='#2E8B57', width=3)
        ))
        
        # Forecast
        fig3.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        fig3.update_layout(
            title="30-Day Usage Forecast",
            xaxis_title="Date",
            yaxis_title="Daily Usage (kWh)",
            height=400
        )
        
        st.plotly_chart(fig3, use_container_width=True)

# Simple tips
st.header("💡 Easy Ways to Save Energy")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="tip-box">
        <h4>🌡️ Heating & Cooling</h4>
        <ul>
            <li>Set thermostat to 68°F in winter, 78°F in summer</li>
            <li>Use fans to feel cooler without changing temperature</li>
            <li>Close curtains during hot days</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-box">
        <h4>🔌 Appliances</h4>
        <ul>
            <li>Unplug devices when not in use</li>
            <li>Use cold water for washing clothes</li>
            <li>Run dishwasher only when full</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tip-box">
        <h4>💡 Lighting</h4>
        <ul>
            <li>Switch to LED bulbs</li>
            <li>Turn off lights when leaving rooms</li>
            <li>Use natural light during the day</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if filtered_df['solar_generated'].sum() > 0:
        st.markdown("""
        <div class="tip-box">
            <h4>☀️ Solar Tips</h4>
            <ul>
                <li>Use heavy appliances during sunny hours (10am-4pm)</li>
                <li>Charge devices during peak solar production</li>
                <li>Consider battery storage for extra solar power</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Monthly summary
if len(filtered_df) > 30:
    st.header("📅 Monthly Breakdown")
    
    monthly_summary = filtered_df.resample('M').agg({
        'daily_usage': 'mean',
        'peak_usage': 'max',
        'solar_generated': 'sum',
        'money_saved': 'sum'
    }).round(1)
    
    if len(monthly_summary) > 0:
        monthly_summary.index = monthly_summary.index.strftime('%B %Y')
        monthly_summary.columns = ['Avg Daily (kWh)', 'Peak Day (kWh)', 'Solar Total (kWh)', 'Money Saved ($)']
        
        st.dataframe(monthly_summary, use_container_width=True)

# Simple data download
st.header("📥 Download Your Data")

col1, col2 = st.columns(2)

with col1:
    csv_data = filtered_df.to_csv()
    st.download_button(
        label="📊 Download Energy Data",
        data=csv_data,
        file_name=f'my_energy_data_{start_date}_to_{end_date}.csv',
        mime='text/csv'
    )

with col2:
    if forecast_values is not None:
        forecast_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'Predicted Usage (kWh)': [round(v, 1) for v in forecast_values]
        })
        forecast_csv = forecast_df.to_csv(index=False)
        
        st.download_button(
            label="🔮 Download Forecast",
            data=forecast_csv,
            file_name='energy_forecast_30_days.csv',
            mime='text/csv'
        )

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>🏠 My Energy Dashboard</strong></p>
    <p>Viewing data from {start_date} to {end_date}</p>
    <p>Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
</div>
""", unsafe_allow_html=True)