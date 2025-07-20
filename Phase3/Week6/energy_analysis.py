import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

class EnergyConsumptionAnalyzer:
    """
    A comprehensive class for energy consumption time series analysis using ARIMA models.
    """
    
    def __init__(self, filepath, sep='\t', datetime_col='datetime', target_col='avg_consumption_kwh'):
        self.filepath = filepath
        self.sep = sep
        self.datetime_col = datetime_col
        self.target_col = target_col
        self.df = None
        self.ts = None
        self.ts_for_modeling = None
        self.model_fit = None
        self.d_order = 0
        self.p_order = 0
        self.q_order = 0
        self.train_ts = None
        self.test_ts = None
        self.predictions = None
        self.anomalies = pd.Series([], dtype='float64')
        self.metrics = {}
        
        # Constants
        self.MIN_OBS_ADF = 5
        self.MIN_OBS_ARIMA = 24 * 7  # One week of hourly data
        self.MIN_OBS_TRAIN_TEST = 2
        self.ANOMALY_THRESHOLD = 3  # Standard deviations for anomaly detection
        
        # Create output directory
        self.output_dir = 'energy_analysis_output'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess the energy consumption data."""
        try:
            self.df = pd.read_csv(self.filepath, sep=self.sep)
            print("✓ Data loaded successfully")
            print(f"Dataset shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            # Handle datetime column
            if self.datetime_col not in self.df.columns:
                print(f"❌ '{self.datetime_col}' column not found.")
                print(f"Available columns: {list(self.df.columns)}")
                return False
                
            self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])
            self.df = self.df.set_index(self.datetime_col).sort_index()
            
            # Handle target column
            if self.target_col not in self.df.columns:
                print(f"❌ '{self.target_col}' column not found.")
                numeric_cols = self.df.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    self.target_col = numeric_cols[0]
                    print(f"✓ Using '{self.target_col}' as target variable")
                else:
                    print("❌ No numeric columns found")
                    return False
            
            # Create time series
            self.ts = self.df[self.target_col].copy()
            
            # Handle missing values
            missing_count = self.ts.isnull().sum()
            if missing_count > 0:
                print(f"⚠️  Found {missing_count} missing values. Filling with forward/backward fill.")
                self.ts = self.ts.ffill().bfill()
            
            # Ensure consistent frequency
            self.ts = self.ts.asfreq('H').ffill().bfill()
            
            print(f"✓ Time series preprocessed: {len(self.ts)} observations")
            print(f"Date range: {self.ts.index.min()} to {self.ts.index.max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def check_stationarity(self, timeseries, title="Series"):
        """Enhanced stationarity check with better error handling."""
        print(f'\n--- Stationarity Check: {title} ---')
        
        nobs = len(timeseries)
        if nobs < self.MIN_OBS_ADF:
            print(f"⚠️  Not enough observations ({nobs}) for reliable ADF test")
            return False
            
        try:
            # Try with automatic lag selection first
            dftest = adfuller(timeseries, autolag='AIC')
        except ValueError:
            # Fallback to manual lag selection
            safe_max_lag = max(1, int(nobs / 3))
            try:
                dftest = adfuller(timeseries, autolag=None, maxlag=safe_max_lag)
            except ValueError as e:
                print(f"❌ ADF test failed: {e}")
                return False
        
        # Format results
        results = pd.Series(dftest[0:4], 
                           index=['Test Statistic', 'p-value', 'Lags Used', 'Observations Used'])
        
        for key, value in dftest[4].items():
            results[f'Critical Value ({key})'] = value
            
        print(results.to_string())
        
        is_stationary = dftest[1] <= 0.05
        conclusion = "✓ Stationary" if is_stationary else "❌ Non-Stationary"
        print(f"Conclusion: {conclusion}")
        
        return is_stationary
    
    def perform_seasonal_decomposition(self):
        """Perform seasonal decomposition to understand data patterns."""
        if len(self.ts) < 2 * 24:  # Need at least 2 days for daily seasonality
            print("⚠️  Insufficient data for seasonal decomposition")
            return
            
        try:
            print("\n--- Seasonal Decomposition ---")
            decomposition = seasonal_decompose(self.ts, model='additive', period=24)
            
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            decomposition.observed.plot(ax=axes[0], title='Original')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ Seasonal decomposition plot saved")
            
        except Exception as e:
            print(f"❌ Seasonal decomposition failed: {e}")
    
    def determine_stationarity_and_differencing(self):
        """Determine if differencing is needed and apply it."""
        print("\n=== STATIONARITY ANALYSIS ===")
        
        # Check original series
        is_stationary = self.check_stationarity(self.ts, "Original Series")
        
        if is_stationary:
            self.ts_for_modeling = self.ts
            self.d_order = 0
            print("✓ Original series is stationary")
        else:
            print("\n🔄 Applying first-order differencing...")
            ts_diff = self.ts.diff().dropna()
            
            if len(ts_diff) < self.MIN_OBS_ADF:
                print("⚠️  Differenced series too short for reliable analysis")
                self.ts_for_modeling = self.ts
                self.d_order = 0
            else:
                is_diff_stationary = self.check_stationarity(ts_diff, "Differenced Series")
                
                if is_diff_stationary:
                    self.ts_for_modeling = ts_diff
                    self.d_order = 1
                    print("✓ Differenced series is stationary")
                else:
                    print("⚠️  Differenced series still non-stationary, using d=1 anyway")
                    self.ts_for_modeling = ts_diff
                    self.d_order = 1
    
    def plot_acf_pacf(self):
        """Plot ACF and PACF for parameter determination."""
        if len(self.ts_for_modeling) < self.MIN_OBS_ADF:
            print("⚠️  Insufficient data for ACF/PACF plots")
            return
            
        print("\n--- ACF/PACF Analysis ---")
        
        # Calculate appropriate number of lags
        max_lags = min(len(self.ts_for_modeling) // 2 - 1, 48)  # Up to 48 for daily patterns
        max_lags = max(1, max_lags)
        
        try:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            plot_acf(self.ts_for_modeling, lags=max_lags, ax=axes[0], 
                    title=f'ACF - {self.target_col}')
            plot_pacf(self.ts_for_modeling, lags=max_lags, ax=axes[1], 
                     title=f'PACF - {self.target_col}')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/acf_pacf_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ ACF/PACF plots saved")
            
        except Exception as e:
            print(f"❌ ACF/PACF plotting failed: {e}")
    
    def split_data(self, train_ratio=0.8):
        """Split data into training and testing sets."""
        if len(self.ts) < self.MIN_OBS_TRAIN_TEST:
            print(f"❌ Insufficient data for train-test split: {len(self.ts)} observations")
            return False
            
        train_size = max(1, int(len(self.ts) * train_ratio))
        
        # Ensure at least one observation in test set
        if len(self.ts) - train_size < 1 and len(self.ts) > 1:
            train_size = len(self.ts) - 1
            
        self.train_ts = self.ts[:train_size]
        self.test_ts = self.ts[train_size:]
        
        print(f"\n--- Data Split ---")
        print(f"Training set: {len(self.train_ts)} observations")
        print(f"Test set: {len(self.test_ts)} observations")
        
        return True
    
    def fit_arima_model(self):
        """Fit ARIMA model with automatic parameter selection."""
        if len(self.train_ts) < self.MIN_OBS_ARIMA:
            print(f"⚠️  Training set too small for reliable ARIMA: {len(self.train_ts)} observations")
            print(f"Minimum recommended: {self.MIN_OBS_ARIMA}")
            
        print(f"\n=== ARIMA MODEL FITTING ===")
        
        # Try different parameter combinations
        param_combinations = [
            (1, self.d_order, 1),
            (2, self.d_order, 2),
            (5, self.d_order, 5),
            (1, self.d_order, 2),
            (2, self.d_order, 1),
        ]
        
        best_aic = float('inf')
        best_params = None
        
        for p, d, q in param_combinations:
            try:
                print(f"Trying ARIMA({p},{d},{q})...")
                model = ARIMA(self.train_ts, order=(p, d, q))
                model_fit = model.fit()
                
                aic = model_fit.aic
                print(f"  AIC: {aic:.2f}")
                
                if aic < best_aic:
                    best_aic = aic
                    best_params = (p, d, q)
                    self.model_fit = model_fit
                    self.p_order, _, self.q_order = p, d, q
                    
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
        
        if self.model_fit is not None:
            print(f"\n✓ Best model: ARIMA{best_params} (AIC: {best_aic:.2f})")
            print("\nModel Summary:")
            print(self.model_fit.summary())
            
            # Diagnostic tests
            self.perform_diagnostic_tests()
            
            return True
        else:
            print("❌ No ARIMA model could be fitted")
            return False
    
    def perform_diagnostic_tests(self):
        """Perform diagnostic tests on the fitted model."""
        print("\n--- Model Diagnostics ---")
        
        try:
            # Ljung-Box test for residual autocorrelation
            residuals = self.model_fit.resid
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            significant_lags = lb_test[lb_test['lb_pvalue'] < 0.05]
            
            if len(significant_lags) == 0:
                print("✓ Ljung-Box test: No significant autocorrelation in residuals")
            else:
                print(f"⚠️  Ljung-Box test: Significant autocorrelation detected at {len(significant_lags)} lag(s)")
                
            # Normality test for residuals
            _, p_value = stats.jarque_bera(residuals)
            if p_value > 0.05:
                print("✓ Jarque-Bera test: Residuals are normally distributed")
            else:
                print("⚠️  Jarque-Bera test: Residuals may not be normally distributed")
                
        except Exception as e:
            print(f"❌ Diagnostic tests failed: {e}")
    
    def forecast_and_evaluate(self):
        """Make forecasts and evaluate model performance."""
        if self.model_fit is None or len(self.test_ts) == 0:
            print("⚠️  Cannot perform forecasting: model not fitted or no test data")
            return
            
        print("\n=== MODEL EVALUATION ===")
        
        try:
            # Make forecasts
            forecast_steps = len(self.test_ts)
            self.predictions = self.model_fit.forecast(steps=forecast_steps)
            self.predictions.index = self.test_ts.index
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.test_ts, self.predictions))
            mae = mean_absolute_error(self.test_ts, self.predictions)
            r2 = r2_score(self.test_ts, self.predictions)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((self.test_ts - self.predictions) / self.test_ts)) * 100
            
            self.metrics = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            }
            
            print(f"Test RMSE: {rmse:.4f}")
            print(f"Test MAE: {mae:.4f}")
            print(f"Test R²: {r2:.4f}")
            print(f"Test MAPE: {mape:.2f}%")
            
            # Plot forecast results
            self.plot_forecast_results()
            
        except Exception as e:
            print(f"❌ Forecasting failed: {e}")
    
    def plot_forecast_results(self):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(15, 8))
        
        # Plot training data
        plt.plot(self.train_ts.index, self.train_ts, 
                label='Training Data', color='gray', alpha=0.7)
        
        # Plot test data
        plt.plot(self.test_ts.index, self.test_ts, 
                label='Actual (Test)', color='blue', linewidth=2)
        
        # Plot predictions
        plt.plot(self.predictions.index, self.predictions, 
                label='Forecast', color='red', linewidth=2, linestyle='--')
        
        plt.title(f'Energy Consumption Forecasting - {self.target_col}')
        plt.xlabel('Date')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/forecast_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Forecast plot saved")
    
    def detect_anomalies(self):
        """Detect anomalies using residual analysis."""
        if self.model_fit is None:
            print("⚠️  Cannot detect anomalies: model not fitted")
            return
            
        print("\n=== ANOMALY DETECTION ===")
        
        try:
            # Get predictions for full series
            full_predictions = self.model_fit.predict(start=0, end=len(self.ts)-1, typ='levels')
            full_predictions.index = self.ts.index
            
            # Calculate residuals
            residuals = self.ts - full_predictions
            
            if len(residuals) < 2:
                print("⚠️  Insufficient data for anomaly detection")
                return
                
            # Define anomaly thresholds
            residual_mean = residuals.mean()
            residual_std = residuals.std()
            
            upper_threshold = residual_mean + self.ANOMALY_THRESHOLD * residual_std
            lower_threshold = residual_mean - self.ANOMALY_THRESHOLD * residual_std
            
            # Identify anomalies
            self.anomalies = residuals[
                (residuals > upper_threshold) | (residuals < lower_threshold)
            ]
            
            print(f"✓ Detected {len(self.anomalies)} anomalies")
            print(f"Anomaly threshold: ±{self.ANOMALY_THRESHOLD} std devs")
            
            if len(self.anomalies) > 0:
                print(f"Anomaly dates: {self.anomalies.index.tolist()[:5]}")  # Show first 5
                
            # Plot anomaly results
            self.plot_anomaly_results(residuals, upper_threshold, lower_threshold)
            
        except Exception as e:
            print(f"❌ Anomaly detection failed: {e}")
    
    def plot_anomaly_results(self, residuals, upper_threshold, lower_threshold):
        """Plot residuals and detected anomalies."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Residuals with thresholds
        axes[0].plot(residuals.index, residuals, label='Residuals', color='gray', alpha=0.7)
        axes[0].axhline(y=0, color='green', linestyle='--', label='Mean Residual')
        axes[0].axhline(y=upper_threshold, color='red', linestyle='-', label='Upper Threshold')
        axes[0].axhline(y=lower_threshold, color='red', linestyle='-', label='Lower Threshold')
        axes[0].scatter(self.anomalies.index, self.anomalies, 
                       color='red', s=50, zorder=5, label='Anomalies')
        axes[0].set_title('Residuals and Anomaly Detection')
        axes[0].set_ylabel('Residual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Original series with anomalies
        axes[1].plot(self.ts.index, self.ts, label=f'Actual {self.target_col}', color='blue')
        if len(self.anomalies) > 0:
            axes[1].scatter(self.anomalies.index, self.ts.loc[self.anomalies.index], 
                           color='red', s=100, zorder=5, label='Anomalies')
        axes[1].set_title('Energy Consumption with Detected Anomalies')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel(self.target_col)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/anomaly_detection.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Anomaly detection plots saved")
    
    def plot_original_series(self):
        """Plot the original time series."""
        plt.figure(figsize=(15, 6))
        plt.plot(self.ts.index, self.ts, label=f'Original {self.target_col}')
        plt.title(f'Energy Consumption Over Time - {self.target_col}')
        plt.xlabel('Date')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/original_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Original series plot saved")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("           ENERGY CONSUMPTION ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   • File: {self.filepath}")
        print(f"   • Target variable: {self.target_col}")
        print(f"   • Total observations: {len(self.ts)}")
        print(f"   • Date range: {self.ts.index.min()} to {self.ts.index.max()}")
        print(f"   • Data frequency: Hourly")
        
        print(f"\n🔍 STATIONARITY ANALYSIS:")
        print(f"   • Differencing order (d): {self.d_order}")
        print(f"   • {'Applied first-order differencing' if self.d_order == 1 else 'Original series used'}")
        
        if self.model_fit is not None:
            print(f"\n📈 MODEL DETAILS:")
            print(f"   • Model: ARIMA({self.p_order}, {self.d_order}, {self.q_order})")
            print(f"   • Training observations: {len(self.train_ts)}")
            print(f"   • Test observations: {len(self.test_ts)}")
            print(f"   • AIC: {self.model_fit.aic:.2f}")
            
            if self.metrics:
                print(f"\n📊 MODEL PERFORMANCE:")
                for metric, value in self.metrics.items():
                    if metric == 'MAPE':
                        print(f"   • {metric}: {value:.2f}%")
                    else:
                        print(f"   • {metric}: {value:.4f}")
        
        print(f"\n🚨 ANOMALY DETECTION:")
        print(f"   • Anomalies detected: {len(self.anomalies)}")
        print(f"   • Detection method: ±{self.ANOMALY_THRESHOLD} standard deviations")
        
        if len(self.anomalies) > 0:
            print(f"   • Anomaly percentage: {len(self.anomalies)/len(self.ts)*100:.2f}%")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   • All plots saved to: {self.output_dir}/")
        print(f"   • Generated plots: original_timeseries.png, forecast_results.png, anomaly_detection.png")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if len(self.ts) < self.MIN_OBS_ARIMA:
            print("   • Consider collecting more data for robust ARIMA modeling")
        print("   • Evaluate SARIMA models for seasonal patterns")
        print("   • Consider external variables (weather, holidays) for improved accuracy")
        print("   • Implement real-time monitoring using the anomaly detection framework")
        
        print("\n" + "="*60)
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("🚀 Starting Energy Consumption Analysis...")
        
        # Step 1: Load and preprocess data
        if not self.load_and_preprocess_data():
            return
        
        # Step 2: Plot original series
        self.plot_original_series()
        
        # Step 3: Seasonal decomposition
        self.perform_seasonal_decomposition()
        
        # Step 4: Determine stationarity
        self.determine_stationarity_and_differencing()
        
        # Step 5: ACF/PACF analysis
        self.plot_acf_pacf()
        
        # Step 6: Split data
        if not self.split_data():
            return
        
        # Step 7: Fit ARIMA model
        if not self.fit_arima_model():
            return
        
        # Step 8: Forecast and evaluate
        self.forecast_and_evaluate()
        
        # Step 9: Detect anomalies
        self.detect_anomalies()
        
        # Step 10: Generate summary
        self.generate_summary_report()
        
        print("\n✅ Analysis complete!")

# Usage example
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = EnergyConsumptionAnalyzer('energy_features01.csv')
    
    # Run the complete analysis
    analyzer.run_full_analysis()