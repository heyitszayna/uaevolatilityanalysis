import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class UAEStockVolatilityAnalysis:
    """Comprehensive volatility analysis for UAE stock markets"""
    
    def __init__(self):
        self.data = {}
        self.returns = {}
        self.results = {}
        
    def load_and_preprocess_data(self, file_paths):
        """Load CSV files and perform data cleaning"""
        print("="*60)
        print("LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        for name, path in file_paths.items():
            try:
                # Read CSV file
                df = pd.read_csv(path)
                
                print(f"\n{name} - Initial columns: {df.columns.tolist()}")
                print(f"First few rows:\n{df.head()}")
                
                # Convert Date column to datetime
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Sort by date
                df = df.sort_values('Date').reset_index(drop=True)
                
                # Clean numeric columns - remove commas, spaces, and currency symbols
                numeric_cols = ['Price', 'Open', 'High', 'Low']
                for col in numeric_cols:
                    if df[col].dtype == 'object':
                        # Remove commas, spaces, dollar signs, and other non-numeric characters
                        df[col] = df[col].astype(str).str.replace(',', '')
                        df[col] = df[col].str.replace('$', '')
                        df[col] = df[col].str.replace('AED', '')
                        df[col] = df[col].str.replace(' ', '')
                        df[col] = df[col].str.strip()
                        # Convert to numeric, coercing errors to NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Handle Change % column
                if 'Change %' in df.columns:
                    if df['Change %'].dtype == 'object':
                        df['Change %'] = df['Change %'].astype(str).str.replace('%', '')
                        df['Change %'] = df['Change %'].str.replace(' ', '')
                        df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce')
                
                # Handle missing values
                df = df.dropna(subset=['Price', 'Open', 'High', 'Low'])
                
                # Remove duplicates
                df = df.drop_duplicates(subset=['Date'], keep='first')
                
                # Remove rows where Price is zero or negative
                df = df[df['Price'] > 0]
                
                # Calculate log returns
                df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
                
                # Remove first row with NaN return and any infinite values
                df = df.dropna(subset=['Log_Return'])
                df = df[np.isfinite(df['Log_Return'])]
                
                # Remove extreme outliers (optional - beyond 10 standard deviations)
                mean_return = df['Log_Return'].mean()
                std_return = df['Log_Return'].std()
                df = df[np.abs(df['Log_Return'] - mean_return) < 10 * std_return]
                
                self.data[name] = df
                self.returns[name] = df['Log_Return'].values
                
                print(f"\n{name} - Successfully loaded:")
                print(f"  - Observations: {len(df)}")
                print(f"  - Date range: {df['Date'].min()} to {df['Date'].max()}")
                print(f"  - Mean return: {df['Log_Return'].mean():.6f}")
                print(f"  - Std deviation: {df['Log_Return'].std():.6f}")
                print(f"  - Min return: {df['Log_Return'].min():.6f}")
                print(f"  - Max return: {df['Log_Return'].max():.6f}")
                
            except Exception as e:
                print(f"\nError loading {name}: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print(f"Traceback:\n{traceback.format_exc()}")
                print(f"Please ensure the file exists at: {path}")
    
    def descriptive_statistics(self, dataset_name):
        """Calculate and display descriptive statistics"""
        if dataset_name not in self.returns:
            print(f"Dataset {dataset_name} not found!")
            return None
        
        returns = self.returns[dataset_name]
        
        stats_dict = {
            'Observations': len(returns),
            'Mean': np.mean(returns),
            'Median': np.median(returns),
            'Maximum': np.max(returns),
            'Minimum': np.min(returns),
            'Std. Dev.': np.std(returns, ddof=1),
            'Skewness': stats.skew(returns),
            'Kurtosis': stats.kurtosis(returns, fisher=False),
            'Jarque-Bera': stats.jarque_bera(returns)[0],
            'JB p-value': stats.jarque_bera(returns)[1]
        }
        
        return pd.DataFrame([stats_dict]).T
    
    def fit_garch(self, dataset_name, p=1, q=1):
        """Fit standard GARCH(p,q) model"""
        if dataset_name not in self.returns:
            print(f"Dataset {dataset_name} not found!")
            return None
        
        returns = self.returns[dataset_name] * 100  # Scale for better convergence
        
        # Fit GARCH model
        model = arch_model(returns, vol='Garch', p=p, q=q, dist='normal')
        results = model.fit(disp='off', show_warning=False)
        
        return results
    
    def fit_tgarch(self, dataset_name, p=1, o=1, q=1):
        """Fit Threshold GARCH (GJR-GARCH) model"""
        if dataset_name not in self.returns:
            print(f"Dataset {dataset_name} not found!")
            return None
        
        returns = self.returns[dataset_name] * 100
        
        # Fit TGARCH model (o parameter captures asymmetry)
        model = arch_model(returns, vol='Garch', p=p, o=o, q=q, dist='normal')
        results = model.fit(disp='off', show_warning=False)
        
        return results
    
    def fit_egarch(self, dataset_name, p=1, q=1):
        """Fit EGARCH model"""
        if dataset_name not in self.returns:
            print(f"Dataset {dataset_name} not found!")
            return None
        
        returns = self.returns[dataset_name] * 100
        
        # Fit EGARCH model
        model = arch_model(returns, vol='EGARCH', p=p, q=q, dist='normal')
        results = model.fit(disp='off', show_warning=False)
        
        return results
    
    def calculate_var(self, results, confidence_level=0.05, position_value=10000000):
        """Calculate Value at Risk (VaR)"""
        # Get conditional volatility forecast
        forecast = results.forecast(horizon=1)
        sigma_next = np.sqrt(forecast.variance.values[-1, 0])
        
        # Calculate VaR (5% and 1%)
        var_5 = stats.norm.ppf(0.05) * sigma_next * position_value / 100
        var_1 = stats.norm.ppf(0.01) * sigma_next * position_value / 100
        
        return {'VaR_5%': abs(var_5), 'VaR_1%': abs(var_1)}
    
    def plot_returns_and_volatility(self, dataset_name):
        """Plot return series and conditional volatility"""
        if dataset_name not in self.data:
            print(f"Dataset {dataset_name} not found!")
            return
        
        df = self.data[dataset_name]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot returns
        axes[0].plot(df['Date'], df['Log_Return'], linewidth=0.8, color='darkblue', alpha=0.7)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=0.8)
        axes[0].set_title(f'{dataset_name} - Log Returns', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date', fontsize=11)
        axes[0].set_ylabel('Log Returns', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot histogram
        axes[1].hist(df['Log_Return'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].set_title(f'{dataset_name} - Return Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Log Returns', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_returns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_conditional_volatility(self, dataset_name, results, model_name):
        """Plot conditional volatility from fitted model"""
        if dataset_name not in self.data:
            print(f"Dataset {dataset_name} not found!")
            return
        
        df = self.data[dataset_name].reset_index(drop=True)
        
        # Get conditional volatility
        cond_vol = results.conditional_volatility
        
        # Ensure matching lengths - align with conditional volatility
        n_obs = len(cond_vol)
        returns_aligned = df['Log_Return'].values[-n_obs:]
        dates_aligned = df['Date'].values[-n_obs:]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot returns with volatility
        axes[0].plot(dates_aligned, returns_aligned, linewidth=0.5, 
                    color='darkblue', alpha=0.6, label='Returns')
        axes[0].set_title(f'{dataset_name} - Returns with {model_name} Conditional Volatility', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Log Returns', fontsize=11)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot conditional volatility
        axes[1].plot(dates_aligned, cond_vol/100, linewidth=1, 
                    color='darkred', alpha=0.8, label='Conditional Volatility')
        axes[1].fill_between(dates_aligned, 0, cond_vol/100, alpha=0.3, color='red')
        axes[1].set_title(f'{dataset_name} - {model_name} Conditional Volatility', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date', fontsize=11)
        axes[1].set_ylabel('Volatility', fontsize=11)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_{model_name}_volatility.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, dataset_name):
        """Fit and compare all models"""
        print("\n" + "="*60)
        print(f"MODEL COMPARISON FOR {dataset_name}")
        print("="*60)
        
        # Fit models
        print("\nFitting GARCH(1,1)...")
        garch_results = self.fit_garch(dataset_name)
        
        print("Fitting TGARCH(1,1)...")
        tgarch_results = self.fit_tgarch(dataset_name)
        
        print("Fitting EGARCH(1,1)...")
        egarch_results = self.fit_egarch(dataset_name)
        
        # Store results
        self.results[dataset_name] = {
            'GARCH': garch_results,
            'TGARCH': tgarch_results,
            'EGARCH': egarch_results
        }
        
        # Compare information criteria
        comparison = pd.DataFrame({
            'Model': ['GARCH(1,1)', 'TGARCH(1,1)', 'EGARCH(1,1)'],
            'Log-Likelihood': [garch_results.loglikelihood, 
                              tgarch_results.loglikelihood,
                              egarch_results.loglikelihood],
            'AIC': [garch_results.aic, tgarch_results.aic, egarch_results.aic],
            'BIC': [garch_results.bic, tgarch_results.bic, egarch_results.bic]
        })
        
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(comparison.to_string(index=False))
        
        # Calculate VaR for each model
        print("\n" + "="*60)
        print("VALUE AT RISK (VaR) - Position: 10,000,000 AED")
        print("="*60)
        
        var_results = []
        for model_name, results in self.results[dataset_name].items():
            var = self.calculate_var(results)
            var_results.append({
                'Model': model_name,
                'VaR (5%)': f"{var['VaR_5%']:,.2f} AED",
                'VaR (1%)': f"{var['VaR_1%']:,.2f} AED"
            })
        
        var_df = pd.DataFrame(var_results)
        print(var_df.to_string(index=False))
        
        return comparison, var_df
    
    def print_model_summary(self, dataset_name, model_name):
        """Print detailed model summary"""
        if dataset_name not in self.results:
            print(f"No results for {dataset_name}. Run compare_models() first.")
            return
        
        if model_name not in self.results[dataset_name]:
            print(f"Model {model_name} not found!")
            return
        
        results = self.results[dataset_name][model_name]
        
        print("\n" + "="*60)
        print(f"{model_name} MODEL SUMMARY - {dataset_name}")
        print("="*60)
        print(results.summary())
    
    def analyze_leverage_effect(self, dataset_name):
        """Analyze asymmetric/leverage effects"""
        if dataset_name not in self.results:
            print(f"No results for {dataset_name}. Run compare_models() first.")
            return
        
        print("\n" + "="*60)
        print(f"LEVERAGE EFFECT ANALYSIS - {dataset_name}")
        print("="*60)
        
        # Check TGARCH asymmetry parameter
        tgarch = self.results[dataset_name]['TGARCH']
        params = tgarch.params
        
        if 'gamma[1]' in params.index:
            gamma = params['gamma[1]']
            print(f"\nTGARCH Asymmetry Parameter (gamma): {gamma:.6f}")
            
            if gamma < 0:
                print("Interpretation: Significant NEGATIVE leverage effect detected.")
                print("Bad news increases volatility more than good news.")
            elif gamma > 0:
                print("Interpretation: POSITIVE asymmetry detected.")
                print("Good news increases volatility more than bad news.")
            else:
                print("Interpretation: No significant leverage effect.")
        
        # Check EGARCH asymmetry
        egarch = self.results[dataset_name]['EGARCH']
        egarch_params = egarch.params
        
        if 'gamma[1]' in egarch_params.index:
            egarch_gamma = egarch_params['gamma[1]']
            print(f"\nEGARCH Asymmetry Parameter (gamma): {egarch_gamma:.6f}")
            
            if egarch_gamma < 0:
                print("Interpretation: Significant leverage effect detected.")
                print("Negative shocks increase volatility more than positive shocks.")
            else:
                print("Interpretation: No significant leverage effect.")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison across all datasets"""
        print("\n" + "="*70)
        print("COMPREHENSIVE COMPARISON ACROSS ALL DATASETS")
        print("="*70)
        
        all_stats = []
        for dataset_name in self.data.keys():
            stats = self.descriptive_statistics(dataset_name)
            stats['Dataset'] = dataset_name
            all_stats.append(stats.T)
        
        if all_stats:
            combined = pd.concat(all_stats, axis=1)
            print("\nDESCRIPTIVE STATISTICS COMPARISON:")
            print(combined.to_string())
        
        # Model comparison across datasets
        if self.results:
            print("\n" + "="*70)
            print("BEST MODEL BY AIC FOR EACH DATASET:")
            print("="*70)
            
            for dataset_name, models in self.results.items():
                aics = {name: res.aic for name, res in models.items()}
                best_model = min(aics, key=aics.get)
                print(f"{dataset_name}: {best_model} (AIC={aics[best_model]:.4f})")


# MAIN EXECUTION
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = UAEStockVolatilityAnalysis()
    
    # Define file paths
    file_paths = {
        'ADX_2001': 'ADX_2001_2005.csv',
        'DFM_2001': 'DFM_2001_2005.csv',
        'ADX_2020': 'ADX_2020_2024.csv',
        'DFM_2020': 'DFM_2020_2024.csv'
    }
    
    print("\n" + "="*70)
    print("UAE STOCK MARKET VOLATILITY ANALYSIS")
    print("="*70)
    
    # Load and preprocess data
    analyzer.load_and_preprocess_data(file_paths)
    
    # Analyze each dataset
    for dataset_name in file_paths.keys():
        if dataset_name in analyzer.data:
            print(f"\n\n{'='*70}")
            print(f"ANALYZING: {dataset_name}")
            print(f"{'='*70}")
            
            # Descriptive statistics
            print("\nDESCRIPTIVE STATISTICS:")
            print("-" * 70)
            stats_df = analyzer.descriptive_statistics(dataset_name)
            if stats_df is not None:
                print(stats_df)
            
            # Plot returns
            analyzer.plot_returns_and_volatility(dataset_name)
            
            # Compare models
            comparison, var = analyzer.compare_models(dataset_name)
            
            # Print detailed summaries
            analyzer.print_model_summary(dataset_name, 'GARCH')
            analyzer.print_model_summary(dataset_name, 'TGARCH')
            analyzer.print_model_summary(dataset_name, 'EGARCH')
            
            # Analyze leverage effect
            analyzer.analyze_leverage_effect(dataset_name)
            
            # Plot conditional volatility for best model
            best_model = comparison.loc[comparison['AIC'].idxmin(), 'Model']
            best_model_key = best_model.split('(')[0]
            analyzer.plot_conditional_volatility(dataset_name, 
                                                analyzer.results[dataset_name][best_model_key],
                                                best_model_key)
    
    # Generate comprehensive comparison report
    analyzer.generate_comparison_report()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
