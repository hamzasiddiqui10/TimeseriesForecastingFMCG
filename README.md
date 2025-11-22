# Time Series Forecasting for FMCG Demand

A comprehensive educational project demonstrating **time-series forecasting** techniques applied to FMCG (Fast-Moving Consumer Goods) demand prediction. This project showcases data exploration, statistical analysis, feature engineering, and machine learning model development with detailed business insights.

## 📊 Project Overview

This project was created to deepen my understanding of **time-series forecasting** concepts and their practical application in supply chain optimization. The notebook provides end-to-end forecasting implementation, from exploratory data analysis to production-ready 30-day demand forecasts.

### Key Learning Objectives
- **Time Series Fundamentals**: Understanding components (trend, seasonality, cyclicity, noise)
- **Stationarity Testing**: ADF tests and time-series differencing
- **Statistical Analysis**: ACF/PACF plots for autocorrelation insights
- **Feature Engineering**: Creating lag features, rolling statistics, and seasonality indicators
- **Forecasting Models**: Implementing ARIMA and RandomForest regression
- **Model Evaluation**: Comprehensive metrics (RMSE, MAE, MAPE) and residual analysis
- **Business Application**: Translating forecasts into actionable supply chain decisions

## 🎯 What This Project Covers

### 1. **Data Loading & Exploration** 📈
- Load and inspect FMCG demand data with 9 features
- Statistical summaries and categorical analysis
- Distribution visualizations across all variables

### 2. **Time Series Visualization** 📉
- Daily aggregated sales trends
- Product category performance comparisons
- Weekly and monthly seasonality patterns
- Promotion impact analysis by category

### 3. **Core Concepts** 🔬
- **Stationarity Testing**: ADF test implementation and interpretation
- **ACF/PACF Analysis**: Identifying lag dependencies
- **Train-Test Split**: Time-based cross-validation (80/20)
- **Moving Averages**: 7-day, 30-day, and exponentially weighted moving averages

### 4. **Feature Engineering** 🔧
- Lag features (1, 3, 7, 14-day lags)
- Rolling statistics (mean, std, min, max)
- Expanding window features
- Category and location encoding
- Feature normalization with StandardScaler

### 5. **Forecasting Models** 🤖
- **ARIMA(1,1,1)**: Classical time-series model with differencing
- **RandomForest Regressor**: ML-based approach with 15 engineered features
- Model comparison and performance analysis

### 6. **Model Evaluation** 📊
- **Metrics**: RMSE, MAE, MAPE comparison
- **Residual Analysis**: Distribution plots and time-series residuals
- **Feature Importance**: Top predictors identified and ranked
- **Predictions vs Actuals**: Visual forecast performance

### 7. **Business Insights** 💡
- **Promotion Impact**: ~40-50% sales uplift quantification
- **Store Location Analysis**: Performance by geography
- **Product Category Performance**: Sales by category
- **Price Elasticity**: Correlation analysis and strategic implications
- **Weekday Patterns**: Demand variation by day-of-week

### 8. **30-Day Production Forecast** 🔮
- Forecast generation using best-performing model
- 90% prediction intervals with confidence bounds
- Weekly forecast summary
- Visualization with historical context

### 9. **Actionable Recommendations** ✅
- Model deployment strategy
- Inventory management guidelines
- Promotion and pricing optimization
- Location-specific recommendations
- Forecast monitoring framework

## 📁 Project Structure

```
TimeseriesForecastingFMCG/
├── FMCG_TimeSeries_Forecasting.ipynb    # Complete interactive notebook (40 cells)
├── extended_fmcg_demand_forecasting.csv # FMCG dataset (~1000 days)
└── README.md                             # This file
```

## 🗂️ Dataset Features

| Feature | Description |
|---------|-------------|
| `Date` | Transaction date |
| `Sales_Volume` | Daily sales units (target variable) |
| `Price` | Product price |
| `Promotion` | Promotion flag (binary) |
| `Product_Category` | Category (Personal Care, Household, Food) |
| `Store_Location` | Location type (Urban, Suburban, Rural) |
| `Weekday` | Day of week (0-6) |
| `Temperature` | External temperature |
| `Humidity` | Environmental humidity |

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Internet connection (for package auto-installation)

### Installation & Running

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hamzasiddiqui10/TimeseriesForecastingFMCG.git
   cd TimeseriesForecastingFMCG
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook FMCG_TimeSeries_Forecasting.ipynb
   ```

3. **Run the notebook**: Execute cells sequentially from top to bottom. The first cell includes auto-installation of required packages.

### Required Libraries

The notebook automatically installs these packages:
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Statistical Analysis**: statsmodels (ARIMA, ADF test, ACF/PACF)
- **Machine Learning**: scikit-learn (RandomForest, preprocessing, metrics)
- **Utilities**: scipy, warnings

## 📊 Key Results

### Model Performance Comparison
| Metric | ARIMA | RandomForest | Winner |
|--------|-------|--------------|--------|
| RMSE | Higher | **Lower** | 🏆 RandomForest |
| MAE | Higher | **Lower** | 🏆 RandomForest |
| MAPE | Higher | **Lower** | 🏆 RandomForest |

**Finding**: RandomForest outperforms ARIMA for this dataset, capturing complex non-linear patterns better than classical time-series methods.

### Business Insights
- **Promotion Effect**: ~40-50% sales increase during promotional periods
- **Best Performing Location**: Urban stores show 20-30% higher sales than rural
- **Top Category**: Consistently high demand across all seasons
- **Price Sensitivity**: Weak correlation suggests opportunity for premium pricing
- **Weekday Pattern**: Peak demand on weekends vs weekdays

## 💡 Learning Outcomes

By working through this project, I learned:

1. ✅ How to properly format and prepare time-series data
2. ✅ Methods to test and ensure stationarity of time series
3. ✅ Techniques for feature engineering in forecasting problems
4. ✅ Implementation of both statistical (ARIMA) and ML (RandomForest) models
5. ✅ Proper model evaluation with domain-specific metrics
6. ✅ How to generate actionable business insights from forecasts
7. ✅ Importance of confidence intervals in production forecasts
8. ✅ Trade-offs between model complexity and interpretability

## 🔍 Notebook Structure (40 Cells)

**Sections**:
1. **Introduction & Setup** (2 cells) - Overview and library imports
2. **Data Loading & Exploration** (6 cells) - Dataset overview and statistics
3. **Time Series Visualization** (5 cells) - Trend, seasonality, and business patterns
4. **Core Concepts** (6 cells) - Stationarity, ACF/PACF, train-test split
5. **Feature Engineering** (3 cells) - Feature creation and scaling
6. **Model Building** (2 cells) - ARIMA and RandomForest implementation
7. **Model Evaluation** (4 cells) - Metrics, residuals, and comparison
8. **Business Insights** (2 cells) - Feature importance and operational insights
9. **Production Forecast** (2 cells) - 30-day forecast with recommendations

## 🎓 Concepts Applied

- **Time Series Components**: Trend, seasonality, cyclicity, noise decomposition
- **Stationarity**: Augmented Dickey-Fuller (ADF) testing
- **Autocorrelation**: ACF and PACF analysis
- **ARIMA Modeling**: Autoregressive, Integrated, Moving Average framework
- **Feature Engineering**: Lag features, rolling statistics, one-hot encoding
- **Random Forests**: Ensemble learning for regression
- **Cross-Validation**: Time-based train-test split methodology
- **Model Metrics**: RMSE, MAE, MAPE for regression evaluation

## 📈 Visualization Highlights

The notebook includes 15+ high-quality visualizations:
- **Distributions**: Histograms and density plots
- **Time Series**: Line plots with trends and patterns
- **Seasonality**: Heatmaps and bar charts for patterns
- **ACF/PACF**: Autocorrelation function plots
- **Residuals**: Distribution and time-series residual plots
- **Comparisons**: Side-by-side model predictions vs actuals
- **Forecast**: 30-day forecast with confidence intervals

## 🔧 Technical Highlights

- **Automated Data Pipeline**: Handles missing values and feature scaling
- **Robust Model Implementation**: Error handling and edge case management
- **Comprehensive Validation**: Multiple evaluation metrics and visualizations
- **Production Ready**: Confidence intervals and uncertainty quantification
- **Business Focused**: Interpretable results with actionable recommendations

## 📚 References & Further Learning

This project covers fundamental concepts from:
- Time series analysis (Box & Jenkins methodology)
- Forecasting best practices (Makridakis & Wheelwright)
- Machine learning for regression (scikit-learn)
- Supply chain optimization

## 🤝 Author

**Hamza Ahmed Siddiqui**

This project was created as a comprehensive learning exercise in time-series forecasting techniques and their real-world application using FMCG data from Kaggle.

## 📝 License

This project is open source and available under the MIT License.

---

**Happy Forecasting!** 📊✨