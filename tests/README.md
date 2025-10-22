# Testing Documentation - Intelligent Forecasting System

## Overview

This directory contains comprehensive end-to-end tests for the Intelligent Time Series Forecasting System. The tests validate the complete pipeline from EDA through model training and report generation.

## Test Suite Components

### 1. **test_end_to_end.py** - Complete Pipeline Validation
Full automated test of the entire forecasting pipeline using direct agent tool calls.

**What it tests:**
- ✅ Proper time series EDA (stationarity, seasonality, ACF/PACF)
- ✅ Intelligent feature recommendations based on data characteristics
- ✅ Feature engineering (balanced, not excessive)
- ✅ SageMaker hyperparameter tuning
- ✅ ARIMA model training with best hyperparameters
- ✅ Comprehensive report generation with diagnostics

**Run:**
```bash
python tests/test_end_to_end.py
```

**Expected Duration:** ~15-20 minutes (includes SageMaker jobs)

**Success Criteria:**
- All validation checks pass
- EDA includes stationarity tests, decomposition, ACF/PACF
- Features count between 10-40 (optimal for SageMaker)
- Tuning job completes with best hyperparameters
- Training job succeeds
- Report includes metrics, forecast with CI, residual analysis

---

### 2. **test_dashboard_simulation.py** - UI Workflow Simulation
Simulates the 7-step Streamlit dashboard workflow by invoking AgentCore directly.

**What it tests:**
- ✅ Complete user-driven workflow
- ✅ AgentCore prompt handling
- ✅ Session management
- ✅ Step-by-step validation
- ✅ Feature selection simulation

**Run:**
```bash
python tests/test_dashboard_simulation.py
```

**Steps Simulated:**
1. Upload Data → S3
2. Run Advanced EDA
3. Get Feature Recommendations
4. Select & Create Features (simulates user selection)
5. Start Hyperparameter Tuning (monitors progress)
6. Train Models (ARIMA)
7. Generate Comprehensive Report

**Expected Duration:** ~15-20 minutes

---

### 3. **test_prompts.sh** - Individual Agent Tool Tests
Enhanced bash script that tests each agent tool independently with validation.

**What it tests:**
- ✅ Advanced EDA tool
- ✅ Feature recommendation tool
- ✅ Feature creation tool
- ✅ SageMaker tuning job creation
- ✅ SageMaker training job creation
- ✅ LSTM training job creation
- ✅ Comprehensive report generation

**Run:**
```bash
chmod +x tests/test_prompts.sh
./tests/test_prompts.sh
```

**Features:**
- JSON validation for all responses
- Field-level validation
- Error detection
- EDA quality checks (time series specific)
- Feature count validation
- Results saved to `tests/results_<timestamp>/`
- Color-coded output

**Expected Duration:** ~5-10 minutes (without SageMaker jobs)

**Note:** SageMaker job tests require manual confirmation to avoid creating expensive resources.

---

## Test Data

All tests create synthetic time series data with:
- **90 days** of daily observations
- **Trend component**: Linear increase from 100 to 150
- **Seasonal component**: 7-day weekly cycle
- **Random noise**: Normal distribution (σ=5)

This ensures consistent, reproducible test results.

---

## Validation Criteria

### EDA Quality Checks
- ✅ Stationarity tests (ADF, KPSS) at multiple differencing levels
- ✅ Seasonal decomposition with strength metrics
- ✅ ACF/PACF analysis with significant lag identification
- ✅ Trend detection with statistical significance
- ✅ HTML report generation

### Feature Engineering Checks
- ✅ Recommendations based on EDA insights (ACF → lags, seasonality → rolling)
- ✅ Feature count: 10-40 (optimal for SageMaker, not overwhelming)
- ✅ Justifications provided for each feature type
- ✅ Priority classification (high/medium/low)
- ✅ Successfully creates selected features

### Model Training Checks
- ✅ Hyperparameter tuning completes successfully
- ✅ Best hyperparameters extracted (p, d, q orders)
- ✅ Training job completes without errors
- ✅ Model artifacts saved to S3

### Report Checks
- ✅ Comprehensive metrics (MAE, RMSE, MAPE, AIC, BIC)
- ✅ 7-day forecast with confidence intervals
- ✅ Backtesting results (actual vs predicted)
- ✅ Residual analysis and diagnostics
- ✅ Interactive Plotly visualizations
- ✅ Executive summary with recommendations

---

## Running the Full Test Suite

### Quick Test (No SageMaker Jobs)
```bash
# Test individual tools only
./tests/test_prompts.sh
# When prompted for SageMaker jobs, press 'N'
```

### Full End-to-End Test
```bash
# Automated pipeline test
python tests/test_end_to_end.py
```

### Dashboard Simulation
```bash
# Simulate UI workflow
python tests/test_dashboard_simulation.py
```

---

## Interpreting Results

### test_end_to_end.py Output
```
✅ Test Dataset Creation: PASS
✅ Advanced EDA: PASS (10/10 checks passed)
✅ Feature Recommendations: PASS (8/8 checks passed)
✅ Feature Creation: PASS (6/6 checks passed, 12 features)
✅ Hyperparameter Tuning: PASS (Best params: p=2, d=1, q=2)
✅ ARIMA Training: PASS (Model: s3://...)
✅ Comprehensive Report: PASS (RMSE=3.45, MAPE=2.1%)
```

### test_dashboard_simulation.py Output
```
📊 STEP 2: Run Advanced EDA
   ✅ Stationarity analysis
   ✅ Decomposition
   ✅ Autocorrelation
   ✅ Trend analysis
   ✅ HTML report

   Recommended differencing: 1
   Seasonal strength: 0.75
```

### test_prompts.sh Output
```
============================================================
TEST 1: Advanced_EDA
============================================================
Prompt: run advanced eda on s3://...
✅ Valid JSON response
✅ No errors in response
✅ Field 'stationarity' exists: {...}
✅ Field 'decomposition' exists: {...}
✅ Advanced_EDA: PASSED
```

---

## Troubleshooting

### Common Issues

#### 1. **Code Interpreter Timeout**
**Error:** `Code execution timeout`
**Solution:** Code Interpreter has a 60-second limit. For large datasets, consider reducing data size for tests.

#### 2. **SageMaker Job Creation Fails**
**Error:** `ValidationException: Invalid execution role`
**Solution:** Ensure `SAGEMAKER_ROLE_ARN` in agent.py is correct and has proper permissions.

#### 3. **Feature Count Too High**
**Warning:** `50 features created (may overwhelm SageMaker)`
**Solution:** Reduce feature selection in `feature_config` to keep count <30.

#### 4. **Tuning Job Never Completes**
**Issue:** Test times out after 10 minutes
**Solution:** Tuning jobs can take longer. Increase `max_wait` in tests or reduce `max_jobs` parameter.

#### 5. **AgentCore Invocation Fails**
**Error:** `Could not connect to endpoint`
**Solution:** Ensure AgentCore is deployed and ARN in `.bedrock_agentcore.yaml` is correct.

---

## Performance Benchmarks

### Expected Execution Times

| Test | Duration | Notes |
|------|----------|-------|
| EDA | 30-60s | Code Interpreter execution |
| Feature Recommendations | 20-40s | Lightweight analysis |
| Feature Creation | 30-60s | Data transformation |
| Hyperparameter Tuning | 5-10 min | SageMaker job (3-10 trials) |
| Model Training | 3-5 min | SageMaker job |
| Report Generation | 60-90s | Full model diagnostics |

### Total End-to-End: 15-20 minutes

---

## Test Artifacts

All tests save artifacts to S3:

```
s3://sagemaker-forecasting-us-east-1-YOUR_ACCOUNT_ID/
├── test_data/                    # Synthetic test datasets
│   ├── synthetic_sales_*.csv
│   └── dashboard_test_*.csv
├── cleaned/                       # Feature-engineered datasets
│   └── *_features.csv
├── reports/                       # HTML reports
│   ├── eda_advanced_report.html
│   └── comprehensive_forecast_report.html
└── sagemaker/
    ├── tuning-output/            # Tuning job results
    ├── models/                   # Trained model artifacts
    └── lstm-models/              # LSTM model artifacts
```

Local artifacts:
```
tests/results_<timestamp>/
├── Advanced_EDA.json
├── Feature_Recommendations.json
├── Feature_Creation.json
├── Tuning_Job_Creation.json
├── Training_Job_Creation.json
└── Comprehensive_Report.json
```

---

## Continuous Integration

For CI/CD pipelines:

```bash
#!/bin/bash
# CI test script (no interactive prompts)

# Run unit tests
python tests/test_end_to_end.py

# Run prompt tests (skip SageMaker jobs)
export SKIP_SAGEMAKER_TESTS=true
./tests/test_prompts.sh

# Check exit codes
if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
    exit 0
else
    echo "❌ Tests failed"
    exit 1
fi
```

---

## Maintenance

### Updating Test Data
Edit synthetic data generation in test files:
```python
# In test_end_to_end.py or test_dashboard_simulation.py
dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
trend = np.linspace(100, 150, 90)
seasonality = 10 * np.sin(2 * np.pi * np.arange(90) / 7)
```

### Adding New Validation Checks
Add to validation functions:
```python
def validate_eda_output(eda_result):
    checks.append(('New check', 'new_field' in eda))
    # ...
```

### Modifying Feature Limits
Adjust feature count warnings:
```python
# Optimal range: 10-40 features
if feature_count > 40:
    print_warning("Too many features")
```

---

## Support

For issues or questions:
1. Check test output logs in `tests/results_*/`
2. Review S3 artifacts (reports, data)
3. Check CloudWatch logs for SageMaker jobs
4. Verify AgentCore deployment status: `agentcore status`

---

## Summary

This test suite ensures:
1. ✅ **Proper EDA for time series** - Not generic stats, but stationarity, seasonality, ACF/PACF
2. ✅ **Intelligent feature engineering** - Data-driven recommendations with justifications
3. ✅ **Balanced features** - Not too few (<10) or too many (>50) for SageMaker
4. ✅ **Successful model training** - Tuning → Training → Deployment pipeline works
5. ✅ **Comprehensive reporting** - Full diagnostics, forecasts, and visualizations

**Run all tests before production deployment to ensure system quality! 🚀**
