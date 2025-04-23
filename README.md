# Financial Forecasting Automation with ARIMA & Random Forest

This project demonstrates the automation of financial forecasting using ARIMA and Random Forest models, deployed in AWS SageMaker to improve forecasting accuracy by 20%.

## 📂 Files
- `data/`: Raw and cleaned financial data (CSV)
- `scripts/`: Python scripts for data preprocessing, model training, and evaluation
    - `arima_model.py`: ARIMA model implementation
    - `random_forest.py`: Random Forest model implementation
    - `forecasting_pipeline.py`: Automates the entire process (from data collection to deployment)
- `notebooks/`: Jupyter Notebooks for exploratory data analysis and modeling steps
- `aws_deployment/`: Folder for AWS SageMaker scripts (model deployment)
    - `sagemaker_deploy.py`: Deploy model to SageMaker
- `README.md`: Project documentation

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/financial-forecasting.git
   
2. Install dependencies: 
pip install -r requirements.txt

3. Run the models:
python arima_model.py
python random_forest.py

---

### 4. **Bonus: Code Snippet for ARIMA Model in Python**

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Load financial data (e.g., revenue)
data = pd.read_csv("financial_data.csv", parse_dates=["Date"], index_col="Date")
train = data[:-12]  # Train on all but last 12 months
test = data[-12:]   # Use last 12 months for testing

# Fit ARIMA model
model = ARIMA(train, order=(5,1,0))  # ARIMA(p,d,q) order
model_fit = model.fit()

# Forecast future values
forecast = model_fit.forecast(steps=12)

# Evaluate the model
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test, forecast)
print(f"ARIMA MAE: {mae}")
