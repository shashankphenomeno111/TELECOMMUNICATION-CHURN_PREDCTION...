import joblib
import pandas as pd
import numpy as np

# Load model and data
model = joblib.load('xgb_churn_model.joblib')
df = pd.read_csv('telecommunications_Dataset.csv')

print("Model feature names:")
print(list(model.feature_names_in_))
print()

# Get actual churned customers
churned = df[df['churn']==1]
features = list(model.feature_names_in_)

print("Testing predictions on ACTUAL CHURNED customers:")
test_df = churned[features].head(10)
probs = model.predict_proba(test_df)[:,1]
for i, (idx, row) in enumerate(churned.head(10).iterrows()):
    print(f"  Customer {idx}: churn_prob={probs[i]*100:.1f}%, service_calls={row['customer_service_calls']}, int_plan={row['international_plan']}, day_mins={row['day_mins']:.0f}")

print()
print("Testing HIGH RISK scenario:")
high_risk = pd.DataFrame([{
    'total_charge': 75.0,
    'customer_service_calls': 6,
    'international_plan': 1,
    'day_mins': 280.0,
    'day_charge': 50.0,
    'voice_mail_messages': 0,
    'international_calls': 5,
    'voice_mail_plan': 0,
    'international_mins': 15.0,
    'evening_charge': 20.0
}])[features]

prob = model.predict_proba(high_risk)[0][1]
print(f"High risk prediction: {prob*100:.1f}%")

print()
print("Testing EXTREME HIGH RISK scenario (9 service calls):")
extreme_risk = pd.DataFrame([{
    'total_charge': 100.0,
    'customer_service_calls': 9,
    'international_plan': 1,
    'day_mins': 350.0,
    'day_charge': 60.0,
    'voice_mail_messages': 0,
    'international_calls': 10,
    'voice_mail_plan': 0,
    'international_mins': 20.0,
    'evening_charge': 25.0
}])[features]

prob = model.predict_proba(extreme_risk)[0][1]
print(f"Extreme risk prediction: {prob*100:.1f}%")
