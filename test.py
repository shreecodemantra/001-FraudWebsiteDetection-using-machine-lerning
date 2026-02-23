from flask import Flask, render_template, request, session, redirect, url_for
import pymysql
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import tensorflow as tf
from scipy import stats
import joblib

def load_models_and_preprocessors():
    models = {}
    model_files = {
        'XGBoost': 'models/model_xgboost.pkl'
    }
    
    for name, path in model_files.items():
        try:
            models[name] = pickle.load(open(path, 'rb'))
            print(f"✓ {name} loaded successfully")
        except Exception as e:
            print(f"✗ Error loading {name}: {e}")
    
    try:
        scaler = joblib.load("models/scaler.joblib")
        label_encoders = joblib.load("models/label_encoders.joblib")
        print("✓ Scaler and Label Encoders loaded successfully")
    except Exception as e:
        print(f"✗ Error loading preprocessors: {e}")
        scaler = None
        label_encoders = None
    
    return models, scaler, label_encoders

def engineer_features(transaction_data, label_encoders, reference_stats=None):
    df = pd.DataFrame([transaction_data])
    
    # 1. Transaction Amount Features
    df['Amount_Log'] = np.log1p(df['Transaction Amount'])
    df['Amount_per_Quantity'] = df['Transaction Amount'] / (df['Quantity'] + 1)
    
    # For z-score, use reference stats if available
    if reference_stats:
        df['Amount_zscore'] = (df['Transaction Amount'] - reference_stats['mean']) / reference_stats['std']
    else:
        df['Amount_zscore'] = 0
    
    # 2. Time-based Features
    df['Hour_Bin'] = pd.cut(df['Transaction Hour'], 
                            bins=[-np.inf, 6, 12, 18, np.inf], 
                            labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    transaction_date = pd.to_datetime(df['Transaction Date'])
    df['Is_Weekend'] = (transaction_date.dt.dayofweek >= 5).astype(int)
    df['Day_of_Week'] = transaction_date.dt.dayofweek
    
    # 3. Customer Profile Features
    df['Age_Category'] = pd.cut(df['Customer Age'], 
                                bins=[0, 25, 35, 50, 65, np.inf], 
                                labels=['Young', 'Young_Adult', 'Adult', 'Senior', 'Elder'])
    df['Account_Age_Weeks'] = df['Account Age Days'] // 7
    df['Is_New_Account'] = (df['Account Age Days'] <= 30).astype(int)
    
    # 4. Transaction Pattern Features
    # For single prediction, we can't use qcut, so use fixed thresholds
    amount = df['Transaction Amount'].iloc[0]
    if amount <= 50:
        transaction_size = 'Very_Small'
    elif amount <= 150:
        transaction_size = 'Small'
    elif amount <= 300:
        transaction_size = 'Medium'
    elif amount <= 600:
        transaction_size = 'Large'
    else:
        transaction_size = 'Very_Large'
    df['Transaction_Size'] = transaction_size
    
    df['Quantity_Log'] = np.log1p(df['Quantity'])
    
    # 5. Location-Device Interaction
    df['Location_Device'] = df['Customer Location'] + '_' + df['Device Used']
    
    # 6. Risk Indicators (use typical thresholds)
    df['High_Amount_Flag'] = (df['Transaction Amount'] > 500).astype(int)
    df['High_Quantity_Flag'] = (df['Quantity'] > 10).astype(int)
    df['Unusual_Hour_Flag'] = ((df['Transaction Hour'] < 6) | (df['Transaction Hour'] > 22)).astype(int)
    
    # Encode categorical features
    categorical_cols = ['Payment Method', 'Product Category', 'Customer Location', 
                       'Device Used', 'Hour_Bin', 'Age_Category', 'Transaction_Size', 
                       'Location_Device']
    
    for col in categorical_cols:
        if col in label_encoders:
            try:
                # Handle unknown categories
                if df[col].iloc[0] not in label_encoders[col].classes_:
                    print(f"Warning: '{df[col].iloc[0]}' not seen during training for {col}. Using default.")
                    df[col] = label_encoders[col].classes_[0]
                df[col] = label_encoders[col].transform(df[col])
            except Exception as e:
                print(f"Error encoding {col}: {e}")
                df[col] = 0
    
    return df

def predict_with_all_models(features, models, scaler):
    """Make predictions using all loaded models"""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    # Define feature order
    feature_cols = [
        'Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days', 'Transaction Hour',
        'Payment Method', 'Product Category', 'Customer Location', 'Device Used',
        'Amount_Log', 'Amount_per_Quantity', 'Amount_zscore',
        'Hour_Bin', 'Is_Weekend', 'Day_of_Week',
        'Age_Category', 'Account_Age_Weeks', 'Is_New_Account',
        'Transaction_Size', 'Quantity_Log',
        'Location_Device',
        'High_Amount_Flag', 'High_Quantity_Flag', 'Unusual_Hour_Flag'
    ]
    
    X = features[feature_cols].values
    X_scaled = scaler.transform(X)
    
    predictions = {}
    
    for model_name, model in models.items():
        try:
            prob = model.predict_proba(X_scaled)[0][1]
            
            prediction = "FRAUDULENT" if prob > 0.5 else "LEGITIMATE"
            predictions[model_name] = (prediction, prob)
            
            print(f"\n{model_name}:")
            print(f"  Prediction: {prediction}")
            print(f"  Fraud Probability: {prob:.2%}")
            print(f"  Confidence: {'HIGH' if abs(prob - 0.5) > 0.3 else 'MEDIUM' if abs(prob - 0.5) > 0.15 else 'LOW'}")
            
        except Exception as e:
            print(f"\n{model_name}: Error - {e}")
    
    return predictions

print("\n🔍 Loading models and preprocessors...")
models, scaler, label_encoders = load_models_and_preprocessors()

if not models or not scaler or not label_encoders:
    print("\n❌ Failed to load required files. Please ensure all model files exist.")
 
transaction_data = {}

trans_amount = "1500"
quantity = "15"
cust_age = "26"  #Get it from user session take it from register and login
acc_age_days = "250"
trans_hour = "15" #take it from current hour
payment_method = "Credit Card" #dropdown   Options: Credit Card, Debit Card, PayPal, Bank Transfer
product_cat = "Electronics"  #dropdown       Options: Electronics, Clothing, Groceries, Home, etc.
cust_loc = "India"  # take it current loc contry
device_used = "Mobile"    #dropdown     Options: Mobile, Desktop, Tablet
trans_date = "2026-01-12"    # take it current date in this format only

# Numeric inputs
transaction_data['Transaction Amount'] = float(trans_amount)
transaction_data['Quantity'] = int(quantity)
transaction_data['Customer Age'] = int(cust_age)
transaction_data['Account Age Days'] = int(acc_age_days)
transaction_data['Transaction Hour'] = int(trans_hour)
# Categorical inputs
transaction_data['Payment Method'] = payment_method
transaction_data['Product Category'] = product_cat
transaction_data['Customer Location'] = cust_loc
transaction_data['Device Used'] = device_used
transaction_data['Transaction Date'] = trans_date

print(f"transaction_data ::: {transaction_data}")
# Engineer features
print("\n🔧 Engineering features...")
features = engineer_features(transaction_data, label_encoders)

# Make predictions
predictions = predict_with_all_models(features, models, scaler)

print(predictions)