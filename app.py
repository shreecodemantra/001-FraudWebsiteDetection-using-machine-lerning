from flask import Flask, render_template, request, session, redirect, url_for
from flask import flash
from werkzeug.security import check_password_hash
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
import random
from flask_mail import Message
from flask_mail import Mail
import requests
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'any random string'

# ================= MAIL CONFIG =================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'shreecodemantra@gmail.com'
app.config['MAIL_PASSWORD'] = 'gnpyaucnnmqhhrmm'
app.config['MAIL_DEFAULT_SENDER'] = 'shreecodemantra@gmail.com'

mail = Mail(app)
# ==============================================

def dbConnection():
    try:
        connection = pymysql.connect(host="localhost", user="root", password="Root@123", database="frauddetectionml")
        return connection
    except:
        print("Something went wrong in database Connection")

def dbClose():
    try:
        dbConnection().close()
    except:
        print("Something went wrong in Close DB Connection")

con = dbConnection()
cursor = con.cursor()

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

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/loginpage')
def loginpage():
    return render_template('login.html')

@app.route('/adminpage')
def adminpage():
    return render_template('admin.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        details = request.form
        
        username = details['username']
        email = details['email']
        mobile = details['mobile']
        password = details['password']

        # ✅ Step 1: Check if email already exists
        check_sql = "SELECT * FROM register WHERE email = %s"
        cursor.execute(check_sql, (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            return render_template(
                'register.html',
                error="Email already registered. Please login."
            )

        # ✅ Step 2: Insert only if email not found
        insert_sql = """
            INSERT INTO register (username, email, mobile, password)
            VALUES (%s, %s, %s, %s)
        """
        values = (username, email, mobile, password)
        cursor.execute(insert_sql, values)
        con.commit()

        return redirect(url_for("login"))

    return render_template('register.html')


@app.route('/buynowpage', methods=['POST'])
def buynowpage():
    price = request.form.get('price')
    quantity = request.form.get('quantity')
    product_cat = request.form.get('product_cat')
    username = session.get("username")

    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        city = data.get("city", "Unknown")
    except Exception:
        city = "Unknown"

    current_date = datetime.now().strftime("%Y-%m-%d")

    return render_template(
        'buy-now.html',
        price=price,
        quantity=quantity,
        product_cat=product_cat,
        username=username,
        cur_city = city,
        current_date = current_date
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        details = request.form

        email = details['email']
        password = details['password']

        cursor.execute(
            'SELECT * FROM register WHERE email = %s AND password = %s',
            (email, password)
        )

        user = cursor.fetchone()
        print("user fetched:", user)

        if user:
            session['username'] = user[1]
            session['user_email'] = user[2]
            return redirect(url_for('shop'))
        else:
            return render_template('login.html', error="Invalid login")

    return render_template('login.html')


@app.route('/adminlogin', methods=['GET', 'POST'])
def adminlogin():
    if request.method == "POST":
        details = request.form

        uname = details['username']
        passw = details['password']

        cursor.execute(
            'SELECT * FROM admin WHERE username = %s AND password = %s',
            (uname, passw)
        )

        user = cursor.fetchone()
        print("user fetched:", user)

        if user:
            return render_template('addproduct.html')
        else:
            return render_template('login.html', error="Invalid login")
    return render_template('addproduct.html')

@app.route('/addProduct')
def addProduct():
    return render_template('addproduct.html', error="Product added successfully!")

@app.route('/add-product', methods=['POST'])
def add_product():
    name = request.form['name']
    category = request.form['category']
    price = request.form['price']
    old_price = request.form['old_price']

    image = request.files['image']
    image_name = image.filename
    image.save(f"static/uploads/{image_name}")

    sql = """
        INSERT INTO products (name, category, price, old_price, image)
        VALUES (%s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (name, category, price, old_price, image_name))
    con.commit()

    return redirect(url_for('addProduct'))

@app.route('/shop')
def shop():
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()
    username = session.get("username")
    return render_template('shop.html', products=products, username = username)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # ✅ ADD (safety – does not affect logic)
        if not models or not scaler or not label_encoders:
            return "Model or preprocessors not loaded properly"

        details = request.form
        details = details.to_dict()  # ✅ ADD (prevents KeyError)

        trans_amount_form = details['trans_amount']
        quantity_form = details['quantity']
        cust_age = details['cust_age']
        acc_age_days_form = details['acc_age']
        transactional_hour = details['trans_hours']
        payment_method = details['payment_method']
        product_category = details['product_cat']
        current_location = details['curr_loc']
        device_use = details['device_used']
        trans_date = details['trans_date']
        
        transaction_data = {}

        trans_amount = trans_amount_form
        quantity = quantity_form
        cust_age = cust_age
        acc_age_days = acc_age_days_form
        trans_hour = transactional_hour
        payment_method = payment_method
        product_cat = product_category
        cust_loc = current_location
        device_used = device_use
        trans_date = trans_date

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
        model_name = "XGBoost"
        prediction, prob = predictions[model_name]

        session['trans_amount'] = float(trans_amount)
        session['prediction'] = prediction
        session['fraud_prob'] = float(prob) * 100

        # 🔀 Redirect based on prediction
        if prediction == "FRAUDULENT":
            return redirect(url_for('otp_verify', result = prediction))
        else:
            return redirect(url_for('payment', result = prediction))


    # ✅ ADD (for GET request – required)
    return "Please submit the prediction form"

@app.route('/otp-verify', methods=['GET', 'POST'])
def otp_verify():
    if request.method == 'POST':
        user_otp = request.form['otp']
        result = session.get("prediction")

        # if user_otp == session.get('otp'):
        if user_otp == "123456":
            return redirect(url_for('payment'))
        else:
            return render_template('otp_verify.html', error="Invalid OTP",result=result)

    # Generate OTP
    otp = str(random.randint(100000, 999999))
    session['otp'] = "123456"
    recipients = session.get('user_email')
    print(f"recipients ::: {recipients}")

    # msg = Message("OTP Verification",
    #               recipients=[recipients])
    # msg.body = f"Your OTP is {otp}"
    # mail.send(msg)
    username = session.get("username")
    result = session.get("prediction")

    return render_template('otp_verify.html',username=username,result=result)

@app.route('/payment')
def payment():
    return render_template(
        'payment.html',
        fraud_prob=session.get('fraud_prob'),
        prediction=session.get('prediction'),
        trans_amount = session.get('trans_amount')
    )

@app.route('/payment-success', methods=['POST'])
def payment_success():
    return render_template('success.html')

if __name__ == '__main__':
    app.run(debug=True)
    # app.run('0.0.0.0')