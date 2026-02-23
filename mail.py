from flask_mail import Message, Mail
from flask import Flask

app = Flask(__name__)
app.secret_key = 'any random string'

# ================= MAIL CONFIG =================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'shreecodemantra@gmail.com'
app.config['MAIL_PASSWORD'] = 'gnpyaucnnmqhhrmm'  # ⚠️ Use App Password
app.config['MAIL_DEFAULT_SENDER'] = 'shreecodemantra@gmail.com'

mail = Mail(app)
# ==============================================

with app.app_context():
    msg = Message(
        subject="OTP Verification",
        recipients=["yash.salvi1209@gmail.com"],  # must be a list
        body="Your OTP is 1254"
    )
    mail.send(msg)

print("Mail sent successfully!")
