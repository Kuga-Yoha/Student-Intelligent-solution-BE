# app.py

from flask import Flask, request, jsonify
import jwt
import datetime
from functools import wraps

app = Flask(__name__)

# Secret key for encoding and decoding JWT tokens
app.config['SECRET_KEY'] = 'your_secret_key'

# Dummy user data (replace with a user database in a real-world application)
users = {'user1': 'password1', 'user2': 'password2'}


# Function to generate a JWT token
def generate_token(username):
    expiration_time = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    payload = {'username': username, 'exp': expiration_time}
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    return token


# Decorator function to check if the user is authenticated
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['username']
        except:
            return jsonify({'message': 'Token is invalid'}), 401

        return f(current_user, *args, **kwargs)

    return decorated


# Login endpoint
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    username = data.get('username')
    password = data.get('password')

    if username in users and users[username] == password:
        token = generate_token(username)
        return jsonify({'token': token})
    else:
        return jsonify({'message': 'Invalid credentials'}), 401


# Logout endpoint (requires authentication)
@app.route('/logout', methods=['POST'])
@token_required
def logout(current_user):
    return jsonify({'message': f'User {current_user} logged out successfully'})


import pickle
import joblib


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = list(data.values())

    model = joblib.load('../model/best_model.pkl')

    output = model.predict(features)
    retur = 2


if __name__ == '__main__':
    app.run(debug=True)
