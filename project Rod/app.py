from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
from model import load_modell, predict
from db import init_db, is_database_empty, authenticate, add_user
from config import MODEL_PATH
from PIL import Image
from io import BytesIO
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

model = load_modell(MODEL_PATH)

current_year = datetime.now().year

# Define a directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def model_metric():
    # Define the data dictionary
    model_metrics = {
        'Healthy': {
            'Precision': 0.75,
            'Recall': 0.99,
            'F1 Score': 0.86,
            'Support': 249
        },
        'Glaucomatous': {
            'Precision': 0.98,
            'Recall': 0.62,
            'F1 Score': 0.76,
            'Support': 210
        }
    }
    
    model_accuracy=0.8191721132897604*100
    
    return model_metrics, model_accuracy

def resize_image(file):
    # Convert the file stream into an image object
    image = Image.open(BytesIO(file.read()))
    
    # Resize the image to 224x224
    resized_image = image.resize((224, 224))
        
    return resized_image

# Define breadcrumbs data
def get_breadcrumbs():
    breadcrumbs=[{'text': 'Logout', 'url': '/logout'}]
    if 'username' in session:
        breadcrumbs.append({'text': 'Dashboard', 'url': '/'})
    return breadcrumbs

# Route for register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if is_database_empty():
        init_db()  # Initialize database if it's empty

    if request.method == 'POST':
        full_name = request.form['full_name']
        username = request.form['username']
        password = request.form['password']
        # Add user to the database
        add_user(full_name, username, password)
        return redirect(url_for('login'))

    return render_template('register.html', year=current_year)

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_database_empty():
        return redirect(url_for('register'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = authenticate(username, password)
        if user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html', year=current_year)


# Route for logging out
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Route for dashboard
@app.route('/')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    breadcrumbs = get_breadcrumbs()
    model_metrics, accuracy = model_metric()
    return render_template('dashboard.html', breadcrumbs=breadcrumbs, year=current_year, model_metrics=model_metrics, accuracy=accuracy)

@app.route('/image1')
def get_image1():
    return send_from_directory('images', 'download.png')

@app.route('/image2')
def get_image2():
    return send_from_directory('images', 'download (1).png')

# Route for uploading an image
@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    breadcrumbs = get_breadcrumbs()
    breadcrumbs.append({'text': 'Upload Image', 'url': '/upload_image'})
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            resized_image = resize_image(file)
            resized_image.save(filename)
            print(filename)
            return redirect(url_for('display_prediction', filename=file.filename))
    return render_template('upload_image.html', breadcrumbs=breadcrumbs, year=current_year)

# Route for displaying predictions
@app.route('/prediction/<filename>')
def display_prediction(filename):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    breadcrumbs = get_breadcrumbs()
    breadcrumbs.append({'text': 'Upload Image', 'url': '/upload_image'})
    breadcrumbs.append({'text': 'Prediction', 'url': '/prediction/{}'.format(filename)})
    
    image_url = url_for('static', filename=os.path.join('uploads', filename))

    # Construct the URL for the uploaded image
    image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img_array = np.array(image)

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize pixel values if necessary
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    
    # Perform prediction using the loaded model
    predictions, claas = predict(model, img_array)

    return render_template('prediction_display.html', predictions=predictions[0][0], classs=claas, image_url=image_url, breadcrumbs=breadcrumbs, year=current_year)

if __name__ == '__main__':
    app.run(debug=True)
