import os
import cv2
import numpy as np
import tempfile
import traceback
from flask import Flask, request, send_file, Response,render_template,jsonify, url_for, flash, session, redirect
from ultralytics import YOLO
import matplotlib.pyplot as plt
import mysql.connector
 
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            user='root',
            password='2003',
            host='localhost',
            database='register'
        )
        if conn.is_connected():
            print("Database connected successfully!")
        return conn
    except Error as e:
        print(f"Database Connection Error: {e}")
        return None
 
 

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = None
        try:
            conn = mysql.connector.connect(user='root', password='2003', host='localhost', database='register')
            cursor = conn.cursor()

            # Fetch password for the given email
            cursor.execute("SELECT password FROM box WHERE email = %s", (email,))
            result = cursor.fetchone()

            if result:
                stored_password = result[0]  # Password is stored as plain text
                if password == stored_password:  # Compare input password with stored password
                    session['user'] = email
                    return redirect(url_for('pcb'))
                else:
                    flash("Invalid password.", "danger")
                    return redirect(url_for('home'))
            else:
                flash("Invalid Email", "danger")
                return redirect(url_for('home'))

        except mysql.connector.Error as e:
            flash(f"Database error: {str(e)}", "error")

        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

    return render_template("login.html")



@app.route('/register', methods=['POST', 'GET'])
def register():
    message = ""
    if request.method == 'POST':
        name = request.form['name']
        lname = request.form['lname']
        email = request.form['email']
        password = request.form['password']

        conn = None
        cursor = None

        try:
            conn = mysql.connector.connect(user='root', password='2003', host='localhost', database='register')
            cursor = conn.cursor()

            # Check if user already exists
            cursor.execute("SELECT email FROM box WHERE email = %s", (email,))
            existing_user = cursor.fetchone()  # Fetch one result

            # Fetch all remaining results to avoid unread result error
            cursor.fetchall()  # This clears any remaining unread results

            if existing_user:  
                flash("Email already exists ", "danger")
                return redirect(url_for('register'))

            # Insert new user
            query = "INSERT INTO box (name, lname, email, password) VALUES (%s, %s, %s, %s)"
            values = (name, lname, email, password)
            cursor.execute(query, values)
            conn.commit()

            flash("Registered successfully!", "success")
            return redirect(url_for('home'))

        except mysql.connector.Error as e:
            flash(f"Database error: {str(e)}", "error")

        finally:
            if cursor:
                cursor.close()  # Close cursor only if it's open
            if conn and conn.is_connected():
                conn.close()

    return render_template('register.html')

 
@app.route("/pcb",methods=["POST","GET"])
def pcb():
    upload_file()
    return render_template("pcb.html")
 
@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully!")
    return redirect(url_for('home'))
 
@app.route("/1", methods=["POST"])
def upload_file():
    print("Inside upload function")
   
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
 
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
 
    # Save uploaded image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    print(f"File uploaded successfully: {filepath}")
 
    # Load and preprocess image for YOLO model
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({"error": "Error reading the image file"}), 500
 
    # Resize image for YOLO model
    image_resized = cv2.resize(image, (512, 512))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
 
    results = model(image_rgb, imgsz=512, conf=0.25, save=True, save_txt=True, save_conf=True)
 
    # Display results
    result_img = results[0].plot()  # Draw detections on the image
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.axis("off")
    plt.show()
 
 
    # Save processed image
    output_filename = "output_" + file.filename
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
    cv2.imwrite(output_path, image)
    print(f"Processed image saved: {output_path}")
 
    # Return JSON with correct image URL
    return jsonify({"image_url": url_for('static', filename=f'uploads/{output_filename}', _external=True)})
 
MODEL_PATH = r'yolov8_model\best.torchscript'
model = YOLO(MODEL_PATH)
 
 
@app.route('/upload', methods=['POST'])
def detect_objects():
    try:
        print(f"[INFO] Received request with content type: {request.content_type}")
       
        # Get raw binary data from request
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle form data (file upload)
            print(f"[INFO] Processing as multipart/form-data, files: {list(request.files.keys())}")
            if 'image' not in request.files:
                print("[ERROR] No image file found in request")
                return "No image provided", 400
            img_binary = request.files['image'].read()
            print(f"[INFO] Image binary size: {len(img_binary)} bytes")
        else:
            # Handle raw binary data directly
            print("[INFO] Processing as raw binary data")
            img_binary = request.data
            print(f"[INFO] Raw binary data size: {len(img_binary)} bytes")
       
        if not img_binary:
            print("[ERROR] Empty image data received")
            return "Empty image data", 400
           
        # Convert binary data to image
        print("[INFO] Converting binary data to image")
        nparr = np.frombuffer(img_binary, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
        if image is None:
            print("[ERROR] Failed to decode image data")
            return "Invalid image format", 400
       
        print(f"[INFO] Successfully decoded image with shape: {image.shape}")
       
        # Resize image to match model input
        print("[INFO] Resizing image to 512x512")
        image_resized = cv2.resize(image, (512, 512))
        print(f"[INFO] Resized image shape: {image_resized.shape}")
       
        # Convert BGR to RGB for the model
        print("[INFO] Converting color from BGR to RGB")
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
       
        # Get confidence threshold from request or use default
        conf_threshold = 0.25
        if request.content_type and 'multipart/form-data' in request.content_type:
            if 'conf' in request.form:
                conf_threshold = float(request.form.get('conf'))
                print(f"[INFO] Using confidence threshold from form: {conf_threshold}")
        elif 'conf' in request.args:
            conf_threshold = float(request.args.get('conf'))
            print(f"[INFO] Using confidence threshold from URL params: {conf_threshold}")
        else:
            print(f"[INFO] Using default confidence threshold: {conf_threshold}")
       
        # Create a temporary file for the result
        print("[INFO] Creating temporary file for results")
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        print(f"[INFO] Temporary file created: {temp_filename}")
       
        # Run inference
        print("[INFO] Running YOLO model inference")
        start_time = cv2.getTickCount()
        results = model(image_rgb,
                       imgsz=512,
                       conf=conf_threshold)
        end_time = cv2.getTickCount()
        inference_time = (end_time - start_time) / cv2.getTickFrequency()
        print(f"[INFO] Inference completed in {inference_time:.4f} seconds")
       
        # Plot results on the image
        print("[INFO] Plotting detection results on image")
        result_img = results[0].plot()
       
        # Get number of detections
        num_detections = len(results[0].boxes)
        print(f"[INFO] Found {num_detections} objects in the image")
       
        # Print detection details
        for i, box in enumerate(results[0].boxes):
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            coords = box.xyxy[0].tolist()
            print(f"[INFO] Detection {i+1}: Class={cls}, Confidence={conf:.4f}, Coordinates={coords}")
       
        # Convert back to BGR for OpenCV to save correctly
        print("[INFO] Converting result image from RGB to BGR for saving")
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
       
        # Save result image to the temporary file
        print(f"[INFO] Saving result image to: {temp_filename}")
        cv2.imwrite(temp_filename, result_img)
       
        # Read the image file and return it as binary response
        print("[INFO] Reading processed image for response")
        with open(temp_filename, 'rb') as f:
            image_binary = f.read()
       
        print(f"[INFO] Response image size: {len(image_binary)} bytes")
       
        # Delete the temporary file
        print(f"[INFO] Cleaning up temporary file: {temp_filename}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print("[INFO] Temporary file deleted successfully")
       
        # Return binary image directly
        print("[INFO] Sending response to client")
        return Response(
            image_binary,
            mimetype='image/jpeg'
        )
       
    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return str(e), 500
 
if __name__ == "__main__":
    app.run(debug=True)
 
 
 