import os
import time
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from mesonet import MesoNetDetector

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Detector (automatically downloads weights on startup if not present)
print("Initializing MesoNet Deepfake Detector...")
detector = MesoNetDetector(weights_path='Meso4_DF.h5')

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_reasoning(verdict, confidence):
    """
    Generates a tailored explanation based on the prediction verdict and confidence.
    """
    if verdict == 'Fake':
        if confidence >= 90.0:
            return (
                "The analysis detected extremely strong markers of generative manipulation (90%+ confidence). "
                "Pronounced boundary artifacts and sub-pixel blending inconsistencies are visible around key facial landmarks "
                "(eyes, nose, and mouth), which are highly characteristic of advanced GAN or diffusion generative models."
            )
        elif confidence >= 70.0:
            return (
                "The system detected clear evidence of digital manipulation (70%-90% confidence). "
                "The image contains structural inconsistencies, lighting mismatches, and mesoscopic texture anomalies "
                "often introduced during face-swapping or neural face-synthesis processes."
            )
        else:
            return (
                "The image shows minor inconsistencies that suggest potential manipulation (50%-70% confidence). "
                "While some textures appear artificial, the score is bordering the safety threshold. "
                "This could indicate either a low-quality deepfake or an authentic photo with severe compression artifacts."
            )
    else:  # Real
        if confidence >= 90.0:
            return (
                "The system detected high authenticity (90%+ confidence). Facial features exhibit "
                "organic micro-textures, consistent lighting gradients, and uniform camera-sensor noise distributions. "
                "No generative blending boundaries or deepfake signatures were detected."
            )
        elif confidence >= 70.0:
            return (
                "The face exhibits solid structural authenticity (70%-90% confidence). Skin textures, shading, and boundary "
                "gradients align with natural photography. No clear signatures of digital fabrication were identified."
            )
        else:
            return (
                "The image leans authentic (50%-70% confidence), though the confidence is low. "
                "The model noted minor texture irregularities, which are commonly caused by aggressive image compression, "
                "low resolution, or poor lighting rather than actual deepfake manipulation."
            )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported extensions: PNG, JPG, JPEG, WEBP.'}), 400
    
    try:
        # Generate unique identifiers to avoid file conflicts
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1].lower()
        if not ext:
            ext = '.jpg'  # Default fallback
            
        original_filename = f"original_{file_id}{ext}"
        crop_filename = f"face_{file_id}{ext}"
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_filename)
        
        # Save uploaded file
        file.save(original_path)
        
        # Record file size
        file_size_bytes = os.path.getsize(original_path)
        file_size_kb = round(file_size_bytes / 1024, 2)
        
        # Run prediction
        start_time = time.time()
        result = detector.detect_and_predict(original_path, output_crop_path=crop_path)
        duration = round(time.time() - start_time, 3)
        
        # Handle prediction error
        if 'error' in result:
            # Clean up files in case of error
            if os.path.exists(original_path):
                os.remove(original_path)
            return jsonify({'error': result['error']}), 500
        
        # Generate explanation
        explanation = generate_reasoning(result['verdict'], result['confidence'])
        
        # Prepare response
        response_data = {
            'file_name': file.filename,
            'file_size_kb': file_size_kb,
            'processing_time_sec': duration,
            'faces_found': result['faces_found'],
            'num_faces': result['num_faces'],
            'bounding_boxes': result['bounding_boxes'],
            'prediction_score': result['prediction_score'],
            'verdict': result['verdict'],
            'confidence': result['confidence'],
            'explanation': explanation,
            'original_image_url': f'/uploads/{original_filename}',
            'cropped_image_url': f'/uploads/{crop_filename}' if result['faces_found'] else None
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f"Internal server error during analysis: {str(e)}"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files (original and cropped faces) to the web UI.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Bind to PORT environment variable (default 5000 for local, Hugging Face uses 7860)
    port = int(os.environ.get('PORT', 5000))
    # Bind to 0.0.0.0 to allow access inside Docker container / local network
    app.run(host='0.0.0.0', port=port, debug=False)
