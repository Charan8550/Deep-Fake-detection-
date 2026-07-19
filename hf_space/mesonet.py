import os
import urllib.request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model

# MesoNet Meso-4 Architecture
def build_meso4(input_shape=(256, 256, 3)):
    """
    Builds the Meso-4 model architecture using the Keras Functional API.
    """
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    # Block 2
    x = Conv2D(8, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    # Block 3
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    
    # Block 4
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4), padding='same')(x)
    
    # Classifier
    y = Flatten()(x)
    y = Dropout(0.5)(y)
    y = Dense(16, activation='relu')(y)
    outputs = Dense(1, activation='sigmoid')(y)
    
    return Model(inputs=inputs, outputs=outputs)


class MesoNetDetector:
    def __init__(self, weights_path='Meso4_DF.h5'):
        self.weights_path = weights_path
        self.model = build_meso4()
        self._load_weights()
        
        # Load OpenCV Face Detector (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("Warning: Failed to load Haar Cascade face detector from OpenCV library.")

    def _load_weights(self):
        """
        Loads the pre-trained weights, downloading them if they do not exist locally.
        """
        if not os.path.exists(self.weights_path):
            print(f"Pre-trained weights not found. Downloading from GitHub...")
            url = 'https://github.com/DariusAf/MesoNet/raw/master/weights/Meso4_DF.h5'
            try:
                # Add User-Agent header to avoid block by GitHub raw content
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url, self.weights_path)
                print(f"Downloaded weights successfully and saved to: {self.weights_path}")
            except Exception as e:
                print(f"Error downloading weights: {e}")
                raise e
        
        self.model.load_weights(self.weights_path)
        print("MesoNet pre-trained weights loaded successfully.")

    def detect_and_predict(self, image_path, output_crop_path=None):
        """
        Detects faces in an image, runs prediction on the face crop, and returns results.
        
        Returns:
            dict: {
                'faces_found': bool,
                'num_faces': int,
                'bounding_boxes': list of lists [[x, y, w, h]],
                'prediction_score': float (sigmoid probability of being REAL, so 1 = Real, 0 = Fake),
                'verdict': str ('Real' or 'Fake'),
                'confidence': float (percentage),
                'error': str (if any)
            }
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {'error': 'Failed to read image file.'}
            
            h_img, w_img, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            faces_found = len(faces) > 0
            bounding_boxes = []
            
            if faces_found:
                # Get the largest face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x_box, y_box, w_box, h_box = largest_face
                
                # Add padding to face crop (15% padding on each side)
                pad_w = int(w_box * 0.15)
                pad_h = int(h_box * 0.15)
                x1 = max(0, x_box - pad_w)
                y1 = max(0, y_box - pad_h)
                x2 = min(w_img, x_box + w_box + pad_w)
                y2 = min(h_img, y_box + h_box + pad_h)
                
                face_crop = img[y1:y2, x1:x2]
                
                # Collect all face coordinates
                for (x, y, w, h) in faces:
                    bounding_boxes.append([int(x), int(y), int(w), int(h)])
            else:
                # Fallback to full image
                face_crop = img
                bounding_boxes = []
            
            # Save cropped face if requested
            if output_crop_path and face_crop.size > 0:
                cv2.imwrite(output_crop_path, face_crop)
            
            # Preprocess the cropped face region
            # Resize to 256x256
            face_resized = cv2.resize(face_crop, (256, 256))
            # Convert BGR to RGB (OpenCV loads BGR, but MesoNet expects RGB)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            face_normalized = face_rgb.astype('float32') / 255.0
            # Add batch dimension
            input_tensor = np.expand_dims(face_normalized, axis=0)
            
            # Run prediction
            # Output is probability of being REAL (1 = Real, 0 = Fake)
            prediction_score = float(self.model(input_tensor, training=False).numpy()[0][0])
            
            # Determine verdict (0.5 threshold)
            if prediction_score >= 0.5:
                verdict = 'Real'
                confidence = prediction_score * 100.0
            else:
                verdict = 'Fake'
                confidence = (1.0 - prediction_score) * 100.0
                
            return {
                'faces_found': faces_found,
                'num_faces': len(faces),
                'bounding_boxes': bounding_boxes,
                'prediction_score': prediction_score,
                'verdict': verdict,
                'confidence': round(confidence, 2)
            }
            
        except Exception as e:
            return {'error': str(e)}
