import os
import urllib.request
from mesonet import MesoNetDetector

def verify():
    print("--- AetherShield AI: Verification Script ---")
    
    # 1. Download sample face image (Lena test image from OpenCV GitHub)
    sample_img_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    sample_img_path = "sample_face.jpg"
    crop_img_path = "sample_face_crop.jpg"
    
    if not os.path.exists(sample_img_path):
        print(f"Downloading sample face image from: {sample_img_url}...")
        try:
            urllib.request.urlretrieve(sample_img_url, sample_img_path)
            print("Sample image downloaded successfully.")
        except Exception as e:
            print(f"Error downloading sample image: {e}")
            return
            
    # 2. Instantiate MesoNet Detector
    print("\nInstantiating MesoNet Detector...")
    try:
        detector = MesoNetDetector(weights_path='Meso4_DF.h5')
    except Exception as e:
        print(f"Error instantiating detector: {e}")
        return
        
    # 3. Run prediction on the sample image
    print("\nRunning face detection and deepfake prediction...")
    try:
        results = detector.detect_and_predict(sample_img_path, output_crop_path=crop_img_path)
    except Exception as e:
        print(f"Prediction failed with exception: {e}")
        return
        
    # 4. Display results
    print("\n--- Diagnostic Results ---")
    if 'error' in results:
        print(f"Error encountered: {results['error']}")
    else:
        print(f"Faces Found: {results['faces_found']}")
        print(f"Number of Faces: {results['num_faces']}")
        print(f"Bounding Boxes: {results['bounding_boxes']}")
        print(f"Prediction Score (Sigmoid Output): {results['prediction_score']:.5f}")
        print(f"Verdict: {results['verdict']}")
        print(f"Confidence: {results['confidence']}%")
        
        # Verify output files exist
        if os.path.exists(crop_img_path):
            print(f"Verification Success: Cropped face saved to {crop_img_path}")
        else:
            print("Verification Warning: Cropped face file was not created.")
            
    # Cleanup files (optional - let's keep them so the user can see them, or clean them up. Let's keep them for reference).
    print("\nVerification process completed.")

if __name__ == '__main__':
    verify()
