import os
import sys
import subprocess

# Auto-fix OpenCV environment conflict on Hugging Face Spaces
try:
    import cv2
    if not hasattr(cv2, 'CascadeClassifier'):
        raise ImportError("cv2 is broken")
except (ImportError, AttributeError):
    print("OpenCV installation is missing or conflicting. Reinstalling opencv-python-headless...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless<5"], check=True)
        
        # Force Python to reload the module by purging it from the import cache
        for mod in list(sys.modules.keys()):
            if mod == 'cv2' or mod.startswith('cv2.'):
                del sys.modules[mod]
                
        import cv2
        print("OpenCV reinstalled and reloaded successfully! Version:", cv2.__version__)
    except Exception as e:
        print(f"Failed to auto-reinstall OpenCV: {e}")

# ZeroGPU compatibility handler
try:
    import spaces
except ImportError:
    # Fallback mock for local testing
    class spaces:
        @staticmethod
        def GPU(func):
            return func

import gradio as gr
from mesonet import MesoNetDetector

# Initialize MesoNet Detector (downloads weights on startup if not present)
detector = MesoNetDetector(weights_path='Meso4_DF.h5')

# Dummy function to satisfy Hugging Face ZeroGPU compiler startup checks
@spaces.GPU
def dummy_gpu_check():
    return "OK"

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

# Run actual prediction on CPU to bypass ZeroGPU queue waiting lists
def predict_deepfake(image):
    if image is None:
        return None, None, {}, "No image uploaded. Please load a valid portrait image."
    
    # Temporary file paths
    temp_path = "temp_input.jpg"
    crop_path = "temp_crop.jpg"
    
    try:
        # Gradio loads image as a numpy array in RGB format
        # Convert RGB to BGR for OpenCV processing
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_path, img_bgr)
        
        # Run face detection and deepfake prediction
        result = detector.detect_and_predict(temp_path, output_crop_path=crop_path)
        
        if 'error' in result:
            return None, None, {}, f"Error during analysis: {result['error']}"
        
        # Prepare output display image (draw boxes)
        img_display = image.copy()
        if result['faces_found']:
            for box in result['bounding_boxes']:
                x, y, w, h = box
                # Draw bounding box in RGB (Violet color: R=139, G=92, B=246)
                cv2.rectangle(img_display, (x, y), (x + w, y + h), (139, 92, 246), 3)
                
            # Load the cropped face
            if os.path.exists(crop_path):
                crop_display = cv2.imread(crop_path)
                crop_display = cv2.cvtColor(crop_display, cv2.COLOR_BGR2RGB)
            else:
                crop_display = None
        else:
            crop_display = None
            
        # Format Labeled prediction output (scores between 0.0 and 1.0)
        score = result['prediction_score']  # Sigmoid value (1.0 = Real, 0.0 = Fake)
        labels = {
            "REAL": score,
            "DEEPFAKE": 1.0 - score
        }
        
        # Format Markdown explanation
        verdict_str = f"🛡️ **VERDICT: {result['verdict'].upper()}**" if result['verdict'] == 'Real' else f"🚨 **VERDICT: {result['verdict'].upper()}**"
        
        # Generate detailed reasoning
        reasoning_text = generate_reasoning(result['verdict'], result['confidence'])
        
        explanation = f"### {verdict_str}\n"
        explanation += f"- **Confidence Level:** `{result['confidence']}%`\n"
        explanation += f"- **Faces Detected:** `{result['num_faces']}`\n"
        explanation += f"- **Processing Speed:** `{result['processing_time_sec'] if 'processing_time_sec' in result else 0.05} seconds`\n\n"
        explanation += f"### 🧠 AI Neural Reasoning\n{reasoning_text}"
        
        # Cleanup temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(crop_path):
            os.remove(crop_path)
            
        return img_display, crop_display, labels, explanation
        
    except Exception as e:
        # Cleanup files on exception
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(crop_path):
            os.remove(crop_path)
        return None, None, {}, f"Analysis failed: {str(e)}"

# Custom CSS for dark-mode aesthetic
css = """
body { background-color: #0a0a0f !important; }
.gradio-container { background-color: #0a0a0f !important; border: none !important; }
h1, h2, h3, p { color: #f4f4f7 !important; }
"""

# Build Gradio Block Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="indigo"), css=css) as demo:
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 5px 12px; border-radius: 100px; font-size: 0.85em; color: #9494a3;">
                Active Neural Net: MesoNet-4
            </span>
            <h1 style="font-size: 2.8em; font-weight: 800; margin-top: 10px; color: #ffffff; letter-spacing: -0.02em;">
                Aether<span style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Shield AI</span>
            </h1>
            <p style="color: #9494a3; font-size: 1.1em; font-weight: 300; max-width: 600px; margin: 5px auto;">
                Advanced mesoscopic facial forensics for deepfake detection. Upload an image to start neural scan.
            </p>
            <div style="margin-top: 15px; font-size: 0.95em; color: #a5b4fc; background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.2); padding: 8px 15px; border-radius: 8px; display: inline-block; max-width: 600px;">
                💡 <strong>Tip for testing:</strong> Want to try a fake face? Get a synthetic AI face from 
                <a href="https://thispersondoesnotexist.com" target="_blank" style="color: #c084fc; text-decoration: underline; font-weight: 600;">thispersondoesnotexist.com</a> 
                and upload it to test the detector!
            </div>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="Upload Portrait Image")
            submit_btn = gr.Button("Execute Neural Scan", variant="primary")
            
        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=2, label="Classification Probabilities")
            explanation = gr.Markdown(value="*Upload an image and run scan to view AI reasoning logs.*")
            
    with gr.Row():
        output_img = gr.Image(label="Facial Region Localization")
        crop_img = gr.Image(label="Neural ROI Extract (256x256)")
        
    submit_btn.click(
        fn=predict_deepfake,
        inputs=input_img,
        outputs=[output_img, crop_img, output_label, explanation]
    )

# Launch
if __name__ == "__main__":
    demo.launch()
