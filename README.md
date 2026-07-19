# AetherShield AI - Advanced Deepfake Facial Detection Web App

AetherShield AI is a modern, production-grade deepfake image detection web application. It features a responsive, glassmorphic dark-themed single-page web interface (HTML5, CSS3, JS) and a Python Flask backend powered by the pre-trained **MesoNet-4** neural network model and **OpenCV** face detection.

This project was upgraded from the original Colab notebook to provide an interactive interface to upload images, run live scanning visual simulations, isolate facial regions of interest, and produce detailed diagnostics reports showing if an image is **REAL** or a **DEEPFAKE**.

---

## Key Features

*   **Facial Recognition ROI:** Automatically detects and marks faces using OpenCV's Haar Cascade classifier, cropping them with 15% padding to analyze boundary textures where blending artifacts are common.
*   **MesoNet-4 Neural Net:** Uses a lightweight convolutional neural network optimized for detecting facial forgeries by analyzing mesoscopic features (noise distribution, sub-pixel blending, and generative artifacts).
*   **Visual Laser Scanner:** A glowing laser scan animation with rolling console logs simulates facial mesh evaluation and structural alignment.
*   **Dynamic Dashboard Report:** Displays diagnostic details: AI classification verdict (REAL/DEEPFAKE), circular gauge confidence score, side-by-side cropped face ROI vs source image with canvas bounding box drawing, processing time, and an automated explanation text.
*   **High Performance:** Optimized inference completes in milliseconds on standard CPU hardware.

---

## File Structure

```text
d:\Documents\Deep-Fake-detection--main\
├── app.py                      # Flask web server and /predict endpoint
├── mesonet.py                  # MesoNet architecture definition & preprocessing module
├── Meso4_DF.h5                 # Downloaded MesoNet model weights (1.1 MB)
├── templates/
│   └── index.html              # Frontend single-page application structure
├── static/
│   ├── css/
│   │   └── style.css           # Premium glassmorphic styling, scan animations & layout
│   └── js/
│       └── main.js             # Drag-and-drop controller, scan simulation & API client
├── uploads/                    # Temporary storage for uploads & face crops
└── README.md                   # Project documentation
```

---

## Installation & Setup

Ensure you have Anaconda or standard Python installed. The application is compatible with **Python 3.10** and **TensorFlow 2.15.0**.

### 1. Set Up Environment
Activate the environment containing TensorFlow and OpenCV:
```bash
# If using Conda
conda activate myenv
```

If these packages are not installed, install them using:
```bash
pip install tensorflow==2.15.0 opencv-python<4.9 numpy<2.0.0 Flask==2.0.0 pillow werkzeug
```
*(Note: NumPy must be kept below version 2.0.0 for compatibility with TensorFlow 2.15.0)*

### 2. Launch the Application
Run the Flask server:
```bash
python app.py
```

On first launch, `app.py` will automatically download the pre-trained MesoNet weights file (`Meso4_DF.h5`, ~1.1 MB) from GitHub into the root folder.

### 3. Open the Browser
Once the server starts, open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## Technical Details

### MesoNet Architecture
MesoNet is a shallow Convolutional Neural Network with approximately 28,000 parameters. It uses four successive convolutional blocks with varying filter sizes ($3 \times 3$ and $5 \times 5$), Batch Normalization, and Max Pooling to capture micro-level textural details rather than macro-level facial structures.

### Face Preprocessing Pipeline
1.  **Read Image:** Load image using OpenCV and convert it to grayscale.
2.  **Detection:** Haar Cascade detects faces. If multiple are found, the largest bounding box is selected.
3.  **Cropping:** Slices the face with 15% margin padding on each side.
4.  **Resizing:** Resizes the cropped face to $256 \times 256$ pixels.
5.  **Color Space:** Converts color from BGR to RGB.
6.  **Normalization:** Rescales pixel values from $[0, 255]$ to $[0, 1.0]$.
7.  **Inference:** Feeds the $1 \times 256 \times 256 \times 3$ tensor to the MesoNet model. Sigmoid output $\ge 0.5$ is classified as **REAL**, and $< 0.5$ as **FAKE** (DEEPFAKE).
