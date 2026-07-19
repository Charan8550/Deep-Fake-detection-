document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const dropzoneContent = document.getElementById('dropzoneContent');
    
    // Preview & Scan elements
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const laserScanner = document.getElementById('laserScanner');
    const statusTitle = document.getElementById('statusTitle');
    const consoleLogs = document.getElementById('consoleLogs');
    
    // Dashboard elements
    const resultsSection = document.getElementById('resultsSection');
    const emptyState = document.getElementById('emptyState');
    const resultsDashboard = document.getElementById('resultsDashboard');
    const verdictBanner = document.getElementById('verdictBanner');
    const verdictIconContainer = document.getElementById('verdictIconContainer');
    const verdictValue = document.getElementById('verdictValue');
    const confidenceValue = document.getElementById('confidenceValue');
    const gaugeFill = document.getElementById('gaugeFill');
    
    // Images Result
    const originalImageResult = document.getElementById('originalImageResult');
    const croppedFaceResult = document.getElementById('croppedFaceResult');
    const noFacePlaceholder = document.getElementById('noFacePlaceholder');
    const boundingBoxCanvas = document.getElementById('boundingBoxCanvas');
    
    // Metadata Details
    const metaFileName = document.getElementById('metaFileName');
    const metaFileSize = document.getElementById('metaFileSize');
    const metaDuration = document.getElementById('metaDuration');
    const metaFacesCount = document.getElementById('metaFacesCount');
    const aiExplanation = document.getElementById('aiExplanation');
    const resetBtn = document.getElementById('resetBtn');

    // Global variables
    let selectedFile = null;
    let bboxes = [];

    // Initialize Lucide Icons
    lucide.createIcons();

    // Event Listeners for Dropzone
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    dropzone.addEventListener('dragenter', dragOverHandler);
    dropzone.addEventListener('dragover', dragOverHandler);
    dropzone.addEventListener('dragleave', dragLeaveHandler);
    dropzone.addEventListener('drop', dropHandler);

    resetBtn.addEventListener('click', resetApp);

    function dragOverHandler(e) {
        e.preventDefault();
        dropzone.classList.add('dragover');
    }

    function dragLeaveHandler(e) {
        e.preventDefault();
        dropzone.classList.remove('dragover');
    }

    function dropHandler(e) {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            processFile(e.dataTransfer.files[0]);
        }
    }

    function handleFileSelect(e) {
        if (e.target.files.length > 0) {
            processFile(e.target.files[0]);
        }
    }

    function processFile(file) {
        if (!file.type.match('image.*')) {
            alert('Invalid file format. Please upload an image (PNG, JPG, JPEG, WEBP).');
            return;
        }
        selectedFile = file;

        // Display local preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            // Switch dropzone view to preview/scan mode
            dropzoneContent.style.opacity = '0';
            setTimeout(() => {
                dropzoneContent.style.display = 'none';
                previewContainer.style.display = 'flex';
                // Trigger Scan
                startNeuralScan();
            }, 300);
        };
        reader.readAsDataURL(file);
    }

    function addConsoleLog(message, delay, type = 'info') {
        return new Promise((resolve) => {
            setTimeout(() => {
                const line = document.createElement('div');
                line.className = `console-line ${type}`;
                
                const timestamp = new Date().toLocaleTimeString();
                line.textContent = `[${timestamp}] ${message}`;
                
                consoleLogs.appendChild(line);
                consoleLogs.scrollTop = consoleLogs.scrollHeight;
                resolve();
            }, delay);
        });
    }

    async function startNeuralScan() {
        // Clear logs
        consoleLogs.innerHTML = '';
        laserScanner.style.display = 'block';
        statusTitle.textContent = 'Analyzing Facial Data...';

        // Prepare upload form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Perform parallel actions: visual scan timeline and backend request
        const apiPromise = fetch('/predict', {
            method: 'POST',
            body: formData
        }).then(res => {
            if (!res.ok) {
                return res.json().then(err => { throw new Error(err.error || 'Server error during scan.') });
            }
            return res.json();
        });

        // Visual scan animation logging sequence (2.5s timeline)
        const logPromise = (async () => {
            await addConsoleLog('AetherShield Core Engine activated.', 0);
            await addConsoleLog('Checking image dimensions and EXIF headers...', 300);
            await addConsoleLog('Initializing OpenCV Haar Cascade model...', 400);
            await addConsoleLog('Locating frontal facial coordinates (ROI)...', 500);
            await addConsoleLog('Slicing image and isolating facial patch...', 400);
            await addConsoleLog('Normalizing sub-pixel pixel tensor values to [0.0, 1.0]...', 400);
            await addConsoleLog('Feeding input tensor (256x256x3) to MesoNet-4...', 300);
            await addConsoleLog('Running convolutional filters & feature maps...', 200);
        })();

        try {
            // Wait for both the minimum visual scan time and the API request to complete
            const [_, result] = await Promise.all([
                logPromise,
                apiPromise
            ]);

            // Add final logs from analysis result
            await addConsoleLog('Evaluation cycle complete.', 100, 'success');
            if (result.faces_found) {
                await addConsoleLog(`OpenCV: Detected ${result.num_faces} face(s) in frame.`, 100);
            } else {
                await addConsoleLog('OpenCV: Face detection failed. Analyzing full frame.', 100, 'warning');
            }
            await addConsoleLog(`Inference execution: ${result.processing_time_sec}s.`, 100);
            await addConsoleLog('Generating report dashboard...', 100, 'success');

            // Brief delay to allow user to read final logs
            setTimeout(() => {
                displayDashboard(result);
            }, 600);

        } catch (error) {
            console.error(error);
            addConsoleLog(`Critical Error: ${error.message}`, 100, 'warning');
            statusTitle.textContent = 'Scan Interrupted';
            laserScanner.style.display = 'none';
            alert(`Analysis failed: ${error.message}`);
            // Wait a moment and reset
            setTimeout(resetApp, 3000);
        }
    }

    function displayDashboard(data) {
        // Reset old styles
        verdictBanner.className = 'verdict-banner';
        
        // Hide upload scan panel and empty state
        previewContainer.style.display = 'none';
        dropzoneContent.style.display = 'flex';
        dropzoneContent.style.opacity = '1';
        emptyState.style.display = 'none';
        
        // Show dashboard
        resultsDashboard.style.display = 'flex';
        
        // 1. Render Verdict Title and Colors
        if (data.verdict === 'Real') {
            verdictBanner.classList.add('real');
            verdictValue.textContent = 'REAL';
            verdictIconContainer.innerHTML = '<i data-lucide="shield-check"></i>';
            document.documentElement.style.setProperty('--accent-color', 'var(--color-real)');
        } else {
            verdictBanner.classList.add('fake');
            verdictValue.textContent = 'DEEPFAKE';
            verdictIconContainer.innerHTML = '<i data-lucide="shield-alert"></i>';
            document.documentElement.style.setProperty('--accent-color', 'var(--color-fake)');
        }
        lucide.createIcons(); // Re-initialize the icon in verdict banner

        // 2. Animate Circular Progress Gauge
        confidenceValue.textContent = `${data.confidence}%`;
        const radius = 34;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (data.confidence / 100) * circumference;
        
        // Set dash offset for animation
        gaugeFill.style.strokeDasharray = `${circumference} ${circumference}`;
        gaugeFill.style.strokeDashoffset = circumference;
        // Trigger reflow to start transition
        gaugeFill.getBoundingClientRect();
        gaugeFill.style.strokeDashoffset = offset;

        // 3. Render Images
        originalImageResult.src = data.original_image_url;
        bboxes = data.bounding_boxes;
        
        // Setup Bounding Box Canvas drawing
        originalImageResult.onload = () => {
            drawBoundingBoxes();
        };

        if (data.faces_found && data.cropped_image_url) {
            croppedFaceResult.src = data.cropped_image_url;
            croppedFaceResult.style.display = 'block';
            noFacePlaceholder.style.display = 'none';
        } else {
            croppedFaceResult.src = '';
            croppedFaceResult.style.display = 'none';
            noFacePlaceholder.style.display = 'flex';
        }

        // 4. Populate Metadata Diagnostics
        metaFileName.textContent = truncateString(data.file_name, 24);
        metaFileSize.textContent = `${data.file_size_kb} KB`;
        metaDuration.textContent = `${data.processing_time_sec} sec`;
        metaFacesCount.textContent = data.faces_found ? `${data.num_faces} Face(s)` : 'None (analyzed full)';
        
        // 5. Populate AI Explanation
        aiExplanation.textContent = data.explanation;
    }

    function drawBoundingBoxes() {
        const img = originalImageResult;
        const canvas = boundingBoxCanvas;
        const ctx = canvas.getContext('2d');

        // Set display dimensions to match rendered image size
        const displayWidth = img.clientWidth;
        const displayHeight = img.clientHeight;
        canvas.width = displayWidth;
        canvas.height = displayHeight;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (bboxes.length === 0) return;

        // Get scaling factors between natural dimensions and displayed dimensions
        const scaleX = displayWidth / img.naturalWidth;
        const scaleY = displayHeight / img.naturalHeight;

        bboxes.forEach(box => {
            const [x, y, w, h] = box;
            const scaledX = x * scaleX;
            const scaledY = y * scaleY;
            const scaledW = w * scaleX;
            const scaledH = h * scaleY;

            // Draw Box
            ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent-color').trim();
            ctx.lineWidth = 3;
            ctx.shadowBlur = 6;
            ctx.shadowColor = ctx.strokeStyle;
            
            // Draw rectangle
            ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
            
            // Draw small corner brackets for HUD/Sci-fi effect
            drawCornerHUD(ctx, scaledX, scaledY, scaledW, scaledH);
        });
    }

    function drawCornerHUD(ctx, x, y, w, h) {
        ctx.shadowBlur = 0; // Reset shadow for fine HUD lines
        ctx.lineWidth = 1;
        ctx.fillStyle = ctx.strokeStyle;
        
        const bracketSize = Math.min(w, h) * 0.15; // 15% of box size
        
        // Draw HUD details on corners (or text label)
        ctx.font = '9px Outfit';
        ctx.fillText('FACE DETECTED', x + 5, y - 6);
    }

    // Redraw bounding boxes on window resize to keep canvas aligned
    window.addEventListener('resize', () => {
        if (resultsDashboard.style.display === 'flex') {
            drawBoundingBoxes();
        }
    });

    function resetApp() {
        fileInput.value = '';
        selectedFile = null;
        bboxes = [];
        
        // Hide preview container
        previewContainer.style.display = 'none';
        laserScanner.style.display = 'none';
        
        // Reset dropzone visibility
        dropzoneContent.style.display = 'flex';
        dropzoneContent.style.opacity = '1';
        
        // Hide dashboard & show empty state
        resultsDashboard.style.display = 'none';
        emptyState.style.display = 'flex';
    }

    function truncateString(str, num) {
        if (str.length <= num) return str;
        return str.slice(0, num) + '...';
    }
});
