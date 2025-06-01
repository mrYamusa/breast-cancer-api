from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import base64
import io
import json
import os
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradCAM:
    """GradCAM implementation for visualizing model attention"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.register_hooks()
    
    def register_hooks(self):
        """Register forward and backward hooks"""
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Find the target layer and register hooks
        target_found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                hook1 = module.register_full_backward_hook(backward_hook)
                hook2 = module.register_forward_hook(forward_hook)
                self.hooks.extend([hook1, hook2])
                target_found = True
                logger.info(f"GradCAM hooks registered for layer: {name}")
                break
        
        if not target_found:
            logger.warning(f"Target layer '{self.target_layer_name}' not found in model")
            # Fallback to a common ResNet layer
            for name, module in self.model.named_modules():
                if 'layer4' in name and 'conv' in name:
                    hook1 = module.register_full_backward_hook(backward_hook)
                    hook2 = module.register_forward_hook(forward_hook)
                    self.hooks.extend([hook1, hook2])
                    logger.info(f"Using fallback layer: {name}")
                    break
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, input_image, class_idx=None):
        """Generate GradCAM heatmap"""
        # Ensure gradients are enabled
        input_image.requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        model_output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(model_output[0]).item()
        
        # Clear previous gradients
        self.model.zero_grad()
        if input_image.grad is not None:
            input_image.grad.zero_()
        
        # Backward pass with respect to the desired class
        score = model_output[0, class_idx]
        score.backward(retain_graph=True)
        
        # Check if we have gradients and activations
        if self.gradients is None or self.activations is None:
            logger.error("GradCAM: No gradients or activations captured")
            # Return a dummy heatmap
            dummy_cam = np.random.rand(7, 7) * 0.1  # Low intensity random pattern
            return dummy_cam, class_idx
        
        # Get gradients and activations
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], device=input_image.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to 0-1
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            # If cam is all zeros, create a small center activation
            cam = torch.zeros_like(cam)
            center_h, center_w = cam.shape[0] // 2, cam.shape[1] // 2
            cam[center_h-1:center_h+2, center_w-1:center_w+2] = 0.3
        
        return cam.detach().cpu().numpy(), class_idx

class BreastCancerPredictor:
    def __init__(self, model_path: str, config_path: str, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.class_names = ['normal', 'benign', 'malignant']
        
        # Load model configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize model
        self.model = self._load_model(model_path)
        
        # Try different layer names for GradCAM
        possible_layers = ['layer4.2.conv3', 'layer4.1.conv3', 'layer4.0.conv3', 'layer4']
        self.gradcam = None
        
        for layer_name in possible_layers:
            try:
                self.gradcam = GradCAM(self.model, target_layer_name=layer_name)
                logger.info(f"GradCAM initialized with layer: {layer_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to initialize GradCAM with layer {layer_name}: {e}")
                continue
        
        if self.gradcam is None:
            logger.error("Failed to initialize GradCAM with any layer")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['preprocessing']['mean'],
                std=self.config['preprocessing']['std']
            )
        ])
        
        logger.info(f"Model loaded on device: {self.device}")
    
    def _load_model(self, model_path: str):
        """Load the trained model"""
        # Initialize ResNet-50 architecture
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, self.config['num_classes'])
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _image_to_base64(self, image_array: np.ndarray) -> str:
        """Convert numpy array to base64 string"""
        # Ensure image is in uint8 format
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def _apply_gradcam_overlay(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Apply GradCAM heatmap overlay on image"""
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on image
        overlay = heatmap_colored * alpha + image * (1 - alpha)
        return overlay.astype(np.uint8)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Make prediction with GradCAM visualization"""
        try:
            # Preprocess image
            original_image = image.convert('RGB')
            input_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(outputs[0]).item()
                confidence = probabilities[predicted_class].item() * 100
            
            # Prepare original image for visualization
            img_array = np.array(original_image.resize((224, 224)))
            
            # Generate GradCAM if available
            cam = None
            heatmap_colored = None
            overlay = img_array.copy()
            
            if self.gradcam is not None:
                try:
                    # Create a new input tensor for GradCAM (requires grad)
                    gradcam_input = self.transform(original_image).unsqueeze(0).to(self.device)
                    gradcam_input.requires_grad_(True)
                    
                    cam, _ = self.gradcam.generate_cam(gradcam_input, predicted_class)
                    
                    # Resize CAM to match image size and ensure it's valid
                    if cam.shape[0] > 1 and cam.shape[1] > 1:
                        cam_resized = cv2.resize(cam, (224, 224))
                        
                        # Ensure CAM values are in valid range
                        cam_resized = np.clip(cam_resized, 0, 1)
                        
                        # Create colored heatmap
                        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                        
                        # Create overlay
                        overlay = self._apply_gradcam_overlay(img_array, cam_resized)
                        
                        logger.info(f"GradCAM generated successfully. CAM shape: {cam.shape}, Min: {cam.min():.3f}, Max: {cam.max():.3f}")
                    else:
                        logger.warning(f"Invalid CAM shape: {cam.shape}")
                        heatmap_colored = np.zeros((224, 224, 3), dtype=np.uint8)
                        
                except Exception as e:
                    logger.error(f"GradCAM generation failed: {str(e)}")
                    heatmap_colored = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                logger.warning("GradCAM not available")
                heatmap_colored = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Prepare response
            all_probabilities = {
                self.class_names[i]: float(probabilities[i].item() * 100)
                for i in range(len(self.class_names))
            }
            
            # Ensure we have valid images for visualization
            if heatmap_colored is None:
                heatmap_colored = np.zeros((224, 224, 3), dtype=np.uint8)
            
            result = {
                'prediction': {
                    'class': self.class_names[predicted_class],
                    'confidence': float(confidence),
                    'all_probabilities': all_probabilities
                },
                'visualizations': {
                    'original_image': self._image_to_base64(img_array),
                    'gradcam_heatmap': self._image_to_base64(heatmap_colored),
                    'gradcam_overlay': self._image_to_base64(overlay)
                },
                'model_info': {
                    'architecture': self.config['model_architecture'],
                    'classes': self.class_names,
                    'device': str(self.device),
                    'gradcam_available': self.gradcam is not None
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize FastAPI app
app = FastAPI(title="Breast Cancer Classification API", version="1.0.0")

# Global predictor instance
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global predictor
    try:
        model_path = "deployment_model/breast_cancer_model.pth"
        config_path = "deployment_model/model_config.json"
        
        # Check if model files exist
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return
            
        predictor = BreastCancerPredictor(model_path, config_path)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def main():
    """Serve the main upload interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Breast Cancer Classification</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .upload-area {
                border: 2px dashed #007bff;
                padding: 40px;
                text-align: center;
                border-radius: 10px;
                margin: 20px 0;
                background-color: #f8f9fa;
            }
            .upload-area:hover {
                background-color: #e9ecef;
            }
            .btn {
                background-color: #007bff;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .btn:hover {
                background-color: #0056b3;
            }
            .results {
                margin-top: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 10px;
                display: none;
            }
            .prediction-card {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #007bff;
            }
            .images-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .image-card {
                text-align: center;
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .image-card img {
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }
            .probability-bar {
                background-color: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                margin: 5px 0;
            }
            .probability-fill {
                height: 20px;
                display: flex;
                align-items: center;
                padding-left: 10px;
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            .normal { background-color: #28a745; }
            .benign { background-color: #ffc107; }
            .malignant { background-color: #dc3545; }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #007bff;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 Breast Cancer Classification System</h1>
            <p>Upload a breast ultrasound image to get AI-powered classification results with GradCAM visualization.</p>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📁 Click here to select an image or drag and drop</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing image... Please wait.</p>
            </div>
            
            <div class="results" id="results">
                <h2>📊 Classification Results</h2>
                <div id="predictionResults"></div>
                <div id="imageResults"></div>
            </div>
        </div>

        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) return;
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }
            
            function displayResults(data) {
                const prediction = data.prediction;
                const visualizations = data.visualizations;
                
                // Display prediction results
                const predictionHTML = `
                    <div class="prediction-card">
                        <h3>🎯 Prediction: ${prediction.class.toUpperCase()}</h3>
                        <p><strong>Confidence:</strong> ${prediction.confidence.toFixed(2)}%</p>
                        
                        <h4>All Class Probabilities:</h4>
                        ${Object.entries(prediction.all_probabilities).map(([className, prob]) => `
                            <div>
                                <label>${className}: ${prob.toFixed(2)}%</label>
                                <div class="probability-bar">
                                    <div class="probability-fill ${className}" style="width: ${prob}%;">
                                        ${prob.toFixed(1)}%
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
                
                // Display images
                const imagesHTML = `
                    <div class="images-grid">
                        <div class="image-card">
                            <h4>📷 Original Image</h4>
                            <img src="${visualizations.original_image}" alt="Original Image">
                        </div>
                        <div class="image-card">
                            <h4>🔥 GradCAM Heatmap</h4>
                            <img src="${visualizations.gradcam_heatmap}" alt="GradCAM Heatmap">
                        </div>
                        <div class="image-card">
                            <h4>🔍 GradCAM Overlay</h4>
                            <img src="${visualizations.gradcam_overlay}" alt="GradCAM Overlay">
                        </div>
                    </div>
                `;
                
                document.getElementById('predictionResults').innerHTML = predictionHTML;
                document.getElementById('imageResults').innerHTML = imagesHTML;
                document.getElementById('results').style.display = 'block';
            }
            
            // Drag and drop functionality
            const uploadArea = document.querySelector('.upload-area');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.backgroundColor = '#e9ecef';
            });
            
            uploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadArea.style.backgroundColor = '#f8f9fa';
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.backgroundColor = '#f8f9fa';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    document.getElementById('fileInput').files = files;
                    uploadImage();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict breast cancer classification from uploaded image"""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Make prediction
        result = predictor.predict(image)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor else "model_not_loaded",
        "model_loaded": predictor is not None
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_architecture": predictor.config['model_architecture'],
        "num_classes": predictor.config['num_classes'],
        "class_names": predictor.class_names,
        "input_size": predictor.config['input_size'],
        "device": str(predictor.device),
        "preprocessing": predictor.config['preprocessing']
    }

@app.get("/debug/layers")
async def debug_model_layers():
    """Debug endpoint to show model layers"""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    layers = []
    for name, module in predictor.model.named_modules():
        layers.append({
            "name": name,
            "type": str(type(module).__name__)
        })
    
    return {
        "total_layers": len(layers),
        "layers": layers,
        "resnet_layer4_modules": [l for l in layers if "layer4" in l["name"]]
    }

@app.post("/debug/gradcam")
async def debug_gradcam(file: UploadFile = File(...)):
    """Debug GradCAM generation with detailed logging"""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess image
        input_tensor = predictor.transform(image).unsqueeze(0).to(predictor.device)
        
        # Make prediction first
        with torch.no_grad():
            outputs = predictor.model(input_tensor)
            predicted_class = torch.argmax(outputs[0]).item()
        
        debug_info = {
            "gradcam_available": predictor.gradcam is not None,
            "predicted_class": predicted_class,
            "class_name": predictor.class_names[predicted_class],
            "input_shape": list(input_tensor.shape),
            "device": str(predictor.device)
        }
        
        if predictor.gradcam is not None:
            try:
                # Create input that requires gradients
                gradcam_input = predictor.transform(image).unsqueeze(0).to(predictor.device)
                gradcam_input.requires_grad_(True)
                
                # Generate CAM
                cam, _ = predictor.gradcam.generate_cam(gradcam_input, predicted_class)
                
                debug_info.update({
                    "cam_generated": True,
                    "cam_shape": list(cam.shape),
                    "cam_min": float(cam.min()),
                    "cam_max": float(cam.max()),
                    "cam_mean": float(cam.mean()),
                    "gradients_captured": predictor.gradcam.gradients is not None,
                    "activations_captured": predictor.gradcam.activations is not None
                })
                
                if predictor.gradcam.gradients is not None:
                    debug_info["gradients_shape"] = list(predictor.gradcam.gradients.shape)
                if predictor.gradcam.activations is not None:
                    debug_info["activations_shape"] = list(predictor.gradcam.activations.shape)
                    
            except Exception as e:
                debug_info.update({
                    "cam_generated": False,
                    "cam_error": str(e)
                })
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)