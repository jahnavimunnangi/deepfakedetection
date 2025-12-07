"""
Deepfake Localization - Visualizing manipulated regions in deepfake images

This module extends the DeepfakeDetector model with visualization capabilities
to highlight which regions of an image contribute most to the deepfake classification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
from PIL import Image
import matplotlib.cm as cm
from typing import Dict, List
import ssl

# Import the necessary components from total_code.py
# Assuming total_code.py is in the same directory
from total_code import DeepfakeDetector, FrequencyTransformer, PREPROCESS_CONFIG


ssl._create_default_https_context = ssl._create_unverified_context



class GradCAM:
    """
    Class for visualizing regions of interest using Grad-CAM technique
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM with a model and target layer

        Args:
            model: The DeepfakeDetector model
            target_layer: The layer to extract feature maps from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        
        # Register hooks
        self.gradients = None
        self.activations = None
        
        # Register forward hook
        self.hook_handle_fwd = self.target_layer.register_forward_hook(self._save_activation)
        
        # Register backward hook
        self.hook_handle_bwd = self.target_layer.register_full_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        """Save activations of the target layer"""
        self.activations = output
        
    def _save_gradient(self, module, grad_input, grad_output):
        """Save gradients of the target layer"""
        self.gradients = grad_output[0]
        
    def remove_hooks(self):
        """Remove registered hooks"""
        self.hook_handle_fwd.remove()
        self.hook_handle_bwd.remove()
        
    def generate_cam(self, input_dict: Dict, target_class: int = 1) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input

        Args:
            input_dict: Dictionary containing 'spatial' and 'frequency' inputs
            target_class: Target class index (1 for fake, 0 for real)
            
        Returns:
            Normalized heatmap as a numpy array
        """
        # Forward pass
        outputs = self.model(input_dict)
        outputs = outputs.squeeze(1)
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(outputs)
        one_hot.fill_(target_class)
        
        # Backward pass
        outputs.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU to remove negative values
        cam = nn.functional.relu(cam)
        
        # Convert to numpy and normalize
        cam = cam.detach().cpu().numpy()[0, 0]
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        
        return cam

class DeepfakeLocalizer:
    """
    Class for localizing and visualizing deepfaked regions in images
    """
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the localizer with a pre-trained model

        Args:
            model_path: Path to the model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize model
        self.model = DeepfakeDetector()
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get the target layer for Grad-CAM
        # For this example, we'll use the last convolutional layer in the spatial stream
        # You can change this to another layer if needed
        self.target_layer = self._get_target_layer()
        
        # Initialize preprocessing tools
        self.freq_transformer = FrequencyTransformer()
        
    def _get_target_layer(self) -> nn.Module:
        """Get the target layer for Grad-CAM visualization"""
        # For Xception model
        if self.model.spatial_stream.backbone_name == 'xception':
            return self.model.spatial_stream.backbone.conv4
        # For ResNet model
        elif self.model.spatial_stream.backbone_name == 'resnet50':
            return self.model.spatial_stream.backbone.layer4[-1]
        else:
            # Default to a reasonable layer if the backbone is unknown
            # You may need to update this based on your model architecture
            return list(self.model.spatial_stream.backbone.modules())[-3]
    
    def preprocess_image(self, image_path: str) -> Dict:
        """
        Preprocess an image for the model

        Args:
            image_path: Path to the image

        Returns:
            Dictionary containing processed inputs for the model
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        if image.size != PREPROCESS_CONFIG['image_size']:
            image = image.resize(PREPROCESS_CONFIG['image_size'])
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Transform to frequency domain
        magnitude, phase = self.freq_transformer.transform(image_np)
        
        # Prepare tensors
        spatial_input = torch.tensor((image_np.astype(np.float32) / 255.0 - 0.5) / 0.5, dtype=torch.float)
        spatial_input = spatial_input.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        freq_input = torch.cat([
            torch.tensor(magnitude, dtype=torch.float).unsqueeze(0),
            torch.tensor(phase, dtype=torch.float).unsqueeze(0)
        ], dim=0).unsqueeze(0).to(self.device)
        
        return {
            'original_image': image,
            'spatial': spatial_input,
            'frequency': freq_input
        }
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze an image and generate heatmap visualization

        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary containing analysis results and visualizations
        """
        # Preprocess image
        inputs = self.preprocess_image(image_path)
        
        # Initialize Grad-CAM
        grad_cam = GradCAM(self.model, self.target_layer)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model({'spatial': inputs['spatial'], 'frequency': inputs['frequency']})
            prob = torch.sigmoid(outputs).item()
            pred = 0 if prob >= 0.5 else 1
            
        # Generate heatmap if predicted as fake
        heatmap = None
        if pred == 1:  # If predicted as fake
            heatmap = grad_cam.generate_cam({
                'spatial': inputs['spatial'], 
                'frequency': inputs['frequency']
            })
            
        # Clean up
        grad_cam.remove_hooks()
        
        return {
            'path': image_path,
            'original_image': inputs['original_image'],
            'prediction': pred,
            'probability': prob,
            'label': 'Fake' if pred == 1 else 'Real',
            'heatmap': heatmap
        }
    
    def visualize_results(self, results: Dict, output_path: str = None, alpha: float = 0.5) -> plt.Figure:
        """
        Visualize analysis results

        Args:
            results: Analysis results from analyze_image
            output_path: Path to save the visualization
            alpha: Transparency of the heatmap overlay
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, axes = plt.subplots(1, 2 if results['heatmap'] is not None else 1, figsize=(15, 7))
        
        # If not a list (only one axis when heatmap is None), convert to list for consistent indexing
        if results['heatmap'] is None:
            axes = [axes]
        
        # Plot original image
        axes[0].imshow(np.array(results['original_image']))
        axes[0].set_title(f"Original Image\nPrediction: {results['label']}")
        axes[0].axis('off')
        
        # Plot heatmap overlay if available
        if results['heatmap'] is not None:
            # Original image
            img = np.array(results['original_image'])
            
            # Convert heatmap to RGB
            heatmap = results['heatmap']
            heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
            
            # Overlay heatmap on original image
            overlay = img * (1 - alpha) + heatmap_colored * 255 * alpha
            overlay = overlay.astype(np.uint8)
            
            # Display overlay
            axes[1].imshow(overlay)
            axes[1].set_title("Manipulated Region Heatmap")
            axes[1].axis('off')
        
        # Add global title
        plt.suptitle(f"Deepfake Analysis: {'Fake detected' if results['prediction'] == 1 else 'No fake detected'}", fontsize=16)
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        return fig

    def process_batch(self, image_paths: List[str], output_dir: str = './outputs/localization'):
        """
        Process a batch of images and save visualizations

        Args:
            image_paths: List of image paths
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for image_path in image_paths:
            try:
                # Get filename for output
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_localized.png")
                
                # Analyze image
                results = self.analyze_image(image_path)
                
                # Visualize and save
                self.visualize_results(results, output_path)
                
                print(f"Processed {filename}: {'Fake' if results['prediction'] == 0 else 'Real'}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Path to your trained model
    model_path = "./best_model.pth"
    
    # Initialize localizer
    localizer = DeepfakeLocalizer(model_path=model_path)
    
    # Example image paths (replace with actual paths)
    image_paths = [
        './fake_1.jpg',
        './fake_2.jpg',
        './fake_3.jpg',
        './yatish.png',
        './vishnu.jpeg'
    ]
    
    # Process a single image and display results
    if len(image_paths) > 0:
        results = localizer.analyze_image(image_paths[0])
        localizer.visualize_results(results)
        plt.show()
        
    # Process a batch of images and save results
    localizer.process_batch(image_paths, output_dir='./outputs/localization') 