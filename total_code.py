import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
from scipy.fft import fft2, fftshift
import timm


# Preprocessing configuration
PREPROCESS_CONFIG = {
    'face_alignment': False,
    'image_size': (224, 224)
}

# Model configuration
MODEL_CONFIG = {
    'spatial': {
        'backbone': 'xception',  
        'pretrained': True,
        'freeze_backbone': False,
        'output_dim': 1024
    },
    'frequency': {
        'backbone': 'custom_cnn',  # Custom CNN for frequency domain
        'input_channels': 2,  # Magnitude and phase spectrum
        'output_dim': 512
    },
    'ensemble': {
        'input_dim': 1024 + 512,  # Combined dimensions from both streams
        'hidden_dims': [512, 128],
        'dropout': 0.5
    }
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 16, 
    'num_epochs': 5,  
    'learning_rate': 0.00005,
    'weight_decay': 0.0001,
    'lr_scheduler': 'cosine',
    'early_stopping_patience': 3, 
    'save_best_only': True
}

#--------------------------------------------------------------------------------
# Preprocessing Utilities
#--------------------------------------------------------------------------------

class FrequencyTransformer:
    def __init__(self):
        """Initialize frequency domain transformer"""
        pass
        
    def transform(self, image):
        """
        Transform image to frequency domain
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Tuple of (magnitude spectrum, phase spectrum)
        """
        # Convert to grayscale if color
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Apply FFT
        fft = fft2(gray)
        fft_shift = fftshift(fft)
        
        # Compute magnitude spectrum (logarithmic scale)
        magnitude = np.log1p(np.abs(fft_shift))
        
        # Normalize magnitude for better visualization and learning
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # Compute phase spectrum
        phase = np.angle(fft_shift)
        phase = (phase + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        
        return magnitude, phase


#--------------------------------------------------------------------------------
# Model Architecture
#--------------------------------------------------------------------------------

class SpatialStream(nn.Module):
    def __init__(self, config=None):
        """
        Spatial stream for deepfake detection
        
        Args:
            config: Configuration for spatial stream
        """
        super(SpatialStream, self).__init__()
        
        if config is None:
            config = MODEL_CONFIG['spatial']
            
        self.backbone_name = config.get('backbone', 'xception')
        self.pretrained = config.get('pretrained', True)
        self.freeze_backbone = config.get('freeze_backbone', False)
        self.output_dim = config.get('output_dim', 1024)
        
        # Initialize backbone
        if self.backbone_name == 'xception':
            self.backbone = timm.create_model('xception', pretrained=self.pretrained)
            self.backbone.fc = nn.Identity()  # Remove classification head
            self.feature_dim = 2048
        elif self.backbone_name == 'resnet50':
            self.backbone = timm.create_model('resnet50', pretrained=self.pretrained)
            self.backbone.fc = nn.Identity()  # Remove classification head
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
                            
        # Additional layers
        self.spatial_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, self.output_dim)
        )
        
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
            
        Returns:
            Feature tensor (batch_size, output_dim)
        """
        features = self.backbone(x)
        return self.spatial_head(features)
    
class FrequencyStream(nn.Module):
    def __init__(self, config=None):
        """
        Frequency stream for deepfake detection
        
        Args:
            config: Configuration for frequency stream
        """
        super(FrequencyStream, self).__init__()
        
        if config is None:
            config = MODEL_CONFIG['frequency']
            
        self.input_channels = config.get('input_channels', 2)  # Magnitude and phase
        self.output_dim = config.get('output_dim', 512)
        
        # CNN for frequency analysis
        self.freq_cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fifth conv block
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # FC layers
        self.freq_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 2, H, W) - magnitude and phase
            
        Returns:
            Feature tensor (batch_size, output_dim)
        """
        features = self.freq_cnn(x)
        return self.freq_head(features)
    
class EnsembleClassifier(nn.Module):
    def __init__(self, config=None):
        """
        Ensemble classifier combining spatial and frequency features
        
        Args:
            config: Configuration for ensemble classifier
        """
        super(EnsembleClassifier, self).__init__()
        
        if config is None:
            config = MODEL_CONFIG['ensemble']
            
        self.input_dim = config.get('input_dim', 1024 + 512)
        self.hidden_dims = config.get('hidden_dims', [512, 128])
        self.dropout = config.get('dropout', 0.5)
        
        # Build classifier
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(self.dropout))
        
        # Hidden layers
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
            
        # Output layer
        layers.append(nn.Linear(self.hidden_dims[-1], 1))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, spatial_features, freq_features):
        """
        Forward pass
        
        Args:
            spatial_features: Spatial features (batch_size, spatial_dim)
            freq_features: Frequency features (batch_size, freq_dim)
            
        Returns:
            Classification logits (batch_size, 1)
        """
        # Concatenate features
        combined = torch.cat([spatial_features, freq_features], dim=1)
        
        # Classify
        return self.classifier(combined)
    
class DeepfakeDetector(nn.Module):
    def __init__(self, spatial_config=None, freq_config=None, ensemble_config=None):
        """
        Complete deepfake detector model
        
        Args:
            spatial_config: Configuration for spatial stream
            freq_config: Configuration for frequency stream
            ensemble_config: Configuration for ensemble classifier
        """
        super(DeepfakeDetector, self).__init__()
        
        # Spatial stream
        self.spatial_stream = SpatialStream(spatial_config)
        
        # Frequency stream
        self.freq_stream = FrequencyStream(freq_config)
        
        # Ensemble classifier
        self.ensemble = EnsembleClassifier(ensemble_config)
        
    def forward(self, data_dict):
        """
        Forward pass
        
        Args:
            data_dict: Dictionary with 'spatial' and 'frequency' keys
            
        Returns:
            Classification logits (batch_size, 1)
        """
        # Extract inputs
        spatial_input = data_dict['spatial']
        freq_input = data_dict['frequency']
        
        # Process through streams
        spatial_features = self.spatial_stream(spatial_input)
        freq_features = self.freq_stream(freq_input)
        
        # Ensemble classification
        logits = self.ensemble(spatial_features, freq_features)
        
        return logits

