import torch
import torch.nn as nn
from torchvision import models
from scipy.linalg import sqrtm
import numpy as np

# Define a function to load the Inception model for feature extraction
class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        inception = models.inception_v3(pretrained=True, transform_input=False)
        self.features = nn.Sequential(*list(inception.children())[:-1])  # Remove the last FC layer
        self.features.eval()

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear')
        x = self.features(x)
        return x.view(x.size(0), -1)

# Load the InceptionV3 model
inception_model = InceptionV3().eval()

def calculate_sifid(real_img, generated_img, model):    
    real_img = nn.functional.interpolate(real_img, size=(299, 299), mode='bilinear', align_corners=False)
    generated_img = nn.functional.interpolate(generated_img, size=(299, 299), mode='bilinear', align_corners=False)
    print('real_img shape:', real_img.shape)
    print('generated_img shape:', generated_img.shape)

    # Extract features for real (style) and generated images
    real_features = model(real_img).detach().numpy()
    generated_features = model(generated_img).detach().numpy()
    
    # Calculate the mean and covariance of features
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_generated, sigma_generated = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    
    # Calculate Frechet Distance
    diff = mu_real - mu_generated
    covmean, _ = sqrtm(sigma_real.dot(sigma_generated), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    sifid = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    
    return sifid

# Function to calculate LPIPS
def calculate_lpips(image1, image2, model):
    """
    Calculates the LPIPS distance between two images.
    
    Args:
        image1: First image tensor of shape [1, 3, H, W] (e.g., style image).
        image2: Second image tensor of shape [1, 3, H, W] (e.g., generated image).
        model: Initialized LPIPS model.
        
    Returns:
        lpips_score: LPIPS score (lower is more perceptually similar).
    """
    with torch.no_grad():
        lpips_score = model(image1, image2)
    return lpips_score.item()