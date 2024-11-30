import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Load and preprocess the images (style and content)
def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Transformations for ViT or a CNN-based model
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing for standard models
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load content and style images
content_image = load_image("content_temple.jpeg", transform)
style_image = load_image("starryNight.jpg", transform)

# Define Adaptive Attention Module (simplified version)
class AdaptiveAttention(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveAttention, self).__init__()
        # Now this module can handle the 512 channels (or any other expected number of channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)  # Output attention map

    def forward(self, x):
        # Compute attention map
        attention_map = self.conv1(x)
        attention_map = self.relu(attention_map)
        attention_map = self.conv2(attention_map)
        attention_map = torch.sigmoid(attention_map)  # Normalize to [0, 1]
        return attention_map * x  # Apply attention to the input feature


# Define the Style Transfer Network using AdaAttN (adaptive attention)
class StyleTransferNetwork(nn.Module):
    def __init__(self, pretrained_model):
        super(StyleTransferNetwork, self).__init__()
        self.model = pretrained_model  # Use a pretrained CNN or Vision Transformer
        
        # Add adaptive attention layers after the model
        self.attn1 = AdaptiveAttention(in_channels=1024)  # Example: after a middle layer
        self.attn2 = AdaptiveAttention(in_channels=256)  # Another attention layer

    def forward(self, content_image, style_image):
        # Extract features from content and style using the pretrained model
        content_features = self.model(content_image)
        style_features = self.model(style_image)

        # Apply adaptive attention to the features
        content_features = self.attn1(content_features)
        style_features = self.attn2(style_features)

        # Combine content and style features (simple averaging here, can be more sophisticated)
        combined_features = 0.7 * content_features + 0.3 * style_features

        return combined_features

# Load pre-trained model (e.g., VGG19 or ResNet) for feature extraction
pretrained_model = models.vgg19(pretrained=True).features.eval()  # For simplicity, using VGG19
model = StyleTransferNetwork(pretrained_model)

# Loss Functions for Style and Content
def compute_content_loss(content_features, target_features):
    return torch.mean((content_features - target_features) ** 2)

def compute_style_loss(style_features, target_features):
    # Gram matrix captures style
    gram_style = gram_matrix(style_features)
    gram_target = gram_matrix(target_features)
    return torch.mean((gram_style - gram_target) ** 2)

def gram_matrix(features):
    """Calculate the Gram matrix of given features"""
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))  # B x C x (H*W)
    return gram / (c * h * w)

# Training loop
epochs = 200
optimizer = optim.Adam([content_image.requires_grad_()], lr=0.01)

for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass
    output = model(content_image, style_image)

    # Compute losses
    content_loss = compute_content_loss(output, content_image)
    style_loss = compute_style_loss(output, style_image)

    # Total loss (weighted content and style losses)
    total_loss = content_loss + 1e6 * style_loss
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}")

    # Save output image at the last epoch
    if epoch == epochs - 1:
        output_image = output.squeeze(0).cpu().detach().numpy()
        output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
        output_image = (output_image + 1) / 2.0  # Denormalize to [0, 1]
        output_image = Image.fromarray((output_image * 255).astype(np.uint8))
        output_image.save(f'output_epoch_{epoch+1}.png')
