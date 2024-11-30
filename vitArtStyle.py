import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


# Load pretrained VGG19 model for feature extraction
vgg = models.vgg19(pretrained=True).features.eval()

# Define the Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Apply convolution to get queries, keys, and values
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        # Compute attention scores
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))

        # Reshape back to the original feature map
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x  # Skip connection
        return out

# Function to load images and apply transforms
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Load content and style images
content_img = load_image("content_temple.jpeg")
style_img = load_image("starryNight.jpg")

# Function to extract features from the VGG19 model
def get_features_with_attention(image, model, attention_layer):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            if name == '21':  # Apply attention after 'conv4_2'
                x = attention_layer(x)
            features[layers[name]] = x
    return features

# Define Gram matrix for style loss
def gram_matrix(imgfeature):
    _, d, h, w = imgfeature.size()
    imgfeature = imgfeature.view(d, h * w)
    gram_mat = torch.mm(imgfeature, imgfeature.t())
    return gram_mat

# Compute Style Loss
def compute_style_loss(styleLayer, targetLayer):
    _, d, h, w = targetLayer.size()
    style_gram = gram_matrix(styleLayer)
    target_gram = gram_matrix(targetLayer)
    return torch.mean((target_gram - style_gram) ** 2) / (d * w * h)

# Compute Content Loss
def content_loss(content, target):
    return torch.mean((content - target) ** 2)

# Initialize the target image (same as content image)
target_img = content_img.clone().requires_grad_(True)  # Allow gradients for optimization
optimizer = torch.optim.Adam([target_img], lr=0.003)

# Set style and content loss weights
content_weight = 100
style_weight = 1e8



# Initialize the self-attention layer
attention_layer = SelfAttention(in_channels=512)  # Apply attention on 'conv4_2'

# Get the features from the content and style images
content_features = get_features_with_attention(content_img, vgg, attention_layer)
style_features = get_features_with_attention(style_img, vgg, attention_layer)

# Lists to store SIFID and LPIPS scores
sifid_scores = []
lpips_scores = []

# Training loop for style transfer
for epoch in range(200):
    print(f"Epoch {epoch}")
    optimizer.zero_grad()
    target_features = get_features_with_attention(target_img, vgg, attention_layer)
    
    # Compute content loss
    c_loss = content_loss(content_features['conv4_2'], target_features["conv4_2"])
    
    # Compute style loss
    s_loss = 0
    for layer in style_features:
        s_loss += compute_style_loss(style_features[layer], target_features[layer])
    
    # Total loss
    total_loss = content_weight * c_loss + style_weight * s_loss
    total_loss.backward(retain_graph=True)
    optimizer.step()


    print(f"Epoch {epoch}, Total Loss: {total_loss.item()}")

# Convert tensor to image and display
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.squeeze(0)  # Remove the batch dimension
    image = image.transpose(1, 2, 0)  # Rearrange to HWC
    image = image * 0.225 + 0.450  # Un-normalize
    image = np.clip(image, 0, 1)
    return image

# Display the output image
output_image = im_convert(target_img)
plt.imshow(output_image)
plt.axis('off')
plt.show()
