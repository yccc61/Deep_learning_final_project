import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from transformers import ViTModel

# Load and preprocess the images (style and content)
def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Transformations for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT expects fixed-size inputs
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

content_image = load_image("content_temple.jpeg", transform)
style_image = load_image("starryNight.jpg", transform)

# Load the pretrained ViT
vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit.eval()  # Set to evaluation mode

class StyleTransferNetwork(nn.Module):
    def __init__(self, vit):
        super(StyleTransferNetwork, self).__init__()
        self.vit = vit  # Pretrained ViT as the feature extractor
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Output normalized to [-1, 1]
        )

    def forward(self, content_image, style_image):
        # Extract features
        content_features = self.vit(content_image).last_hidden_state
        style_features = self.vit(style_image).last_hidden_state
        
        # Combine features (e.g., weighted sum for simplicity)
        combined_features = 0.7 * content_features + 0.3 * style_features
        
        # Calculate number of patches (assuming a square grid of patches)
        num_patches = combined_features.shape[1]  # This will be 196 for ViT with 16x16 patches
        
        # Calculate patch size (height and width of the grid)
        patch_size = int(num_patches ** 0.5)  # Should be 14 for 196 patches

        # Reshape combined features to (batch_size, embedding_dim, height, width)
        combined_features = combined_features.view(-1, 768, patch_size, patch_size)

        # Decode into image
        output = self.decoder(combined_features)
        return output

    
def compute_content_loss(content_features, target_features):
    return torch.mean((content_features - target_features) ** 2)

def compute_style_loss(style_features, target_features):
    # Gram matrix captures style
    gram_style = torch.matmul(style_features, style_features.transpose(1, 2))
    gram_target = torch.matmul(target_features, target_features.transpose(1, 2))
    return torch.mean((gram_style - gram_target) ** 2)

# Initialize network and optimizer
model = StyleTransferNetwork(vit)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"Epoch [{epoch+1}/{epochs}]")
    optimizer.zero_grad()
    
    # Forward pass
    output = model(content_image, style_image)
    
    # Compute losses
    content_loss = compute_content_loss(model.vit(content_image).last_hidden_state.detach(),
                                        model.vit(output).last_hidden_state)
    style_loss = compute_style_loss(model.vit(style_image).last_hidden_state.detach(),
                                    model.vit(output).last_hidden_state)
    total_loss = content_loss + 1e-3 * style_loss
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()
    
    print(f"Loss: {total_loss.item():.4f}")
