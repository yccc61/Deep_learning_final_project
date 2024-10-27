import torch
from torchvision import models
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
#We use vgg19 for art style transfer
vgg = models.vgg19(pretrained=True).features.eval()


def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

content_img = load_image("content_temple.jpeg")
style_img = load_image("starryNight.jpg")




def get_features(image, model):
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
            features[layers[name]] = x
    return features

content_features = get_features(content_img, vgg)
style_features = get_features(style_img, vgg)

def gram_matrix(imgfeature):
    _,d,h,w = imgfeature.size()
    imgfeature = imgfeature.view(d,h*w)
    gram_mat = torch.mm(imgfeature,imgfeature.t())
    
    return gram_mat

def computerStyleLoss(styleLayer, targetLayer):
    _,d,h,w = targetLayer.size()
    style_gram=gram_matrix(styleLayer)
    target_gram=gram_matrix(targetLayer)
    return torch.mean((target_gram-style_gram)**2)/d*w*h

def content_loss(content, target):
    return torch.mean((content - target) ** 2)

target_img = content_img.clone().requires_grad_(True)  # Allow gradients for optimization
optimizer = torch.optim.Adam([target_img], lr=0.003)
# style_weight = 1000000
# content_weight = 1
content_weight = 100
style_weight = 1e8

for i in range(100):
    optimizer.zero_grad()
    target_features=get_features(target_img, vgg)
    c_loss=content_loss(content_features['conv4_2'], target_features["conv4_2"])
    s_loss=0
    for layer in style_features:
        s_loss+=computerStyleLoss(target_features[layer], style_features[layer])

    total_loss = content_weight * c_loss + style_weight * s_loss
    total_loss.backward(retain_graph=True)
    optimizer.step()

    print(i)

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.squeeze(0)  # Remove the batch dimension
    image = image.transpose(1, 2, 0)  # Rearrange to HWC
    image = image * 0.225 + 0.450  # Un-normalize
    image = np.clip(image, 0, 1)
    return image

print("Here")
output_image = im_convert(target_img)
plt.imshow(output_image)
plt.axis('off')
plt.show()
