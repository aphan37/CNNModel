# gradcam_and_cli.py
"""
Includes:
1. Grad-CAM visualizer for CNN predictions
2. Command-line interface to classify a single image
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os

from cnn_app_model import AlzhiNet  # assume you moved your model class into cnn_app_model.py

# === CONFIG ===
model_path = 'best_model.pth'
image_size = (224, 224)
class_names = ['NoAlzheimers', 'Mild', 'CognitivelyIntact', 'Moderate', 'Severe']

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(img_path):
    image = Image.open(img_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # add batch dimension
    return tensor, image

def get_last_conv_layer(model):
    # Assumes model.features is nn.Sequential and conv3 is the last conv block
    return model.features[-3]

def apply_gradcam(model, input_tensor, class_index=None):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    last_conv = get_last_conv_layer(model)
    forward_handle = last_conv.register_forward_hook(forward_hook)
    backward_handle = last_conv.register_backward_hook(backward_hook)

    output = model(input_tensor)
    if class_index is None:
        class_index = torch.argmax(output, dim=1).item()

    one_hot = torch.zeros_like(output)
    one_hot[0, class_index] = 1
    model.zero_grad()
    output.backward(gradient=one_hot)

    grads = gradients[0][0].detach().cpu().numpy()
    acts = activations[0][0].detach().cpu().numpy()

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, image_size)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

def show_gradcam_on_image(pil_image, cam):
    img = np.array(pil_image.resize(image_size)) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    combined = heatmap + img
    combined = combined / np.max(combined)
    plt.imshow(combined)
    plt.title("Grad-CAM")
    plt.axis('off')
    plt.show()

def predict_image(model, tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        return predicted.item(), conf.item()

# === CLI ===
def cli():
    parser = argparse.ArgumentParser(description="Classify MRI image using AlzhiNet and Grad-CAM")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--gradcam", action="store_true", help="Show Grad-CAM visualization")
    args = parser.parse_args()

    if not os.path.exists(model_path):
        print("❌ Trained model not found. Train and save 'best_model.pth' first.")
        return

    model = AlzhiNet(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    input_tensor, original_image = preprocess_image(args.image)
    class_idx, confidence = predict_image(model, input_tensor)

    print(f"🧠 Prediction: {class_names[class_idx]} ({confidence * 100:.2f}% confidence)")

    if args.gradcam:
        cam = apply_gradcam(model, input_tensor, class_idx)
        show_gradcam_on_image(original_image, cam)

if __name__ == '__main__':
    cli()
