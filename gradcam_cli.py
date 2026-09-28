"""
gradcam_cli.py

Grad-CAM visualizer + single-image CLI classifier for AlzhiNet.

This replaces the original gradcamCLI.py, which was broken:
  1. It imported `from cnn_app_model import AlzhiNet` — no such file
     exists (the actual file is AppModel.py, whose header comment says
     "# cnn_app_model.py" — looks like a rename that never got the
     import statement updated). Fixed: imports from AppModel.py, the
     single shared model definition.
  2. class_names used the old, medically incorrect CDR mapping
     (['NoAlzheimers', 'Mild', 'CognitivelyIntact', 'Moderate', 'Severe']).
     Fixed: uses config.CLASS_ORDER, the same standard CDR order used in
     data_pipeline.py and train.py.
  3. `model.features[-3]` hardcoded the last conv layer's index — broke as
     soon as BatchNorm layers were added, since the indices shifted.
     Fixed: dynamically finds the last nn.Conv2d layer, robust to future
     architecture changes.
  4. Used ImageNet normalization stats, inconsistent with training.
     Fixed: loads the same dataset_stats.json that train.py uses.
  5. `register_backward_hook` is deprecated in current PyTorch.
     Fixed: uses `register_full_backward_hook`.
  6. Only called plt.show() — fails in a headless/SSH environment with
     no display. Fixed: always saves the output image; --show is optional.

Usage:
    python gradcam_cli.py --image path/to/scan.jpg
    python gradcam_cli.py --image path/to/scan.jpg --gradcam
    python gradcam_cli.py --image path/to/scan.jpg --gradcam --show
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from AppModel import AlzhiNet

IMAGE_SIZE = (config.IMAGE_SIZE, config.IMAGE_SIZE)


def load_dataset_stats():
    if os.path.exists(config.DATASET_STATS_PATH):
        with open(config.DATASET_STATS_PATH) as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]
    print(f"  [WARN] {config.DATASET_STATS_PATH} not found — run "
          f"`python preprocessing.py stats` first. Using a neutral "
          f"placeholder, which may hurt prediction accuracy.")
    return [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]


def build_transform():
    mean, std = load_dataset_stats()
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def preprocess_image(img_path: str, transform):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor, image


def get_last_conv_layer(model: nn.Module) -> nn.Conv2d:
    """Finds the last Conv2d layer in model.features dynamically, instead
    of hardcoding an index that breaks whenever the architecture changes
    (e.g. adding BatchNorm shifted every subsequent index by one)."""
    conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No Conv2d layer found in model.features — check the architecture.")
    return conv_layers[-1]


def apply_gradcam(model: nn.Module, input_tensor: torch.Tensor, class_index: int = None):
    model.eval()
    gradients, activations = [], []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    last_conv = get_last_conv_layer(model)
    forward_handle = last_conv.register_forward_hook(forward_hook)
    backward_handle = last_conv.register_full_backward_hook(backward_hook)  # non-deprecated

    try:
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
        cam = cv2.resize(cam, IMAGE_SIZE)
        cam = cam - np.min(cam)
        max_val = np.max(cam)
        cam = cam / max_val if max_val > 0 else cam
        return cam
    finally:
        forward_handle.remove()
        backward_handle.remove()


def render_gradcam(pil_image: Image.Image, cam: np.ndarray, out_path: str, show: bool = False):
    img = np.array(pil_image.resize(IMAGE_SIZE)) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    combined = heatmap + img
    combined = combined / np.max(combined)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(np.uint8(255 * combined), cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM visualization to {out_path}")

    if show:
        import matplotlib.pyplot as plt
        plt.imshow(combined)
        plt.title("Grad-CAM")
        plt.axis("off")
        plt.show()


def predict_image(model: nn.Module, tensor: torch.Tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
    return predicted.item(), conf.item()


def cli():
    parser = argparse.ArgumentParser(description="Classify MRI image using AlzhiNet and Grad-CAM")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualization")
    parser.add_argument("--show", action="store_true", help="Also open a plot window (needs a display)")
    args = parser.parse_args()

    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"Trained model not found at {config.BEST_MODEL_PATH}. Run train.py first.")
        return

    class_names = config.CLASS_ORDER  # standard CDR order — must match train.py's OrderedImageFolder
    model = AlzhiNet(num_classes=len(class_names))
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    transform = build_transform()
    input_tensor, original_image = preprocess_image(args.image, transform)
    class_idx, confidence = predict_image(model, input_tensor)
    print(f"Prediction: {class_names[class_idx]} ({confidence * 100:.2f}% confidence)")

    if args.gradcam:
        cam = apply_gradcam(model, input_tensor, class_idx)
        image_stem = os.path.splitext(os.path.basename(args.image))[0]
        out_path = os.path.join(config.RESULTS_DIR, f"gradcam_{image_stem}.png")
        render_gradcam(original_image, cam, out_path, show=args.show)


if __name__ == "__main__":
    cli()
