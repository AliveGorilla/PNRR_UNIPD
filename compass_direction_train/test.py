import os
import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from model import ResNet18Keypoints
from dataset import KeypointDataset

# Paths and constants
DATA_DIR = 'data'
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
JSON_FILE = os.path.join(DATA_DIR, 'annotations.json')
CKPT = 'best_model.pth'
IMG_SIZE = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformation should match the training pipeline
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def draw_keypoints_arrow(img, pred, color, width=4):
    """
    Draws keypoints and an arrow from Center to North on an image.

    Args:
        img (PIL.Image): Input image (will be modified in place).
        pred (list or np.ndarray): [center_x, center_y, north_x, north_y], normalized [0,1].
        color (str or tuple): Arrow and point color.
        width (int): Arrow width.
    Returns:
        PIL.Image: Image with annotation overlay.
    """
    W, H = img.size
    cx, cy = int(pred[0]*W), int(pred[1]*H)
    nx, ny = int(pred[2]*W), int(pred[3]*H)
    draw = ImageDraw.Draw(img)

    # Draw a visible white border around the image
    draw.rectangle([(0,0),(W-1,H-1)], outline="white", width=6)

    # Draw center and north points
    draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill=color, outline='black', width=2)
    draw.ellipse([nx-6, ny-6, nx+6, ny+6], fill=color, outline='black', width=2)

    # Draw arrow from center to north
    draw.line([cx, cy, nx, ny], fill=color, width=width)
    # Draw arrowhead
    angle = np.arctan2(ny-cy, nx-cx)
    L = 20
    for a in [-0.4, 0.4]:
        ex = int(nx - L * np.cos(angle+a))
        ey = int(ny - L * np.sin(angle+a))
        draw.line([nx, ny, ex, ey], fill=color, width=width)

    # Draw legend
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    draw.text((10,10), "Center", fill=color, font=font)
    draw.text((10,30), "North", fill=color, font=font)
    return img

def main():
    """
    Loads the best trained model and visualizes predictions
    on 5-10 random images from the test set.
    Saves the images with both ground truth (blue) and prediction (red) arrows.
    """
    # Load dataset (no transform to keep original image for visualization)
    ds = KeypointDataset(JSON_FILE, IMAGE_DIR, transform=None)
    idxs = random.sample(range(len(ds)), min(10, len(ds)))
    model = ResNet18Keypoints().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    for idx in idxs:
        img, keyps, fname = ds[idx]
        # Load and resize the original image
        orig_img = Image.open(os.path.join(IMAGE_DIR, fname)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        # Preprocess for model input
        input_img = transform(orig_img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_img).cpu().numpy()[0]
        # Draw ground truth (blue) and predicted (red) arrows
        vis_img = orig_img.copy()
        vis_img = draw_keypoints_arrow(vis_img, keyps, color="blue")
        vis_img = draw_keypoints_arrow(vis_img, pred, color="red")
        vis_img.save(f"test_vis_{os.path.splitext(fname)[0]}.png")
        print(f"Saved: test_vis_{os.path.splitext(fname)[0]}.png")

if __name__ == '__main__':
    main()