import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class KeypointDataset(Dataset):
    """
    Custom Dataset for loading images and keypoint annotations from a JSON file.

    Each sample consists of:
        - An image (loaded and transformed)
        - A tensor of 4 normalized keypoint coordinates:
            [center_x, center_y, north_x, north_y]
        - The original image file name
    """
    def __init__(self, json_file, image_dir, transform=None):
        """
        Args:
            json_file (str): Path to the JSON annotation file.
            image_dir (str): Directory with input images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        for entry in self.data:
            file_name = entry['file_upload']
            img_path = os.path.join(self.image_dir, file_name)
            annots = entry['annotations']
            for annot in annots:
                kp_dict = {}
                for res in annot['result']:
                    # Extract label and normalized (x, y) coordinates (range [0, 1])
                    label = res['value']['keypointlabels'][0]
                    x = res['value']['x'] / 100.0
                    y = res['value']['y'] / 100.0
                    kp_dict[label] = (x, y)
                # Only use samples with both "Center" and "North" keypoints
                if 'Center' in kp_dict and 'North' in kp_dict:
                    self.samples.append({
                        'img_path': img_path,
                        'center': kp_dict['Center'],
                        'north': kp_dict['North'],
                        'file_name': file_name
                    })

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns one sample (image, keypoint coordinates, file name).

        Args:
            idx (int): Index of the item.

        Returns:
            tuple:
                img (Tensor): Transformed image tensor.
                keypoints (Tensor): 1D tensor of 4 normalized coordinates.
                file_name (str): Image file name.
        """
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # Construct keypoints tensor: [center_x, center_y, north_x, north_y]
        keypoints = torch.tensor([
            sample['center'][0], sample['center'][1],
            sample['north'][0], sample['north'][1]
        ], dtype=torch.float32)
        return img, keypoints, sample['file_name']