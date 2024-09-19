import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import random
import cv2
import numpy as np
from loadParam import IMG_DIR, MASK_DIR, BATCH_SIZE, NUM_WORKERS, IMG_SIZE
import re

class WindowDataset(Dataset):
    def __init__(self, img_dir=IMG_DIR, mask_dir=MASK_DIR, img_size=IMG_SIZE, num_augments=5):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.num_augments = num_augments  # Number of augmentations to perform per image
        
        # Get list of all mask filenames
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.img_filenames = sorted(os.listdir(img_dir))

        # Extract numbers from filenames and create the mapping
        def extract_number(filename):
            return re.findall(r'\d+', filename)[0]  # Extract the number part from the filename

        # Map images to masks by comparing the numeric part of filenames
        self.img_to_mask_mapping = {
            img_file: mask_file
            for img_file in self.img_filenames
            for mask_file in self.mask_filenames
            if extract_number(img_file) == extract_number(mask_file)
        }

        if len(self.img_to_mask_mapping) == 0:
            raise ValueError("No matching image-mask pairs found in the dataset.")

        # Transformations: Resizing images and masks to a common size
        self.resize = transforms.Resize(self.img_size)

        # Define the pool of augmentations
        self.augmentations_pool = [
            self.color_jitter,
            self.random_gamma,
            self.random_occlusion,
            self.random_homography,
            self.motion_blur,
            self.defocus_blur
        ]

    def __len__(self):
        # The dataset length is the number of original pairs * number of augmentations
        return len(self.img_to_mask_mapping) * (self.num_augments + 1)  # +1 to include the original images

    def color_jitter(self, img):
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        return color_jitter(img)

    def random_gamma(self, img):
        return transforms.functional.adjust_gamma(img, gamma=random.uniform(0.8, 1.2))

    def random_occlusion(self, img):
        img_np = np.array(img)
        h, w, _ = img_np.shape
        num_occlusions = random.randint(1, 5)  # Random number of occlusions

        for _ in range(num_occlusions):
            occlusion_type = random.choice(['rectangle', 'square', 'circle'])
            if occlusion_type in ['rectangle', 'square']:
                # Random size and position for rectangles/squares
                occl_width = random.randint(10, 50)
                occl_height = occl_width if occlusion_type == 'square' else random.randint(10, 50)
                x1 = random.randint(0, w - occl_width)
                y1 = random.randint(0, h - occl_height)
                cv2.rectangle(img_np, (x1, y1), (x1 + occl_width, y1 + occl_height), (0, 0, 0), -1)
            elif occlusion_type == 'circle':
                # Random position and radius for circles
                radius = random.randint(5, 10)  # Radius not more than 10 pixels
                center_x = random.randint(radius, w - radius)
                center_y = random.randint(radius, h - radius)
                cv2.circle(img_np, (center_x, center_y), radius, (0, 0, 0), -1)  # Black circle
        return Image.fromarray(img_np)

    def random_homography(self, img):
        img_np = np.array(img)
        h, w, _ = img_np.shape
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[random.randint(0, w//4), random.randint(0, h//4)],
                           [w - random.randint(0, w//4), random.randint(0, h//4)],
                           [random.randint(0, w//4), h - random.randint(0, h//4)],
                           [w - random.randint(0, w//4), h - random.randint(0, h//4)]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_np = cv2.warpPerspective(img_np, matrix, (w, h))
        return Image.fromarray(img_np)

    def motion_blur(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 5)))

    def defocus_blur(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(5, 10)))

    def apply_random_augmentations(self, img, mask):
        # Randomly select 5 augmentations from the pool and apply them
        selected_augmentations = random.sample(self.augmentations_pool, k=5)
        for aug in selected_augmentations:
            img = aug(img)  # Apply the augmentation to the image
        return img, mask  # Mask remains the same (no need for augmentation)


    
    def __getitem__(self, idx):
        # Get the original image and mask based on the index
        img_idx = idx // (self.num_augments + 1)  # Each image has num_augments + 1 entries
        img_filename = list(self.img_to_mask_mapping.keys())[img_idx]
        mask_filename = self.img_to_mask_mapping[img_filename]

        img_path = os.path.join(self.img_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale for thresholding


        # Resize the image and mask
        img = self.resize(img)
        mask = self.resize(mask)

        if idx % (self.num_augments + 1) != 0:
            # Apply augmentations to this sample (skip augmentation for original)
            img, mask = self.apply_random_augmentations(img, mask)

        # Convert to tensor
        img = transforms.ToTensor()(img)
        # mask = torch.tensor(np.array(mask) // 255, dtype=torch.float32)  # Convert binary mask to tensor (0 or 1)
        mask = transforms.ToTensor()(mask)

        return img, mask


# Test the dataloader
if __name__ == "__main__":
    dataset = WindowDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    for imgs, masks in dataloader:
        print(f'Image batch shape: {imgs.size()}')
        print(f'Mask batch shape: {masks.size()}')
        break
