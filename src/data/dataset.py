import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ThermalAnomalyDataset(Dataset):
    """Dataset for thermal anomaly detection."""
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (32, 32)
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Load data
        self.images, self.labels = self._load_data()
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load thermal images and labels."""
        images_path = self.data_path / "thermal_images.npy"
        labels_path = self.data_path / "labels.npy"
        
        if images_path.exists() and labels_path.exists():
            images = np.load(images_path)
            labels = np.load(labels_path)
        else:
            # Try alternative file names
            patches_path = self.data_path / "patches.npy"
            labels_path = self.data_path / "labels.npy"
            
            if patches_path.exists() and labels_path.exists():
                images = np.load(patches_path)
                labels = np.load(labels_path)
            else:
                raise FileNotFoundError(f"Data files not found in {self.data_path}")
        
        # Handle different image shapes
        if len(images.shape) == 4:  # (N, C, H, W)
            images = images
        elif len(images.shape) == 3:  # (N, H, W)
            images = images[:, np.newaxis, :, :]  # Add channel dimension
        else:
            raise ValueError(f"Unexpected image shape: {images.shape}")
        
        return images, labels
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get image and label
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to uint8 for albumentations
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            image_norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        else:
            image_norm = image
        
        # Handle single channel vs multi-channel
        if len(image_norm.shape) == 3 and image_norm.shape[0] == 1:
            image_norm = image_norm[0]  # Remove channel dimension for albumentations
        elif len(image_norm.shape) == 3 and image_norm.shape[0] > 1:
            image_norm = np.transpose(image_norm, (1, 2, 0))  # CHW to HWC
        
        # Resize if needed
        if image_norm.shape[:2] != self.target_size:
            image_norm = cv2.resize(image_norm, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply transforms
        if self.transform:
            if len(image_norm.shape) == 2:  # Grayscale
                transformed = self.transform(image=image_norm)
                image_tensor = transformed['image']
            else:  # Multi-channel
                transformed = self.transform(image=image_norm)
                image_tensor = transformed['image']
        else:
            # Convert to tensor manually
            if len(image_norm.shape) == 2:  # Grayscale
                image_tensor = torch.from_numpy(image_norm[np.newaxis, :, :]).float() / 255.0
            else:  # Multi-channel
                image_tensor = torch.from_numpy(np.transpose(image_norm, (2, 0, 1))).float() / 255.0
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor

def get_transforms(split: str = 'train', target_size: Tuple[int, int] = (32, 32)) -> A.Compose:
    """Get data transforms for training/validation."""
    
    if split == 'train':
        transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.Normalize(mean=[0.485], std=[0.229]),  # Adjust for single channel
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ])
    
    return transform

def create_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (32, 32),
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Get transforms
    train_transform = get_transforms('train', target_size)
    val_transform = get_transforms('val', target_size)
    
    if val_path is None:
        # Split training data
        full_dataset = ThermalAnomalyDataset(
            data_path=train_path,
            split='train',
            transform=None,  # Apply transforms later
            target_size=target_size
        )
        
        # Split indices
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=val_split, 
            random_state=42,
            stratify=full_dataset.labels
        )
        
        # Create subset datasets
        train_images = full_dataset.images[train_indices]
        train_labels = full_dataset.labels[train_indices]
        val_images = full_dataset.images[val_indices]
        val_labels = full_dataset.labels[val_indices]
        
        # Create temporary datasets
        class SubsetDataset(Dataset):
            def __init__(self, images, labels, transform, target_size):
                self.images = images
                self.labels = labels
                self.transform = transform
                self.target_size = target_size
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                image = self.images[idx]
                label = self.labels[idx]
                
                # Similar processing as ThermalAnomalyDataset
                if image.dtype != np.uint8:
                    image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
                
                if len(image.shape) == 3 and image.shape[0] == 1:
                    image = image[0]
                elif len(image.shape) == 3 and image.shape[0] > 1:
                    image = np.transpose(image, (1, 2, 0))
                
                if image.shape[:2] != self.target_size:
                    image = cv2.resize(image, self.target_size)
                
                if self.transform:
                    if len(image.shape) == 2:
                        transformed = self.transform(image=image)
                        image_tensor = transformed['image']
                    else:
                        transformed = self.transform(image=image)
                        image_tensor = transformed['image']
                else:
                    if len(image.shape) == 2:
                        image_tensor = torch.from_numpy(image[np.newaxis, :, :]).float() / 255.0
                    else:
                        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
                
                return image_tensor, torch.tensor(label, dtype=torch.long)
        
        train_dataset = SubsetDataset(train_images, train_labels, train_transform, target_size)
        val_dataset = SubsetDataset(val_images, val_labels, val_transform, target_size)
        
    else:
        # Use separate validation path
        train_dataset = ThermalAnomalyDataset(
            data_path=train_path,
            split='train',
            transform=train_transform,
            target_size=target_size
        )
        
        val_dataset = ThermalAnomalyDataset(
            data_path=val_path,
            split='val',
            transform=val_transform,
            target_size=target_size
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

class FireDataModule:
    """Data module for fire detection dataset."""
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        target_size: Tuple[int, int] = (32, 32),
        val_split: float = 0.2
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.val_split = val_split
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders."""
        return create_dataloaders(
            train_path=self.data_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            target_size=self.target_size,
            val_split=self.val_split
        )

if __name__ == "__main__":
    # Test dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    train_loader, val_loader = create_dataloaders(
        train_path=args.data_path,
        batch_size=args.batch_size
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test first batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        break
