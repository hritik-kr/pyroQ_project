import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
import geopandas as gpd
from typing import Tuple, List, Optional, Dict, Any
import cv2
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
import argparse
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

logger = logging.getLogger(__name__)

class EnhancedThermalPreprocessor:
    """Enhanced preprocessor with cloud masking and advanced features."""
    
    # Constants for different satellite sensors
    CLOUD_THRESHOLDS = {
        'MODIS': {'reflectance': 0.3, 'temperature': 273.15},
        'LANDSAT': {'reflectance': 0.25, 'temperature': 280.0},
        'SENTINEL': {'reflectance': 0.2, 'temperature': 285.0}
    }
    
    def __init__(
        self, 
        patch_size: int = 32, 
        overlap: float = 0.1,
        sensor_type: str = 'MODIS',
        enable_cloud_masking: bool = True,
        scaler_type: str = 'robust'
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.sensor_type = sensor_type
        self.enable_cloud_masking = enable_cloud_masking
        
        # Initialize scaler
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        # Cloud detection parameters
        self.cloud_params = self.CLOUD_THRESHOLDS.get(sensor_type, self.CLOUD_THRESHOLDS['MODIS'])
        
        # Fire detection parameters
        self.fire_threshold_temp = 320.0  # Kelvin
        self.fire_threshold_std = 3.0
        
    def load_modis_tile(self, filepath: str, apply_quality_mask: bool = True) -> Tuple[np.ndarray, dict]:
        """Enhanced MODIS tile loading with quality filtering."""
        with rasterio.open(filepath) as src:
            # Read all bands
            data = src.read()
            
            # Quality mask (if available)
            if apply_quality_mask and data.shape[0] > 4:
                quality_band = data[-1]  # Assuming last band is quality
                quality_mask = self._create_quality_mask(quality_band)
                
                # Apply quality mask to all bands
                for i in range(data.shape[0] - 1):
                    data[i] = np.where(quality_mask, data[i], np.nan)
            
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'shape': data.shape,
                'nodata': src.nodata,
                'sensor': self.sensor_type,
                'quality_filtered': apply_quality_mask
            }
            
        return data, metadata
    
    def _create_quality_mask(self, quality_band: np.ndarray) -> np.ndarray:
        """Create quality mask from quality assurance band."""
        # MODIS quality bit flags (simplified)
        # Bit 0-1: cloud state (00 = clear, 01 = cloudy, 10 = mixed, 11 = not set)
        cloud_mask = (quality_band & 0b11) == 0  # Clear pixels only
        
        # Additional quality checks can be added here
        
        return cloud_mask
    
    def apply_cloud_mask(
        self, 
        thermal_data: np.ndarray, 
        visible_bands: Optional[np.ndarray] = None,
        temperature_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced cloud masking using multiple criteria."""
        if not self.enable_cloud_masking:
            return thermal_data, np.ones_like(thermal_data, dtype=bool)
        
        cloud_mask = np.ones_like(thermal_data, dtype=bool)
        
        # Temperature-based cloud detection
        if temperature_data is not None:
            temp_cloud_mask = temperature_data > self.cloud_params['temperature']
            cloud_mask &= ~temp_cloud_mask
        
        # Reflectance-based cloud detection
        if visible_bands is not None:
            # Use visible bands to detect bright clouds
            reflectance = np.mean(visible_bands, axis=0)
            refl_cloud_mask = reflectance > self.cloud_params['reflectance']
            cloud_mask &= ~refl_cloud_mask
        
        # Statistical outlier detection for clouds
        mean_temp = np.nanmean(thermal_data)
        std_temp = np.nanstd(thermal_data)
        outlier_mask = np.abs(thermal_data - mean_temp) > (4 * std_temp)
        cloud_mask &= ~outlier_mask
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cloud_mask = cv2.morphologyEx(cloud_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply cloud mask
        masked_thermal = np.where(cloud_mask, thermal_data, np.nan)
        
        return masked_thermal, cloud_mask.astype(bool)
    
    def extract_enhanced_thermal_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive thermal and spectral features."""
        features = {}
        
        if data.shape[0] >= 4:  # Multi-spectral data
            red = data[0].astype(np.float32)
            nir = data[1].astype(np.float32)
            swir1 = data[2].astype(np.float32) if data.shape[0] > 2 else nir
            thermal = data[3].astype(np.float32) if data.shape[0] > 3 else swir1
            
            # Basic thermal features
            features['thermal'] = thermal
            features['thermal_celsius'] = thermal - 273.15  # Convert to Celsius
            
            # Vegetation indices
            features['ndvi'] = self._safe_divide(nir - red, nir + red)
            features['evi'] = 2.5 * self._safe_divide(nir - red, nir + 6*red - 7.5*swir1 + 1)
            
            # Fire indices
            features['nbr'] = self._safe_divide(nir - swir1, nir + swir1)  # Normalized Burn Ratio
            features['bai'] = self._safe_divide(1, (0.1 - red)**2 + (0.06 - nir)**2)  # Burn Area Index
            features['fire_index'] = thermal / (swir1 + 1e-8)
            
            # Advanced thermal features
            features['thermal_anomaly'] = self._compute_thermal_anomaly(thermal)
            features['temporal_change'] = self._compute_temporal_change(thermal)
            
            # Texture features
            features['thermal_texture'] = self._compute_texture_features(thermal)
            
        else:
            # Single band thermal data
            thermal = data[0] if len(data.shape) == 3 else data
            features['thermal'] = thermal
            features['thermal_celsius'] = thermal - 273.15
            features['thermal_anomaly'] = self._compute_thermal_anomaly(thermal)
            features['thermal_texture'] = self._compute_texture_features(thermal)
        
        return features
    
    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Safe division avoiding division by zero."""
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    def _compute_thermal_anomaly(self, thermal_data: np.ndarray) -> np.ndarray:
        """Compute thermal anomaly using local adaptive thresholding."""
        # Local statistics using sliding window
        from scipy import ndimage
        
        # Compute local mean and std
        local_mean = ndimage.uniform_filter(thermal_data, size=5)
        local_var = ndimage.uniform_filter(thermal_data**2, size=5) - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Adaptive threshold
        threshold = local_mean + self.fire_threshold_std * local_std
        anomaly_score = (thermal_data - threshold) / (local_std + 1e-8)
        
        return np.maximum(anomaly_score, 0)  # Only positive anomalies
    
    def _compute_temporal_change(self, thermal_data: np.ndarray) -> np.ndarray:
        """Compute temporal change indicators (placeholder for time-series data)."""
        # This would be implemented for time-series analysis
        # For now, return gradient magnitude as proxy for change
        grad_y, grad_x = np.gradient(thermal_data)
        return np.sqrt(grad_x**2 + grad_y**2)
    
    def _compute_texture_features(self, thermal_data: np.ndarray) -> np.ndarray:
        """Compute texture features using GLCM-inspired approach."""
        from scipy import ndimage
        
        # Simple texture measures
        # Variance (local)
        variance = ndimage.generic_filter(thermal_data, np.var, size=3)
        
        # Entropy (approximation)
        # Quantize data for entropy calculation
        quantized = ((thermal_data - np.nanmin(thermal_data)) / 
                    (np.nanmax(thermal_data) - np.nanmin(thermal_data)) * 255).astype(int)
        
        def local_entropy(values):
            """Compute local entropy."""
            values = values[~np.isnan(values)]
            if len(values) == 0:
                return 0
            _, counts = np.unique(values, return_counts=True)
            probs = counts / len(values)
            return -np.sum(probs * np.log2(probs + 1e-8))
        
        entropy = ndimage.generic_filter(quantized.astype(float), local_entropy, size=3)
        
        # Combine texture features
        texture = variance + entropy
        
        return texture
    
    def detect_advanced_thermal_anomalies(
        self, 
        features: Dict[str, np.ndarray], 
        use_clustering: bool = True
    ) -> np.ndarray:
        """Advanced thermal anomaly detection using multiple criteria."""
        thermal = features['thermal']
        
        # Multi-criteria approach
        anomaly_scores = []
        
        # 1. Statistical threshold
        if 'thermal_anomaly' in features:
            anomaly_scores.append(features['thermal_anomaly'])
        
        # 2. Fire index threshold
        if 'fire_index' in features:
            fire_threshold = np.nanpercentile(features['fire_index'], 95)
            fire_anomaly = (features['fire_index'] > fire_threshold).astype(float)
            anomaly_scores.append(fire_anomaly)
        
        # 3. Temperature absolute threshold
        temp_anomaly = (thermal > self.fire_threshold_temp).astype(float)
        anomaly_scores.append(temp_anomaly)
        
        # 4. Texture-based detection
        if 'thermal_texture' in features:
            texture_threshold = np.nanpercentile(features['thermal_texture'], 90)
            texture_anomaly = (features['thermal_texture'] > texture_threshold).astype(float)
            anomaly_scores.append(texture_anomaly)
        
        # Combine scores
        combined_score = np.mean(anomaly_scores, axis=0)
        
        # Clustering-based refinement
        if use_clustering:
            combined_score = self._refine_with_clustering(thermal, combined_score)
        
        # Final threshold ta-da
        final_anomaly = combined_score > 0.5
        
        # Post-processing
        kernel = np.ones((3, 3), np.uint8)
        final_anomaly = cv2.morphologyEx(final_anomaly.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        final_anomaly = cv2.morphologyEx(final_anomaly, cv2.MORPH_CLOSE, kernel)
        
        return final_anomaly.astype(bool)
    
    def _refine_with_clustering(self, thermal_data: np.ndarray, anomaly_scores: np.ndarray) -> np.ndarray:
        """Refine anomaly detection using clustering."""
        # Prepare features for clustering
        valid_mask = ~np.isnan(thermal_data) & ~np.isnan(anomaly_scores)
        
        if np.sum(valid_mask) < 100:  # Not enough valid pixels
            return anomaly_scores
        
        features = np.column_stack([
            thermal_data[valid_mask],
            anomaly_scores[valid_mask]
        ])
        
        # K-means clustering
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            
            # Identify fire cluster (highest temperature and anomaly score)
            cluster_means = []
            for i in range(3):
                mask = clusters == i
                if np.sum(mask) > 0:
                    mean_temp = np.mean(features[mask, 0])
                    mean_anomaly = np.mean(features[mask, 1])
                    cluster_means.append((i, mean_temp, mean_anomaly))
            
            # Fire cluster is the one with highest combined score
            cluster_means.sort(key=lambda x: x[1] + x[2], reverse=True)
            fire_cluster_id = cluster_means[0][0]
            
            # Update anomaly scores
            refined_scores = anomaly_scores.copy()
            fire_pixels = valid_mask.copy()
            fire_pixels[valid_mask] = clusters == fire_cluster_id
            
            refined_scores[fire_pixels] = np.maximum(refined_scores[fire_pixels], 0.8)
            
            return refined_scores
            
        except Exception as e:
            logger.warning(f"Clustering refinement failed: {e}")
            return anomaly_scores
    
    def create_patches_enhanced(
        self, 
        features: Dict[str, np.ndarray], 
        labels: Optional[np.ndarray] = None,
        stratify: bool = True
    ) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Enhanced patch creation with stratification and metadata."""
        
        # Get primary feature for patch extraction
        primary_feature = features.get('thermal', list(features.values())[0])
        
        if len(primary_feature.shape) == 2:
            height, width = primary_feature.shape
        else:
            _, height, width = primary_feature.shape
        
        patches = []
        patch_labels = []
        patch_metadata = []
        
        step_size = int(self.patch_size * (1 - self.overlap))
        
        for y in range(0, height - self.patch_size + 1, step_size):
            for x in range(0, width - self.patch_size + 1, step_size):
                
                # Extract patches from all features
                patch_dict = {}
                for feature_name, feature_data in features.items():
                    if len(feature_data.shape) == 2:
                        patch_dict[feature_name] = feature_data[y:y+self.patch_size, x:x+self.patch_size]
                    else:
                        patch_dict[feature_name] = feature_data[:, y:y+self.patch_size, x:x+self.patch_size]
                
                # Stack features into multi-channel patch
                feature_stack = []
                for feature_name in ['thermal', 'ndvi', 'nbr', 'fire_index', 'thermal_anomaly']:
                    if feature_name in patch_dict:
                        feature_stack.append(patch_dict[feature_name])
                
                if not feature_stack:
                    feature_stack = [primary_feature[y:y+self.patch_size, x:x+self.patch_size]]
                
                # Create multi-channel patch
                if len(feature_stack[0].shape) == 2:
                    multi_channel_patch = np.stack(feature_stack, axis=0)
                else:
                    multi_channel_patch = np.concatenate(feature_stack, axis=0)
                
                patches.append(multi_channel_patch)
                
                # Extract label
                if labels is not None:
                    label_patch = labels[y:y+self.patch_size, x:x+self.patch_size]
                    # Use fire pixel percentage for labeling
                    fire_percentage = np.sum(label_patch) / (self.patch_size * self.patch_size)
                    patch_label = 1 if fire_percentage > 0.1 else 0  # 10% threshold
                    patch_labels.append(patch_label)
                else:
                    patch_labels.append(0)
                
                # Store metadata
                metadata = {
                    'position': (y, x),
                    'fire_percentage': np.sum(label_patch) / (self.patch_size * self.patch_size) if labels is not None else 0,
                    'mean_temperature': np.mean(patch_dict.get('thermal', primary_feature[y:y+self.patch_size, x:x+self.patch_size])),
                    'valid_pixels': np.sum(~np.isnan(multi_channel_patch[0]))
                }
                patch_metadata.append(metadata)
        
        return patches, patch_labels, patch_metadata
    
    def balance_dataset(
        self, 
        patches: List[np.ndarray], 
        labels: List[int], 
        metadata: List[Dict],
        target_ratio: float = 0.3
    ) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Balance dataset to achieve target fire/no-fire ratio."""
        
        fire_indices = [i for i, label in enumerate(labels) if label == 1]
        no_fire_indices = [i for i, label in enumerate(labels) if label == 0]
        
        n_fire = len(fire_indices)
        n_no_fire = len(no_fire_indices)
        
        logger.info(f"Original distribution: Fire={n_fire}, No-fire={n_no_fire}")
        
        if n_fire == 0:
            logger.warning("No fire samples found!")
            return patches, labels, metadata
        
        # Calculate target counts
        if n_fire / (n_fire + n_no_fire) < target_ratio:
            # Need more fire samples - augment
            target_fire = int(n_no_fire * target_ratio / (1 - target_ratio))
            fire_patches_aug, fire_labels_aug, fire_metadata_aug = self._augment_fire_samples(
                [patches[i] for i in fire_indices],
                [labels[i] for i in fire_indices],
                [metadata[i] for i in fire_indices],
                target_fire - n_fire
            )
            
            # Combine with original
            balanced_patches = patches + fire_patches_aug
            balanced_labels = labels + fire_labels_aug
            balanced_metadata = metadata + fire_metadata_aug
            
        else:
            # Need fewer no-fire samples - subsample
            target_no_fire = int(n_fire * (1 - target_ratio) / target_ratio)
            selected_no_fire = np.random.choice(no_fire_indices, min(target_no_fire, n_no_fire), replace=False)
            
            # Combine fire and selected no-fire
            selected_indices = fire_indices + selected_no_fire.tolist()
            balanced_patches = [patches[i] for i in selected_indices]
            balanced_labels = [labels[i] for i in selected_indices]
            balanced_metadata = [metadata[i] for i in selected_indices]
        
        logger.info(f"Balanced distribution: {np.bincount([balanced_labels])}")
        
        return balanced_patches, balanced_labels, balanced_metadata
    
    def _augment_fire_samples(
        self, 
        fire_patches: List[np.ndarray], 
        fire_labels: List[int], 
        fire_metadata: List[Dict],
        n_augment: int
    ) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Augment fire samples with sophisticated transformations."""
        
        augmented_patches = []
        augmented_labels = []
        augmented_metadata = []
        
        n_original = len(fire_patches)
        if n_original == 0:
            return augmented_patches, augmented_labels, augmented_metadata
        
        # Define augmentation strategies
        augmentations = [
            lambda x: np.rot90(x, 1, axes=(-2, -1)),  # 90° rotation
            lambda x: np.rot90(x, 2, axes=(-2, -1)),  # 180° rotation
            lambda x: np.rot90(x, 3, axes=(-2, -1)),  # 270° rotation
            lambda x: np.flip(x, axis=-1),              # Horizontal flip
            lambda x: np.flip(x, axis=-2),              # Vertical flip
            lambda x: self._add_noise(x, 0.05),         # Add noise
            lambda x: self._adjust_brightness(x, 0.1),  # Brightness adjustment
            lambda x: self._elastic_transform(x, 0.1),  # Elastic deformation
        ]
        
        for i in range(n_augment):
            # Select random original patch
            orig_idx = np.random.randint(0, n_original)
            patch = fire_patches[orig_idx].copy()
            
            # Apply random augmentation
            aug_func = np.random.choice(augmentations)
            try:
                augmented_patch = aug_func(patch)
                augmented_patches.append(augmented_patch)
                augmented_labels.append(fire_labels[orig_idx])
                
                # Update metadata
                new_metadata = fire_metadata[orig_idx].copy()
                new_metadata['augmented'] = True
                new_metadata['original_index'] = orig_idx
                augmented_metadata.append(new_metadata)
                
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
                # Use original patch as fallback
                augmented_patches.append(patch)
                augmented_labels.append(fire_labels[orig_idx])
                augmented_metadata.append(fire_metadata[orig_idx])
        
        return augmented_patches, augmented_labels, augmented_metadata
    
    def _add_noise(self, patch: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to patch."""
        noise = np.random.normal(0, noise_level * np.std(patch), patch.shape)
        return patch + noise
    
    def _adjust_brightness(self, patch: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness of patch."""
        adjustment = np.random.uniform(-factor, factor) * np.mean(patch)
        return patch + adjustment
    
    def _elastic_transform(self, patch: np.ndarray, alpha: float) -> np.ndarray:
        """Apply elastic transformation to patch."""
        # Simplified elastic transformation
        random_state = np.random.RandomState(None)
        
        if len(patch.shape) == 3:
            channels, height, width = patch.shape
            transformed = np.zeros_like(patch)
            
            for c in range(channels):
                # Generate displacement fields
                dx = random_state.uniform(-alpha, alpha, (height, width))
                dy = random_state.uniform(-alpha, alpha, (height, width))
                
                # Apply transformation (simplified)
                y_indices, x_indices = np.indices((height, width))
                new_y = np.clip(y_indices + dy, 0, height - 1).astype(int)
                new_x = np.clip(x_indices + dx, 0, width - 1).astype(int)
                
                transformed[c] = patch[c, new_y, new_x]
            
            return transformed
        else:
            return patch  # Return unchanged if not 3D

def process_modis_directory_parallel(
    input_dir: str,
    output_dir: str,
    patch_size: int = 32,
    overlap: float = 0.1,
    max_workers: int = 4,
    sensor_type: str = 'MODIS'
):
    """Process MODIS directory with parallel processing."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all TIF files
    tif_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.TIF"))
    
    if not tif_files:
        logger.warning(f"No TIF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(tif_files)} TIF files to process")
    
    def process_single_file(tif_file: Path) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Process single TIF file."""
        try:
            preprocessor = EnhancedThermalPreprocessor(
                patch_size=patch_size, 
                overlap=overlap,
                sensor_type=sensor_type,
                enable_cloud_masking=True
            )
            
            logger.info(f"Processing {tif_file.name}")
            
            # Load data
            data, metadata = preprocessor.load_modis_tile(str(tif_file), apply_quality_mask=True)
            
            # Extract enhanced features
            features = preprocessor.extract_enhanced_thermal_features(data)
            
            # Apply cloud masking
            if 'thermal' in features:
                masked_thermal, cloud_mask = preprocessor.apply_cloud_mask(
                    features['thermal'], 
                    visible_bands=data[:2] if data.shape[0] > 2 else None
                )
                features['thermal'] = masked_thermal
                features['cloud_mask'] = cloud_mask
            
            # Detect anomalies
            anomaly_mask = preprocessor.detect_advanced_thermal_anomalies(features, use_clustering=True)
            
            # Create patches
            patches, labels, patch_metadata = preprocessor.create_patches_enhanced(
                features, anomaly_mask, stratify=True
            )
            
            return patches, labels, patch_metadata
            
        except Exception as e:
            logger.error(f"Error processing {tif_file}: {e}")
            return [], [], []
    
    # Process files in parallel
    all_patches = []
    all_labels = []
    all_metadata = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks(finally)
        future_to_file = {executor.submit(process_single_file, tif_file): tif_file 
                         for tif_file in tif_files}
        
        # Collect results
        for future in as_completed(future_to_file):
            tif_file = future_to_file[future]
            try:
                patches, labels, metadata = future.result()
                all_patches.extend(patches)
                all_labels.extend(labels)
                all_metadata.extend(metadata)
                logger.info(f"Completed {tif_file.name}: {len(patches)} patches")
            except Exception as e:
                logger.error(f"Failed to process {tif_file}: {e}")
    
    if not all_patches:
        logger.error("No patches were created!")
        return
    
    # Balance dataset
    preprocessor = EnhancedThermalPreprocessor(patch_size=patch_size, overlap=overlap)
    balanced_patches, balanced_labels, balanced_metadata = preprocessor.balance_dataset(
        all_patches, all_labels, all_metadata, target_ratio=0.3
    )
    
    # Normalize patches
    logger.info("Normalizing patches...")
    normalized_patches = []
    for patch in balanced_patches:
        # Handle multi-channel patches
        if len(patch.shape) == 3:
            normalized_patch = np.zeros_like(patch)
            for c in range(patch.shape[0]):
                channel = patch[c]
                # Robust normalization
                p5, p95 = np.nanpercentile(channel, [5, 95])
                normalized_channel = np.clip((channel - p5) / (p95 - p5 + 1e-8), 0, 1)
                normalized_patch[c] = normalized_channel
        else:
            p5, p95 = np.nanpercentile(patch, [5, 95])
            normalized_patch = np.clip((patch - p5) / (p95 - p5 + 1e-8), 0, 1)
        
        normalized_patches.append(normalized_patch)
    
    # Save processed data
    logger.info("Saving processed data...")
    np.save(output_path / "patches.npy", np.array(normalized_patches))
    np.save(output_path / "labels.npy", np.array(balanced_labels))
    
    # Save metadata
    import pickle
    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump(balanced_metadata, f)
    
    # Save processing statistics
    stats = {
        'total_patches': len(balanced_patches),
        'fire_patches': sum(balanced_labels),
        'no_fire_patches': len(balanced_labels) - sum(balanced_labels),
        'fire_ratio': sum(balanced_labels) / len(balanced_labels),
        'patch_size': patch_size,
        'overlap': overlap,
        'sensor_type': sensor_type,
        'files_processed': len(tif_files)
    }
    
    with open(output_path / "processing_stats.json", "w") as f:
        import json
        json.dump(stats, f, indent=2)
    
    logger.info(f"Processing completed!")
    logger.info(f"Total patches: {len(balanced_patches)}")
    logger.info(f"Fire patches: {sum(balanced_labels)}")
    logger.info(f"No-fire patches: {len(balanced_labels) - sum(balanced_labels)}")
    logger.info(f"Fire ratio: {sum(balanced_labels) / len(balanced_labels):.3f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Enhanced MODIS thermal imagery preprocessing')
    parser.add_argument('--input', type=str, required=True, help='Input directory with MODIS tiles')
    parser.add_argument('--output', type=str, required=True, help='Output directory for processed patches')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of patches')
    parser.add_argument('--overlap', type=float, default=0.1, help='Overlap between patches')
    parser.add_argument('--sensor', type=str, default='MODIS', choices=['MODIS', 'LANDSAT', 'SENTINEL'])
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    process_modis_directory_parallel(
        input_dir=args.input,
        output_dir=args.output,
        patch_size=args.patch_size,
        overlap=args.overlap,
        max_workers=args.workers,
        sensor_type=args.sensor
    )
