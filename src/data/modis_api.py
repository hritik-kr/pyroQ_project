import requests
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional
import argparse

class MODISDataDownloader:
    """Download MODIS data from NASA APIs."""
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.base_url = "https://modis.ornl.gov/rst/api/v1"
        self.earthdata_url = "https://n5eil01u.ecs.nsidc.org"
        
        # Authentication for NASA Earthdata
        self.username = username or os.getenv("NASA_USERNAME")
        self.password = password or os.getenv("NASA_PASSWORD")
        
        if not self.username or not self.password:
            print("Warning: NASA credentials not provided. Some functions may not work.")
    
    def get_modis_products(self) -> List[Dict]:
        """Get available MODIS products."""
        url = f"{self.base_url}/products"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()['products']
        except requests.RequestException as e:
            print(f"Error fetching MODIS products: {e}")
            return []
    
    def get_product_dates(self, product: str, latitude: float, longitude: float) -> List[str]:
        """Get available dates for a product at given coordinates."""
        url = f"{self.base_url}/{product}/dates"
        params = {
            'latitude': latitude,
            'longitude': longitude
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()['dates']
        except requests.RequestException as e:
            print(f"Error fetching dates: {e}")
            return []
    
    def get_modis_data(
        self, 
        product: str, 
        latitude: float, 
        longitude: float, 
        start_date: str, 
        end_date: str,
        kmAboveBelow: int = 0,
        kmLeftRight: int = 0
    ) -> Dict:
        """Get MODIS data for specified location and time range."""
        url = f"{self.base_url}/{product}/subset"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'startDate': start_date,
            'endDate': end_date,
            'kmAboveBelow': kmAboveBelow,
            'kmLeftRight': kmLeftRight
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching MODIS data: {e}")
            return {}
    
    def download_fire_data(
        self,
        bbox: Tuple[float, float, float, float],  # (min_lon, min_lat, max_lon, max_lat)
        start_date: str,
        end_date: str,
        output_dir: str
    ) -> List[str]:
        """Download MODIS fire/thermal data for a bounding box."""
        min_lon, min_lat, max_lon, max_lat = bbox
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = []
        
        # MODIS Terra and Aqua fire products
        products = [
            "MOD14A1",  # MODIS/Terra Thermal Anomalies/Fire Daily
            "MYD14A1",  # MODIS/Aqua Thermal Anomalies/Fire Daily
        ]
        
        # Sample locations within bounding box
        lat_points = [min_lat + i * (max_lat - min_lat) / 4 for i in range(5)]
        lon_points = [min_lon + i * (max_lon - min_lon) / 4 for i in range(5)]
        
        for product in products:
            for lat in lat_points:
                for lon in lon_points:
                    try:
                        print(f"Downloading {product} data for {lat:.2f}, {lon:.2f}")
                        
                        data = self.get_modis_data(
                            product=product,
                            latitude=lat,
                            longitude=lon,
                            start_date=start_date,
                            end_date=end_date,
                            kmAboveBelow=50,  # ~100km square
                            kmLeftRight=50
                        )
                        
                        if data and 'subset' in data:
                            filename = f"{product}_{lat:.2f}_{lon:.2f}_{start_date}_{end_date}.json"
                            filepath = output_path / filename
                            
                            with open(filepath, 'w') as f:
                                json.dump(data, f, indent=2)
                            
                            downloaded_files.append(str(filepath))
                            print(f"Saved: {filename}")
                        
                        # Rate limiting
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Error downloading {product} at {lat}, {lon}: {e}")
                        continue
        
        return downloaded_files
    
    def get_recent_fire_alerts(self, bbox: Tuple[float, float, float, float], hours: int = 24) -> List[Dict]:
        """Get recent fire alerts from FIRMS API."""
        # NASA FIRMS (Fire Information for Resource Management System)
        firms_url = "https://firms.modaps.eosdis.nasa.gov/api/country/csv/your_api_key/MODIS_C6_1"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours)
        
        # This is a simplified version - you would need to register for FIRMS API key
        print(f"To get real-time fire alerts, register at: https://firms.modaps.eosdis.nasa.gov/api/")
        
        # Mock data structure for development
        return [
            {
                'latitude': bbox[1] + 0.1,
                'longitude': bbox[0] + 0.1,
                'brightness': 350.5,
                'confidence': 85,
                'acq_date': start_date.strftime('%Y-%m-%d'),
                'acq_time': start_date.strftime('%H:%M')
            }
        ]

def create_sample_data(output_dir: str, num_samples: int = 1000) -> None:
    """Create sample thermal anomaly data for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    import numpy as np
    from scipy import ndimage
    
    # Create synthetic thermal imagery
    images = []
    labels = []
    
    for i in range(num_samples):
        # Base temperature map
        base_temp = np.random.normal(290, 10, (64, 64))  # Kelvin
        
        # Add thermal anomalies (fires) to some images
        if np.random.random() < 0.3:  # 30% fire probability
            # Add hotspots
            num_hotspots = np.random.randint(1, 4)
            label = 1
            
            for _ in range(num_hotspots):
                # Random hotspot location
                y, x = np.random.randint(10, 54, 2)
                
                # Create hotspot with Gaussian distribution
                y_grid, x_grid = np.mgrid[0:64, 0:64]
                hotspot = 50 * np.exp(-((y_grid - y)**2 + (x_grid - x)**2) / (2 * 3**2))
                
                base_temp += hotspot
        else:
            label = 0
        
        # Add noise
        base_temp += np.random.normal(0, 2, base_temp.shape)
        
        # Smooth slightly
        base_temp = ndimage.gaussian_filter(base_temp, sigma=0.5)
        
        images.append(base_temp)
        labels.append(label)
    
    # Save data
    np.save(output_path / "thermal_images.npy", np.array(images))
    np.save(output_path / "labels.npy", np.array(labels))
    
    print(f"Created {num_samples} sample images")
    print(f"Fire samples: {sum(labels)}")
    print(f"No-fire samples: {len(labels) - sum(labels)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download MODIS fire data')
    parser.add_argument('--bbox', type=str, required=True, 
                       help='Bounding box as "min_lon,min_lat,max_lon,max_lat"')
    parser.add_argument('--start_date', type=str, 
                       default=(datetime.now() - timedelta(days=7)).strftime('A%Y%j'),
                       help='Start date (format: A2023001)')
    parser.add_argument('--end_date', type=str,
                       default=datetime.now().strftime('A%Y%j'),
                       help='End date (format: A2023001)')
    parser.add_argument('--hours', type=int, default=24,
                       help='Hours of recent data to fetch')
    parser.add_argument('--out', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--sample', action='store_true',
                       help='Create sample data instead of downloading')
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_data(args.out)
    else:
        # Parse bounding box
        bbox_str = args.bbox.split(',')
        bbox = tuple(map(float, bbox_str))
        
        # Initialize downloader
        downloader = MODISDataDownloader()
        
        # Download data
        files = downloader.download_fire_data(
            bbox=bbox,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.out
        )
        
        print(f"Downloaded {len(files)} files")
