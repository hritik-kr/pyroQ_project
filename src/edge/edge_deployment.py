import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import io
import json
import geojson
from typing import List, Dict, Tuple, Optional, Any
import asyncio
from pathlib import Path
import logging
import cv2
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point, Polygon
import time
import hashlib
import jwt
from datetime import datetime, timedelta
import redis
from collections import defaultdict
import psutil
import sys
import os
import gc
import secrets

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import settings
from src.hybrid.hybrid_model import PyroQHybridModel
from src.quantum.qcnn import EnhancedQCNN

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.log_file) if settings.log_file else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class RateLimiter:
    """Enhanced Redis-based rate limiter with configuration support."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        try:
            self.redis = redis_client or redis.from_url(settings.redis_url, decode_responses=True)
            # Test connection
            self.redis.ping()
        except (redis.RedisError, ConnectionError) as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            self.redis = None
            self._memory_cache = defaultdict(dict)
        
        self.limits = {
            'detect': {'requests': settings.rate_limit_requests_per_hour, 'window': 3600},
            'detect_geotiff': {'requests': settings.rate_limit_geotiff_per_hour, 'window': 3600},
        }
    
    async def is_allowed(self, client_id: str, endpoint: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed based on rate limits."""
        limit_config = self.limits.get(endpoint, {'requests': 1000, 'window': 3600})
        
        key = f"rate_limit:{client_id}:{endpoint}"
        window = limit_config['window']
        max_requests = limit_config['requests']
        
        if self.redis:
            try:
                return await self._redis_rate_limit(key, max_requests, window)
            except redis.RedisError as e:
                logger.warning(f"Redis rate limiting failed: {e}")
                return await self._memory_rate_limit(client_id, endpoint, max_requests, window)
        else:
            return await self._memory_rate_limit(client_id, endpoint, max_requests, window)
    
    async def _redis_rate_limit(self, key: str, max_requests: int, window: int) -> Tuple[bool, Dict[str, Any]]:
        """Redis-based rate limiting."""
        current = self.redis.get(key)
        if current is None:
            self.redis.setex(key, window, 1)
            return True, {'remaining': max_requests - 1, 'reset_time': time.time() + window}
        
        current = int(current)
        if current >= max_requests:
            ttl = self.redis.ttl(key)
            return False, {'remaining': 0, 'reset_time': time.time() + ttl}
        
        self.redis.incr(key)
        ttl = self.redis.ttl(key)
        return True, {'remaining': max_requests - current - 1, 'reset_time': time.time() + ttl}
    
    async def _memory_rate_limit(self, client_id: str, endpoint: str, max_requests: int, window: int) -> Tuple[bool, Dict[str, Any]]:
        """Memory-based rate limiting fallback."""
        now = time.time()
        key = f"{client_id}:{endpoint}"
        
        if key not in self._memory_cache:
            self._memory_cache[key] = {'count': 1, 'window_start': now}
            return True, {'remaining': max_requests - 1, 'reset_time': now + window}
        
        cache_data = self._memory_cache[key]
        
        # Reset window if expired
        if now - cache_data['window_start'] > window:
            self._memory_cache[key] = {'count': 1, 'window_start': now}
            return True, {'remaining': max_requests - 1, 'reset_time': now + window}
        
        # Check limit
        if cache_data['count'] >= max_requests:
            reset_time = cache_data['window_start'] + window
            return False, {'remaining': 0, 'reset_time': reset_time}
        
        # Increment counter
        cache_data['count'] += 1
        reset_time = cache_data['window_start'] + window
        return True, {'remaining': max_requests - cache_data['count'], 'reset_time': reset_time}

class SecurityManager:
    """Enhanced security manager with JWT key rotation and secure password hashing."""
    
    def __init__(self):
        self.users_db = {
            "admin": {
                "username": "admin",
                "hashed_password": self._hash_password("admin123", salt=os.urandom(16).hex()),
                "permissions": ["detect", "detect_geotiff", "admin"],
                "salt": os.urandom(16).hex()
            },
            "user": {
                "username": "user",
                "hashed_password": self._hash_password("user123", salt=os.urandom(16).hex()),
                "permissions": ["detect"],
                "salt": os.urandom(16).hex()
            }
        }
        
        # Key rotation configuration
        self.key_rotation = {
            "current": settings.secret_key,
            "previous": [],
            "rotation_interval": timedelta(days=1),
            "max_previous_keys": 3
        }
        self.last_rotation = datetime.utcnow()
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Secure password hashing using PBKDF2-HMAC-SHA256."""
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000,
            dklen=32
        ).hex()
    
    def rotate_keys(self):
        """Rotate JWT signing keys according to schedule."""
        now = datetime.utcnow()
        if now - self.last_rotation >= self.key_rotation["rotation_interval"]:
            # Move current key to previous keys
            self.key_rotation["previous"].insert(0, self.key_rotation["current"])
            
            # Generate new key
            self.key_rotation["current"] = secrets.token_hex(32)
            
            # Trim old keys
            if len(self.key_rotation["previous"]) > self.key_rotation["max_previous_keys"]:
                self.key_rotation["previous"] = self.key_rotation["previous"][:self.key_rotation["max_previous_keys"]]
            
            self.last_rotation = now
            logger.info("Rotated JWT signing keys")
    
    def verify_password(self, plain_password: str, username: str) -> bool:
        """Verify password against stored hash with proper salt."""
        user = self.get_user(username)
        if not user:
            return False
            
        salt = user.get("salt", "")
        hashed_attempt = self._hash_password(plain_password, salt)
        return secrets.compare_digest(hashed_attempt, user["hashed_password"])
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user from database."""
        return self.users_db.get(username)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user."""
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, username):
            return None
        return user
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT token with key rotation support."""
        self.rotate_keys()
        
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        try:
            return jwt.encode(to_encode, self.key_rotation["current"], algorithm=settings.algorithm)
        except Exception as e:
            logger.warning(f"Failed to encode with current key: {e}")
        
        for key in self.key_rotation["previous"]:
            try:
                return jwt.encode(to_encode, key, algorithm=settings.algorithm)
            except Exception:
                continue
                
        raise ValueError("No valid signing key available")
    
    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> Dict:
        """Verify JWT token with key rotation support."""
        credentials_exception = HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        for key in [self.key_rotation["current"]] + self.key_rotation["previous"]:
            try:
                payload = jwt.decode(token, key, algorithms=[settings.algorithm])
                username: str = payload.get("sub")
                if username is None:
                    continue
                    
                user = self.get_user(username)
                if user is None:
                    continue
                    
                return user
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=401,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            except jwt.PyJWTError:
                continue
                
        raise credentials_exception

class OptimizedInferenceEngine:
    """Enhanced inference engine with improved model loading."""
    
    def __init__(self, model_path: str = None, use_onnx: bool = None, enable_caching: bool = None):
        self.model_path = model_path or settings.model_path
        self.use_onnx = use_onnx if use_onnx is not None else settings.use_onnx
        self.enable_caching = enable_caching if enable_caching is not None else settings.enable_caching
        
        self.model = None
        self.onnx_session = None
        self.model_type = "unknown"
        
        # Performance monitoring
        self.inference_times = []
        self.memory_usage = []
        self.request_count = 0
        
        # Caching setup
        if self.enable_caching:
            try:
                self.cache = redis.from_url(settings.redis_url)
                self.cache.ping()
            except (redis.RedisError, ConnectionError):
                logger.warning("Redis cache not available, using memory cache")
                self.cache = {}
        
        # Load model with enhanced error handling
        self._load_model()
    
    def _load_model(self):
        """Enhanced model loading with multiple fallback strategies."""
        try:
            self._load_quantum_model()
            self.model_type = "quantum"
            logger.info("Successfully loaded quantum model")
        except Exception as e:
            logger.warning(f"Quantum model loading failed: {e}")
            try:
                self._load_classical_model()
                self.model_type = "classical"
                logger.info("Successfully loaded classical model")
            except Exception as e:
                logger.error(f"Classical model loading failed: {e}")
                self._create_fallback_model()
                self.model_type = "fallback"
                logger.info("Using fallback model")
    
    def _load_quantum_model(self):
        """Load quantum-enhanced model."""
        if self.use_onnx and Path(self.model_path.replace('.pt', '.onnx')).exists():
            onnx_path = self.model_path.replace('.pt', '.onnx')
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(self.model_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                model_config = checkpoint.get('model_config', {})
            else:
                state_dict = checkpoint
                model_config = {}
            
            self.model = EnhancedQCNN(
                input_size=(32, 32), 
                num_classes=2,
                quantum_enabled=settings.quantum_enabled
            )
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(device)
    
    def _load_classical_model(self):
        """Load classical fallback model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(self.model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.model = EnhancedQCNN(
            input_size=(32, 32), 
            num_classes=2,
            quantum_enabled=False
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
    
    def _create_fallback_model(self):
        """Create simple fallback model."""
        logger.warning("Creating minimal fallback model")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(32 * 16, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        for layer in self.model:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        self.model.eval()
        self.model.to(device)
    
    def _validate_input(self, image: np.ndarray) -> np.ndarray:
        """Enhanced input validation."""
        if image.nbytes > settings.max_file_size:
            raise ValueError(f"Image too large: {image.nbytes} bytes > {settings.max_file_size}")
        
        if len(image.shape) > 3:
            raise ValueError(f"Invalid image dimensions: {image.shape}")
        
        if len(image.shape) >= 2:
            height, width = image.shape[:2]
            if height > settings.max_image_height or width > settings.max_image_width:
                raise ValueError(f"Image dimensions too large: {height}x{width} > {settings.max_image_height}x{settings.max_image_width}")
        
        if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            raise ValueError(f"Unsupported image dtype: {image.dtype}")
        
        return image
    
    def _get_cache_key(self, image: np.ndarray) -> str:
        """Generate cache key for image."""
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        return f"inference:{self.model_type}:{image_hash}"
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
        """Enhanced preprocessing with validation."""
        image = self._validate_input(image)
        
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                image = image[:, :, 0]
        
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        image_norm = image.astype(np.float32) / 255.0
        image_tensor = image_norm[np.newaxis, np.newaxis, :, :]
        
        return image_tensor
    
    async def predict(self, image: np.ndarray, use_cache: bool = True) -> Tuple[int, float, np.ndarray]:
        """Enhanced prediction with comprehensive error handling."""
        start_time = time.time()
        
        try:
            if use_cache and self.enable_caching:
                cache_key = self._get_cache_key(image)
                
                if isinstance(self.cache, dict):
                    cached_result = self.cache.get(cache_key)
                else:
                    try:
                        cached_result = self.cache.get(cache_key)
                        if cached_result:
                            cached_result = json.loads(cached_result)
                    except:
                        cached_result = None
                
                if cached_result:
                    logger.debug("Cache hit for inference")
                    return (cached_result['prediction'], 
                           cached_result['confidence'], 
                           np.array(cached_result['probabilities']))
            
            input_tensor = self.preprocess_image(image)
            
            if self.onnx_session:
                logits = self._onnx_inference(input_tensor)
            else:
                logits = self._pytorch_inference(input_tensor)
            
            probabilities = self._softmax(logits)
            predicted_class = np.argmax(logits, axis=1)[0]
            confidence = probabilities[0][predicted_class]
            
            if use_cache and self.enable_caching:
                result = {
                    'prediction': int(predicted_class),
                    'confidence': float(confidence),
                    'probabilities': probabilities[0].tolist(),
                    'model_type': self.model_type
                }
                
                if isinstance(self.cache, dict):
                    self.cache[cache_key] = result
                else:
                    try:
                        self.cache.setex(cache_key, 3600, json.dumps(result))
                    except:
                        pass
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)
            self.request_count += 1
            
            if len(self.inference_times) > 1000:
                self.inference_times = self.inference_times[-1000:]
                self.memory_usage = self.memory_usage[-1000:]
            
            return predicted_class, confidence, probabilities[0]
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    def _onnx_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """ONNX model inference."""
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: input_tensor})
        return outputs[0]
    
    def _pytorch_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """PyTorch model inference with quantum fallback."""
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_tensor)
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                input_tensor = input_tensor.cuda()
            
            if self.model_type == "quantum" and hasattr(self.model, 'forward'):
                try:
                    logits = self.model(input_tensor, use_quantum=True)
                except Exception as e:
                    logger.warning(f"Quantum inference failed: {e}")
                    logits = self.model(input_tensor, use_quantum=False)
            else:
                logits = self.model(input_tensor)
            
            return logits.cpu().numpy()
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.inference_times:
            return {"error": "No inference data available"}
        
        return {
            "total_requests": self.request_count,
            "avg_inference_time_ms": np.mean(self.inference_times) * 1000,
            "p50_inference_time_ms": np.percentile(self.inference_times, 50) * 1000,
            "p95_inference_time_ms": np.percentile(self.inference_times, 95) * 1000,
            "p99_inference_time_ms": np.percentile(self.inference_times, 99) * 1000,
            "avg_memory_usage_mb": np.mean(self.memory_usage),
            "current_memory_usage_mb": self.memory_usage[-1] if self.memory_usage else 0,
            "model_type": self.model_type,
            "cache_enabled": self.enable_caching
        }

class EnhancedWildfireDetectionAPI:
    """Production-ready FastAPI application with all improvements."""
    
    def __init__(self, model_path: str = None):
        self.app = FastAPI(
            title="PyroQ: Quantum-Enhanced Wildfire Detection",
            description="Production-ready wildfire detection using quantum-classical hybrid models",
            version="2.1.0",
            docs_url="/docs" if not settings.is_production else None,
            redoc_url="/redoc" if not settings.is_production else None
        )
        
        self.inference_engine = OptimizedInferenceEngine(model_path)
        self.security_manager = SecurityManager()
        self.rate_limiter = RateLimiter()
        
        self.request_stats = defaultdict(int)
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Enhanced middleware setup with security headers."""
        
        if settings.is_production:
            self.app.add_middleware(HTTPSRedirectMiddleware)
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        
        self.app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=settings.trusted_hosts
        )
        
        @self.app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            response = await call_next(request)
            
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Content-Security-Policy"] = "default-src 'self'"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
            
            response.headers.pop("server", None)
            
            return response
        
        @self.app.middleware("http")
        async def log_and_monitor_requests(request: Request, call_next):
            start_time = time.time()
            
            client_ip = request.headers.get("X-Forwarded-For", request.client.host)
            user_agent = request.headers.get("User-Agent", "Unknown")
            
            response = await call_next(request)
            
            process_time = time.time() - start_time
            
            logger.info(
                f"{client_ip} - {request.method} {request.url.path} - "
                f"{response.status_code} - {process_time:.3f}s - {user_agent}"
            )
            
            self.request_stats[request.url.path] += 1
            
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-API-Version"] = "2.1.0"
            
            return response
    
    def _setup_routes(self):
        """Setup API routes with enhanced functionality."""
        
        @self.app.post("/token", tags=["Authentication"])
        async def login(form_data: OAuth2PasswordRequestForm = Depends()):
            user = self.security_manager.authenticate_user(form_data.username, form_data.password)
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
            access_token = self.security_manager.create_access_token(
                data={"sub": user["username"]}, expires_delta=access_token_expires
            )
            return {
                "access_token": access_token, 
                "token_type": "bearer",
                "expires_in": settings.access_token_expire_minutes * 60
            }
        
        @self.app.get("/", tags=["General"])
        async def root():
            return {
                "message": "PyroQ Wildfire Detection API v2.1", 
                "status": "ready",
                "model_type": self.inference_engine.model_type,
                "environment": settings.environment,
                "documentation": "/docs" if not settings.is_production else "Contact admin for documentation"
            }
        
        @self.app.get("/health", tags=["Monitoring"])
        async def health_check():
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk_usage = psutil.disk_usage('/')
            
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "model": {
                    "loaded": self.inference_engine.model is not None or self.inference_engine.onnx_session is not None,
                    "type": self.inference_engine.model_type,
                    "quantum_enabled": settings.quantum_enabled
                },
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "memory_available_mb": memory_info.available / 1024 / 1024,
                    "disk_free_gb": disk_usage.free / 1024 / 1024 / 1024
                },
                "configuration": {
                    "environment": settings.environment,
                    "cache_enabled": settings.enable_caching,
                    "metrics_enabled": settings.enable_metrics
                }
            }
        
        @self.app.post("/detect", tags=["Detection"])
        async def detect_fire(
            request: Request,
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            current_user: dict = Depends(self.security_manager.get_current_user)
        ):
            if "detect" not in current_user.get("permissions", []):
                raise HTTPException(status_code=403, detail="Permission denied")
            
            client_id = f"{current_user['username']}_{request.client.host}"
            allowed, limit_info = await self.rate_limiter.is_allowed(client_id, "detect")
            
            if not allowed:
                raise HTTPException(
                    status_code=429, 
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(settings.rate_limit_requests_per_hour),
                        "X-RateLimit-Remaining": str(limit_info["remaining"]),
                        "X-RateLimit-Reset": str(int(limit_info["reset_time"]))
                    }
                )
            
            try:
                if file.size > settings.max_file_size:
                    raise HTTPException(status_code=413, detail="File too large")
                
                allowed_types = ["image/jpeg", "image/png", "image/tiff", "application/octet-stream"]
                if file.content_type not in allowed_types:
                    raise HTTPException(status_code=400, detail="Invalid file type")
                
                contents = await file.read()
                
                try:
                    image = Image.open(io.BytesIO(contents))
                    image_np = np.array(image)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
                
                prediction, confidence, probabilities = await self.inference_engine.predict(image_np)
                
                alert_level = "none"
                if prediction == 1:
                    if confidence > 0.9:
                        alert_level = "critical"
                    elif confidence > 0.7:
                        alert_level = "high"
                    else:
                        alert_level = "medium"
                
                result = {
                    "prediction": "fire" if prediction == 1 else "no_fire",
                    "confidence": float(confidence),
                    "probabilities": {
                        "no_fire": float(probabilities[0]),
                        "fire": float(probabilities[1])
                    },
                    "alert_level": alert_level,
                    "model_type": self.inference_engine.model_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "user": current_user["username"]
                }
                
                headers = {
                    "X-RateLimit-Limit": str(settings.rate_limit_requests_per_hour),
                    "X-RateLimit-Remaining": str(limit_info["remaining"]),
                    "X-RateLimit-Reset": str(int(limit_info["reset_time"]))
                }
                
                background_tasks.add_task(
                    self._log_detection_result, 
                    current_user["username"], 
                    file.filename, 
                    result
                )
                
                return JSONResponse(content=result, headers=headers)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in fire detection: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/detect_geotiff", tags=["Detection"])
        async def detect_fire_geotiff(
            request: Request,
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            confidence_threshold: float = 0.8,
            current_user: dict = Depends(self.security_manager.get_current_user)
        ):
            if "detect_geotiff" not in current_user.get("permissions", []):
                raise HTTPException(status_code=403, detail="Permission denied")
            
            client_id = f"{current_user['username']}_{request.client.host}"
            allowed, limit_info = await self.rate_limiter.is_allowed(client_id, "detect_geotiff")
            
            if not allowed:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            try:
                if file.size > settings.max_file_size:
                    raise HTTPException(status_code=413, detail="File too large")
                
                temp_path = f"/tmp/{hashlib.md5((file.filename + str(time.time())).encode()).hexdigest()}.tif"
                with open(temp_path, "wb") as f:
                    f.write(await file.read())
                
                detections = await self._process_geotiff_advanced(temp_path, confidence_threshold)
                
                result = {
                    "detections": detections,
                    "total_detections": len(detections),
                    "confidence_threshold": confidence_threshold,
                    "model_type": self.inference_engine.model_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "user": current_user["username"]
                }
                
                background_tasks.add_task(
                    self._log_detection_result,
                    current_user["username"],
                    file.filename,
                    result
                )
                
                return JSONResponse(content=result)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in GeoTIFF processing: {str(e)}")
                raise HTTPException(status_code=500, detail="GeoTIFF processing error")
        
        @self.app.get("/system/monitor", tags=["Monitoring"])
        async def system_monitor(current_user: dict = Depends(self.security_manager.get_current_user)):
            if "admin" not in current_user.get("permissions", []):
                raise HTTPException(status_code=403, detail="Admin access required")
            
            cpu_times = psutil.cpu_times()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return {
                "inference_metrics": self.inference_engine.get_performance_metrics(),
                "system": {
                    "cpu": {
                        "percent": psutil.cpu_percent(interval=1),
                        "count": psutil.cpu_count(),
                        "times": cpu_times._asdict()
                    },
                    "memory": {
                        "total_gb": memory.total / 1024 / 1024 / 1024,
                        "available_gb": memory.available / 1024 / 1024 / 1024,
                        "percent": memory.percent,
                        "used_gb": memory.used / 1024 / 1024 / 1024
                    },
                    "disk": {
                        "total_gb": disk.total / 1024 / 1024 / 1024,
                        "free_gb": disk.free / 1024 / 1024 / 1024,
                        "used_gb": disk.used / 1024 / 1024 / 1024,
                        "percent": (disk.used / disk.total) * 100
                    },
                    "network": {
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv,
                        "packets_sent": network.packets_sent,
                        "packets_recv": network.packets_recv
                    }
                },
                "api_stats": {
                    "total_requests": sum(self.request_stats.values()),
                    "endpoint_stats": dict(self.request_stats)
                },
                "configuration": {
                    "model_path": settings.model_path,
                    "quantum_enabled": settings.quantum_enabled,
                    "cache_enabled": settings.enable_caching,
                    "environment": settings.environment
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/metrics", tags=["Monitoring"])
        async def get_metrics(current_user: dict = Depends(self.security_manager.get_current_user)):
            return {
                "performance": self.inference_engine.get_performance_metrics(),
                "model_info": {
                    "type": self.inference_engine.model_type,
                    "quantum_enabled": settings.quantum_enabled,
                    "cache_enabled": settings.enable_caching
                },
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _process_geotiff_advanced(self, geotiff_path: str, confidence_threshold: float = 0.8) -> List[Dict]:
        """Enhanced GeoTIFF processing with tile streaming and memory management."""
        detections = []
        
        try:
            with rasterio.open(geotiff_path) as src:
                if src.count < 1:
                    raise ValueError("No bands found in GeoTIFF")
                
                block_shapes = src.block_shapes
                if not block_shapes:
                    block_shapes = [(1024, 1024)] * src.count
                
                for band_idx in range(min(src.count, 4)):
                    block_height, block_width = block_shapes[band_idx]
                    
                    n_blocks_x = int(np.ceil(src.width / block_width))
                    n_blocks_y = int(np.ceil(src.height / block_height))
                    
                    for block_x in range(n_blocks_x):
                        for block_y in range(n_blocks_y):
                            x_offset = block_x * block_width
                            y_offset = block_y * block_height
                            width = min(block_width, src.width - x_offset)
                            height = min(block_height, src.height - y_offset)
                            
                            window = Window(x_offset, y_offset, width, height)
                            band_data = src.read(band_idx + 1, window=window)
                            transform = src.window_transform(window)
                            
                            patch_size = 32
                            stride = 16
                            
                            for y in range(0, height - patch_size + 1, stride):
                                for x in range(0, width - patch_size + 1, stride):
                                    patch = band_data[y:y+patch_size, x:x+patch_size]
                                    
                                    if np.isnan(patch).mean() > 0.5:
                                        continue
                                        
                                    try:
                                        prediction, confidence, _ = await self.inference_engine.predict(patch)
                                        
                                        if prediction == 1 and confidence >= confidence_threshold:
                                            center_x = x + patch_size // 2 + x_offset
                                            center_y = y + patch_size // 2 + y_offset
                                            lon, lat = rasterio.transform.xy(transform, center_y, center_x)
                                            
                                            detections.append({
                                                "id": len(detections) + 1,
                                                "band": band_idx + 1,
                                                "latitude": float(lat),
                                                "longitude": float(lon),
                                                "confidence": float(confidence),
                                                "pixel_coordinates": {
                                                    "x": int(center_x),
                                                    "y": int(center_y)
                                                }
                                            })
                                            
                                    except Exception as e:
                                        logger.warning(f"Patch inference failed at {x},{y}: {e}")
                                        continue
                            
                            del band_data
                            gc.collect()
        
        except Exception as e:
            logger.error(f"GeoTIFF processing failed: {e}")
            raise HTTPException(status_code=500, detail="GeoTIFF processing error")
        finally:
            try:
                Path(geotiff_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {geotiff_path}: {e}")
        
        return detections
    
    async def _log_detection_result(self, username: str, filename: str, result: Dict):
        """Log detection results for audit trail."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": username,
            "filename": filename,
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "model_type": result.get("model_type"),
            "total_detections": result.get("total_detections", 1 if result.get("prediction") == "fire" else 0)
        }
        
        logger.info(f"Detection result: {json.dumps(log_entry)}")

def create_app(model_path: str = None) -> FastAPI:
    """Factory function to create the FastAPI app."""
    api = EnhancedWildfireDetectionAPI(model_path)
    return api.app

if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description='PyroQ Enhanced Edge Deployment')
    parser.add_argument('--model', type=str, help='Path to trained model (overrides config)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='API host')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    api = EnhancedWildfireDetectionAPI(args.model)
    
    config = {
        "app": api.app,
        "host": args.host,
        "port": args.port,
        "log_level": settings.log_level.lower(),
        "access_log": True,
        "workers": args.workers if not args.reload else 1,
        "reload": args.reload
    }
    
    if settings.is_production:
        config.update({
            "ssl_keyfile": os.getenv("SSL_KEYFILE"),
            "ssl_certfile": os.getenv("SSL_CERTFILE"),
        })
    
    logger.info(f"Starting PyroQ API v2.1 in {settings.environment} mode")
    logger.info(f"Model type: {api.inference_engine.model_type}")
    logger.info(f"Quantum enabled: {settings.quantum_enabled}")
    
    uvicorn.run(**config)