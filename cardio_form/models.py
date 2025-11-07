import os
import yaml
import hashlib
import urllib.request
from pathlib import Path
from tqdm import tqdm

# --- Helper for TQDM progress bar ---
class TqdmUpTo(tqdm):
    """Provides `update_to(block_num, block_size, total_size)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# --- The Main Class ---
class ModelManager:
    """
    Manages the downloading, caching, and retrieval of model weights.
    Reads a manifest file (models.yaml) to know about available models.
    """
    def __init__(self, manifest_path=None, cache_dir=None):
        if manifest_path is None:
            # Default to a 'models.yaml' in the project root
            manifest_path = Path(__file__).parent.parent / 'models.yaml'
        if cache_dir is None:
            # Default to a user-level cache directory
            cache_dir = Path.home() / '.cache' / 'cardio_form'

        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Model manifest not found at {self.manifest_path}")
            
        self._manifest = self._load_manifest()

    def _load_manifest(self):
        with open(self.manifest_path, 'r') as f:
            return yaml.safe_load(f)

    def _verify_hash(self, file_path, expected_hash):
        """Verifies the SHA256 hash of a file."""
        if expected_hash is None:
            return True
        
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        actual_hash = hasher.hexdigest()
        if actual_hash != expected_hash:
            raise IOError(
                f"File hash mismatch for {file_path}.\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hash}"
            )
        return True

    def get_model_path(self, model_name: str, version: str = 'default') -> str:
        """
        The main public method. Gets the local path to a model, downloading it if necessary.

        Args:
            model_name (str): The logical name of the model (e.g., 'reconstruction_3d').
            version (str): The desired version tag (e.g., 'v1.1'). If 'default',
                           uses the default version from the manifest.

        Returns:
            The absolute path to the cached model file.
        """
        if model_name not in self._manifest:
            raise KeyError(f"Model '{model_name}' not found in manifest.")
        
        model_info = self._manifest[model_name]
        
        if version == 'default':
            version = model_info['default']
        
        if version not in model_info['versions']:
            raise KeyError(f"Version '{version}' for model '{model_name}' not found in manifest.")
            
        version_info = model_info['versions'][version]
        url = version_info['url']
        expected_hash = version_info.get('sha256')
        
        # Handle local file paths defined in the manifest
        if url.startswith("file://"):
            local_path = Path(url[7:])
            if not local_path.exists():
                raise FileNotFoundError(f"Local model file not found: {local_path}")
            return str(local_path)

        # Handle remote files
        filename = os.path.basename(url)
        cached_path = self.cache_dir / filename

        if cached_path.exists():
            print(f"Found model '{filename}' in cache. Verifying integrity...")
            try:
                if self._verify_hash(cached_path, expected_hash):
                    print("Integrity check passed.")
                    return str(cached_path)
            except IOError as e:
                print(f"Integrity check failed: {e}. Re-downloading...")
                os.remove(cached_path)

        # File needs to be downloaded
        print(f"Downloading model '{filename}' from {url}...")
        try:
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, filename=str(cached_path), reporthook=t.update_to)
            
            print("Download complete. Verifying integrity...")
            if self._verify_hash(cached_path, expected_hash):
                print("Integrity check passed.")
                return str(cached_path)
        except Exception as e:
            # Clean up partial download on failure
            if cached_path.exists():
                os.remove(cached_path)
            raise RuntimeError(f"Failed to download or verify model '{filename}'. Error: {e}")

# For convenience, you can create a singleton instance to be used across the library.
default_model_manager = ModelManager()