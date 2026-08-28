import os
import yaml
import hashlib
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

from cardio_form.utils import configure_logging
from cardio_form.config import config_path, dev_root
logger = configure_logging(__name__)

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
    Manages downloading, caching, unzipping, and retrieving model weights.
    Reads a manifest file (models.yaml) to know about available models.
    """
    def __init__(self, manifest_path=None, cache_dir=None):
        if manifest_path is None:
            manifest_path = config_path('models.yaml')
        if cache_dir is None:
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
        if not expected_hash:
            return True # Skip verification if no hash is provided
        
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
        Gets the local path to a model file, handling downloads and unzipping.
        
        - If the URL points to a .zip, it downloads, verifies, and unzips it.
          It then returns the path to a specific file *inside* the unzipped folder.
        - If the URL is a local file://, it returns that path directly.
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
        
        # --- Handle local files directly ---
        if url.startswith("file://"):
            # Dev weights are written relative to the repo root, not to the
            # manifest's own directory (the manifest now ships inside the package).
            local_path = dev_root() / url[7:]
            if not local_path.exists():
                raise FileNotFoundError(f"Local model file not found: {local_path}")
            return str(local_path.resolve())

        # --- Handle remote files ---
        filename = os.path.basename(url)
        cached_zip_path = self.cache_dir / filename
        
        # Determine the final destination for unzipped content
        # e.g., 'segment_sax.zip' -> '~/.cache/cardio_form/segment_sax/'
        unzip_dir = self.cache_dir / filename.replace('.zip', '')

        # --- THE CRUCIAL UNZIP LOGIC ---
        # For nnU-Net, we need the path to the 'checkpoint_final.pth' inside the unzipped folder
        if model_name.startswith('segment_'):
            final_model_path = unzip_dir / 'fold_all' / 'checkpoint_final.pth'
        else: # For reconstruction models
            # Assumes the zip contains a single .pth file
            final_model_path = unzip_dir / os.path.basename(filename).replace('.zip', '.pth')

        # If the final, unzipped file already exists, we are done!
        if final_model_path.exists():
            logger.info(f"Found prepared model in cache: {final_model_path}")
            return str(final_model_path)

        # If the zip file exists but content is not unzipped, verify and unzip
        if cached_zip_path.exists():
            logger.info(f"Found model archive '{filename}' in cache. Verifying integrity...")
            try:
                if self._verify_hash(cached_zip_path, version_info.get('sha256')):
                    logger.info("Integrity check passed. Unzipping...")
                    with zipfile.ZipFile(cached_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(unzip_dir)
                    logger.info(f"Successfully unzipped to {unzip_dir}")
                    return str(final_model_path)
            except IOError as e:
                logger.warning(f"Integrity check failed: {e}. Re-downloading...")
                os.remove(cached_zip_path)

        # --- File needs to be downloaded ---
        logger.info(f"Downloading model '{filename}' from {url}...")
        try:
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, filename=str(cached_zip_path), reporthook=t.update_to)
            
            logger.info("Download complete. Verifying integrity...")
            if self._verify_hash(cached_zip_path, version_info.get('sha256')):
                logger.info("Integrity check passed. Unzipping...")
                with zipfile.ZipFile(cached_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(unzip_dir)
                logger.info(f"Successfully unzipped to {unzip_dir}")
                return str(final_model_path)
        except Exception as e:
            if cached_zip_path.exists():
                os.remove(cached_zip_path)
            raise RuntimeError(f"Failed to download, verify, or unzip model '{filename}'. Error: {e}")

# Create the singleton instance for use across the library
default_model_manager = ModelManager()