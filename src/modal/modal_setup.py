"""
Modal.com setup utilities.
Contains container image, volume, and app definitions.
"""
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    print("Warning: Modal not installed. Modal features will be disabled.")

from configs.modal_config import ModalConfig


def create_modal_app(config: ModalConfig = None):
    """
    Create Modal app with configured settings.
    
    Args:
        config: Modal configuration
        
    Returns:
        Modal App instance
    """
    if not MODAL_AVAILABLE:
        raise ImportError("Modal is not installed. Run: pip install modal")
    
    if config is None:
        config = ModalConfig()
    
    app = modal.App(name=config.app_name)
    return app


def create_volume(config: ModalConfig = None):
    """
    Create Modal volume for persistent storage.
    
    Args:
        config: Modal configuration
        
    Returns:
        Modal Volume instance
    """
    if not MODAL_AVAILABLE:
        raise ImportError("Modal is not installed. Run: pip install modal")
    
    if config is None:
        config = ModalConfig()
    
    volume = modal.Volume.from_name(config.volume_name, create_if_missing=True)
    return volume


def create_image(config: ModalConfig = None):
    """
    Create Modal container image with required dependencies.
    
    Args:
        config: Modal configuration
        
    Returns:
        Modal Image instance
    """
    if not MODAL_AVAILABLE:
        raise ImportError("Modal is not installed. Run: pip install modal")
    
    if config is None:
        config = ModalConfig()
    
    image = (
        modal.Image.debian_slim(python_version=config.image_python_version)
        .pip_install(*config.pip_packages)
    )
    
    return image


def get_gpu_config(config: ModalConfig = None) -> str:
    """
    Get GPU configuration string for Modal.
    
    Args:
        config: Modal configuration
        
    Returns:
        GPU config string (e.g., "A100:1")
    """
    if config is None:
        config = ModalConfig()
    
    return config.gpu_config
