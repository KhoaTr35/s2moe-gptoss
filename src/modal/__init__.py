# Modal modules for cloud GPU training

from .modal_setup import create_modal_app, create_volume
from .modal_entry import train_remote, evaluate_remote, infer_remote
from .modal_utils import get_secrets, setup_modal_environment

__all__ = [
    "create_modal_app",
    "create_volume",
    "train_remote",
    "evaluate_remote",
    "infer_remote",
    "get_secrets",
    "setup_modal_environment",
]
