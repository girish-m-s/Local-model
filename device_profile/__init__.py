"""Phase 1: CPU / memory DeviceProfile detection (no benchmarks)."""

from .models import DeviceProfile
from .detect import detect_device_profile

__all__ = ["DeviceProfile", "detect_device_profile"]
__version__ = "0.1.0"
