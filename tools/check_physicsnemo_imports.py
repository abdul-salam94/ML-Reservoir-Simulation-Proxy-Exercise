"""Test which physicsnemo import paths work with the installed version."""
import physicsnemo
print(f"physicsnemo version: {getattr(physicsnemo, '__version__', 'unknown')}")
print()

# Test the EXACT imports that xmgn train.py / inference.py use
from physicsnemo.distributed import DistributedManager
print("DistributedManager:  OK (from physicsnemo.distributed)")

# Old vs new logger path
old_log_ok = False
new_log_ok = False
try:
    from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper, LaunchLogger
    print("OLD path  physicsnemo.utils.logging:           OK")
    old_log_ok = True
except ImportError as e:
    print(f"OLD path  physicsnemo.utils.logging:           FAIL ({e})")

try:
    from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper, LaunchLogger
    print("NEW path  physicsnemo.launch.logging:          OK")
    new_log_ok = True
except ImportError as e:
    print(f"NEW path  physicsnemo.launch.logging:          FAIL ({e})")

# Old vs new mlflow path
try:
    from physicsnemo.utils.logging.mlflow import initialize_mlflow
    print("OLD path  physicsnemo.utils.logging.mlflow:    OK")
except ImportError as e:
    print(f"OLD path  physicsnemo.utils.logging.mlflow:    FAIL")

try:
    from physicsnemo.launch.logging.mlflow import initialize_mlflow
    print("NEW path  physicsnemo.launch.logging.mlflow:   OK")
except ImportError as e:
    print(f"NEW path  physicsnemo.launch.logging.mlflow:   FAIL")

# Old vs new checkpoint path
try:
    from physicsnemo.utils import load_checkpoint, save_checkpoint
    print("OLD path  physicsnemo.utils.{load,save}_checkpoint:    OK")
except ImportError as e:
    print(f"OLD path  physicsnemo.utils.{{load,save}}_checkpoint:    FAIL")

try:
    from physicsnemo.launch.utils.checkpoint import load_checkpoint, save_checkpoint
    print("NEW path  physicsnemo.launch.utils.checkpoint:         OK")
except ImportError as e:
    print(f"NEW path  physicsnemo.launch.utils.checkpoint:         FAIL")

from physicsnemo.models.meshgraphnet import MeshGraphNet
print("MeshGraphNet (path is same in both versions): OK")
