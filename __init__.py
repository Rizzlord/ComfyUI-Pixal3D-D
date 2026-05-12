import os
import sys

node_dir = os.path.dirname(os.path.abspath(__file__))


def _purge_external_module(module_name: str) -> None:
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file and not os.path.abspath(module_file).startswith(node_dir):
        stale_keys = [key for key in sys.modules if key == module_name or key.startswith(f"{module_name}.")]
        for key in stale_keys:
            sys.modules.pop(key, None)


if node_dir in sys.path:
    sys.path.remove(node_dir)
sys.path.insert(0, node_dir)

for local_module in ("pixal3d", "pixal3dpipeline", "pixal3dpipeline2stage"):
    _purge_external_module(local_module)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
