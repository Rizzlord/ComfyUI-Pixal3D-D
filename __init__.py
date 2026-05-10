import os
import sys

# Add the current directory to sys.path to allow absolute imports of pixal3d
node_dir = os.path.dirname(os.path.abspath(__file__))
if node_dir not in sys.path:
    sys.path.append(node_dir)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
