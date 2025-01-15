from .system_check import SystemCheckNode, api_route
from .system_viz import SystemVizNode

NODE_CLASS_MAPPINGS = {
    "SystemCheck": SystemCheckNode,
    "SystemViz": SystemVizNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SystemCheck": "System Check",
    "SystemViz": "System Visualization"
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

def register_extension(server):
    api_route(server) 