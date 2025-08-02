from .custom_nodes import WanVideoEnhancer_F2

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "CustomWanVideoEnhancer_F2": WanVideoEnhancer_F2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomWanVideoEnhancer_F2": "Custom Wan Video Enhancer (F2)"
}
