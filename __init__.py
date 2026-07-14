"""ComfyDoctor - ComfyUI environment diagnostics with one-click repair.

Loaded by ComfyUI at startup. Three things happen here and nothing else:
  1. the node mappings are exported,
  2. the HTTP API is registered on the running server,
  3. WEB_DIRECTORY points at the sidebar panel.

Registration is defensive on purpose. A diagnostic extension that itself fails
to import is a bad joke, so if anything goes wrong we degrade to "no panel" and
print how to run the CLI - which is the thing you'd want anyway if your
environment is broken enough to break us.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:
    from .comfydoctor.api import register

    if register():
        print("[ComfyDoctor] Ready - open the Doctor tab in the sidebar.")
    else:
        print("[ComfyDoctor] Loaded, but the API could not attach to the server.")
except Exception as e:  # pragma: no cover
    print(f"[ComfyDoctor] Panel unavailable ({type(e).__name__}: {e}).")
    print("[ComfyDoctor] The CLI still works:  python -m comfydoctor")
