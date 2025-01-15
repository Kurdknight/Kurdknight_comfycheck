import torch
import psutil
import os
import platform
import sys
import time
import importlib.util
import subprocess
import pkg_resources
import site
import json
from datetime import datetime
from pathlib import Path

class SystemCheckNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "check_type": (["BASIC", "DETAILED"], {"default": "DETAILED"}),
            "save_report": ("BOOLEAN", {"default": False}),
        }}
    
    RETURN_TYPES = ("STRING", "SYSTEM_INFO")
    RETURN_NAMES = ("report", "data")
    FUNCTION = "check_system"
    CATEGORY = "utils"

    def __init__(self):
        # Save reports in ComfyUI's output directory
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                     "output", "system_reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def _check_module_exists(self, module_name):
        """Helper function to check if a Python module exists"""
        return importlib.util.find_spec(module_name) is not None

    def _get_module_version(self, module_name):
        """Helper function to safely get module version"""
        try:
            module = importlib.import_module(module_name)
            return getattr(module, '__version__', 'Version not found')
        except:
            return 'Not installed'

    def _get_pip_version(self):
        """Helper function to get pip version"""
        try:
            return pkg_resources.get_distribution('pip').version
        except:
            return 'Not found'

    def _get_conda_info(self):
        """Helper function to get conda information"""
        try:
            result = subprocess.run(['conda', 'info'], capture_output=True, text=True)
            if result.returncode == 0:
                return "Conda is installed"
            return "Conda not found"
        except:
            return "Conda not found"

    def _get_python_path(self):
        """Helper function to get Python executable path"""
        return sys.executable

    def save_report_to_file(self, report):
        """Save the system report to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"system_report_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        
        return filepath

    def parse_info_to_dict(self, info_lines):
        """Convert info lines to a structured dictionary"""
        result = {}
        current_section = None
        current_data = {}
        
        for line in info_lines:
            if line.startswith("=== ") and line.endswith(" ==="):
                if current_section:
                    result[current_section] = current_data
                current_section = line.strip("= ").strip()
                current_data = {}
            elif ":" in line and current_section:
                key, value = line.split(":", 1)
                current_data[key.strip()] = value.strip()
        
        if current_section:
            result[current_section] = current_data
            
        return result

    def check_system(self, check_type, save_report=False):
        info = []
        
        # System Information
        info.append("=== System Information ===")
        info.append(f"OS: {platform.system()} {platform.version()}")
        info.append(f"OS Platform: {platform.platform()}")
        info.append(f"Machine: {platform.machine()}")
        if platform.system() == 'Windows':
            info.append(f"Windows Edition: {platform.win32_edition()}")
        
        # Python Environment
        info.append("\n=== Python Environment ===")
        info.append(f"Python Version: {sys.version}")
        info.append(f"Python Implementation: {platform.python_implementation()}")
        info.append(f"Python Compiler: {platform.python_compiler()}")
        info.append(f"Python Path: {self._get_python_path()}")
        info.append(f"Pip Version: {self._get_pip_version()}")
        info.append(f"Conda Status: {self._get_conda_info()}")
        
        # GPU Information
        info.append("\n=== GPU Information ===")
        info.append(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'No GPU detected'}")
        if torch.cuda.is_available():
            info.append(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
            info.append(f"VRAM Used: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            info.append(f"VRAM Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        
        # CUDA Configuration
        info.append("\n=== CUDA Configuration ===")
        info.append(f"CUDA Available: {torch.cuda.is_available()}")
        info.append(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        if torch.cuda.is_available():
            info.append(f"Current Device: {torch.cuda.current_device()}")
            info.append(f"FP16 Available: {torch.cuda.is_bf16_supported()}")
        
        # AI Libraries Check
        info.append("\n=== AI Libraries Status ===")
        # Triton Check
        triton_available = self._check_module_exists('triton')
        info.append(f"Triton Available: {triton_available}")
        if triton_available:
            info.append(f"Triton Version: {self._get_module_version('triton')}")
        
        # xFormers Check
        xformers_available = self._check_module_exists('xformers')
        info.append(f"xFormers Available: {xformers_available}")
        if xformers_available:
            info.append(f"xFormers Version: {self._get_module_version('xformers')}")
            try:
                import xformers
                info.append(f"xFormers CUDA Available: {xformers.ops.memory_efficient_attention.available_backends()}")
            except:
                info.append("xFormers CUDA Status: Error checking backends")

        # Flash Attention Check
        flash_attn_available = self._check_module_exists('flash_attn')
        info.append(f"Flash Attention Available: {flash_attn_available}")
        if flash_attn_available:
            info.append(f"Flash Attention Version: {self._get_module_version('flash_attn')}")
        
        # SDPA Information
        info.append("\n=== SDPA Configuration ===")
        info.append(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
        info.append(f"Math SDP enabled: {torch.backends.cuda.math_sdp_enabled()}")
        info.append(f"Mem efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        
        if check_type == "DETAILED":
            # Python Detailed Configuration
            info.append("\n=== Python Detailed Configuration ===")
            info.append(f"Python Build: {platform.python_build()}")
            info.append(f"Python Revision: {sys.version_info}")
            info.append(f"Site Packages Location: {site.getsitepackages()[0] if 'site' in sys.modules else 'Not available'}")
            info.append(f"User Site Packages: {site.getusersitepackages() if 'site' in sys.modules else 'Not available'}")
            info.append("\nPython Path (sys.path):")
            for path in sys.path:
                info.append(f"  - {path}")
            
            # PyTorch Detailed Configuration
            info.append("\n=== PyTorch Detailed Configuration ===")
            info.append(f"PyTorch Debug Mode: {torch.version.debug}")
            info.append(f"OpenMP Enabled: {torch.backends.openmp.is_available()}")
            info.append(f"MKL Enabled: {torch.backends.mkl.is_available()}")
            if torch.cuda.is_available():
                info.append(f"MAGMA Available: {torch.cuda.get_device_capability()}")
            
            # CUDA Detailed Configuration
            if torch.cuda.is_available():
                info.append("\n=== CUDA Detailed Configuration ===")
                device = torch.cuda.current_device()
                capabilities = torch.cuda.get_device_capability(device)
                info.append(f"CUDA Device Capability: {capabilities[0]}.{capabilities[1]}")
                info.append(f"CUDA Device Count: {torch.cuda.device_count()}")
                info.append(f"CUDA Device Properties:")
                props = torch.cuda.get_device_properties(device)
                info.append(f"  - Name: {props.name}")
                info.append(f"  - Total Memory: {props.total_memory / 1024**2:.0f}MB")
                info.append(f"  - Multi Processor Count: {props.multi_processor_count}")
                
                # Safely check for additional properties
                try:
                    info.append(f"  - Max Threads Per Block: {props.max_threads_per_block}")
                except AttributeError:
                    info.append("  - Max Threads Per Block: Not available")
                    
                try:
                    info.append(f"  - Max Threads Per MP: {props.max_threads_per_multi_processor}")
                except AttributeError:
                    info.append("  - Max Threads Per MP: Not available")
                    
                # Add more basic CUDA properties with error handling
                try:
                    info.append(f"  - Max Shared Memory Per Block: {props.max_shared_memory_per_block / 1024:.0f}KB")
                except AttributeError:
                    info.append("  - Max Shared Memory Per Block: Not available")
                
                try:
                    info.append(f"  - Clock Rate: {props.clock_rate / 1000:.0f}MHz")
                except AttributeError:
                    info.append("  - Clock Rate: Not available")
                
                try:
                    info.append(f"  - Memory Clock Rate: {props.memory_clock_rate / 1000:.0f}MHz")
                except AttributeError:
                    info.append("  - Memory Clock Rate: Not available")
                
                try:
                    info.append(f"  - Memory Bus Width: {props.memory_bus_width}bit")
                except AttributeError:
                    info.append("  - Memory Bus Width: Not available")
            
            # cuDNN Configuration
            if torch.cuda.is_available():
                info.append("\n=== cuDNN Configuration ===")
                info.append(f"cuDNN Version: {torch.backends.cudnn.version()}")
                info.append(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
                info.append(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
                info.append(f"cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
            
            # Additional AI Libraries
            info.append("\n=== Additional AI Libraries ===")
            # Check for common ML/DL libraries
            libraries = [
                'transformers', 'diffusers', 'accelerate', 'bitsandbytes',
                'einops', 'safetensors', 'pytorch_lightning', 'timm',
                'numpy', 'scipy', 'pandas', 'pillow', 'opencv-python',
                'matplotlib', 'scikit-learn', 'tqdm'
            ]
            for lib in libraries:
                info.append(f"{lib}: {self._get_module_version(lib)}")
            
            # Face and Vision Libraries
            info.append("\n=== Face and Vision Libraries ===")
            face_libraries = [
                'insightface', 'dlib', 'mediapipe', 'face-recognition',
                'deepface', 'facial-recognition', 'retinaface',
                'face-alignment', 'facenet-pytorch'
            ]
            for lib in face_libraries:
                info.append(f"{lib}: {self._get_module_version(lib)}")
            
            # Additional Vision Libraries
            info.append("\n=== Additional Vision Libraries ===")
            vision_libraries = [
                'torchvision', 'albumentations', 'imgaug', 'kornia',
                'detectron2', 'ultralytics', 'supervision'
            ]
            for lib in vision_libraries:
                info.append(f"{lib}: {self._get_module_version(lib)}")

            # System Memory
            info.append("\n=== System Memory Details ===")
            mem = psutil.virtual_memory()
            info.append(f"Total: {mem.total/1024**3:.2f}GB")
            info.append(f"Available: {mem.available/1024**3:.2f}GB")
            info.append(f"Used: {mem.used/1024**3:.2f}GB")
            info.append(f"Percentage: {mem.percent}%")
            
            # Environment Variables
            info.append("\n=== AI-Related Environment Variables ===")
            env_vars = [
                'PYTHONPATH',
                'PYTHON_HOME',
                'CONDA_PREFIX',
                'VIRTUAL_ENV',
                'PYTORCH_CUDA_ALLOC_CONF',
                'PYTORCH_ENABLE_FLASH_SDP',
                'PYTORCH_ENABLE_MATH_SDP',
                'PYTORCH_ENABLE_MEM_EFFICIENT_SDP',
                'CUDA_HOME',
                'CUDA_PATH',
                'CUDA_VISIBLE_DEVICES',
                'CUDA_LAUNCH_BLOCKING',
                'TORCH_CUDA_ARCH_LIST',
                'TORCH_EXTENSIONS_DIR',
                'XFORMERS_ENABLE_VERSION_CHECK',
                'BITSANDBYTES_CUDA_VERSION',
                'HF_HOME',
                'TRANSFORMERS_CACHE'
            ]
            for var in env_vars:
                info.append(f"{var}: {os.environ.get(var, 'Not set')}")
            
            # CPU Information
            info.append("\n=== CPU Information ===")
            info.append(f"CPU Cores: {psutil.cpu_count()}")
            info.append(f"Physical Cores: {psutil.cpu_count(logical=False)}")
            info.append(f"CPU Usage: {psutil.cpu_percent()}%")
            if hasattr(psutil, "cpu_freq"):
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    info.append(f"CPU Frequency: Current={cpu_freq.current:.0f}MHz, Min={cpu_freq.min:.0f}MHz, Max={cpu_freq.max:.0f}MHz")
        
        # Join all information into a single string
        report = "\n".join(info)
        
        # Save report if requested
        if save_report:
            self.save_report_to_file(report)
        
        # Convert to structured data
        data = self.parse_info_to_dict(info)
        
        return (report, data)

# Web UI integration
def get_system_check_js():
    """Get the path to the JavaScript file"""
    js_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "js", "system_check.js")
    return js_path

def api_route(server):
    """Register API routes for the web UI"""
    @server.route("/kurdknight/systemcheck/refresh", methods=["GET"])
    def refresh_system_check():
        node = SystemCheckNode()
        report = node.check_system("DETAILED")[0]
        return {"message": report}
    
    @server.route("/kurdknight/systemcheck/save", methods=["GET"])
    def save_system_check():
        node = SystemCheckNode()
        report = node.check_system("DETAILED", save_report=True)[0]
        return {"message": "Report saved successfully"}

NODE_CLASS_MAPPINGS = {
    "SystemCheck": SystemCheckNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SystemCheck": "System Check"
}

WEB_DIRECTORY = "./js" 