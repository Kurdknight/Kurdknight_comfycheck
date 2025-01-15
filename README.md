# KurdKnight ComfyUI System Check Node

A comprehensive system information node for ComfyUI that provides detailed information about your system, GPU, CUDA, and AI libraries configuration. Works on both Windows and Linux systems.
![alt text](https://github.com/Kurdknight/Kurdknight_comfycheck/blob/main/comfycheck.png?raw=true)

## Installation
1. Navigate to your ComfyUI custom nodes directory
2. Clone this repository or download and extract the zip file into a new directory named `Kurdknight_comfycheck`
3. Restart ComfyUI

That's it! No additional setup needed.

## Usage

### As a Node
1. In ComfyUI, find the node under "utils" category named "System Check"
2. The node has one input parameter:
   - `check_type`: Choose between "BASIC" or "DETAILED" information output
3. Connect the output to any node that accepts a STRING input, or use a "Print" node to view the information

### Quick Access
- Click the "System Check" button in the ComfyUI menu
- View information in a nice dialog with:
  - Color-coded status indicators
  - Save report functionality
  - Refresh button to update information

## Features

### Basic Check Includes:
- System Information:
  - OS Details (Windows/Linux)
  - Platform Information
  - Machine Architecture
- Python Environment:
  - Python Version and Implementation
  - Python Compiler
  - Python Path
  - Pip Version
  - Conda Status
- GPU Information:
  - GPU Name and Model
  - VRAM Usage and Capacity
- CUDA Configuration
- AI Libraries Status:
  - Triton
  - xFormers
  - Flash Attention
- SDPA Settings

### Detailed Check Adds:
- Python Detailed Configuration:
  - Build Information
  - Site Packages Locations
  - Python Path Details
- PyTorch Configuration:
  - Debug Mode
  - OpenMP/MKL Status
  - MAGMA Status
- Detailed CUDA Properties:
  - Device Capabilities
  - Memory Information
  - Thread Limits
- Additional AI Libraries:
  - Core ML/DL: numpy, scipy, pandas
  - Vision: opencv-python, pillow
  - Deep Learning: transformers, diffusers
  - Utilities: einops, safetensors
- Face and Vision Libraries:
  - Face Analysis: insightface, dlib, mediapipe
  - Face Recognition Libraries
  - Vision Processing Tools
- System Resources:
  - Memory Usage
  - CPU Information
  - Environment Variables

## Requirements
- ComfyUI
- torch
- psutil

## Cross-Platform Support
- Works on both Windows and Linux
- Automatically detects and shows relevant information for your system
- Proper handling of platform-specific features 
