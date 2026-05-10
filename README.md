# ComfyUI-Pixal3D-D

A modular ComfyUI integration for **Pixal3D-D**, a state-of-the-art 3D generation pipeline featuring a hybrid Dense-to-Sparse architecture for high-quality, 1024-resolution 3D assets.

![3D Preview](https://github.com/TencentARC/Pixal3D-D/raw/main/assets/teaser.png)

## Features
- **Hybrid 2-Stage Pipeline**: Combines a Dense model for stable global topology with a Sparse DiT for high-frequency details.
- **1024 Resolution Support**: Native support for 1024-resolution refinement.
- **Refine Mode Optimization**: Includes a memory-efficient `refine` mode that runs on **16GB VRAM** cards by leveraging cross-resolution conditioning.
- **Mesh Refinement**: Enhance existing `.glb` or `.obj` meshes by re-voxelizing them through the 1024-sparse model.
- **Advanced Post-Processing**: Integrated `meshlib` (mrmeshpy) for high-performance decimation and automatic interior/floater removal.
- **VRAM/RAM Management**: Built-in "Keep Model Loaded" toggle and explicit unloading to free system resources after generation.

## Installation

### 1. Clone the repository
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YourUsername/ComfyUI-Pixal3D-D
```

### 2. Install Dependencies
Ensure you have the required libraries installed in your ComfyUI environment:
```bash
pip install trimesh mrmeshpy numpy torch torchvision scipy scikit-image
```

### 3. Download Models
Models are **not** automatically downloaded. You must place them manually:

1. Create a folder: `ComfyUI/models/pixal3d-d/`
2. Download the model directories from [TencentARC/Pixal3D-D on Hugging Face](https://huggingface.co/TencentARC/Pixal3D-D/tree/main).
3. Place the `dense`, `sparse512`, and `sparse1024` folders inside the `pixal3d-d` directory.

### 🚀 Cleaning up disk space
If you have already moved your weights to the central `ComfyUI/models/pixal3d-d/` folder, you can delete the redundant weight files in the custom node folder to save space. 

**WARNING:** Do NOT delete the entire `models` folder, as it contains the Python source code for the model architectures. 

Only delete the weights subdirectory:
`path/to/ComfyUI/custom_nodes/ComfyUI-Pixal3D-D/pixal3d/models/Pixal3D-D/`

**Structure:**
```text
ComfyUI/models/pixal3d-d/
├── dense/
│   ├── dit/
│   ├── vae/
│   └── ...
├── sparse512/
│   ├── dit/
│   └── ...
└── sparse1024/
    ├── dit/
    └── ...
```

## Node Descriptions

### 🧩 Pixal3D-D Model Loader
Loads the hybrid pipeline. 
- **keep_model_loaded**: If set to `False`, the system will completely purge VRAM and RAM after the generation is finished.

### 🧩 Pixal3D-D Generate Dense Index
The first stage of the pipeline.
- **optimize_mesh_scale**: Automatically fits the object into the 3D grid for maximum sharpness.
- **dense_threshold**: Controls the initial surface extraction.

### 🧩 Pixal3D-D Refine Sparse Latents
The second and third stages (512 & 1024).
- **mode_1024**: Use `refine` for 16GB cards. Use `full` for 24GB+ cards if you want maximum detail.
- **remove_interior**: Uses `meshlib` to remove all internal shells and keep only the largest outer component.

### 🧩 Pixal3D-D Refine Existing Mesh
Takes an existing `TRIMESH` and runs it through the 1024-resolution refinement model to "enhance" it with Pixal3D details.

## Credits
This project is an integration of the research by **Tencent ARC**.
- Original Repo: [Pixal3D-D](https://github.com/TencentARC/Pixal3D-D)
- Architecture inspired by the modularity of Direct3D-S2.
