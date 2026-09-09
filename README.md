# GANG-Jittor-Render: Scene-Level Relightable Neural Gaussian Rendering with Jittor

**English** | [简体中文](README_CN.md)

GANG-Jittor-Render implements the complete GANG inference pipeline using the Jittor deep learning framework and the JGaussian rendering library, covering checkpoint conversion, scene rendering, and relighting. Once the weights are converted, rendering and relighting run independently in Jittor without PyTorch or nvdiffrast.

We optimize lighting computation for complex scenes. On the same NVIDIA RTX 4090, the Jittor renderer maintains average reconstruction quality close to the original PyTorch implementation while reducing full-forward latency from **87.83 ms/frame to 59.63 ms/frame**, achieving **16.77 FPS** and reducing the sampled peak GPU memory usage by approximately **40.6%**. Test conditions and reproduction instructions are provided below.

[Gallery](#gallery) · [Quality & Performance](#results) · [Quick Start](#quick-start) · [Implementation](#implementation) · [Scope](#scope) · [References](#references)

<a id="gallery"></a>

![Garden DSC08066 relit with the learned environment and 14 external HDR environment maps](assets/envmap_relighting_garden_dsc08066_3x5.png)

Geometry, materials, and camera remain fixed; only the environment map changes. The first panel uses the learned environment, and the others use 14 TensoIR HDR maps. Each panel shows the environment map used in its lower-right corner. This demonstration disables SG and point lights and uses a fixed linear-clamp display transform. Differences in HDR intensity may cause highlight clipping.

<details>
<summary>View all 24 rendered test views</summary>

The overview below shows scene structure, material details, and reconstruction results from different viewpoints in Garden.

![Overview of the 24 Garden test views](assets/garden_multiview_24views.jpg)

</details>

## Part 1 From 3DGS to Scene-Level Relighting

### 1.1 Challenges in Scene-Level Relighting

3D Gaussian Splatting (3DGS) excels at real-time novel-view synthesis, but its direct representation of view-dependent color makes it difficult to separate geometry, materials, and illumination. The Jittor-based JGaussian library has supported advances in object and human relighting. For real scenes with diverse materials and complex structures, however, storing material parameters per Gaussian still incurs substantial storage costs and can compromise local material consistency.

GANG (Geometrically-Aligned Neural Gaussians) combines an anchor-based neural Gaussian representation with physically based rendering to reconstruct complex real scenes for high-quality relighting. It supports editing scene materials and illumination. The paper appeared in IEEE TVCG in 2026.

![Relighting comparison across methods in real scenes](assets/图%202真实场景下不同算法的重光照对照.png)

### 1.2 Core Architecture of the GANG Rendering Pipeline

GANG organizes a scene around anchors. Lightweight MLPs decode compact features into geometry and PBR material attributes for multiple neural Gaussians. Gaussians within an anchor share features and decoders to maintain local material consistency. In the paper's efficiency experiments across three datasets, average model storage was approximately 1/25 that of R3DG.

During rendering, the decoders use viewing direction and distance to generate Gaussian attributes. The lighting module computes colors from material properties and illumination, and rasterization produces the final image.

![Overview of the GANG architecture](assets/图%203%20GANG%20架构整体流程图.svg)

GANG combines cubemap environment lighting with spherical Gaussian (SG) local lighting. The environment map provides overall illumination, while SGs describe spatially varying local illumination, adding directional highlights and shading variations. Both lighting components use a Cook–Torrance material model to compute diffuse and specular reflection, reproducing the appearance of different materials.

![Hybrid lighting with multiple spherical Gaussians](assets/图%204多球面高斯混合光照示意图.png)

<a id="implementation"></a>

## Part 2 Implementation and Optimization in Jittor

### 2.1 From Pretrained Models to Jittor Inference

This project reuses JGaussian's CUDA rasterizer framework, environment lighting model, and BRDF lookup-table integration (FG_LUT), and incorporates the material and lighting components required by GANG. Inference calls the CUDA forward pass directly without constructing a Tape or retaining backward-pass buffers. Intermediate tensors are released between stages to reduce memory usage during multi-view rendering.

The repository includes a `.pth → .npz` converter for original GANG PBR models and their lighting states, allowing pretrained weights to be reused without retraining. Supported structures and LOD metadata requirements are described in the [checkpoint conversion guide](docs/checkpoint_conversion.md).

### 2.2 Anchor-Based PBR Material Decoding

The Jittor implementation includes geometry and material decoders for GANG's anchor-based representation. They combine anchor features with viewing direction, distance, and other inputs to predict Gaussian geometry and material parameters such as albedo, roughness, and metallicity. Their outputs feed directly into lighting computation and rasterization for PBR rendering.

### 2.3 Hybrid Lighting and SG Optimization

The Jittor implementation integrates environment-light precomputation with local SG lighting. It generates diffuse irradiance maps and specular prefiltered mipmaps from the cubemap, computes spatially varying SG contributions, and combines these with material parameters for shading.

Dedicated CUDA operators accelerate three-component dot products and norms in SG lighting while preserving the original FP32/FP64 precision, epsilon values, and clamps.

The optimized `vector3_cuda` backend is inference-only. General visualization entry points still default to `native` and require explicit opt-in. The measured entry point, `tools/render_measured.py`, selects `vector3_cuda` and the corresponding settings. See the [SG three-component reduction guide](docs/sg_vector3.md).

### 2.4 Native Jittor Environment Texture Sampling and Relighting

Original GANG uses nvdiffrast for environment texture queries. We implement the required texture sampling in Jittor, supporting 2D textures, interpolation across cubemap faces, and mipmap queries. This preserves environment-light continuity and supports specular reflection at different roughness levels.

The implementation supports both the learned environment and replacement HDR maps, showing how scene materials and shading respond to different illumination.

<a id="results"></a>

## Part 3 Rendering Quality and Inference Performance

Both renderers were tested sequentially on the same NVIDIA RTX 4090, loading identical weights and rendering the same 24 fixed views at the same resolution. Each implementation ran in three independent processes. Each process performed two warm-up rounds and five timed rounds, yielding 360 timed frames per implementation. Full-forward timing synchronizes the device at the end of each frame without synchronization between stages. Model loading, compilation, warm-up, and image saving are excluded.

The test used the Garden 40K model with all 597027 anchors, PBR, and 16 SGs, at `res=4` and an output size of 1297×840. The Jittor version was 1.3.11.0, with the `vector3_cuda` backend and `flattened` offset layout.

### 3.1 Rendering Quality

The following metrics are averaged over 24 views against ground-truth (GT) images resized using the same procedure:

| Metric | Original PyTorch | Jittor Renderer |
| --- | ---: | ---: |
| PSNR ↑ | 28.342778582 dB | 28.342778142 dB |
| SSIM ↑ | 0.897338535 | 0.897338543 |
| LPIPS ↓ | 0.073406972 | 0.073407122 |

PSNR measures pixel error and SSIM measures structural similarity; higher is better for both. LPIPS measures perceptual difference, so lower is better. These metrics compare each renderer with GT, rather than directly comparing the two renderers' outputs. The LPIPS configuration is documented in the [measurement guide](docs/measured_inference.md).

### 3.2 Inference Performance

The full forward pass includes Gaussian generation, material and lighting computation, and rasterization.

| Metric | Original PyTorch | Jittor Renderer | Improvement |
| --- | ---: | ---: | ---: |
| Full-forward latency | 87.83 ms/frame | 59.63 ms/frame | 32.1% lower |
| Inference speed | 11.39 FPS | 16.77 FPS | 47.3% higher |
| Sampled peak GPU memory | 8.126 GiB | 4.829 GiB | 40.6% lower |

Full-forward latency is the computation time required to generate one frame. FPS is the number of frames generated per second. Sampled peak GPU memory is the highest memory usage recorded during sampling.

These results apply to the specified test environment. Full statistics, per-frame records, standalone rasterizer and RGB-only results, and reproduction conditions are available in the [measurement guide](docs/measured_inference.md) and [raw statistics](docs/benchmarks/20260908/three_run_summary.json). Performance may vary with GPU, driver, and software environment.

<a id="quick-start"></a>

## Quick Start

Run all commands from the repository root. Model weights, scene data, third-party HDR maps, and precompiled libraries are not distributed with the source. The detailed guides linked below are currently in Chinese.

### Installation and Build

Use a Jittor CUDA environment on Linux or WSL with the CUDA Toolkit, a C++ compiler, and CMake. The measured environment used an RTX 4090 with SM 89. Select the appropriate architecture for other GPUs.

```bash
pip install -r requirements.txt
pip install nvidia-ml-py
GANG_CUDA_ARCHS=89 bash submodules/light_gaussian/_rebuild.sh stable
```

### Reproduce the Measured Inference Path

The weights directory must contain the measured `model.npz` and `outputs.log`; the latter supplies the model's LOD parameters. The fixed camera list is included in the repository. Rendering and timing do not require GT images; quality evaluation against GT does.

```bash
# Check three fixed views first.
python tools/render_measured.py --weights /path/to/garden_40k \
  --output outputs/measured_smoke --smoke --rounds 1

# Full forward: two warm-up rounds and five timed rounds.
python tools/render_measured.py --weights /path/to/garden_40k \
  --output outputs/measured_full_A
```

Existing output directories are rejected. Use new B and C directories for repeated tests. This entry point reproduces a fixed experiment and does not accept arbitrary model structures or checkpoint formats. See the [measurement guide](docs/measured_inference.md) for the PyTorch reference environment and material parameter ordering in the historical model.

### Convert Weights and Render

Prepare the model, lighting state, and LOD metadata according to the [checkpoint conversion guide](docs/checkpoint_conversion.md), and verify the material parameter order. The converter does not support arbitrary PyTorch models or cross-framework training resumption.

Run conversion in a separate environment with PyTorch and NumPy installed. After producing the NPZ, switch to the Jittor inference environment to render. The installation steps above configure only the Jittor inference environment.

```bash
# Run in the PyTorch conversion environment.
python tools/convert_pytorch_checkpoint.py \
  --checkpoint /path/to/chkpnt40000.pth \
  --light /path/to/Hybridlight40000.npy \
  --metadata /path/to/metadata.json \
  --output /path/to/model_named.npz \
  --trust-pickle

# Switch to the Jittor inference environment before rendering.
python tools/render_learned_light.py \
  --model-npz /path/to/model_named.npz \
  --camera-json /path/to/cameras.json \
  --views '0 1 2' --res 4 --base-res 256 \
  --sg-reduce-backend vector3_cuda \
  --output-dir outputs/learned_light
```

Use `--trust-pickle` only with trusted files. Supply the camera JSON separately; `--views` selects indices in its camera array. Match `--res` to the model's training resolution and `--base-res` to the lighting cubemap size. This visualization entry point uses an ACES display transform and does not replace the measured entry point.

### Replace the Environment Map

This example keeps geometry, materials, and camera fixed, disables SG and point lights, and replaces only the cubemap. Third-party HDR maps are available from the [TensoIR envmap archive](https://drive.google.com/file/d/10WLc4zk2idf4xGb6nPL43OXTTHvAXSR3/view) referenced by original GANG.

```bash
python tools/relight_envmap.py \
  --model-npz /path/to/model_named.npz \
  --camera-json /path/to/cameras.json \
  --views '120' --comparison-view 120 \
  --res 4 --base-res 256 \
  --hdr-path /path/to/night.hdr \
  --replacement-label night \
  --scene-output paper_linear_clamp \
  --output-dir outputs/envmap_night
```

Index 120 corresponds to DSC08066 in the homepage experiment. Verify the index if you use a different camera list. The script saves images, linear HDR arrays, A/B comparisons, and parameter records. For multiple HDR maps, use a separate process and output directory for each map to avoid keeping multiple environment textures in GPU memory simultaneously.

<a id="scope"></a>

## Scope

- This repository provides inference and relighting tools, not a complete training entry point.
- Named NPZ and legacy item-sequence NPZ formats are not interchangeable. See the [conversion guide](docs/checkpoint_conversion.md) for missing-state and `_extra_level` handling.
- The standard SG path does not compute occlusion visibility. Point lights and shadows are experimental branches disabled by default and are outside the quality and performance results on this page.
- Public PBR entry points preserve the original model's normal convention. Replacing the environment map does not change model materials or normals.
- The current license permits non-commercial research and evaluation only. See [LICENSE](LICENSE) for the terms.

<a id="references"></a>

## References

GANG: D. Li, S.-S. Huang, H. Fu, and H. Huang, *GANG: Geometrically-Aligned Neural Gaussians for Efficient and Realistic Relighting*, IEEE TVCG, 2026. DOI: [10.1109/TVCG.2026.3687668](https://doi.org/10.1109/TVCG.2026.3687668). Original project: [wanglids/GANG](https://github.com/wanglids/GANG).

This work reuses the rasterizer framework and PBR components of the Jittor-based [JGaussian](https://github.com/IGLICT/JGaussian) library.

GANG-Jittor-Render and JGaussian are built on the [Jittor deep learning framework](https://cg.cs.tsinghua.edu.cn/jittor/). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party attribution and licenses.
