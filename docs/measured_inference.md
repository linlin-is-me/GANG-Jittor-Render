# 59.63 ms/帧对应的推理版本

本发布恢复了 2026-09-08 的实际运行源码，不再只移植 SG 算子。高斯生成、索引、模型状态适配、SG/PBR、cubemap、stable 光栅封装、同步回收策略与测速工具均来自当时上传包。逐文件比对结果和来源见 [source_inventory.json](benchmarks/20260908/source_inventory.json)。仅统一文本换行，没有重写这些运行模块或计算额外哈希。

模型类中原有训练方法保留，避免删改共享模块影响推理；不发布训练入口、手工梯度实验和候选光栅生成器。只支持 stable 光栅，不启动训练。权重、数据集、动态库与缓存不随代码分发。

## 固定条件与数据

- GPU：NVIDIA GeForce RTX 4090；Jittor 1.3.11.0。
- 同一 PyTorch Garden 40K 模型，597027 个锚点，全锚点，PBR + 16 SG，res4。
- 24 个固定测试视角，1297×840，共用 CPU 相机矩阵。
- 独立进程 A/B/C，每进程预热 2 轮，正式计时 5 轮，共 360 帧。
- 完整前向末尾设备同步；不插入分段同步，不开 profiler。排除加载、编译、预热、图片保存和下载。
- 优化开关为 `vector3_cuda`，offset 布局为 `flattened`；固定 stable 光栅。

| 路径 | 平均 ms/帧 | FPS | P95 ms |
| --- | ---: | ---: | ---: |
| PyTorch 完整前向 | 87.8299 | 11.3856 | 93.8521 |
| Jittor 完整前向 | 59.6289 | 16.7704 | 64.9270 |
| PyTorch 光栅器 | 5.4570 | 183.2517 | — |
| Jittor 光栅器 | 4.2417 | 235.7536 | — |
| Jittor RGB-only | 57.6099 | 17.3581 | 61.6418 |

完整统计见 [three_run_summary.json](benchmarks/20260908/three_run_summary.json)。同目录的 `full-{local,pytorch}-{A,B,C}.jsonl` 保留完整前向的逐帧原始记录。光栅器不包含高斯生成和 PBR；RGB-only 不等同于完整辅助输出。完整前向显存采样峰值分别为 PyTorch 8.126 GiB、Jittor 4.829 GiB。

历史同 GT 评价：PyTorch/Jittor 平均 PSNR 为 28.342778582/28.342778142 dB，SSIM 为 0.897338535/0.897338543，LPIPS-Alex 为 0.073406972/0.073407122。GT 使用 BICUBIC 缩放；SSIM 为 11×11 高斯窗口、sigma 1.5、零填充；LPIPS 为 Alex v0.1。并非逐像素完全相等。

## 使用实测入口

在 Linux/WSL 的 Jittor CUDA 环境中安装依赖并构建：

```bash
pip install -r requirements.txt
pip install nvidia-ml-py
GANG_CUDA_ARCHS=89 bash submodules/light_gaussian/_rebuild.sh stable
```

准备目录包含当时的 `model.npz` 和 `outputs.log`。前者为无 object 的命名 NPZ，后者提供原 checkpoint 未保存的 voxel_size、levels、init_level；这两个文件不发布。固定相机参数已包含在 [garden_res4.json](benchmarks/20260908/garden_res4.json)。只渲染和测速无需 GT 图片；对 GT 做质量评价仍需原数据集。

```bash
# 先检查前三个固定视角，保存图像和计时。
python tools/render_measured.py --weights /path/to/garden_40k \
  --output outputs/measured_smoke --smoke --rounds 1

# 完整前向：固定两轮预热、五轮计时。
python tools/render_measured.py --weights /path/to/garden_40k \
  --output outputs/measured_full_A

# 原生 RGB-only 路径。
python tools/render_measured.py --weights /path/to/garden_40k \
  --output outputs/measured_rgb_A --mode rgb
```

输出目录存在时拒绝覆盖。重新测量 B、C 时使用新的目录。包装入口仅指定参数并启动归档 worker，不修改计时和渲染逻辑。原 worker 也保留，可用 `--framework pytorch` 测原版，或 `--mode export-raster/raster` 测光栅。PyTorch 参考固定为 `wanglids/GANG` 的 `de4ca224f09b879285411b69b816a36710e0c87b`，其独立环境需安装原版 CUDA 扩展。

worker 的 `--weights` 指向目录。PyTorch 路径还需要可信的 `chkpnt40000.pth`、`Hybridlight40000.npy`，会按历史 albedo、roughness、metallic 顺序适配；它使用 pickle 加载，只能用于可信模型。请勿拿其他 capture 顺序代入该固定实验。

新添加的通用转换工具是独立便利入口，不是历史测速流程的一步。若重转本模型，必须显式选择 `--pbr-order albedo-roughness-metallic` 并核对转换结果；不能用转换器默认的原版顺序代替历史模型顺序。

## 验证边界

此次发布验证归档源码一致性、CPU 测试和入口参数，没有重新开启 AutoDL。59.6289 ms 是已有实测数据，不是本次整理后新测的一轮。重新编译、驱动、CUDA、Jittor 版本或 GPU 状态不同均可能改变速度。当前源码对应实测版本，并不意味着任意电脑必然得到相同耗时。

旧 `render_learned_light.py` 和 `relight_envmap.py` 继续提供可视化功能，已适配 optimizer-free restore 和扁平 offset，但它们的显示变换及相机入口不属于上述固定测速条件。复现性能应使用 `render_measured.py`。
