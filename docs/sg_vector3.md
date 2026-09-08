# SG 三分量归约：可选推理实现

本次更新从本地研究代码中单独提取 `dot3`、`squared_norm3`、`norm3`，用于 SG 点积与范数计算。每个 CUDA 线程处理一个三维向量，不使用原子累加，支持广播、FP32 和 FP64。原有精度转换、epsilon、clamp、平方根和 16 路光照求和保持不变。

默认仍使用 `native`。`vector3_cuda` 只支持 CUDA 推理，必须位于 `jt.no_grad()` 内；它不提供反向，梯度开启时立即报错。

## 使用

已有命名 NPZ 模型和相机清单时：

```bash
python tools/render_learned_light.py \
  --model-npz /path/to/model.npz \
  --camera-json /path/to/cameras.json \
  --views '0 1 2' --res 4 \
  --sg-reduce-backend vector3_cuda \
  --output-dir outputs/sg_vector3
```

沿用该入口的 ACES 显示变换，原始浮点结果另存为 NPY。此显示方式不等同于下述测速实验的 `[0,1]` 裁剪质量评估。

旧 item-sequence NPZ 入口也支持 `tools/render_views.py --is_pbr 1 --sg-reduce-backend vector3_cuda`，其余参数保持原样。未启用 PBR 时拒绝选择该后端。环境贴图替换工具不新增此开关，因为其既有实验关闭学习到的 SG。

Python 调用方式：

```python
light.sg_reduce_backend = 'vector3_cuda'
with jt.no_grad():
    result = render(camera, model, pipeline, background,
                    is_pbr=True, light=light, is_training=False)
```

## 本地版本与发布基线的差异

比较基线为本仓库 `05217fa`。按同名源码文件比较，本地研究目录在 `scene`、`gaussian_renderer`、`submodules/light_gaussian`、`utils` 中有 26 个文件与基线不同，不包含本地新增文件。

| 部分 | 本地额外改动 | 此次提交 |
| --- | --- | --- |
| SG 光照 | 三分量归约、同步回收策略、训练光照状态管理 | 仅三分量归约与显式开关 |
| 高斯生成与渲染入口 | 索引、训练诊断、损失与状态生命周期 | 不移植 |
| GaussianModel | 训练恢复、Adam、rotation、offset 布局 | 不移植 |
| 相机与数据加载 | 驻留策略、训练输入和状态适配 | 不移植 |
| 光栅器 | 反向实现、工作区及诊断变体 | 不移植 |
| cubemap 与工具函数 | 训练反向及数值、诊断支持 | 不移植 |

本次保留发布基线原有的 SG 中途同步与显存回收，因此它并非整份本地测速管线的复制。权重格式、光栅器、cubemap 和训练代码不随此次更新改变。历史实验、权重、数据及缓存不纳入提交。

## 已有测量与适用范围

2026-09-08，本地研究管线在同一 RTX 4090 上使用 PyTorch Garden 40K 权重、resolution 4、24 个测试视角和全锚点完成对照。每个进程预热两轮、计时五轮；三次独立运行共 360 帧。排除模型加载、编译、图片保存及下载，整帧前向末尾同步，不插入分段计时同步。

| 指标 | 原版 PyTorch | 本地 Jittor + vector3_cuda |
| --- | ---: | ---: |
| 完整前向平均耗时 | 87.830 ms | 59.629 ms |
| 完整前向 FPS | 11.386 | 16.770 |
| 完整前向 P95 | 93.852 ms | 64.927 ms |
| GPU 显存采样峰值 | 8.126 GiB | 4.829 GiB |
| 独立光栅器平均耗时 | 5.457 ms | 4.242 ms |
| 对 GT 的平均 PSNR | 28.342779 dB | 28.342778 dB |
| 对 GT 的平均 SSIM | 0.89733854 | 0.89733854 |
| 对 GT 的平均 LPIPS-Alex | 0.07340697 | 0.07340712 |

GT 使用同一组原图，以 BICUBIC 缩放至 1297×840；SSIM 使用 11×11、sigma=1.5 的高斯窗口和零填充，LPIPS 使用 Alex v0.1。独立光栅器计时不包含高斯生成与 PBR。显存为运行期间采样峰值，不能理解为单个算子工作区。

以上是本地完整研究管线的既有测量，**不是此次选择性移植后的公共仓库实测结果**，也不证明训练提速。原生 SG 与 vector3_cuda 的本地 24 视角对照已通过浮点 RGB 最大误差 ≤1e-4、裁剪后 PSNR ≥80 dB、SSIM ≥0.99999 的检查。跨框架输出仍有小范围差异，不能宣称逐值一致。

发布版本保留额外同步与旧加载路径，最终速度必须另行实测。AutoDL 已关闭，本次整理不重新开启实例，不把既有 GPU 证据作为移植后的端到端验证。

## 检查

无需 CUDA 的入口检查：

```bash
python -m unittest discover -s tests -p test_sg_inference_contract.py -v
```

有 Jittor/CUDA 环境时的算子检查：

```bash
python tests/test_sg_vector3_cuda.py
```

发布版本的端到端验证尚待执行：相同权重和相机分别运行 native 与 vector3_cuda，先比较前三个视角的浮点输出，再按 native-A、optimized-A、optimized-B、native-B 的顺序测量 24 视角。不得将加载、编译、图片保存计入前向耗时。若质量或显存不合适，切回默认 native。
