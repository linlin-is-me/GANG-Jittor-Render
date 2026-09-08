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

## 发布范围与验证

当前仓库已从上传归档恢复完整的实测推理模块，不再使用最初仅移植 SG 算子的发布方案。来源、构建方法、性能数据和复现入口见 [实测版本说明](measured_inference.md)。

使用 `tools/render_measured.py` 会显式选择优化后端和实测参数。旧可视化入口仍默认 native，需要手动添加开关。原生 SG 与 vector3_cuda 的历史 24 视角对照已通过浮点 RGB 最大误差 ≤1e-4、裁剪后 PSNR ≥80 dB、SSIM ≥0.99999。源码恢复不等同于在新环境中重新测速。

CPU 检查：

```bash
python -m unittest discover -s tests -p test_sg_inference_contract.py -v
```

Jittor CUDA 算子检查：

```bash
python tests/test_sg_vector3_cuda.py
```
