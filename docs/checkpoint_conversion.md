# PyTorch GANG PBR 检查点转换

`tools/convert_pytorch_checkpoint.py` 在 CPU 上把原版 GANG 的 `(capture_list, iteration)` 转成命名 NPZ，供 `render_learned_light.py` 和 `relight_envmap.py` 读取。转换阶段需要 NumPy 和支持 `weights_only` 参数的 PyTorch；渲染阶段仍只需要 Jittor。

## 支持范围

- 21 项 PBR capture：前 18 项为原版 GANG 模型状态，后 3 项为 albedo、metallic、roughness MLP。
- FP32 权重，offset 布局 `[N,K,3]`，无 feature bank、appearance embedding、额外距离/level 输入或 normal-detail MLP。
- RGB SG 灯光 `[M,10]`，一个 BRDF SG，六面正方形 cubemap。
- 保留锚点及 MLP 原值，不转置 Linear 权重，不降低精度，不导出优化器或训练统计量，不生成哈希。

不支持任意 PyTorch `state_dict`、其他 3DGS 项目的 `.pth`、18 项非 PBR/25K 检查点或 Jittor 训练检查点。不能把输出交给旧的 item-sequence 格式入口 `render_views.py --npz`。本工具是推理转换器，不用于跨框架续训。

## 输入文件

1. `chkpnt40000.pth`：模型 capture 和 iteration。
2. 同一迭代的 `Hybridlight40000.npy`：原版保存的灯光字典；也可使用无 object 的 NPZ。
3. `metadata.json`：训练配置与日志中恢复的 LOD 参数。`.pth` 没有完整保存它们，转换器不根据场景名称猜测。

元数据结构示例，数值仅用于展示格式，必须替换成对应模型的实际值：

```json
{
  "standard_dist": 10.0,
  "voxel_size": 0.01,
  "levels": 10,
  "init_level": 5,
  "fork": 2,
  "base_layer": 10,
  "dist2level": "round",
  "progressive": true,
  "extend": 1.1
}
```

`standard_dist` 从训练的场景/LOD 状态取得；不要用任意相机距离代替。`voxel_size`、`levels`、`init_level` 应使用初始化完成后的实际值，而非配置中的自动推断占位值。特征维度与 offset 数量直接从权重形状读取。

原版 capture 也未保存 `_extra_level`。本工具沿用历史推理约定填零，并在 NPZ 的 `extra_level_origin` 中说明。这不等同于恢复完整训练状态；若模型依赖非零自适应 extra levels，应先补充相应状态，不能宣称精确复现。

## 转换与渲染

只对自己训练或已经确认可信的文件使用 `--trust-pickle`。原版灯光 NPY 使用 pickle；该开关也允许 PyTorch 非受限反序列化，恶意文件可以执行代码。默认不自动回退到此模式。

```bash
python tools/convert_pytorch_checkpoint.py \
  --checkpoint /path/to/chkpnt40000.pth \
  --light /path/to/Hybridlight40000.npy \
  --metadata /path/to/metadata.json \
  --output /path/to/model_named.npz \
  --trust-pickle

python tools/render_learned_light.py \
  --model-npz /path/to/model_named.npz \
  --camera-json /path/to/cameras.json \
  --views '0 1 2' --res 4 --base-res 256 \
  --output-dir outputs/converted_model
```

`--base-res` 应匹配灯光 cubemap 的尺寸，`--res` 应匹配训练分辨率。相机文件不是权重的一部分，仍需另行提供。转换器不修改输入，输出路径存在时立即拒绝覆盖；父目录需要预先建立。

默认末三项顺序为原版源码的 `albedo-metallic-roughness`。某些历史分支使用 `albedo-roughness-metallic`，此时必须显式传入同名的 `--pbr-order`。两个标量 MLP 的形状相同，不能靠形状可靠判断，不会自动交换。

输出 NPZ 不包含 object 数组，可用 `np.load(path, allow_pickle=False)` 检查。公共命名加载器会把 LOD 元数据传给模型，并把两个标量 MLP 放到 Jittor 需要的位置。

## 验证状态

CPU 测试覆盖命名 NPZ 写入后重新读取、公共加载器参数映射、两种 PBR 顺序、元数据传递、错误形状、非有限值、精度检查、pickle 授权及拒绝覆盖。

```bash
python -m unittest discover -s tests -p test_checkpoint_conversion.py -v
```

此轮没有重新转换大型真实模型或启动 GPU。转换后的实际模型仍应先与 PyTorch 对比相同三个视角，确认参数顺序、LOD 和灯光配置正确。
