# Android 本地生成

Android 独立配套应用使用 Chaquopy Python 3.10、NumPy 1.23.3、SciPy 1.8.1、soxr 0.3.6 和 ONNX Runtime Android 1.20。通过系统文件选择器读写，通过 MediaCodec 解码音频，前台服务执行任务；无需 Unity、桌面主机或网络服务。

## 推理链

六个神经网络阶段分别是 genelive_stack、genelive_recurrent、genelive_head、acoustic、selector_step、layout_step。patterns.json 保存 236 类段落模式、森林树、簇中心及转移概率；expression.json 保存节奏条件 flick/critical 模型；slide_paths.json 保存 64 类独立长条路径；dsp.npz 保存对应 Hann 窗、mel 滤波器和静音特征。清单校验 11 项模型资产。

大的模式森林使用无损压缩 NumPy 数组保存，解压为只读视图；数值不量化，避免巨型 JSON 树列表造成额外解析内存。手机升级后按内置资产 SHA-256 替换旧模型缓存。

CNN 用 640 帧核心块和 128 帧上下文边界限制内存。GenéLive 循环层先前向分块传递 hidden/cell，再反向分块，只替换反向半部。NS 和布局 GRU 逐事件传递状态。阶段结束后释放会话，失败或取消也释放资源。

采音阈值、静音过滤、时间校正、原点约束、模式排键和双指投影复用桌面算法。核验实际导出的 SUS，转换为社区版 score.json 后才发布。已有歌曲包保留前置和媒体，不直接修改原文件。

自动 BPM 使用便携谱通量、自相关与相位估计，与 Windows 的 librosa 方法不同。两端自动估计可能不同；填写同一 BPM/起点可减少差异。压缩音频由设备解码器支持范围决定；便携 WAV 读取器目前支持 PCM16。

## 验证与限制

桌面已验证便携链的真实音频生成、mel 数值对照、CNN 分块边界、循环状态、NS 声学分块和模式森林概率。桌面验证不能证明 Android JNI、文件选择器、后台生命周期和实际设备内存均正常。

当前 APK 为 ARM64 调试签名预览版，Android 7+；Chaquopy 数值库使用 4 KB ELF 页对齐，16 KB 内存页设备不支持。正式发布需要开发者自己的签名密钥，并在目标手机完成断网生成、长曲、取消、已有包、后台切换及 OpenSekai Live/成绩测试。尚未完成真机实测。
