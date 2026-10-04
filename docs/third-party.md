# 来源与许可记录

- Unity 基础项目：[kamcdev/OpenSekai_Community](https://github.com/kamcdev/OpenSekai_Community)，
  本次基线提交 `0cd2e6212d71671d48373732fbaa3559c16a765a`，原 MIT 文件保留在仓库根。
- GenéLive：现有工具中的 KLab `notes_generator` 源码，MIT 文件保留在 `backend/models/genelive/LICENSE`。
- AutoOsu：现有工具中的 [issyun/AutoOsu](https://github.com/issyun/AutoOsu)，
  模型适配器记录提交 `b81dc6f43f6274eb37b6cffa0945e3e5e23cf1dd`。
  本地来源目录没有 LICENSE 文件，未给它补写 MIT 授权。
- 项目定制的 NS、布局模型和模式库来源于用户现有自动采音工具；校验值记录在 `backend/model-lock.json`。
- FFmpeg：现有工具的 Windows 运行目录；复制时同时保留其 LICENSE。完整发布包须保留该许可。
- PyTorch、torchaudio、NumPy、SciPy、librosa、scikit-learn、ONNX Runtime 各保留自身许可。
- Android 桥接：[Chaquopy](https://github.com/chaquo/chaquopy)，音频重采样 soxr，数值库随 Android wheel 分发。源码未将它们改为本项目 MIT。

社区版中的字体、贴图、音频、Prefab、Shader 和可能来自原作的资源不自动受代码 MIT 许可覆盖。
当前独立配套工具源码 ZIP 排除 Unity 原型及上述游戏资源；程序依赖已有社区版玩家端。
发布源码时保留社区版关于这些资源的权利说明；模型文件也不因进入本地打包产物而自动改变授权。
本项目与 Project Sekai 的原作、发行方和权利方没有官方关联。
