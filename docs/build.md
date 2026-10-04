# 独立工具构建

本方案不需要 Unity。运行时依赖和模型随发布包提供；以下环境只供开发者构建。

## Windows

准备 Python 3.11，按 backend/requirements.txt 安装 CPU 推理依赖。用 tools/setup_backend.ps1 配置模型包与完整 FFmpeg 目录；私有路径记录在 backend/runtime.local.json。权重校验清单为 backend/model-lock.json。

```powershell
./tools/build_worker.ps1 -Python <已配置推理环境的python.exe>
./tools/build_companion.ps1 -Python <含PyInstaller的python.exe>
```

发布目录为 artifacts/companion/OurSekai/。必须同时提供 EXE、_internal/、AutoChart/，不能单独发送 EXE。构建脚本对采音目录使用本地硬链接节约空间，导出的 ZIP 包含独立文件。

## Android

```powershell
python tools/setup_android_build.py
<推理环境python.exe> -m pip install onnx==1.17.0
<推理环境python.exe> tools/export_portable_models.py
<推理环境python.exe> tools/prepare_mobile_models.py
<推理环境python.exe> tools/build_android_companion.py
```

本地工具链包含 JDK 17、Gradle 8.9、Android SDK 35 和 Python 3.10 构建解释器。下载与首次依赖解析需要网络，应用运行无需网络。配置在 .build-tools/android/build-env.json，不提交 Git。构建脚本本身需要 NumPy/SciPy/soxr，所以应使用已配置推理环境的解释器。

也可将已保存的便携模型包解压到 artifacts/portable-models/，跳过模型导出。build_android_companion.py 会校验清单、复制指定 Python 源码和模型，再构建 APK。

输出 artifacts/OurSekai-Companion-Android-arm64.apk 为调试版。调试签名密钥在本地工具目录，不进入源码 ZIP；正式更新应使用自己的持久签名密钥。支持范围和验收见 [Android 说明](android-port.md)。

0.2 的桌面与便携模型备份为 our-sekai-models-v3.zip、our-sekai-mobile-models-v3.zip。旧版模型包不满足新版清单。若已有六阶段 ONNX 文件，可运行 tools/export_rich_mobile.py 更新训练后的模式，再运行 tools/prepare_mobile_models.py，避免重复导出未变更的神经网络。

## 源码与验证

```powershell
python tools/export_source.py --companion
Push-Location backend
<推理环境python.exe> -m unittest discover -s tests -v
Pop-Location
<推理环境python.exe> -m unittest companion.test_song_package -v
<推理环境python.exe> tools/verify_portable_runtime.py
python tools/verify_companion_release.py
```

源码 ZIP 包含独立程序和构建脚本，排除 Unity 原型、模型、环境、歌曲和缓存。额外保存桌面与便携模型包。真正可游玩全流程仍须使用目标社区版和手机验证。
