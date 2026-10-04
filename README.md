# Our Sekai Companion

配合已安装的 [OpenSekai 社区版](https://github.com/kamcdev/OpenSekai_Community) 使用的离线自动写谱工具。Windows 和 Android 各自完成“选择难度 → 导入歌曲 → 生成歌曲包”，再在社区版中导入并进入 Live。应用不提供谱面编辑。

Windows 发布包包含 EXE、采音后端、模型与 FFmpeg；Android APK 包含本地 Python、数值计算库、ONNX 模型和音频解码流程。运行时没有服务器或联网推理依赖。Android 当前为 ARM64 调试预览版，需要 Android 7 或更新系统、4 KB 内存页；16 KB 页设备不支持。真机测试状态见 [验证记录](docs/verification.md)。

## 功能

- EASY、NORMAL、HARD、EXPERT、MASTER 五档预设；参数集中在高级设置。
- GenéLive 采音、NS 键型、236 类段落模式、64 类独立长条路径及节奏条件 flick/critical 模型、双指约束、SUS 导出后网格与重叠核验。
- 新歌曲默认实际加入 9 秒静音。裁剪、谱面时长、试听和节拍起点相应换算。
- 已有歌曲包重新生成沿用原前置，保留歌曲信息、封面和视频。
- 核验后发布歌曲包；失败与取消保留日志，不发布半成品。
- 可持久设置输出位置：Windows 目录；Android 系统文件夹授权，生成后自动保存 ZIP。
- ZIP 内含社区版所需的 score.json、manifest.json、音频、SUS、可选封面和视频。

Live、自动游玩、成绩、视频和游戏设置由已安装的社区版提供。外部工具不会移除原游戏安装中的编辑器入口。

## 使用

Windows 完整解压发布 ZIP 后运行 OurSekai.exe。Android 安装 APK 后通过系统文件选择器导入歌曲；生成后点“保存 ZIP”，再在社区版导入。两端均可在高级设置填写 BPM 和原曲节拍起点。

详细步骤见 [使用说明](docs/companion-user-guide.md)，开发与打包见 [构建说明](docs/build.md)。

## 文件结构

```text
companion/             歌曲包格式转换、Windows 界面、测试
android/               Android 界面、音频解码、后台生成服务、ONNX 桥接
backend/               采音与排键；portable/ 为手机本地推理链
tools/                 配置、模型导出、打包及验证脚本
docs/                  使用、设计、构建、移植、验证和来源说明
artifacts/             本地发布结果与模型包（不入 Git）
.build-tools/          本地编译工具及缓存（不入 Git）
```

此仓库仅包含独立配套工具源码，不包含之前的 Unity 集成原型或游戏资源。外部工具不需要 Unity；玩家需另外安装 OpenSekai 社区版。

## 时间轴

原曲 120 秒加默认前置后，音频实际长 129 秒，fillerSec = 9，secForMusicScoreMaker = 122（音乐区间加 2 秒结尾缓冲，不含前置），默认试听从第 9 秒开始。节拍起点按原曲填写，程序换算为带静音的时间。已有歌曲包不再次加静音。

## GitHub 与模型

独立源码 ZIP 可整体解压到新仓库。不要上传歌曲、权重、私有配置、环境或构建缓存；它们已被 .gitignore 排除。模型包另行保存或放在 Release，并保留 SHA-256 清单。桌面模型与便携模型包互不替代。没有权重时，源码不会自动下载或伪造模型。

第三方代码与模型保留其来源及权利归属，见 [来源与许可](docs/third-party.md)。

0.2 版训练方法与限制见 [模型扩展记录](docs/training-v3.md)。训练数据 600 首 MASTER 谱面，按歌曲分组；模型数目不等同于主观手感验收。
