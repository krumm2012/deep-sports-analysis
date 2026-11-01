# Codex 教练分析复刻与 5090 视频归档

日期：2026-10-31（Asia/Shanghai）。GPU 主机 `ubuntu32`，项目 `/root/deep-sports-analysis`。

## 已实现的两种结果

1. **公开测量重放**：从原文章 `#ex-data` 读取 Peter / Richard 各 24 项有效指标，使用明确指定的 `gpt-6.1-sol`、Codex CLI、high reasoning 生成新中文建议。输入、业务提示词、schema、原始答案和来源均逐人保存。没有调用 Claude / Anthropic API；没有把历史 Claude 文字改名。
2. **下载录像的真实 GPU 验证**：通过 yt-dlp 下载 Peter Wang 频道 `kbBfuwNfNj8`（Richard Peter 2025.06.14，696 秒、720p、60fps），另取前 60 秒降至 30fps，在 RTX 5090 D 从头执行检测、球定位、跟踪和姿态。1800 帧，6400 个姿态裁剪、1503 帧检出球，GPU 作业耗时 54.1 秒；后续 Codex 球场标定、CPU 场景分析、两段可视化及中文结果解读也已完成。模型预测 21 次击球（6 发球、11 正手、4 反手），双球员覆盖 90%；这些是模型输出和覆盖计数，不是准确率。两段可视化均已完整解码检查。

公开测量重放不是从原始比赛重新计算职业基准。Peter/Richard 的逐类有效样本数和 IQR 未公开，保留为 unknown/null。不能用 5 个展示片段推断样本数。新下载的 2025 手机录像与截图中的 2026 私人录像不是同一场，不得替代后声称完全重现原结果。

Codex 使用已登录的 Mac 执行，媒体、源码与最终结果部署到 5090；未复制 Codex 凭据到服务器。复刻模型经固定输入生成建议，不在 GPU 上本地运行 Codex 模型。

## 视频与下载清单

- 文章全部 **306** 个独立公开 MP4 已归档到 `/root/deep-sports-analysis/data/published-media/`，合计 **77,086,439** 字节；每个文件保存 SHA-256、ffprobe 元数据与来源 URL，306 成功、0 失败。
- `data/published-media/manifest.json` 是完整清单。本机镜像逐文件 SHA-256 与远端清单一致。
- `data/published-demo/offline.html` 将原页面的视频引用指向上述本地文件，保留历史署名和原始文字。字体及原站导航仍可能联网。
- `config/atp-download-manifest.json` 包含 **34** 个职业比赛上传 ID（17 个原目录和 17 个 MORE 候选）及 **1** 个公开 2025 手机录像。该清单不是对 35 个资源全部可用或全部下载完成的声明。
- yt-dlp 按 `docs/QUIZ.md` 的策略保留完整上传，选择不超过 720p 的 MP4 视频流，无音频；每次一个视频，片段并发 8。检测到 aria2c 时使用单视频 8 路 HTTP 下载。记录标题、频道、上传日期、时长、文件哈希和元数据。
- 完整源录像放在 `data/raw/{atp_hardcourt,own_public_2025}/`；原始上传保持原帧率，分析输入另行通过 FFmpeg 截取和降帧。
- 原作者完整 2026 手机录像未随源码公开。公开的短片和 23 秒示例已下载，不能因此宣称获取了完整 2026 原片。

远端实时状态：

```bash
ssh ubuntu32-sam5090 'cd /root/deep-sports-analysis && cat data/raw/reference-download-status.json'
ssh ubuntu32-sam5090 'cd /root/deep-sports-analysis && tail -10 verification/reference-download.log'
```

截至本次核验，35 个完整上传的队列已有 2 个完成、0 个失败：2025 手机录像，以及澳网官方 Kyrgios–Medvedev 2022 全场（9072.16 秒，1,795,430,051 字节）。第三项 Alcaraz–Tiafoe 2022 美网半决赛正在下载，其余尚未完成。实时状态优先于本文快照。

下载状态在每个视频开始/结束时保存，进行中的 aria2c 文件可能是稀疏文件，不能把逻辑文件大小当成已下载字节数。日志和 `.part/.aria2` 文件用于续传。磁盘不足 20GiB 时停止；连续失败三次时停止，避免无效重试。YouTube 资源是否删除、受地区限制或需要登录，以每个文件的实际结果为准。

## 网络与恢复命令

5090 直连 YouTube 超时，本机现有 HTTP 代理为 `127.0.0.1:1082`。本次仅通过 SSH 绑定远端回环地址 `127.0.0.1:18882`，没有开放公网代理或修改主机密钥校验。

在 Mac 上保持代理与隧道可用（已有隧道时不要重复占用端口）：

```bash
ssh -o BatchMode=yes -o ControlMaster=no -o ControlPath=none \
  -o ExitOnForwardFailure=yes -N \
  -R 127.0.0.1:18882:127.0.0.1:1082 ubuntu32-sam5090
```

下载启动器通过 flock 防止重复运行。当前作业进行中时不要再启动第二份。需要恢复时运行：

```bash
ssh ubuntu32-sam5090 \
  'cd /root/deep-sports-analysis && bash deploy/run-reference-download-5090.sh'
```

Mac 休眠、代理关闭或隧道断开可能中断下载，恢复网络后可以续传。`verification/reference-download.exit` 只在本次作业结束时写入；不能将历史 exit 文件当作当前状态。

## 报告与输入

本机镜像：`http://127.0.0.1:8768/reproduction/`。

启动本机预览应使用项目服务器（中文文本显式返回 UTF-8，支持视频 Range）：

```bash
PYTHONPATH=src .venv/bin/python -m dsa.pages.serve data 8768
```

不要用没有显式字符集的默认静态服务器替代；`prompt.txt` 等中文文本可能被浏览器错误解码。

远端页面使用 Python HTTP server，仅绑定 `127.0.0.1:8768`；本次 SSH 转发到本机 `8769`：`http://127.0.0.1:8769/reproduction/`。本机镜像不依赖 SSH。`deployment-status.html` 显示下载队列：远端地址每 10 秒刷新，本机地址明确标注最近同步快照；源录像播放链接指向 5090。

报告目录：`data/reproduction/`：

- `index.html`：Peter/Richard 切换、真实示例、职业代表片段、建议和指标依据。
- `{peter,richard}/input.json`、`prompt.txt`、`schema.json`、`answer.json`、`provenance.json`、`cache/*.usage.json`：测量输入与可审计模型输出。
- `reference-review.html`、`references.json`：原文全部 9 条引用和方法核查。
- `manifest.json`：本次复刻模式与模型信息。

所有视频仅用于网页展示，教练建议的模型调用不接收视频或图片，只接收汇总测量文本。输出必须引用真实、同类动作的 metric ID；缺测和高机位敏感指标不进入新建议。原网页已发布数据不二次应用相机缩放；未独立验证原作者的校正。

重新生成：

```bash
PYTHONPATH=src .venv/bin/python -m dsa.pages.published_coach \
  --model gpt-6.1-sol --media-base ../published-media/
```

## 显存问题与回归

首次 60 秒新录像在 RTMW 的 Conv_6 分配显存失败。复现发现：姿态批次默认 128、缓冲区可超过阈值，以及手机 pass2 开始时 RF-DETR/WASB 仍被局部/闭包引用持有。此前复用 pass1 缓存的测试未覆盖这一首次运行情况。

修复：`process(..., pose_batch=32)` 对每次姿态推理实行硬上限；手机 pass1 后仅释放本函数拥有的检测模型并清空空闲 CUDA 缓存；不释放调用者提供的共享模型，也不提前释放 TV 的 court/pose。已保留失败日志与旧 pass1 缓存，并从头重跑 60 秒原输入，退出码 0。GPU 从头运行各阶段：detect 13.8s、ball 5.7s、pose 20.7s，墙钟合计 54.1s；各阶段存在 CPU/准备工作，不能解释为纯 CUDA kernel 时间。

本机与 5090 均通过 54 项测试，包含跨帧/框顺序、姿态批次硬上限、模型释放及共享模型保留。源码增量同步前，关键文件 SHA-256 与本机一致；远端是无 `.git` 的部署副本，不能仅凭仓库 HEAD 声称完整版本一致。

Graphify 报告不存在，按 AGENTS.md 尝试重建时因缺少 `graphify` 模块失败；不影响上述运行和测试。

## 同步核对记录

| 组件 | 本机 / Git | 5090 部署 | 分类与处理 |
|---|---|---|---|
| 源码基点 | main `2d742add1f05f1bd27fda28d5ec2e1b1ebd911cb`，已有未提交改动 | 无 `.git` 的源码副本 | 版本来源不能仅用提交号证明；同步前核对了 codex/projects/models/others_tab 四个文件 SHA-256 一致，保留既有改动 |
| 本次新增模块 | 视频归档、下载队列、Codex 教练报告与回归 | 增量上传，54 项测试通过 | 本次模块已部署；未执行 pull/reset/清理仓库 |
| 公开媒体 | 306 个本机镜像 | 306 个，77,086,439 字节 | same：逐文件 SHA-256 与远端清单一致 |
| 新视频可视化 | 60 秒源片、render 和 render_plain | 同样的两段成品 | same：成品及教练页 SHA-256 对应，远端 HTTP 200 |
| 完整源录像 | 本机仅保存状态快照 | 额外源录像队列持续写入 | ahead-deployed：以远端实时 JSON 为准，不把本机快照当实时 |
| Codex 登录 | 本机已有登录 | 不复制凭据 | 使用认证工作机生成建议，上传可审计输出 |

此部署为 Python 进程与文件目录，本次没有 Docker 或数据库变更。操作记录包括 `git status/rev-parse/ls-remote`、关键文件 SHA-256、SSH GPU/磁盘检查、ffprobe、FFmpeg 解码、本地/远端 pytest 和 HTTP/浏览器验证。

## 仍未完成的完整重算

后台源比赛下载不等于完成职业参考重算。完整复刻仍需完成可用源录像下载、主机位筛选、职业球员 GPU 推理、身份核对、逐拍测量、有效样本统计及相机校正，再以新统计生成报告。额外比赛的选用和原作者未提交的 pad 清单也会使数据集不同；必须记录差异，不能声称与原文逐数值一致。

详细问题与原始来源：[ARTICLE_REFERENCE_REVIEW.md](ARTICLE_REFERENCE_REVIEW.md)。
