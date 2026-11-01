# camera05：72段人工标定后的完整监督训练

结果页：http://127.0.0.1:8768/projects/camera05_custom/semantic-pose-v5/

人工标定：http://127.0.0.1:8768/projects/camera05_custom/semantic-pose-v5/edit/

2026-11-01从用户明确指定的Chrome原页面导出camera05-pose-v4-adjusted-labels (2).json。已确认72/72段、1,184个区间。原人工任务字典逐项保留；新编辑页使用独立存储键，全部72项仍为已确认，不用模型建议替换人工边界。完整导出快照为semantic-pose-v5/user-annotation.json。

本轮使用17,592个人工确认且姿态质量合格的单阶段帧，自动或代理训练标签0帧。有效72段共17,996帧，378帧人工未知/无效/混合区间及26帧姿态质量不足不参与拟合。五阶段帧数为准备3,021、引拍4,228、挥拍3,430、随挥2,277、回位/移动4,636。保持原25fps、19身体关键点和原始划分。

模型为共享五阶段姿态集成模型，加正手引拍/挥拍/随挥专用分支。正手核心分支使用69段中的9,384帧；准备与回位共享。类型来自人工区间记录或用户明确确认，不训练正手/单反/双反分类器。没有可靠类型信息的视频使用共享分支，原21段留出视频均如此。

内部核对使用5组连续时间块，训练与测试块之间清除不足90秒间隔的训练片段。每组训练/测试/隔离段数分别48/15/9、39/15/18、44/14/14、44/14/14、54/14/4。所有17,592帧各测试一次，合并一致率0.978115、宏F1为0.976677。该结果来自同次录像、且已知类型由人工提供，不代表独立留出准确率，也不代表跨人或跨机位效果。原21段验证从未参与模型拟合、专家拟合和阈值估计，当前仅生成预测，尚无其人工语义准确率。

9段继续整体剔除：04_30、04_40、08_10、09_00、11_19、11_30、11_39、12_10、13_10。原视频保留。预测文件共93段（训练72、留出21）。旧模型及102段合并MoSeq重训暂停决定不变。本轮是Mac本地CPU监督训练，耗时约33秒。

复现环境：Python3.14.7、NumPy2.5.4、scikit-learn1.8.0、joblib1.5.3。远端旧虚拟环境版本不同，加载模型应先匹配训练环境。training-code.py、annotation-registry.json、curation.json及training.npz保存本轮代码、校验规则、排除规则及精确训练数组。推理使用SupervisedPhaseModel，类型未知默认共享分支。

SHA256：

- 原manifest：71c79acecf0996c69a9647dd65f3d598012c3c1c3aad9919c38271870ad57313
- 人工导出：8bd8d967cc99b70b894dae8fb0581bed91f298aa548c47088e6a4b74f1513e92
- 模型：f3cda9744021f814d20c0d2ba8b93cc530ce1c0495a325ae37ee9281b59df275
- 训练代码：ec13f7c64a203b306ae2974b777382a28b0e7a7340ca1b13118f9553e551438a

verification.json记录完整核验：人工导出校验、训练特征/目标/类型逐帧重建、加权标准化参数、专家支持、5组90秒隔离、全部93段模型重载后预测/置信度/筛选掩码一致、72段编辑任务与原导出完全相同。42项相关测试通过。current-semantic.json及项目首页指向v5，历史页面显示新入口。

复现调用：使用同一项目根和人工快照，向全新输出目录调用dsa.measure.semantic_supervised.run(root, output, annotation)，避免覆盖已完成版本。该入口要求所有有效训练段已确认；不完整导出会直接拒绝训练。

## 参数与外部调用

输入为恒定25fps、已跟踪同一人的19关节二维骨架。frame_features产生192维特征：前/当前/后姿态114维、速度38维、局部变化38维、髋部移动速度2维。含前后帧平滑及整段中位数归一化，因此当前按完整片段离线推理，不能直接声称已支持实时流。输入预处理保持drill.export的关节顺序、坐标与置信度规则；直接MP4需要先提取姿态。

共享模型和正手核心专家分别使用StandardScaler、LogisticRegression(C=0.2,max_iter=3000,random_state=31)与ExtraTreesClassifier(n_estimators=240,min_samples_leaf=3,max_features=0.7,n_jobs=6,random_state=31)，分类分数两路各占50%。类别权重在每次拟合内部平衡。共享逻辑回归实际100次迭代，正手专家76次；max_iter为上限。model.joblib约57.1MB。

推理保留条件：肩/肘/腕/髋/膝/踝置信度最低0.55，19关节原始观测比例至少0.8；线性与树分类器一致；最高分至少0.70、第一二名差至少0.25；到训练样本的标准化最近邻距离不超过内部折外95%分位0.637447405。类型专家参与时也检查专家一致性与距离。阶段切换两侧各屏蔽2帧，孤立通过段少于3帧也屏蔽。分数未做概率校准。准备静止本身不会被剔除。

严格筛选后训练视频13,562/17,996帧通过（75.36%），留出视频3,977/5,298帧通过（75.07%）。这是预测覆盖率，不是准确率；内部97.8%是在全部可核对折外人工帧上的原始分类一致率，不是筛选后覆盖率。当前没有独立留出语义真值、边界误差或完整击球周期准确率指标。

已提供可执行调用示例examples/camera05_phase_predict.py，输入现有规范NPZ，输出JSON时间区间。示例核对18_08留出视频，与保存预测的逐段边界、阶段和分数一致。默认击球类型无法判断，使用共享模型；只有外部已知为正手时才传--stroke-type 正手调用专用分支。其他类型暂无专家，回退共享模型。脚本检查模型SHA、25fps、关节顺序、连续帧和原始观测掩码；输出phase=null表示不确定，时间与帧的终点均不含在区间中。原始分类分数不是校准后的正确概率。

```bash
cd /Users/krum5539/Documents/deep-sports-analysis
PYTHONPATH=src .venv/bin/python examples/camera05_phase_predict.py \
  --model-dir data/projects/camera05_custom/semantic-pose-v5 \
  --input data/projects/camera05_custom/clips/18_08/moseq-input.npz \
  --output /tmp/camera05-phases.json
```

长期运行的Python服务可复用read_input、frame_features、predict_frames，在进程启动时加载joblib模型和summary，逐次请求仅做特征及预测。示例命令为独立进程调用，每次会重载模型。提供可兼容的运行环境及src/dsa代码，不能只发送joblib文件。

8768当前是绑定127.0.0.1的静态结果服务器，不提供POST推理接口。已算好的93段预测可GET /projects/camera05_custom/semantic-pose-v5/predictions/{clip_id}.json；其他机器访问其自己的127.0.0.1无法到达本机。接入新视频的HTTP服务需要另行封装姿态预处理与推理，不应把静态展示地址当作模型服务。没有在本轮增加公共端口。

## Mac统一工作台：第一阶段已部署

入口：http://127.0.0.1:8777/。原8768静态报告继续保留。工作台通过用户级LaunchAgent常驻运行，运行副本和数据位于`~/Library/Application Support/DeepSportsWorkbench/`；源码仍在Git checkout。为避开launchd访问Documents工作目录的限制，运行代码与环境放在Application Support。既有项目文件采用硬链接（跨磁盘时复制），运行流程只读取旧数据，新模型、导入和标注为独立文件；不能将这份链接镜像视为独立备份。

当前项目原72段完整人工标定已带入本机SQLite，并保存不可变JSON修订。编辑页自动同步，旧浏览器的修改不能覆盖已保存的新版本。训练冻结提交时的标注快照；未确认段不进入完整监督训练。每个项目一次仅运行一个后台任务，独立进程运行、保存日志，失败可使用保留输入重试。旧v5仍为默认模型，新训练版本可单独选择预测和查看报告。

实际验收：

- 工作台完整训练任务`01fd32f39b0b459697535fa96a04ecf9`成功，新目录`semantic-pose-wb-01fd32f39b0b`。x/y/weight/groups/frames/stroke_types/origins全部与v5逐项一致；三个训练/验证样例的预测、分数和接受掩码一致。
- 用现有18_08视频副本验证导入，不作为新训练数据。导入任务`ce848d19030d40a9898d5305e3595e6d`成功，250帧25fps；5090任务`2727251f42f04efea70b73d78a22e571`成功。人体检测、跟踪、姿态在5090运行后，规范NPZ自动取回Mac，浏览器可播放视频并预测阶段，通过筛选68.4%。该比例是覆盖率。
- 原manifest哈希仍为71c79acecf0996c69a9647dd65f3d598012c3c1c3aad9919c38271870ad57313；72个人工任务字典完全相同，项目训练/留出预测视频仍共93段。
- 新增持久化、并发冲突、训练快照、留出隔离、跨站写入拦截、私有文件保护和归档路径保护检查；全量177项测试通过。实际wheel构建包含工作台及标注HTML/JS，修复此前网页模板遗漏。

导入限制是固定机位MP4短片，最长120秒、最大500MB；上传后至少保留2GiB磁盘空间。新视频只做独立推理，不会自动改变原数据划分或加入训练。5090配置使用已有SSH别名，源码和视频仅传入隔离的`data/workbench-incoming/{导入ID}`，不会覆盖原camera05数据。当前没有实现MPS/CoreML视频计算，也没有声称完成跨机位识别、长视频批处理或职业参考完整重算。

运行副本中的`data/.workbench/acceptance.json`保存本次验收记录。持久标注在`data/.workbench/state.sqlite3`及`data/.workbench/annotations/`；新视频在`data/.workbench/imports/`；日志与训练快照在`data/.workbench/jobs/`。这些数据均不进入GitHub提交，应另做独立备份。

本次另外创建了独立的`.venv-workbench`运行环境（不继承系统site-packages），按轻量requirements安装37个包，完成全量CPU检查；LaunchAgent已切换到该环境。源仓库`.github/workflows/workbench.yml`在推送和PR时执行同一组CPU检查。部署脚本可start/restart/stop/status，安装包与源码运行均包含所需HTML/JS。
