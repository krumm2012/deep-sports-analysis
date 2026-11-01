# camera05：逐段完整视频标注

入口：http://127.0.0.1:8768/projects/camera05_custom/semantic-clip-labeling/

81段训练视频按原始时间顺序列出，25fps。21段留出验证视频继续单独保存。MoSeq原状态仅提供可折叠的边界参考，不自动决定动作阶段。原模型与弱监督模型保持原样，本轮没有重新训练。

先看完整视频，在动作边界暂停并拆分区间，再选择准备、引拍、挥拍、随挥、回位/移动。支持1帧（0.04秒）调整。无效动作、无法判断、混合/需调整区间单独保留。每段确认前必须连续覆盖全片。

20段视频带入旧人工区间作为待核对草稿；00_40的214帧（8.56秒）手工边界保留。自动建议阶段不会被当作新的人工标签。所有完整视频初始均未确认。

进度自动保存在浏览器，也可随时“导出标签与进度”，用“恢复导出文件”恢复。导出文件camera05-training-clip-labels.json使用schema_version=3、annotation_mode=complete_clip_intervals；records只包含整段已确认视频，annotation_progress包含全部草稿。单阶段interval的phase_usable=true；invalid/mixed/uncertain的phase=null、phase_usable=false，只有invalid的excluded=true。无效区间保留原始帧范围，未删除视频或MoSeq状态。

完整视频导出采用独立schema，不能直接交给旧schema2小样本训练器。后续训练需先验证来源、边界覆盖与人工确认，并保留21段独立验证。

```sh
PYTHONPATH=src .venv/bin/python -m dsa.pages.semantic_clip_labeling validate \
  data/projects/camera05_custom \
  data/projects/camera05_custom/semantic-clip-labeling/candidates.json \
  /absolute/path/camera05-training-clip-labels.json
```

用户导入的camera05-old-model-state-names (2).json仅应用到旧30fps示例。五个状态均已确认，其中状态3为挥拍＋随挥混合状态；它不会作为单阶段训练标签。输出在camera05/2025_1830/user-state-phases-v1/，不转移到正式25fps的8状态模型。
