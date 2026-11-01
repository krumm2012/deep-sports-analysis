"""A five-stage trial report that distinguishes weak agreement from human checks."""
from __future__ import annotations

from html import escape
import json

from dsa.measure.semantic_phases import PHASES, phase_schema


def report(output, summary, samples):
    schema = phase_schema()
    rows = []
    for phase in PHASES:
        human = summary['source_interval_counts']['human'][phase]
        weak = summary['source_interval_counts']['weak'][phase]
        rows.append(f'<tr><td>{phase}</td><td>{human}</td><td>{weak}</td><td>{human+weak}</td></tr>')
    evaluation = summary['evaluation']
    human = evaluation['human_reference_check']
    weak = evaluation['weak_label_agreement']
    hscore = human['metrics']
    wscore = weak['metrics']
    chips = ''.join(f'<span class="chip" style="background:{d["background"]};color:{d["color"]}">{i+1}. {d["label"]}</span>'
                    for i, d in enumerate(schema['definitions']))
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>五阶段弱监督训练 · v1</title><style>
body{{max-width:1150px;margin:32px auto;padding:0 22px;color:#192b3b;font:16px/1.7 system-ui}}h1{{font-size:28px}}a{{color:#145b94}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #d6dfe5}}
.notice{{padding:14px 16px;background:#fff2d6;border:1px solid #e4d1a5;border-radius:8px;margin:16px 0}}
.legend{{display:flex;gap:8px;flex-wrap:wrap}}.chip{{padding:5px 10px;border-radius:6px;font-weight:600}}.muted{{color:#53667b;font-size:14px}}
video{{width:100%;max-height:480px;background:#101820;border-radius:8px}}.play-status{{margin:10px 0;font-weight:600}}
.segments{{display:flex;gap:3px;flex-wrap:wrap;margin:12px 0}}button{{font:inherit;border:1px solid #aebdcc;border-radius:5px;padding:7px 9px;cursor:pointer}}
.sample{{padding:18px;border:1px solid #d8e0e7;border-radius:9px;margin:18px 0}}.scroll{{overflow-x:auto}}
</style></head><body><h1>五阶段弱监督训练 · v1</h1>
<p><a href="../">项目首页</a> · <a href="../semantic-labeling-round2/">继续微调与确认</a> · <a href="../semantic-pilot/">前一版人工小样本报告</a>
 · <a href="summary.json">训练摘要</a> · <a href="models.json">五阶段模型参数</a> · <a href="training-samples.json">样本来源与权重</a></p>
<div class="legend">{chips}</div><p class="notice">本版已按五阶段完成弱监督试训。
引拍、随挥当前只有自动建议，尚无人工确认参考。下面的“弱标签一致性”只衡量模型对候选标注的复现程度，不能解释为真实动作识别准确率。
原21段留出视频只生成预测，没有进入模型拟合。</p>
<p>使用训练池中{summary['training_recordings']}段视频的{summary['training_intervals']}个阶段区间。每个人工样本来源权重1，弱标签来源权重0.25，
并在每折训练数据内按各类的权重总和进行平衡。模型为身体姿态与运动统计上的五分类逻辑回归。
分类训练和逐视频推断已完成；本轮未重训MoSeq。</p>
<h2>训练样本来源</h2><table><tr><th>阶段</th><th>人工确认</th><th>弱标签，未确认</th><th>合计</th></tr>{''.join(rows)}</table>
<h2>按整段视频留出的内部检查</h2><p>每次留出一整段视频，同时移除该视频的人工样本与弱标签，重新拟合标准化和分类器。
同人、同次录像的相邻视频仍可能相关；这些检查不替代独立的人工语义验证。</p><table><tr><th>参考来源</th><th>区间</th><th>一致比例</th><th>宏F1</th><th>说明</th></tr>'''
    if hscore:
        html += f'<tr><td>人工参考检查</td><td>{human["intervals"]}</td><td>{hscore["accuracy"]:.1%}</td><td>{human["macro_f1_on_reference_phases"]:.3f}</td><td>宏F1仅覆盖有人工参考的{len(human["reference_phases"])}类</td></tr>'
    if wscore:
        html += f'<tr><td>弱标签一致性</td><td>{weak["intervals"]}</td><td>{wscore["accuracy"]:.1%}</td><td>{wscore["macro_f1"]:.3f}</td><td>五类自动建议；并非人工准确率</td></tr>'
    html += '</table>'
    if hscore:
        html += f'<p class="muted">同样分折下，多数类基线在人工参考上的一致比例为{human["majority_baseline"]["accuracy"]:.1%}；当前样本不足以证明五阶段识别能力已可靠改善。</p>'
    html += '<p class="notice">完整五阶段的人工准确率：尚不能计算。需要先确认引拍与随挥的标注，并在独立视频上检查所有阶段与边界。</p>'
    html += '<h2>21段留出视频：五阶段预测示例</h2><p>下面展示其中3段。模型分数未经校准。时间平滑使用五阶段循环顺序和最短4帧约束，阶段边界仍需人工复核。</p>'
    previews = {}
    style = {d['label']: d for d in schema['definitions']}
    for c in summary['preview_clips']:
        data = json.loads((output / 'predictions' / f'{c}.json').read_text())
        previews[c] = data
        html += f'<section class="sample" data-clip="{c}"><h3>{c}</h3><video controls preload="metadata" playsinline src="../clips/{c}/source.mp4"></video><p class="play-status">准备播放</p><div class="segments">'
        for i, s in enumerate(data['segments']):
            d = style[s['phase']]
            html += f'<button data-segment="{i}" style="background:{d["background"]};color:{d["color"]}" title="模型分数 {s["mean_model_score"]:.2f}">{s["start_seconds"]:.2f}–{s["end_seconds"]:.2f}s · {s["phase"]}</button>'
        html += f'</div><p class="muted"><a href="predictions/{c}.json">这段视频的分段JSON</a> · <a href="predictions/{c}.npz">逐帧结果</a></p></section>'
    human_lookup = {s['id']: s for s in samples}
    html += '<h2>人工参考区间：逐项检查</h2><table><tr><th>视频区间</th><th>人工阶段</th><th>留出预测</th><th>结果</th></tr>'
    for p in evaluation['out_of_fold']:
        if not p['human_reviewed']:
            continue
        s = human_lookup[p['id']]
        html += f'<tr><td><a href="../clips/{s["clip_id"]}/source.mp4#t={s["start_frame"]/25},{s["end_frame_exclusive"]/25}">{escape(s["id"])} · {s["clip_id"]}</a></td><td>{p["reference"]}</td><td>{p["prediction"]}</td><td>{"一致" if p["reference"]==p["prediction"] else "不一致"}</td></tr>'
    html += '''</table><p>下一步：在五阶段标注页微调并确认自动建议，再以更高比例的人工样本重新训练。
独立验证标注仍在单独页面保存。</p><p class="muted">加权拟合依据：
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">StandardScaler</a>、
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">LogisticRegression</a>。</p>'''
    payload = json.dumps(previews, ensure_ascii=False).replace('<', '\\u003c')
    html += f'<script type="application/json" id="previews">{payload}</script>'
    html += '''<script>
const clips=JSON.parse(document.getElementById('previews').textContent);
for(const section of document.querySelectorAll('[data-clip]')){const data=clips[section.dataset.clip],video=section.querySelector('video'),status=section.querySelector('.play-status');let stop=null;
video.addEventListener('timeupdate',()=>{const s=data.segments.find(s=>video.currentTime>=s.start_seconds&&video.currentTime<s.end_seconds);status.textContent=s?`${video.currentTime.toFixed(2)}秒 · ${s.phase} · 模型分数 ${s.mean_model_score.toFixed(2)}`:'播放结束';if(stop!==null&&video.currentTime>=stop){video.pause();stop=null}});
section.querySelectorAll('[data-segment]').forEach(button=>button.onclick=async()=>{const s=data.segments[Number(button.dataset.segment)];video.currentTime=s.start_seconds;stop=s.end_seconds;try{await video.play()}catch(error){status.textContent='点击视频播放按钮继续';}});
}</script></body></html>'''
    (output / 'index.html').write_text(html)
