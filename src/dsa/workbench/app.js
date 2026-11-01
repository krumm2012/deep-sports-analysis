'use strict';
const $=s=>document.querySelector(s),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let project=null,summary=null,prediction=null,busy=false;
const headers={'Content-Type':'application/json','X-Workbench-Request':'1'};
async function api(path,options={}){const r=await fetch(path,options);const value=await r.json();if(!r.ok)throw Error(value.error||`请求失败 ${r.status}`);return value}
const base=()=>'/api/projects/'+encodeURIComponent(project);
function message(text,error=false){$('#message').textContent=text;$('#message').className=error?'error':'muted'}
async function action(fn){try{await fn()}catch(e){message(e.message,true)}}
function download(value,name){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function options(el,rows,previous){el.innerHTML=rows.map(r=>`<option value="${esc(r.value)}">${esc(r.label)}</option>`).join('');if(rows.some(r=>r.value===previous))el.value=previous}
async function refresh(){
 if(!project)return;const next=await api(base());summary=next;
 $('#saved').textContent='本机标定版本 '+next.annotation_revision+' · '+new Date(next.saved_at).toLocaleString();
 $('#annotation-summary').textContent=`已确认 ${next.completed} / ${next.total} 段视频，可继续查看或修正。`;
 $('#editor').href=next.editor_url;$('#report').href=next.report_url;
 const current=next.models.find(m=>m.version===next.current.version);
 $('#metrics').innerHTML=[['人工标定',`${next.completed}/${next.total}`],['当前模型',next.current.version.replace('semantic-pose-','')],['人工训练帧',current?.training_frames.toLocaleString()||'—'],['内部核对一致率',current?`${(100*current.accuracy).toFixed(1)}%`:'—']].map(([name,value])=>`<div class="metric"><strong>${esc(value)}</strong><span>${esc(name)}</span></div>`).join('');
 const model=$('#model').value;options($('#model'),next.models.map(m=>({value:m.version,label:(m.version.includes('-wb-')?'工作台训练 '+new Date(m.created_at).toLocaleString():m.version.replace('semantic-pose-',''))+' · '+m.human_clips+'段人工'})),model||next.current.version);
 const clip=$('#clip').value;
 const imports=next.jobs.filter(j=>j.action==='import_video'&&j.status==='succeeded');
 const ready=new Set(next.jobs.filter(j=>j.action==='remote_pose'&&j.status==='succeeded').map(j=>j.result.import_id));
 options($('#clip'),[...next.clips.map(c=>({value:'clip:'+c.id,label:c.id+' · '+(c.split==='validation'?'留出验证':'训练')})),...imports.filter(j=>ready.has(j.id)).map(j=>({value:'import:'+j.id,label:j.result.filename+' · 新导入'}))],clip);
 const running=next.jobs.some(j=>['queued','running'].includes(j.status));
 $('#train').disabled=running||next.completed!==next.total;$('#import').disabled=running;
 $('#imports').innerHTML=imports.length?imports.map(j=>`<div class="row"><span>${esc(j.result.filename)} · ${j.result.frames}帧</span>${ready.has(j.id)?'<span class="pill">姿态已取回 · 可在预测视频中选择</span>':`<button data-pose="${j.id}" ${running?'disabled':''}>提交5090提取姿态</button>`}</div>`).join(''):'还没有导入的新视频';
 const names={train:'Mac监督训练',import_video:'导入与标准化',remote_pose:'5090检测与姿态'};
 const statuses={queued:'等待中',running:'运行中',succeeded:'已完成',failed:'失败',interrupted:'已中断'};
 $('#jobs').innerHTML=next.jobs.length?`<table><thead><tr><th>任务</th><th>状态</th><th>创建时间</th><th>结果</th><th>操作</th></tr></thead><tbody>${next.jobs.map(j=>`<tr><td>${names[j.action]||esc(j.action)}</td><td>${statuses[j.status]||esc(j.status)}</td><td>${new Date(j.created_at).toLocaleString()}</td><td>${j.result?.report_url?`<a target="_blank" href="${esc(j.result.report_url)}">打开新模型报告 ↗</a>`:esc(j.error||j.result?.filename||'—')}</td><td><button data-log="${j.id}">日志</button> ${['failed','interrupted'].includes(j.status)?`<button data-retry="${j.id}" ${running?'disabled':''}>重试</button>`:''}</td></tr>`).join('')}</tbody></table>`:'<p class="empty">任务会在这里显示；既有v5模型已经可以直接预测。</p>';
}
$('#project').onchange=()=>action(async()=>{project=$('#project').value;await refresh()});
$('#refresh').onclick=()=>action(refresh);
$('#train').onclick=()=>action(async()=>{const job=await api(base()+'/train',{method:'POST',headers});message('已创建监督训练任务，将保留旧模型。');await refresh()});
$('#export').onclick=()=>action(async()=>download(await api(base()+'/export'),project+'-workbench-labels.json'));
$('#check-backend').onclick=()=>action(async()=>{$('#backend').textContent='正在检查…';try{const b=await api(base()+'/backend-check',{method:'POST',headers});$('#backend').textContent=b.online?'已连接 · '+b.device:(b.error||'未配置或暂时离线')}catch(e){$('#backend').textContent='连接失败';throw e}});
$('#import').onclick=()=>action(async()=>{const file=$('#upload').files[0];if(!file)throw Error('先选择一个MP4短片');$('#import').disabled=true;message('正在上传到Mac…');await api(base()+'/imports?filename='+encodeURIComponent(file.name),{method:'POST',headers:{'X-Workbench-Request':'1','Content-Type':'video/mp4'},body:file});$('#upload').value='';message('上传完成，正在后台标准化视频。');await refresh()});
$('#imports').onclick=e=>{const b=e.target.closest('[data-pose]');if(b)action(async()=>{await api(base()+'/remote-pose',{method:'POST',headers,body:JSON.stringify({import_id:b.dataset.pose})});message('已提交5090，完成后将自动取回姿态。');await refresh()})};
$('#jobs').onclick=e=>{const log=e.target.closest('[data-log]'),retry=e.target.closest('[data-retry]');if(log)action(async()=>{const v=await api('/api/jobs/'+log.dataset.log+'/log');$('#job-log').hidden=false;$('#job-log').textContent=v.log||'等待任务输出…'});if(retry)action(async()=>{await api('/api/jobs/'+retry.dataset.retry+'/retry',{method:'POST',headers});message('已用保留的输入重新创建任务。');await refresh()})};
const colors={'准备':'#e6f0ff','引拍':'#f0e9ff','挥拍':'#fff0db','随挥':'#e1f4ea','回位/移动':'#eaf0f3'};
$('#predict').onclick=()=>action(async()=>{const selected=$('#clip').value;if(!selected)throw Error('没有可预测的视频');const [kind,id]=selected.split(':');const request={version:$('#model').value,stroke_type:$('#stroke').value,[kind==='import'?'import_id':'clip_id']:id};$('#predict').disabled=true;try{prediction=await api(base()+'/predict',{method:'POST',headers,body:JSON.stringify(request)});$('#prediction').hidden=false;$('#video').src=kind==='import'?'/api/imports/'+id+'/video':'/projects/'+project+'/clips/'+encodeURIComponent(id)+'/source.mp4';$('#timeline').innerHTML=prediction.segments.map(s=>`<button class="phase" data-start="${s.start_seconds}" style="flex:${s.end_frame_exclusive-s.start_frame};background:${colors[s.phase]||'#e3e5e4'}" title="${s.start_seconds.toFixed(2)}–${s.end_seconds.toFixed(2)}秒 · ${esc(s.phase||'不确定')}">${esc(s.phase||'·')}</button>`).join('');$('#prediction-stats').textContent=`${prediction.frames}帧 · 通过筛选 ${(100*prediction.accepted_fraction).toFixed(1)}% · ${prediction.core_expert_used?'正手专用分支':'共享阶段模型'}。分数未经概率校准。`;message('预测完成，可点击阶段色条定位。')}finally{$('#predict').disabled=false}});
$('#timeline').onclick=e=>{const b=e.target.closest('[data-start]');if(b){$('#video').pause();$('#video').currentTime=Number(b.dataset.start)}};
$('#download-prediction').onclick=()=>{if(prediction)download(prediction,'phase-prediction.json')};
async function init(){const h=await api('/api/health');$('#health').textContent='Mac在线 · 剩余 '+h.free_gib+'GiB';if(h.free_gib<20)message('磁盘空间偏少，批量视频建议使用外置存储。');const projects=await api('/api/projects');options($('#project'),projects.map(p=>({value:p.id,label:p.id})));project=$('#project').value;if(project)await refresh();else message('尚未发现带有阶段模型的项目。',true)}
action(init);setInterval(()=>{if(!document.hidden&&project)action(refresh)},5000);
