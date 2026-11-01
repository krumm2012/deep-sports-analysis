"""Run the same pure JS used by browser sliders against interval invariants."""
import json
from pathlib import Path
import subprocess


LOGIC = Path(__file__).parents[1]/'src/dsa/pages/semantic_timeline.js'


def js_check(body):
    subprocess.run(['node', '-e', LOGIC.read_text()+'\nconst assert=require("node:assert/strict");\n'+body], check=True, capture_output=True, text=True)


def test_shared_boundary_drag_rounds_and_clamps_without_changing_outer_coverage():
    js_check('''
const rows=[{start_frame:0,end_frame_exclusive:20,phase:'引拍'},
{start_frame:20,end_frame_exclusive:35,phase:'挥拍'},
{start_frame:35,end_frame_exclusive:60,phase:'随挥'}];
assert.equal(PhaseTimeline.moveBoundary(rows,0,24.6),true);
assert.equal(rows[0].end_frame_exclusive,25);assert.equal(rows[1].start_frame,25);
PhaseTimeline.moveBoundary(rows,0,1000);assert.equal(rows[0].end_frame_exclusive,34);
assert.equal(rows[1].end_frame_exclusive-rows[1].start_frame,1);
PhaseTimeline.moveBoundary(rows,1,-1000);assert.equal(rows[1].end_frame_exclusive,35);
assert.equal(rows[0].start_frame,0);assert.equal(rows[2].end_frame_exclusive,60);
for(let i=1;i<rows.length;i++)assert.equal(rows[i-1].end_frame_exclusive,rows[i].start_frame);
assert.equal(PhaseTimeline.moveBoundary(rows,0,NaN),false);
''')


def test_cycle_continuity_preserves_all_frames_and_original_provenance():
    js_check('''
const phase=['引拍','无法判断','挥拍','无法判断','随挥'],bounds=[0,20,24,35,40,60];
const rows=phase.map((p,i)=>({phase:p,start_frame:bounds[i],end_frame_exclusive:bounds[i+1]}));
const before=JSON.stringify(rows),r=PhaseTimeline.connectStages(rows,10);
assert.equal(JSON.stringify(rows),before);assert.equal(r.filledFrames,9);
assert.deepEqual(r.segments.map(s=>s.phase),['引拍','挥拍','随挥']);
assert.equal(r.segments[0].end_frame_exclusive,22);assert.equal(r.segments[1].start_frame,22);
assert.equal(r.segments[1].end_frame_exclusive,38);assert.equal(r.segments[2].start_frame,38);
assert.equal(r.segments[0].start_frame,0);assert.equal(r.segments[2].end_frame_exclusive,60);
assert.ok(r.segments[1].continuity_sources.length>=3);
''')


def test_invalid_mixed_long_gaps_and_reverse_phase_order_are_not_filled():
    cases = [('无效动作', 4, '引拍', '挥拍'), ('混合/需调整区间', 4, '引拍', '挥拍'),
             ('无法判断', 11, '引拍', '挥拍'), ('无法判断', 4, '随挥', '引拍'),
             ('无法判断', 4, '准备', '引拍')]
    js_check('const cases='+json.dumps(cases, ensure_ascii=False)+''';
for(const [phase,gap,left,right] of cases){
 const rows=[{start_frame:0,end_frame_exclusive:20,phase:left},
 {start_frame:20,end_frame_exclusive:20+gap,phase},
 {start_frame:20+gap,end_frame_exclusive:60,phase:right}];
 const result=PhaseTimeline.connectStages(rows,10);assert.equal(result.joined,0);assert.deepEqual(result.segments,rows);
}
''')


def test_same_phase_fragments_merge_but_real_uncovered_gaps_are_untouched():
    js_check('''
const rows=[{start_frame:0,end_frame_exclusive:10,phase:'引拍'},
 {start_frame:10,end_frame_exclusive:13,phase:'无法判断'},
 {start_frame:13,end_frame_exclusive:25,phase:'引拍'}];
const r=PhaseTimeline.connectStages(rows,10);assert.equal(r.segments.length,1);assert.equal(r.segments[0].end_frame_exclusive,25);
rows[1].start_frame=11;assert.equal(PhaseTimeline.connectStages(rows).joined,0);
assert.equal(PhaseTimeline.moveBoundary(rows,0,8),false);
''')


def test_stroke_type_applies_to_one_cycle_only_and_keeps_shared_phases_separate():
    js_check('''
const phase=['准备','引拍','挥拍','随挥','引拍','挥拍','随挥','回位/移动'];
const rows=phase.map((p,i)=>({start_frame:i*10,end_frame_exclusive:(i+1)*10,phase:p}));
assert.deepEqual(PhaseTimeline.strokeCycle(rows,2),{start:1,end:3});
assert.deepEqual(PhaseTimeline.strokeCycle(rows,5),{start:4,end:6});
assert.equal(PhaseTimeline.strokeCycle(rows,0),null);
assert.equal(PhaseTimeline.strokeCycle(rows,7),null);
''')
