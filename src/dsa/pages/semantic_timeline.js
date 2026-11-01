/* Shared-boundary editing and conservative continuity repair. No DOM dependency. */
const PhaseTimeline = (() => {
  const cycle = ['引拍', '挥拍', '随挥'];
  const compatibleType = (a, b) => !a.stroke_type || !b.stroke_type || a.stroke_type === '无法判断' || b.stroke_type === '无法判断' || a.stroke_type === b.stroke_type;
  function moveBoundary(rows, index, frame, minimum = 1) {
    const left = rows[index], right = rows[index + 1];
    if (!left || !right || left.end_frame_exclusive !== right.start_frame || !Number.isFinite(frame)) return false;
    const low = left.start_frame + minimum, high = right.end_frame_exclusive - minimum;
    if (low > high) return false;
    const value = Math.max(low, Math.min(high, Math.round(frame)));
    if (value === left.end_frame_exclusive) return false;
    left.end_frame_exclusive = value;
    right.start_frame = value;
    return true;
  }
  function connectStages(input, maximumGap = 10) {
    const rows = input.map(s => ({...s}));
    if (rows.some((s, i) => s.start_frame >= s.end_frame_exclusive || (i && rows[i - 1].end_frame_exclusive !== s.start_frame)))
      return {segments: rows, joined: 0, filledFrames: 0};
    let joined = 0, filledFrames = 0;
    const source = s => s.continuity_sources || [{start_frame: s.start_frame, end_frame_exclusive: s.end_frame_exclusive, phase: s.phase}];
    const merge = (a, b) => ({...a, end_frame_exclusive: b.end_frame_exclusive,
      notes: [...new Set([a.notes, b.notes].filter(Boolean))].join('；'),
      continuity_sources: [...source(a), ...source(b)]});
    // Adjacent uncertainty fragments still form one gap; never bridge an invalid/mixed region.
    for (let i = 0; i < rows.length - 1;) {
      if (rows[i].phase === rows[i + 1].phase && compatibleType(rows[i], rows[i+1]) && (cycle.includes(rows[i].phase) || rows[i].phase === '无法判断')) {
        const meaningful = cycle.includes(rows[i].phase);
        rows.splice(i, 2, merge(rows[i], rows[i + 1])); if (meaningful) joined++;
      } else i++;
    }
    for (let i = 0; i < rows.length - 2;) {
      const [left, gap, right] = rows.slice(i, i + 3);
      const l = cycle.indexOf(left.phase), r = cycle.indexOf(right.phase), frames = gap.end_frame_exclusive - gap.start_frame;
      if (l >= 0 && (r === l || r === l + 1) && compatibleType(left,right) && gap.phase === '无法判断' && frames <= maximumGap) {
        const provenance = [...source(left), ...source(gap), ...source(right)];
        if (l === r) {
          rows.splice(i, 3, {...merge(left, right), continuity_sources: provenance});
        } else {
          const boundary = Math.round((gap.start_frame + gap.end_frame_exclusive) / 2);
          rows.splice(i, 3, {...left, end_frame_exclusive: boundary, continuity_sources: provenance},
            {...right, start_frame: boundary, continuity_sources: provenance});
        }
        joined++; filledFrames += frames;
        if (i) i--;
      } else i++;
    }
    return {segments: joined ? rows : input.map(s => ({...s})), joined, filledFrames};
  }
  function strokeCycle(rows, index) {
    if (!rows[index] || !cycle.includes(rows[index].phase)) return null;
    let start = index, end = index;
    while (start > 0 && cycle.includes(rows[start-1].phase) && rows[start-1].end_frame_exclusive === rows[start].start_frame && cycle.indexOf(rows[start-1].phase) <= cycle.indexOf(rows[start].phase)) start--;
    while (end+1 < rows.length && cycle.includes(rows[end+1].phase) && rows[end].end_frame_exclusive === rows[end+1].start_frame && cycle.indexOf(rows[end].phase) <= cycle.indexOf(rows[end+1].phase)) end++;
    return {start, end};
  }
  return {moveBoundary, connectStages, strokeCycle};
})();
