# User-defined names for existing MoSeq states

The workflow now lets the user directly name frozen model states. The user
defines the action phase; no automatic classifier supplies these names.
Several states may share a phase. Uncertain states may remain unnamed.

Two independent mapping workspaces are available:

| Model | State IDs | FPS | Workspace |
|---|---|---:|---|
| `camera05/2025_1830`, reference seed0 | 0–4 | 30 | `/projects/camera05/2025_1830/state-labeling/` |
| `camera05_custom`, formal seed0 trained on 81 clips | 0–7 | 25 | `/projects/camera05_custom/state-labeling/` |

The old short example has 13 representative occurrences; the formal workspace
uses 24 existing training candidates from distinct time periods, three per
state. Every displayed range is checked against the raw state array. The
formal model's state counts include the existing 81 training and 21 validation
outputs. Naming applies to that same model; identifiers are not shared across
the two checkpoints or other random seeds.

The editor supports state-specific video playback, surrounding context, full
clip playback, the original raw-state timeline, representative thumbnails,
five phase choices, notes, explicit confirmation, browser-local saving, and
JSON export/import. Names begin empty. Changing a phase clears confirmation.
The exported mapping contains checkpoint SHA256, dataset ID, model ID, FPS,
and one entry for each state, including pending entries.

Build the workspaces:

```sh
PYTHONPATH=src .venv/bin/python -m dsa.pages.semantic_state_labeling build \
  data/projects/camera05/2025_1830 \
  data/projects/camera05/2025_1830/state-labeling --mode single

PYTHONPATH=src .venv/bin/python -m dsa.pages.semantic_state_labeling build \
  data/projects/camera05_custom \
  data/projects/camera05_custom/state-labeling --mode dataset
```

After the user defines and returns a mapping, apply its confirmed entries to
the existing outputs, preserving all raw IDs and temporal boundaries:

```sh
PYTHONPATH=src .venv/bin/python -m dsa.pages.semantic_state_labeling apply \
  data/projects/camera05_custom/state-labeling/states.json \
  /path/to/camera05-81clip-model-state-names.json \
  data data/projects/camera05_custom/user-state-phases-v1
```

Application checks checkpoint, result hashes, FPS and state IDs before writing
to a fresh directory. Pending states keep `phase: null`. Model fitting is not
part of naming or application. The weak semantic classifier reports remain
separate prior experiments, and the proposed second classifier retraining was
not started after the user clarified this state-naming workflow.

## Mixed, invalid and uncertain states

Mappings now export schema version 2 with `annotation_type` set to `single`,
`mixed`, `invalid`, or `uncertain` (empty for an unassigned draft). Existing
single-stage confirmations and notes migrate under the same browser key.
Version-1 JSON files remain importable.

Mixed states carry at least two `mixed_phases` once confirmed and have no
single `phase`. Invalid states set `excluded: true` only when confirmed.
Uncertain states remain explicitly unresolved. Only a confirmed `single`
annotation sets `usable_for_single_phase: true`. Applying mixed or invalid
annotations retains raw state IDs and temporal boundaries; it never assigns
one arbitrary phase to a mixture or removes frames from the time axis.

The user's old-model mapping currently confirms states 0 as ready, 1 and 2 as
backswing, and 4 as recovery/movement. The note on state 3 says it contains
swing and follow-through. That note is represented as a pending mixed-stage
draft; it is not an invalid-action label. Existing four confirmations and the
original note are retained. The pre-upgrade mapping export is backed up as
`state-labeling/mapping-before-state-types.json` in the old example project.
