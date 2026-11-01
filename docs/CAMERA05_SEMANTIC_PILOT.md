# Camera05 supervised pilot, 2026-10-31

The 24 human-confirmed intervals exported at `2026-10-31T11:51:43.892Z`
were used to fit two interval classifiers on `ubuntu32` through the existing
`ubuntu32-sam5090` SSH alias. The original 81/21 split and seed0 checkpoint
remain the sources of clip membership and candidate segmentation.

Results are in `data/projects/camera05_custom/semantic-pilot/` on this Mac and
under `/root/deep-sports-analysis/` on the 5090 host. The report is served at
<http://127.0.0.1:8768/projects/camera05_custom/semantic-pilot/>.

| Task | Intervals / recordings | Grouped correct | Macro F1 | Majority correct |
|---|---:|---:|---:|---:|
| Shot type | 23 / 19 | 22/23 | 0.888 | 20/23 |
| Phase | 10 / 10 | 8/10 | 0.500 | 7/10 |

These are training-pool leave-one-recording-out checks, **not** the semantic
accuracy of the original 21 validation recordings. No semantic labels exist
for that set; `validation_semantic_metrics` is explicitly null. The data are
candidate-sampled, from the same person/session. Neighbouring recordings can
remain dependent. No temporal guard was added within this small pilot.

Shot labels are 20 forehand and 3 volley intervals (volleys from only two
recordings). Phase labels are 7 recovery/movement, 2 ready, 1 swing. Mixed and
unknown phase labels are excluded only from that head. The singleton swing
has no training example in its held-out fold; that error remains in the
aggregate metrics. Missing actions cannot be recognized by this classifier.

Each interval contributes one sample. Features use only normalized body
coordinates inside the interval: hip-centered pose mean, standard deviation,
net change, joint speed statistics, and hip motion. No MoSeq IDs or latent
features enter the classifier. Each fold separately fits StandardScaler and
class-balanced LogisticRegression with fixed C=0.1 and random_state=0. There
was no hyperparameter selection based on these results. The portable model
parameters are stored in JSON; scores are not calibrated probabilities.

The final two heads were fitted on all eligible reviewed training intervals.
Their provisional predictions cover 81 training and 21 validation clips,
using the existing seed0 segment boundaries. Segments shorter than four
frames receive no semantic prediction. These boundaries are not semantic
phase boundaries. All outputs are marked unreviewed model candidates.

Reproduce on the existing 5090 environment (scikit-learn 1.9.1, NumPy 1.26.4):

```sh
cd /root/deep-sports-analysis
PYTHONPATH=src .venv-moseq/bin/python -m dsa.measure.semantic_pilot \
  data/projects/camera05_custom \
  data/projects/camera05_custom/semantic-labeling/human-labels-20261031T115143Z.json \
  data/projects/camera05_custom/semantic-pilot-reproduction
```

An existing output directory is rejected to preserve earlier results. The
summary records code, labels, manifest and checkpoint hashes, versions, and
every held-out fold. Per-clip outputs also record input and segmentation
hashes. The original label export is retained byte-for-byte as `labels.json`.

Next data work: add independent examples of volley, ready and swing; split
the 11 mixed intervals into single phases; annotate a separate held-out set
before estimating semantic validation performance.

## Second annotation round

The follow-up annotation workspace is at
<http://127.0.0.1:8768/projects/camera05_custom/semantic-labeling-round2/>.
It has 11 mixed-interval tasks and 12 new candidates from distinct training
recordings (3 selected by the volley head, 4 by ready, 5 by swing). These are
model-selected proposals; the categories are not human ground truth. All
new labels begin empty. Existing human-confirmed shot labels are shown on
the mixed tasks. No MoSeq training was started for this follow-up.

The editor supports slow playback, frame stepping, splitting at the current
frame, time-range edits, multiple intervals per task, explicit confirmation,
browser-local draft saving, and JSON export/import. Mixed tasks must retain
continuous coverage of the original reviewed interval. Completed replacements
replace their original parent record in the export; uncompleted tasks retain
the original reviewed record. The other original labels are preserved.

`validation.html` supplies the first three complete validation clips without
model predictions. Its labels are exported separately and explicitly rejected
by the training entry point. These three clips are an initial annotation
batch, not a replacement for evaluating the whole 21-clip holdout.

Rebuild the workspace from the existing pilot outputs:

```sh
PYTHONPATH=src .venv/bin/python -m dsa.pages.semantic_labeling \
  data/projects/camera05_custom \
  data/projects/camera05_custom/semantic-labeling-round2
```

After the user returns actual reviewed supplemental labels, train a new
semantic pilot directory using the explicit registry (do not overwrite v1):

```sh
PYTHONPATH=src .venv-moseq/bin/python -m dsa.measure.semantic_pilot \
  data/projects/camera05_custom \
  /path/to/camera05-round2-training-labels.json \
  data/projects/camera05_custom/semantic-pilot-round2 \
  --candidates data/projects/camera05_custom/semantic-labeling-round2/candidates.json
```

Registry hashes, parent-child identities, allowed ranges, source membership,
and overlapping supervised targets are checked before fitting. Use the
validation export only for a separate comparison after annotation.

### Editable automatic segmentation

At the user's request, the 23 training tasks now have 65 proposed intervals,
all pending human confirmation. `auto-segments.json` stores these separately
from the candidate registry and labels. It does not change the queue ID or
the browser storage key. Pristine tasks receive proposals automatically;
modified or confirmed tasks retain their existing values. The pre-automatic
browser export is backed up as `progress-before-auto-segmentation.json`.

Boundaries minimize within-segment variation of smoothed hip-relative limb
positions with a small duration penalty, subject to at least four frames per
segment. The number of segments is a modest editing budget: prior visual
phase order for mixed tasks, otherwise 2–5 segments based on interval length.
This is a pose-change proposal, not a trained semantic-boundary detector.
New candidates include up to ten context frames on either side.

Mixed-task phase suggestions use the earlier provisional visual review's
phase order, where available. Other phase and shot suggestions use the
already-fitted pilot's portable parameters. Ambiguous phase names and weak
model scores remain unknown. Backswing/follow-through names in the mixed
tasks are prior visual suggestions, not additional trained categories.
No model was fitted or refitted for this operation.

The editor supports a clickable phase timeline, linked adjacent boundaries,
one-frame boundary nudges with video preview, merging adjacent intervals,
and restoring the pre-automatic draft. Automatic suggestions remain outside
exported reviewed records until the user explicitly confirms a task.
Validation annotations retain the separate page without semantic suggestions.

```sh
PYTHONPATH=src .venv/bin/python -m dsa.pages.semantic_autosegment \
  data/projects/camera05_custom \
  data/projects/camera05_custom/semantic-labeling-round2
```

## Five-stage weak supervision trial

The user standardized the taxonomy to ready → backswing → swing → follow-through
→ recovery/movement, then explicitly selected a weak supervision trial using
both human labels and automatic suggestions. The result is kept separately in
`semantic-five-phase-weak-v1/`; no MoSeq model was refitted.

The annotation editor now uses five consistent stage colors. Unknown and mixed
are review states, not additional learned stages. Candidate action cycles are
proposed around wrist-motion peaks, with pose-based boundaries constrained to
the stage order. Partial cycles omit stages with insufficient frame support.
The user's manually adjusted 8.56-second boundary in candidate-01 is preserved.
Untouched earlier automatic drafts can migrate to the current proposal revision;
manual and confirmed drafts remain unchanged. There are now 82 proposed draft
intervals across 23 tasks; 3 have unresolved phases and are excluded from fitting.

The actual trial uses 89 intervals from 29 training recordings: 10 human
references and 79 weak references. Human/weak counts by stage are 2/14 ready,
0/18 backswing, 1/16 swing, 0/17 follow-through, and 7/14 recovery/movement.
Weak samples remain `human_reviewed: false` in `training-samples.json`.
Source weights are 1 for human and 0.25 for weak labels. Each fold balances
classes using the sum of source weights and fits its weighted scaler separately.

Leave-one-recording-out checks remove both human and weak samples from the
held-out recording. The human check agrees on 7/10 intervals, covering only
3 stages (supported-stage macro F1 0.544). The majority baseline also agrees
on 7/10. Weak-label agreement is 50/79 = 63.3%, macro F1 0.633; it is not human
accuracy. The result does not establish five-stage generalization or semantic
boundary accuracy. No hyperparameters were tuned on these checks.

The frozen classifier generates predictions for the existing 81/21 clips.
Prediction features use an 11-frame sliding window. Cyclic stage-order smoothing
and a four-frame minimum duration are explicit priors; the model is not a
five-stage ground-truth boundary detector. Parameters, provenance, checks,
per-frame NPZ outputs and per-clip JSON are retained in the new directory.

```sh
PYTHONPATH=src .venv-moseq/bin/python -m dsa.measure.semantic_weak \
  data/projects/camera05_custom \
  data/projects/camera05_custom/semantic-labeling-round2/five-phase-weak-training-input.json \
  data/projects/camera05_custom/semantic-labeling-round2/candidates.json \
  data/projects/camera05_custom/semantic-five-phase-weak-reproduction \
  --weak-weight 0.25
```

Review at <http://127.0.0.1:8768/projects/camera05_custom/semantic-five-phase-weak-v1/>.
Next: confirm the five-stage proposals, especially backswing and follow-through,
then refit and evaluate separately annotated holdout recordings.
