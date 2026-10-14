# E38c probe criteria r2 (FROZEN, supersedes the GUI-Odyssey section of r1)

r1 measured GUI-Odyssey task_done mappability at 99 % clean — but every clean
row was label `true` (zero IMPOSSIBLE episodes in the sample): as an eval
family it is degenerate. r2 measures whether the **negative class can be
constructed from mid-episode screenshots by rule alone**. CORD and DocLayNet
are not re-probed (r1 verdicts stand: PASS and FAIL respectively).

Frozen before any r2 row or screenshot is fetched. After freezing, the sample
rule, the negative-construction rule and the threshold do not change;
anything else needs an `-r3` file. No training, no quota, no full downloads;
asset fetches capped at 200 screenshots plus the 20-case agreement subset.

## Sample rule (frozen)

Source repo `OpenGVLab/GUI-Odyssey`, config `default`, split `all`, same
`/rows` fetch as r1 (first min(2 000, split_size) rows in server order,
streaming fallback if `/rows` fails). From those rows keep only episodes whose
final step action is `COMPLETE`, compute `sha256` over the server row JSON as
in r1, and take the **first 100 episodes in ascending sha256 order**.

## Negative construction (frozen)

For a sampled episode with `n = len(steps)` steps and final action COMPLETE:

- `k = int(sha256(episode_id) % floor(n / 2))` — the negative screenshot is
  `steps[k].screenshot` (first half of the episode, chosen by rule; never the
  penultimate step, which can already show the post-"send" result);
- positive screenshot = `steps[-1].screenshot` (the r1 rule).

Label: positive screenshot → `true`; negative screenshot → `false`. Both
cases of one episode share `group_id = episode_id`.

A pair is **clean** iff: the negative screenshot file exists in the repo file
tree and downloads individually; the negative path differs from the final
path; and the task is **not trivially already completed** in the negative.

Declared limitation: "not trivially already completed" cannot be fully
decided by rule (an early frame may already show the goal achieved, e.g. for
"open app X" tasks). The rule-based proxy is the first-half construction
above; residual judgment is handled by an agreement subset: for a
seed-frozen choice of 20 clean pairs (seed 7 over the sampled episode ids),
both screenshots are stored under `data/e38c-probe/gui_odyssey/r2_pairs/` and
the probe operator's provisional per-case verdict is recorded in
`runs/e38c-probe/probe_r2.json`; the coordinator reviews the same 20 pairs and
the agreement rate is reported before the E38c criteria are frozen. The 20
pairs are inspection artifacts, not eval rows.

## Threshold (frozen)

r2 passes iff ≥ 70 of the 100 sampled episodes yield a clean pair (≥ 70 %).
Download-failure of either screenshot counts against clean. The threshold is
not relaxed after measurement.

## Outputs

- `data/e38c-probe/gui_odyssey/r2_rows.jsonl` (per-episode: k, negative and
  final paths, local copies, clean flag, reason), screenshots under
  `r2_shots/` and the 20 inspection pairs under `r2_pairs/`, `manifest.json`
  with SHA-256 of stored files.
- `runs/e38c-probe/probe_r2.json`: clean share, rejection reasons, and the
  operator's provisional verdicts for the 20 inspection pairs.
