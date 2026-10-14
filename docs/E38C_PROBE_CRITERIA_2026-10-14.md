# E38c probe criteria (FROZEN 2026-10-14)

E38c-probe answers one question: **what share of real permissive rows can be
turned into contract questions with labels derived by rule alone — no human
judgment, decisive value visible in the image, no ambiguity?** It is a
measurement gate in front of any bulk download: if a source cannot map, we
learn it for the cost of ~100 rows, not ~8-30 GB.

Frozen before any row is fetched. Dataset cards and schemas were consulted;
no data row has been read. After freezing, sample rules, mapping tables and
thresholds do not change; anything else needs a new criteria file.

No training, no Kaggle quota, no full downloads. Asset fetches are capped at
100 images per source; annotation/metadata fetches are capped at the first
2 000 rows per split.

## Sources (exact repos, frozen)

| Source | Hub repo | Config | Split | Rows |
|---|---|---|---|---|
| CORD v2 | `naver-clova-ix/cord-v2` | `default` | `test` (100 rows, non-train) | all 100 |
| DocLayNet (small mirror) | `pierreguillou/DocLayNet-small` | `DocLayNet_2022.08_processed_on_2023.01` | `test` (non-train) | first 2 000 eligible |
| GUI-Odyssey | `OpenGVLab/GUI-Odyssey` | `default` | `all` | first 2 000 eligible |

Declared limitations, frozen with the criteria:

- DocLayNet: the official repo (`docling-project/DocLayNet`, renamed from
  `ds4sd`) is a legacy dataset script that the datasets-server cannot serve;
  the probe uses the parquet mirror `pierreguillou/DocLayNet-small`
  (CDLA-Permissive-1.0 data, conversion by the mirror author, provenance
  declared). The mirror exposes layout labels but **no document-level
  `doc_category`**, which the doc_type mapping needs; rows failing for this
  reason are recorded as `missing_doc_category`, and the probe result for
  DocLayNet is an **access-route result**, not a verdict on the core dataset.
- GUI-Odyssey: the parquet conversion ships a single `all` split; train/test
  membership lives in separate split files inside the annotation zips, which
  this probe does not download. Sampled rows are marked `split=unknown`;
  E38c proper must re-derive membership from the official split files before
  any evaluation use.
- Licenses: CORD CC BY 4.0, DocLayNet CDLA-Permissive-1.0, GUI-Odyssey
  CC BY 4.0 — re-verified from primary texts in `docs/data-licenses.md`;
  probe use is evaluation-side data inspection, attribution obligations noted.

## Sample rule (frozen)

For each source: fetch the first `min(2 000, split_size)` rows of the frozen
split via the datasets-server `/rows` API in ascending `row_idx` order
(offset pagination, length 100). Compute `sha256` over each row's server JSON
(`{"row_idx":…, "row":…}`, UTF-8, key order as returned). Take the **first
100 rows in ascending sha256 order**. For CORD `test` (100 rows) this
degenerates to the full split. No other filtering before the hash sort.

## Mapping table (frozen)

| Source | Family | Label rule (ground-truth only, no judgment) | Clean row requires |
|---|---|---|---|
| CORD | `field_threshold` (+ fixed class `receipt` for `doc_type`, no discrimination claimed) | exactly one total price derivable from the ground-truth JSON (`gt_parse.total.*_price`, else sum of unique `total.*_price` values from `valid_line` entries); parse as number after removing currency separators; value > 0 | the total's text string appears among the row's annotation `text` values (decisive value printed on the image) |
| DocLayNet | `doc_type` | the row's document category equals one of the six contract classes (`invoice`, `receipt`, `contract`, `form`, `letter`, `other`); synonym table: **empty by design** — no interpretive mapping is allowed at probe stage | category present in row metadata |
| GUI-Odyssey | `task_done` | last step `action` == `COMPLETE` → `true`; == `IMPOSSIBLE` → `false`; anything else → reject (`no_terminal_action`) | final step's screenshot file exists in the source repo file tree and downloads individually (one PNG per sampled row) |

A **clean row** satisfies all of: (i) its label is derivable by the frozen
rule alone; (ii) the decisive value is visible (CORD: total text in annotation
strings; GUI-Odyssey: final screenshot fetched OK; DocLayNet: n/a beyond
metadata); (iii) no ambiguity flag from the rule set below.

Ambiguity rejections (any hit fails the row): CORD total text not parseable as
a number by the frozen parse rule — strip currency symbols; ≥ 2 separators →
thousands; exactly one separator followed by exactly 3 digits → thousands
(ID convention, IDR totals carry no cents); one separator followed by 1-2
digits → decimal; both separators present → EN pattern `,` thousands + `.`
decimal, else ID pattern, else reject as ambiguous; GUI-Odyssey final action
both COMPLETE-flagged and IMPOSSIBLE-flagged in `ps` notes; DocLayNet category
outside the six classes (→ `no_doc_type_match`, expected for
laws/tenders/reports — that IS the measurement).

## Threshold (frozen)

A source **passes** iff ≥ 70 of its 100 sampled rows are clean (≥ 70 %).
Below the threshold the source degrades to **declared evaluation background**
(not a scored eval family). If no source passes, the fallback per the council
verdict is ~300 hand-labelled real screenshots as the primary eval floor.
The threshold is not relaxed after measurement.

## Outputs

- `data/e38c-probe/` (git-ignored): per-source `rows.jsonl` (server row JSON
  + derived fields + rejection reason), fetched images, `manifest.json` with
  SHA-256 of every stored file.
- `runs/e38c-probe/probe.json`: clean share per source and per family, per-row
  rejection reasons, estimated evaluation-subset size per passing source
  (fetched-rows basis, extrapolated only as a note, not a commitment).
