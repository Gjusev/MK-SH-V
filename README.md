# MK-SH-V

A small open **decision model that reads documents and screenshots**: given a
state (text and images), a question and described options, it returns
calibrated probabilities, a choice, and an explicit abstention when confidence
is too low. Visual sibling of [MK-SH-1](https://huggingface.co/Gjusev/MK-SH-1).

**Status: E37 executed (GO-A1), pre-data.** No trained checkpoint is released
yet. The feasibility gate passed on one Kaggle T4: NeoMME-260M + the marker
head trains at 768/1024/1536 px, marker tokens resize and receive gradients
in both factorized tables, text parity holds (testC 1.000 vs MK-SH-1's 0.979;
task 0.837 vs 0.800, one seed, exploratory), and the visual smoke test reads
the class from pixels (dev 1.000, image-swap 0.000). Design:
[`docs/design.md`](docs/design.md). Data and model licenses:
[`docs/data-licenses.md`](docs/data-licenses.md). Feasibility criteria:
[`docs/E37_CRITERIA_2026-10-14.md`](docs/E37_CRITERIA_2026-10-14.md).

License: Apache-2.0.
