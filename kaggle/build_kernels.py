"""Generate self-contained Kaggle kernel scripts.

Kaggle script kernels are single files, and dataset-upload permissions are
not available to the MCP token, so each protocol script (kaggle/e37_*.py) is
concatenated with an embedded copy of src/mksh_v/* into
kaggle/gen/<name>.gen.py. The kernel materialises the package under
/kaggle/working/pkg and imports it. Source of truth stays in src/; regenerate
after every edit:

    python kaggle/build_kernels.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "mksh_v"
KAGGLE = ROOT / "kaggle"
GEN = KAGGLE / "gen"
MODULES = ("__init__.py", "sequence.py", "splitting.py", "visual.py",
           "synth_doc_type.py", "arch_a1v.py", "arch_a1v_aux.py", "train_compare.py", "rules_baseline.py")

HEADER = '''# AUTO-GENERATED from kaggle/{name}.py + src/mksh_v/* by kaggle/build_kernels.py.
# Do not edit: edit the sources and regenerate.
from __future__ import annotations

import pathlib
import sys

PKG_FILES = {pkg_repr}
ROOT_FILES = {root_repr}

_root = pathlib.Path("/kaggle/working/pkg")
for _rel, _src in PKG_FILES.items():
    _p = _root / "mksh_v" / _rel
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_text(_src, encoding="utf-8")
for _rel, _src in ROOT_FILES.items():  # sibling modules, e.g. e39_train for E39-r2
    (_root / _rel).write_text(_src, encoding="utf-8")
sys.path.insert(0, str(_root))

'''
# kernels that import another protocol script as a module (shipped at pkg root)
ROOT_MODULES = {"e39r2_comparators": ("e39_train.py",), "e41_train": ("e39_train.py",),
                "e44_train": ("e39_train.py", "e41_train.py"),
                "e45_train": ("e39_train.py", "e41_train.py", "e44_train.py"),
                **{k: ("e39_train.py", "e41_train.py", "e44_train.py")
                   for k in ("e46_smoke", "e46a_train", "e46b_train", "e47_train", "e49_train",
                             "e52_train", "e54_train")},
                "e48_calib": ("e42_calib.py",), "e53_calib": ("e42_calib.py",)}


def build(name: str) -> pathlib.Path:
    pkg = {rel: (SRC / rel).read_text(encoding="utf-8") for rel in MODULES}
    root = {rel: (KAGGLE / rel).read_text(encoding="utf-8") for rel in ROOT_MODULES.get(name, ())}
    body = (KAGGLE / f"{name}.py").read_text(encoding="utf-8")
    # the header already carries the future import; a second one would not parse
    body = body.replace("from __future__ import annotations\n", "", 1)
    text = HEADER.format(name=name, pkg_repr=repr(pkg), root_repr=repr(root)) + body
    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / f"{name}.gen.py"
    out.write_text(text, encoding="utf-8")
    compile(text, str(out), "exec")  # syntax gate before pushing
    print(f"built {out} ({len(text)} bytes, syntax OK)")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*",
                    default=["e37_preflight", "e37_c1", "e37_c2", "e37_c4", "e37_c3",
                             "e39_ocrprep", "e39_train"])
    args = ap.parse_args()
    for name in args.names:
        build(name)


if __name__ == "__main__":
    main()
