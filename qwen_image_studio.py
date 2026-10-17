#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen Image Studio
=================
A desktop studio for Qwen-Image-2.1 (and compatible Qwen-Image) models in GGUF
or safetensors format. Two workflows:

  * CREATE - text-to-image
  * EDIT   - upload one or more images and describe the change in plain words
             (add / remove / replace objects, change style or background,
             edit text inside the picture, make the background transparent ...)
  * MARK   - open an image, paint over the part you want changed (brush, eraser,
             adjustable size, undo/redo) and say what should happen there
             (add / remove / change). The area outside your marking can be kept
             pixel-identical, and an optional "focus" mode edits just a crop
             around the marking for better detail on small objects. Smart selection
             tools (magic wand, smart brush, box select) outline objects for you using
             plain colour/edge analysis - no extra libraries or models needed.

How it works
------------
The studio is a front-end for stable-diffusion.cpp's `sd-cli`, the runtime the
Qwen-Image-2.1 GGUF files are published for.  Because the quantisation type is
read from the file itself by the runtime, ANY quant works: Q2_K ... Q8_0,
K-quants, mixed-precision "HQ" builds, merged distilled builds (e.g. UltraFast),
or the original safetensors.  Nothing in this program is tied to one quant.

Model set (all configurable on the "Models" tab):
  * diffusion model   - the Qwen-Image-2.1 .gguf / .safetensors (any quant)
  * VAE               - qwen_image_2.1_vae_bf16.safetensors
  * text encoder      - Qwen3-VL-8B-Instruct (GGUF or safetensors)
  * vision projector  - mmproj-Qwen3VL-8B-Instruct-F16.gguf (needed for EDIT
                        when the text encoder is a GGUF)

Setup
-----
    pip install PyQt6
    python qwen_image_studio.py

Then open the Models tab:  "Get sd-cli..."  ->  "Download companions..."  ->
"Download from Hugging Face..." (pick any Qwen-Image-2.1 GGUF repo + file).

Only PyQt6 and the standard library are required.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import random
import re
import shlex
import shutil
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    from PyQt6.QtCore import (QEvent, QFile, QObject, QPointF, QProcess, QRect, QRectF, QSize, Qt,
                              QThread, QTimer, QUrl, pyqtSignal)
    from PyQt6.QtGui import (QAction, QBrush, QColor, QDesktopServices, QFont, QIcon,
                             QImage, QImageReader, QKeySequence, QPainter, QPen,
                             QPixmap, QPolygonF, QShortcut)
    from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
                                 QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
                                 QFrame, QGraphicsPixmapItem, QGraphicsScene,
                                 QGraphicsView, QGroupBox, QHBoxLayout, QLabel,
                                 QLineEdit, QListView, QListWidget, QListWidgetItem,
                                 QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
                                 QProgressBar, QPushButton, QScrollArea, QSlider,
                                 QSpinBox, QSplitter, QStackedWidget, QTabWidget,
                                 QVBoxLayout, QWidget, QButtonGroup)
except ImportError:  # pragma: no cover
    print("PyQt6 is required:  pip install PyQt6")
    sys.exit(1)

APP_NAME = "Qwen Image Studio"
APP_VERSION = "1.0"

ROOT = Path(os.environ.get("QWEN_STUDIO_HOME", Path.home() / "QwenImageStudio"))
MODELS_DIR = ROOT / "models"
TOOLS_DIR = ROOT / "tools"
DEFAULT_OUT_DIR = ROOT / "outputs"
CLIPBOARD_DIR = ROOT / "clipboard"
SETTINGS_FILE = ROOT / "settings.json"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_EXTS = {".gguf", ".safetensors"}
SD_CLI_NAMES = ("sd-cli", "sd-cli.exe", "sd", "sd.exe")

# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def round32(v: float, lo: int = 64) -> int:
    return max(lo, int(round(v / 32.0)) * 32)


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", s)


def fmt_elapsed(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def ensure_dirs() -> None:
    for d in (ROOT, MODELS_DIR, TOOLS_DIR, DEFAULT_OUT_DIR, CLIPBOARD_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  GGUF header reader (any quant, no third-party dependency)
# --------------------------------------------------------------------------- #

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}
_FLOAT_TYPES = {"F32", "F16", "BF16", "F64"}
# (elements per block, bytes per block) - only types whose layout is certain; anything else
# disables the completeness check instead of risking a false alarm.
_TYPE_LAYOUT = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "F64": (1, 8), "I8": (1, 1), "I16": (1, 2),
    "I32": (1, 4), "I64": (1, 8), "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22),
    "Q5_1": (32, 24), "Q8_0": (32, 34), "Q8_1": (32, 36), "Q2_K": (256, 84), "Q3_K": (256, 110),
    "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
    "IQ4_NL": (32, 18), "IQ4_XS": (256, 136),
}
_SCALARS = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
            5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8),
            12: ("d", 8)}
_BLOCK_RE = re.compile(r"(?:^|\.)([a-z_]*blocks?|layers)\.(\d+)\.")


@dataclass
class GGUFInfo:
    path: str = ""
    ok: bool = False
    error: str = ""
    version: int = 0
    arch: str = ""
    name: str = ""
    tensor_count: int = 0
    size_bytes: int = 0
    quant: str = ""            # dominant quantisation
    quant_summary: str = ""    # e.g. "Q4_K 71%, Q6_K 9%, F32 0.1%"
    blocks: int = 0            # transformer block / layer count (heuristic)
    expected_size: int = 0     # minimum file size implied by the tensor table (0 = unknown)
    truncated: bool = False    # file is smaller than the tensor table says


def _rd_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    if n > (1 << 24):
        raise ValueError("implausible string length")
    return f.read(n).decode("utf-8", "replace")


def _rd_value(f, vt: int):
    """Read one GGUF value; arrays are skipped (returns None)."""
    if vt in _SCALARS:
        fmt, size = _SCALARS[vt]
        return struct.unpack("<" + fmt, f.read(size))[0]
    if vt == 8:
        return _rd_str(f)
    if vt == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if et in _SCALARS:
            f.seek(n * _SCALARS[et][1], 1)
        elif et == 8:
            for _ in range(n):
                (ln,) = struct.unpack("<Q", f.read(8))
                f.seek(ln, 1)
        elif et == 9:
            for _ in range(n):
                _rd_value(f, 9)
        else:
            raise ValueError(f"unknown array element type {et}")
        return None
    raise ValueError(f"unknown value type {vt}")


def read_gguf_info(path: str | Path, tensors: bool = True) -> GGUFInfo:
    """Parse a GGUF header. tensors=False stops as soon as the architecture is
    known (fast, safe for multi-GB LLM files with huge tokenizer arrays)."""
    p = Path(path)
    info = GGUFInfo(path=str(p))
    try:
        info.size_bytes = p.stat().st_size
        with open(p, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError("not a GGUF file")
            (info.version,) = struct.unpack("<I", f.read(4))
            if info.version < 2:
                raise ValueError(f"GGUF v{info.version} is not supported")
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            info.tensor_count = n_tensors
            kv = {}
            for _ in range(n_kv):
                key = _rd_str(f)
                (vt,) = struct.unpack("<I", f.read(4))
                kv[key] = _rd_value(f, vt)
                if not tensors and "general.architecture" in kv:
                    break
            info.arch = str(kv.get("general.architecture", "") or "")
            info.name = str(kv.get("general.name", "") or "")
            if not tensors:
                info.ok = True
                return info

            elements: dict[str, int] = {}
            max_end, layout_known = 0, True
            groups: dict[str, list[int]] = {}   # prefix -> [count, max_idx]
            for _ in range(n_tensors):
                tname = _rd_str(f)
                (nd,) = struct.unpack("<I", f.read(4))
                dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd)) if nd else ()
                (tt,) = struct.unpack("<I", f.read(4))
                (t_off,) = struct.unpack("<Q", f.read(8))
                n_el = 1
                for d in dims:
                    n_el *= d
                tn = GGML_TYPES.get(tt, f"type{tt}")
                lay_ = _TYPE_LAYOUT.get(tn)
                if lay_ and n_el % lay_[0] == 0:
                    max_end = max(max_end, t_off + n_el // lay_[0] * lay_[1])
                else:
                    layout_known = False
                elements[tn] = elements.get(tn, 0) + n_el
                m = _BLOCK_RE.search(tname)
                if m:
                    g = groups.setdefault(m.group(1), [0, -1])
                    g[0] += 1
                    g[1] = max(g[1], int(m.group(2)))
            if layout_known and n_tensors:
                align = int(kv.get("general.alignment") or 32)
                if align > 0:
                    data_start = -(-f.tell() // align) * align
                    info.expected_size = data_start + max_end
                    info.truncated = info.size_bytes < info.expected_size
            total = sum(elements.values()) or 1
            quantised = {k: v for k, v in elements.items() if k not in _FLOAT_TYPES}
            pool = quantised or elements
            if pool:
                info.quant = max(pool, key=pool.get)
            top = sorted(elements.items(), key=lambda kv_: -kv_[1])[:3]
            info.quant_summary = ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in top)
            if groups:
                best = max(groups.values(), key=lambda g: g[0])
                info.blocks = best[1] + 1
        info.ok = True
    except Exception as e:  # noqa: BLE001
        info.error = f"{type(e).__name__}: {e}"
    return info


def safetensors_problem(path: Path) -> str:
    """'' when the safetensors header is sane and the file is complete, else a description."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return "the file is almost empty"
            (n,) = struct.unpack("<Q", raw)
            if n < 2 or n > (1 << 27) or 8 + n > size:
                return "it is not a valid safetensors file (bad header)"
            head = json.loads(f.read(n).decode("utf-8"))
        end = max((v["data_offsets"][1] for k, v in head.items()
                   if k != "__metadata__" and isinstance(v, dict) and "data_offsets" in v), default=0)
        if size < 8 + n + end:
            return (f"the download is incomplete ({human_size(size)} on disk, "
                    f"{human_size(8 + n + end)} expected)")
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return "it is not a valid safetensors file (header unreadable)"
    return ""


_LLM_ARCHS = ("qwen3vl", "qwen3", "qwen2vl", "qwen25vl", "qwen2", "llama", "gemma", "clip")
_CHECK_CACHE: dict = {}


def inspect_model_file(path: str, role: str) -> tuple[bool, str]:
    """role: diffusion | vae | llm | mmproj.  Returns (ok, one-line text)."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError as e:
        return False, f"cannot read the file ({e.strerror or e})"
    key = (str(p), role, st.st_size, st.st_mtime_ns)
    if key in _CHECK_CACHE:
        return _CHECK_CACHE[key]
    if p.suffix.lower() == ".gguf":
        inf = read_gguf_info(p, tensors=True)
        if not inf.ok:
            res = (False, "it is not a readable GGUF file - usually an unfinished or broken download "
                          f"(e.g. a web page saved as .gguf). Download it again. ({inf.error})")
        elif inf.truncated:
            res = (False, f"the download is incomplete ({human_size(inf.size_bytes)} on disk, at least "
                          f"{human_size(inf.expected_size)} expected). Download it again.")
        else:
            arch = inf.arch.lower()
            if role in ("diffusion", "vae") and arch in _LLM_ARCHS:
                where = "Vision projector" if arch == "clip" else "Text encoder (LLM)"
                res = (False, f"this is a '{inf.arch}' file, not a {role} model - it belongs in the "
                              f"{where} field.")
            elif role == "llm" and arch == "clip":
                res = (False, "this is the vision projector (mmproj) - it belongs in the Vision "
                              "projector field.")
            elif role == "llm" and arch and arch not in _LLM_ARCHS and \
                    classify_model_file(p, inf) == "diffusion":
                res = (False, f"this looks like a diffusion model (architecture '{inf.arch}') - it "
                              "belongs in the Diffusion model field.")
            elif role == "mmproj" and arch and arch != "clip":
                res = (False, f"this is not a vision projector (architecture '{inf.arch}').")
            else:
                res = (True, f"{inf.quant or inf.arch or 'GGUF'}, {human_size(inf.size_bytes)}, "
                             f"{inf.tensor_count} tensors"
                             + (", complete" if inf.expected_size else ""))
    else:
        bad = safetensors_problem(p)
        res = (False, bad + ". Download it again.") if bad else (True, f"safetensors, {human_size(st.st_size)}, complete")
    _CHECK_CACHE[key] = res
    return res


_ROLE_LABELS = (("diffusion", "Diffusion model"), ("vae", "VAE"), ("llm", "Text encoder (LLM)"),
                ("llm_vision", "Vision projector"))


def check_model_files(cfg: "EngineConfig", report: Optional[list] = None) -> list[str]:
    """Problems (one sentence each) with the chosen model files; fills `report` with every line."""
    errs: list[str] = []
    for key, label in _ROLE_LABELS:
        path = getattr(cfg, key)
        if not path or not Path(path).is_file():
            continue
        ok, text = inspect_model_file(path, "mmproj" if key == "llm_vision" else key)
        if report is not None:
            report.append(f"{'OK     ' if ok else 'PROBLEM'}  {label}: {Path(path).name} - {text}")
        if not ok:
            errs.append(f"{label} file '{Path(path).name}': {text}")
    return errs


def classify_model_file(path: Path, info: Optional[GGUFInfo] = None) -> str:
    """Return diffusion | vae | llm | mmproj | unknown (filename first, then GGUF arch)."""
    n = path.name.lower()
    arch = (info.arch if info else "").lower()
    if "mmproj" in n or arch == "clip":
        return "mmproj"
    if "vae" in n:
        return "vae"
    if any(k in n for k in ("qwen3vl", "qwen3-vl", "qwen2.5-vl", "qwen2_5_vl",
                            "qwen_2.5_vl", "text_encoder", "qwen3_vl")):
        return "llm"
    if arch.startswith(("qwen3", "qwen2", "llama", "gemma")) and "image" not in n:
        return "llm"
    if "qwen" in n and "image" in n:
        return "diffusion"
    if info and info.blocks >= 20:
        return "diffusion"
    return "unknown"


def find_sd_cli(extra: str = "") -> str:
    cands: list[Path] = []
    if extra:
        cands.append(Path(extra))
    for nm in SD_CLI_NAMES:
        w = shutil.which(nm)
        if w:
            cands.append(Path(w))
    here = Path(__file__).resolve().parent
    for base in (here, here / "bin", here / "build" / "bin" / "Release", TOOLS_DIR):
        if base.exists():
            for nm in SD_CLI_NAMES:
                cands.extend(base.rglob(nm) if base == TOOLS_DIR else [base / nm])
    for c in cands:
        try:
            if c.is_file():
                return str(c)
        except OSError:
            pass
    return ""


# --------------------------------------------------------------------------- #
#  Settings
# --------------------------------------------------------------------------- #

DEFAULT_SETTINGS = {
    "sd_cli": "", "diffusion": "", "vae": "", "llm": "", "llm_vision": "",
    "out_dir": str(DEFAULT_OUT_DIR), "models_dir": str(MODELS_DIR),
    "offload": True, "flash_attn": False, "vae_tiling": False, "vae_cpu": False,
    "clip_cpu": False, "prefix_cache": True, "verbose": False, "threads": 0,
    "extra_args": "", "edit_max_side": 1024,
    "steps": 30, "cfg": 6.0, "seed": -1, "sampler": "euler", "scheduler": "default",
    "batch": 1, "transparent": False, "size_preset": "1024 x 1024  (1:1)",
    "width": 1024, "height": 1024, "edit_size_mode": "Match input image",
    "last_models_browse": "", "last_image_browse": "",
    "mark_color": "Magenta", "mark_style": "Tint", "mark_wrap": True, "mark_crop": False,
    "mark_pad": 128, "mark_strict": True, "mark_feather": 12, "mark_clean": False,
    "sel_tol": 40, "sel_reach": 50, "sel_contig": True, "sel_prec": "Normal", "sel_mode": "Add",
}


class Settings:
    def __init__(self) -> None:
        self.data = dict(DEFAULT_SETTINGS)
        try:
            self.data.update(json.loads(SETTINGS_FILE.read_text("utf-8")))
        except Exception:  # noqa: BLE001
            pass

    def __getitem__(self, k):
        return self.data.get(k, DEFAULT_SETTINGS.get(k))

    def __setitem__(self, k, v) -> None:
        self.data[k] = v

    def save(self) -> None:
        try:
            ROOT.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(self.data, indent=2), "utf-8")
        except OSError:
            pass


# --------------------------------------------------------------------------- #
#  Command construction
# --------------------------------------------------------------------------- #

SAMPLERS = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
            "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd", "default"]
SCHEDULERS = ["default", "discrete", "karras", "exponential", "ays", "gits",
              "smoothstep", "sgm_uniform", "simple", "kl_optimal", "lcm"]
# (steps, cfg): "Standard" suits the base model, "Turbo" the 8-step distilled build.
SAMPLING_PRESETS = {"Standard": (30, 6.0), "Turbo": (8, 1.0)}


def model_family(path: str) -> str:
    """'turbo' for distilled Turbo builds (by file name), 'standard' for other files, '' if unset."""
    n = Path(path).name.lower() if path else ""
    if not n:
        return ""
    return "turbo" if "turbo" in n else "standard"


SIZE_PRESETS = {
    "1024 x 1024  (1:1)": (1024, 1024),
    "1216 x 832   (3:2)": (1216, 832),
    "832 x 1216   (2:3)": (832, 1216),
    "1344 x 768   (16:9)": (1344, 768),
    "768 x 1344   (9:16)": (768, 1344),
    "1152 x 896   (4:3)": (1152, 896),
    "896 x 1152   (3:4)": (896, 1152),
    "1536 x 1024  (3:2 large)": (1536, 1024),
    "512 x 512    (quick test)": (512, 512),
    "Custom": (0, 0),
}
EDIT_TEMPLATES = [
    ("Insert a template...", ""),
    ("Remove an object", "Remove the [object] from the image. Keep everything else unchanged."),
    ("Add an object", "Add [a red umbrella] next to [the person on the left]. Keep everything else unchanged."),
    ("Replace an object", "Replace the [object] with [new object]. Keep everything else unchanged."),
    ("Change color", "Change the color of the [object] to [color]."),
    ("Change background", "Change the background to [a sunset beach]. Keep the subject unchanged."),
    ("Change style", "Convert the image to [watercolor painting] style."),
    ("Edit text in image", "Change the text '[old text]' to '[new text]', keeping the same font."),
    ("Make background transparent", "Remove the background of the image, keeping only the main subject."),
]


@dataclass
class EngineConfig:
    sd_cli: str = ""
    diffusion: str = ""
    vae: str = ""
    llm: str = ""
    llm_vision: str = ""
    offload: bool = True
    flash_attn: bool = False
    vae_tiling: bool = False
    vae_cpu: bool = False
    clip_cpu: bool = False
    prefix_cache: bool = True
    verbose: bool = False
    threads: int = 0
    extra_args: str = ""


@dataclass
class JobSpec:
    mode: str = "create"            # create | edit
    prompt: str = ""
    negative: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg: float = 6.0
    seed: int = -1
    sampler: str = "euler"
    scheduler: str = "default"
    batch: int = 1
    refs: list = field(default_factory=list)
    transparent: bool = False
    user_prompt: str = ""           # what the user typed (mark mode wraps it)


def wrap_transparent(prompt: str) -> str:
    """Official RGBA prompt format for Qwen-Image-2.1 (works for create and edit)."""
    p = prompt.strip().rstrip(".")
    return (f"This is an RGBA image with transparency. {p}. "
            "The image has alpha channel and the background is transparent.")


def validate(cfg: EngineConfig, job: JobSpec) -> list[str]:
    errs = []
    if not cfg.sd_cli or not Path(cfg.sd_cli).is_file():
        errs.append("sd-cli executable not found (Models tab -> 'Get sd-cli...' or Browse).")
    if not cfg.diffusion or not Path(cfg.diffusion).is_file():
        errs.append("Diffusion model file is missing (any Qwen-Image-2.1 .gguf / .safetensors).")
    if not cfg.vae or not Path(cfg.vae).is_file():
        errs.append("VAE file is missing (qwen_image_2.1_vae_bf16.safetensors).")
    if not cfg.llm or not Path(cfg.llm).is_file():
        errs.append("Text encoder is missing (Qwen3-VL-8B-Instruct GGUF or safetensors).")
    if not errs:
        errs += check_model_files(cfg)
    if job.mode in ("edit", "mark"):
        if job.mode == "edit" and not job.refs:
            errs.append("Add at least one input image on the Edit tab.")
        if cfg.llm.lower().endswith(".gguf") and not (cfg.llm_vision and Path(cfg.llm_vision).is_file()):
            errs.append("Editing with a GGUF text encoder needs the vision projector "
                        "(mmproj-Qwen3VL-8B-Instruct-F16.gguf) on the Models tab.")
    if not job.prompt.strip():
        errs.append("The prompt is empty.")
    return errs


def build_command(cfg: EngineConfig, job: JobSpec, out_path: Path) -> list[str]:
    prompt = wrap_transparent(job.prompt) if job.transparent else job.prompt.strip()
    c = [cfg.sd_cli, "--diffusion-model", cfg.diffusion, "--vae", cfg.vae, "--llm", cfg.llm]
    if cfg.llm_vision:
        c += ["--llm_vision", cfg.llm_vision]
    for r in job.refs:
        c += ["-r", str(r)]
    c += ["-p", prompt]
    if job.negative.strip():
        c += ["-n", job.negative.strip()]
    c += ["-W", str(job.width), "-H", str(job.height), "--steps", str(job.steps),
          "--cfg-scale", f"{job.cfg:g}", "-s", str(job.seed), "-b", str(job.batch)]
    if job.sampler and job.sampler != "default":
        c += ["--sampling-method", job.sampler]
    if job.scheduler and job.scheduler != "default":
        c += ["--scheduler", job.scheduler]
    if cfg.offload:
        c.append("--offload-to-cpu")
    if cfg.flash_attn:
        c.append("--diffusion-fa")
    if cfg.vae_tiling:
        c.append("--vae-tiling")
    if cfg.vae_cpu:
        c.append("--vae-on-cpu")
    if cfg.clip_cpu:
        c.append("--clip-on-cpu")
    if not cfg.prefix_cache:
        c += ["--model-args", "qwen_image_2_1_prefix_cache=false"]
    if cfg.threads and cfg.threads > 0:
        c += ["-t", str(cfg.threads)]
    if cfg.verbose:
        c.append("-v")
    if cfg.extra_args.strip():
        try:
            c += shlex.split(cfg.extra_args, posix=(os.name != "nt"))
        except ValueError:
            pass
    c += ["-o", str(out_path)]
    return c


def cmdline_for_display(cmd: list[str]) -> str:
    if os.name == "nt":
        return " ".join(f'"{a}"' if re.search(r"\s", a) else a for a in cmd)
    return " ".join(shlex.quote(a) for a in cmd)


_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*-\s*([\d.]+)\s*(s/it|it/s)")


def diagnose(tail: str, exit_code: int) -> str:
    t = tail.lower()
    if re.search(r"out of memory|failed to allocate|cudamalloc|alloc_buffer|vk::|bad_alloc", t) \
            or exit_code in (3221226505, -1073740791):
        return ("The runtime ran out of memory. Try: a smaller resolution, a smaller quant "
                "(Q4_K / Q3_K), 'Offload weights to CPU' on, 'VAE tiling' on, "
                "'Text encoder on CPU' on, or disable the prefix cache (Settings tab).")
    if "get sd version from file failed" in t:
        return ("The runtime could not identify the model from the files it was given. This usually "
                "means one file is incomplete or damaged (re-download the Diffusion model first), "
                "or a file is in the wrong field. Use 'Check files' on the Models tab - it names the "
                "file that is wrong.")
    if re.search(r"unknown model|unsupported model|unknown architecture|unrecognized model", t):
        return ("This sd-cli build does not recognise the model. Qwen-Image-2.1 needs a recent "
                "stable-diffusion.cpp build - use 'Get sd-cli...' to fetch the latest release.")
    if re.search(r"unknown argument|invalid argument|invalid parameter|unrecognized|"
                 r"unknown option", t):
        return ("Your sd-cli build rejected one of the flags (see the command in the log). "
                "Turn off the matching option on the Settings tab or update sd-cli.")
    if "mmproj" in t or "llm_vision" in t or "vision" in t and "fail" in t:
        return "Vision projector problem - check the mmproj file on the Models tab."
    if re.search(r"load.*fail|init.*fail|new_sd_ctx|failed to load|cannot open|no such file", t):
        return ("A model failed to load. Check that the diffusion model, VAE and text encoder "
                "belong to the same model family (Qwen-Image-2.1 needs its own VAE).")
    if exit_code in (3221225477, 3221226356, -1073741819):
        return "sd-cli crashed natively - usually out of VRAM/RAM. See the out-of-memory tips above."
    return ""


# --------------------------------------------------------------------------- #
#  Process runner
# --------------------------------------------------------------------------- #


class Runner(QObject):
    log = pyqtSignal(str)
    stage = pyqtSignal(str)
    progress = pyqtSignal(int, int)          # current step, total steps
    done = pyqtSignal(int, bool, str)        # exit code, cancelled, log tail

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.proc: Optional[QProcess] = None
        self._buf = ""
        self._cancelled = False
        self._tail: list[str] = []

    def running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning

    def start(self, cmd: list[str], cwd: str) -> None:
        self._buf, self._cancelled, self._tail = "", False, []
        p = QProcess(self)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.setWorkingDirectory(cwd)
        if sys.platform == "win32":
            try:  # hide the console window
                def _mod(args):
                    args.flags |= 0x08000000  # CREATE_NO_WINDOW
                p.setCreateProcessArgumentsModifier(_mod)
            except Exception:  # noqa: BLE001
                pass
        p.readyReadStandardOutput.connect(self._on_ready)
        p.finished.connect(self._on_finished)
        p.errorOccurred.connect(self._on_error)
        self.proc = p
        p.start(cmd[0], cmd[1:])

    def cancel(self) -> None:
        if self.running():
            self._cancelled = True
            self.proc.kill()

    # -- output handling ---------------------------------------------------
    def _on_ready(self) -> None:
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._buf += data
        parts = re.split(r"[\r\n]+", self._buf)
        self._buf = parts.pop()
        for line in parts:
            self._handle(line)
        self._try_progress(self._buf)

    def _try_progress(self, text: str) -> bool:
        m = _PROGRESS_RE.search(strip_ansi(text))
        if m:
            self.progress.emit(int(m.group(1)), int(m.group(2)))
            return True
        return False

    def _handle(self, raw: str) -> None:
        line = strip_ansi(raw).strip()
        if not line:
            return
        if self._try_progress(line):
            return
        low = line.lower()
        if "loading" in low or "init from" in low:
            self.stage.emit("Loading models...")
        elif "get_learned_condition" in low or "encoding" in low:
            self.stage.emit("Encoding prompt / images...")
        elif "sampling" in low and "completed" not in low:
            self.stage.emit("Sampling...")
        elif "decoding" in low or "decode_first_stage" in low:
            self.stage.emit("Decoding image...")
        self._tail.append(line)
        del self._tail[:-60]
        self.log.emit(line)

    def _on_finished(self, code: int, status) -> None:
        if self._buf.strip():
            self._handle(self._buf)
            self._buf = ""
        crashed = status == QProcess.ExitStatus.CrashExit and not self._cancelled
        self.done.emit(code if not crashed else (code or -1), self._cancelled, "\n".join(self._tail))

    def _on_error(self, err) -> None:
        if err == QProcess.ProcessError.FailedToStart:
            msg = "sd-cli failed to start (wrong file, missing DLLs or blocked by antivirus)."
            self.log.emit(msg)
            self.done.emit(-2, False, msg)


# --------------------------------------------------------------------------- #
#  Background tasks: HTTP / downloads
# --------------------------------------------------------------------------- #

UA = f"QwenImageStudio/{APP_VERSION}"


def _headers(url: str, extra: Optional[dict] = None) -> dict:
    h = {"User-Agent": UA}
    host = urllib.parse.urlparse(url).netloc
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok and host.endswith("huggingface.co"):
        h["Authorization"] = f"Bearer {tok}"
    if extra:
        h.update(extra)
    return h


def http_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers=_headers(url))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def hf_list_files(repo: str) -> list[tuple[str, int]]:
    data = http_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
    out = []
    for s in data.get("siblings", []):
        name = s.get("rfilename", "")
        if Path(name).suffix.lower() in MODEL_EXTS:
            out.append((name, int(s.get("size") or 0)))
    return sorted(out)


def hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(filename)}"


@dataclass
class DlSpec:
    url: str
    dest: Path
    role: str = ""
    extract: bool = False


def download_file(url: str, dest: Path, progress: Callable[[int, int], None],
                  cancelled: Callable[[], bool]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    pos = part.stat().st_size if part.exists() else 0
    extra = {"Range": f"bytes={pos}-"} if pos else None
    req = urllib.request.Request(url, headers=_headers(url, extra))
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and part.exists():      # already complete
            part.replace(dest)
            return
        raise
    with resp:
        total = 0
        cr = resp.headers.get("Content-Range")
        if resp.status == 206 and cr and "/" in cr:
            total = int(cr.rsplit("/", 1)[1])
            mode = "ab"
        else:
            total = int(resp.headers.get("Content-Length") or 0)
            pos, mode = 0, "wb"
        done = pos
        with open(part, mode) as f:
            while True:
                if cancelled():
                    raise InterruptedError("cancelled")
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                progress(done, total)
    if total and part.stat().st_size != total:
        raise IOError("download incomplete - run it again to resume")
    part.replace(dest)


class Task(QThread):
    """Run a callable off the GUI thread."""
    result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable, parent=None) -> None:
        super().__init__(parent)
        self.fn = fn

    def run(self) -> None:
        try:
            self.result.emit(self.fn())
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


class DownloadThread(QThread):
    status = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(list)      # list[DlSpec] completed
    failed = pyqtSignal(str)

    def __init__(self, specs: list[DlSpec], parent=None) -> None:
        super().__init__(parent)
        self.specs = specs
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        done = []
        try:
            for i, s in enumerate(self.specs, 1):
                self.status.emit(f"[{i}/{len(self.specs)}] {s.dest.name}")
                if s.dest.exists() and s.dest.stat().st_size > 0:
                    done.append(s)
                    continue
                download_file(s.url, s.dest, lambda a, b: self.progress.emit(a, b),
                              lambda: self._cancel)
                if s.extract:
                    self.status.emit(f"Extracting {s.dest.name} ...")
                    with zipfile.ZipFile(s.dest) as z:
                        z.extractall(s.dest.parent)
                done.append(s)
            self.finished_ok.emit(done)
        except InterruptedError:
            self.failed.emit("Download cancelled (partial files are kept and will resume).")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


COMPANIONS = [
    # role, repo, matcher(name)->bool, sub-folder
    ("vae", "Comfy-Org/Qwen-Image-2.1",
     lambda n: n.lower().endswith("qwen_image_2.1_vae_bf16.safetensors"), "vae"),
    ("llm", "Qwen/Qwen3-VL-8B-Instruct-GGUF",
     lambda n: "q4_k_m" in n.lower() and "mmproj" not in n.lower(), "text_encoders"),
    ("mmproj", "Qwen/Qwen3-VL-8B-Instruct-GGUF",
     lambda n: "mmproj" in n.lower() and re.search(r"[-_]F16\.gguf$", n) is not None,
     "text_encoders"),
]


def resolve_companions(models_dir: Path, have: dict) -> list[DlSpec]:
    specs, cache = [], {}
    for role, repo, match, sub in COMPANIONS:
        if have.get(role) and Path(have[role]).is_file():
            continue
        if repo not in cache:
            cache[repo] = hf_list_files(repo)
        hit = next((n for n, _ in cache[repo] if match(n)), None)
        if not hit:
            raise RuntimeError(f"Could not find the {role} file in {repo}.")
        specs.append(DlSpec(hf_url(repo, hit), models_dir / sub / Path(hit).name, role))
    return specs


# --------------------------------------------------------------------------- #
#  Widgets
# --------------------------------------------------------------------------- #


def checker_brush() -> QBrush:
    pm = QPixmap(24, 24)
    pm.fill(QColor("#10161f"))
    p = QPainter(pm)
    p.fillRect(0, 0, 12, 12, QColor("#161e2a"))
    p.fillRect(12, 12, 12, 12, QColor("#161e2a"))
    p.end()
    return QBrush(pm)


class ImageViewer(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(checker_brush())
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._fit = True
        self._hint = "Your image appears here"

    def set_hint(self, text: str) -> None:
        self._hint = text
        self.viewport().update()

    def set_image(self, pix: Optional[QPixmap]) -> None:
        self._item.setPixmap(pix if pix is not None else QPixmap())
        self._scene.setSceneRect(self._item.boundingRect())
        if self._fit:
            self.fit()
        self.viewport().update()

    def has_image(self) -> bool:
        return not self._item.pixmap().isNull()

    def fit(self) -> None:
        if self.has_image():
            self.resetTransform()
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        self._fit = True

    def actual_size(self) -> None:
        self.resetTransform()
        self._fit = False

    def wheelEvent(self, e) -> None:  # noqa: N802
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self._fit = False
        self.scale(f, f)

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        self.fit()

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        if self._fit:
            self.fit()

    def paintEvent(self, e) -> None:  # noqa: N802
        super().paintEvent(e)
        if not self.has_image():
            p = QPainter(self.viewport())
            p.setPen(QColor("#4d5b73"))
            f = QFont(self.font())
            f.setPointSize(13)
            p.setFont(f)
            p.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self._hint)
            p.end()


class RefList(QListWidget):
    """Reference-image list: drop files, reorder by dragging."""
    filesDropped = pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setIconSize(QSize(72, 72))
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setMinimumHeight(110)
        self.setMaximumHeight(190)

    def _urls(self, e) -> list[str]:
        out = []
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p and Path(p).suffix.lower() in IMAGE_EXTS:
                out.append(p)
        return out

    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            files = self._urls(e)
            if files:
                self.filesDropped.emit(files)
            e.acceptProposedAction()
        else:
            super().dropEvent(e)

    def paths(self) -> list[str]:
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]


class PathRow(QWidget):
    changed = pyqtSignal(str)

    def __init__(self, filt: str, placeholder: str = "") -> None:
        super().__init__()
        self.filt = filt
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.btn = QPushButton("...")
        self.btn.setFixedWidth(34)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.btn)
        self.edit.editingFinished.connect(lambda: self.changed.emit(self.edit.text().strip()))

    def text(self) -> str:
        return self.edit.text().strip()

    def set_text(self, t: str) -> None:
        self.edit.setText(t)
        self.changed.emit(t)


def scrolled(w: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setWidget(w)
    return sa


# --------------------------------------------------------------------------- #
#  Mark & edit: marker rendering, mask helpers, compositing
# --------------------------------------------------------------------------- #

MAX_CANVAS_SIDE = 4096
MARK_COLORS = {
    "Magenta": (255, 0, 255), "Red": (255, 0, 0), "Green": (0, 255, 0),
    "Blue": (0, 0, 255), "Yellow": (255, 255, 0), "Cyan": (0, 255, 255),
}
MARK_STYLES = ["Tint", "Solid", "Outline"]
MARK_TEMPLATES = [
    ("Insert a template...", ""),
    ("Remove what is in the marked area",
     "Remove the object in the marked area and fill it naturally with the surrounding background."),
    ("Add something there", "Add [a small wooden boat] in the marked area."),
    ("Replace with something else", "Replace what is in the marked area with [a vase of flowers]."),
    ("Change its color", "Change the color of the object in the marked area to [blue]."),
    ("Change its material / style", "Change the object in the marked area so it looks like [polished gold]."),
    ("Clean up / repair", "Repair the marked area and make it look clean and natural."),
    ("Edit text there", "Change the text in the marked area to '[new text]', keeping the same font."),
]
SEL_HINTS = {
    "wand": "Click an area to select all connected pixels of a similar color. Drag the Tolerance "
            "slider to grow or shrink the selection. Click again to add another area.",
    "smart": "Scribble lightly over the object with the left button (use a small brush). "
             "The selection snaps to its edges after every stroke. Optionally scribble the "
             "background with the right button to exclude it.",
    "box": "Drag a box around the object (a tight box works best). Raise Reach if parts of the "
           "object are missing, lower it if too much is selected.",
}
_PREMULT = QImage.Format.Format_ARGB32_Premultiplied
_IGNORE = Qt.AspectRatioMode.IgnoreAspectRatio
_SMOOTH = Qt.TransformationMode.SmoothTransformation
_DEST_IN = QPainter.CompositionMode.CompositionMode_DestinationIn


def mark_prompt(instr: str, color: str, style: str, clean_ref: bool) -> str:
    """Wrap the user's instruction so the model knows where to apply it."""
    c = color.lower()
    marking = f"a {c} outline" if style == "Outline" else f"a {c} overlay"
    instr = instr.strip()
    if instr and instr[-1] not in ".!?":
        instr += "."
    if clean_ref:
        return (f"Image 1 is the original photo. Image 2 is the same photo with an area marked by "
                f"{marking}. In the marked area only: {instr} Keep everything outside the marked "
                f"area exactly as in image 1, and do not leave any {c} marking in the result.")
    return (f"The image contains an area marked by {marking}. In the marked area only: {instr} "
            f"Keep everything outside the marked area exactly unchanged, and remove the {c} "
            f"marking from the result.")


def flatten_on_white(img: QImage) -> QImage:
    out = QImage(img.size(), QImage.Format.Format_RGB32)
    out.fill(QColor("white"))
    p = QPainter(out)
    p.drawImage(0, 0, img)
    p.end()
    return out


_ALPHA_FLOOR = bytes(0 if i < 26 else i for i in range(256))


def mask_bbox(mask: QImage) -> Optional[QRect]:
    """Bounding box (slightly padded) of everything painted, or None if the mask is empty."""
    if mask.isNull():
        return None
    W, H = mask.width(), mask.height()
    f = max(1, -(-max(W, H) // 512))                      # ceil -> work on <= 512 px
    small = mask if f == 1 else mask.scaled(max(1, W // f), max(1, H // f), _IGNORE, _SMOOTH)
    a = small.convertToFormat(QImage.Format.Format_Alpha8)
    w, h, bpl = a.width(), a.height(), a.bytesPerLine()
    ptr = a.constBits()
    ptr.setsize(bpl * h)
    data = bytes(ptr).translate(_ALPHA_FLOOR)           # ignore faint residue (< ~10 %)
    y0, y1, x0, x1 = -1, -1, w, -1
    for y in range(h):
        row = data[y * bpl: y * bpl + w]
        s = row.lstrip(b"\0")
        if not s:
            continue
        if y0 < 0:
            y0 = y
        y1 = y
        x0 = min(x0, w - len(s))
        x1 = max(x1, len(row.rstrip(b"\0")) - 1)
    if y0 < 0:
        return None
    r = QRect(x0 * f, y0 * f, (x1 - x0 + 1) * f, (y1 - y0 + 1) * f)
    return r.adjusted(-f, -f, f, f).intersected(QRect(0, 0, W, H))


def render_marker(base: QImage, mask: QImage, color: QColor, style: str) -> QImage:
    """Draw the user's marking onto the picture that is sent to the model."""
    out = base.convertToFormat(_PREMULT)
    size = out.size()

    def shaped(m: QImage) -> QImage:
        layer = QImage(size, _PREMULT)
        layer.fill(color)
        p = QPainter(layer)
        p.setCompositionMode(_DEST_IN)
        p.drawImage(0, 0, m)
        p.end()
        return layer

    colored = shaped(mask)
    orig = out.copy()
    p = QPainter(out)
    if style == "Outline":
        t = max(3, round(0.008 * max(size.width(), size.height())))
        ring = QImage(size, _PREMULT)
        ring.fill(Qt.GlobalColor.transparent)
        rp = QPainter(ring)
        for k in range(16):
            ang = 2 * math.pi * k / 16
            rp.drawImage(round(t * math.cos(ang)), round(t * math.sin(ang)), colored)
        rp.end()
        p.drawImage(0, 0, ring)
        inner = orig.copy()                      # keep the inside of the outline untouched
        ip = QPainter(inner)
        ip.setCompositionMode(_DEST_IN)
        ip.drawImage(0, 0, mask)
        ip.end()
        p.drawImage(0, 0, inner)
    else:
        p.setOpacity(0.6 if style == "Tint" else 1.0)
        p.drawImage(0, 0, colored)
    p.end()
    return out


def feather_mask(m: QImage, radius: int) -> QImage:
    """Soft-edged copy of a mask (grow a little, then blur by down/up-scaling)."""
    if radius <= 0:
        return m
    w, h = m.width(), m.height()
    grown = QImage(m.size(), _PREMULT)
    grown.fill(Qt.GlobalColor.transparent)
    d = max(1, radius // 2)
    p = QPainter(grown)
    for dx, dy in ((0, 0), (d, 0), (-d, 0), (0, d), (0, -d), (d, d), (-d, -d), (d, -d), (-d, d)):
        p.drawImage(dx, dy, m)
    p.end()
    k = max(2, radius)
    small = grown.scaled(max(1, w // k), max(1, h // k), _IGNORE, _SMOOTH)
    return small.scaled(w, h, _IGNORE, _SMOOTH)


def composite_mark(raw_path: str, ctx: dict) -> QImage:
    """Blend the model output back into the original so only the marked area changes."""
    raw = QImage(raw_path)
    if raw.isNull():
        raise IOError(f"cannot read result {raw_path}")
    rect: QRect = ctx["rect"]
    region = raw.scaled(rect.size(), _IGNORE, _SMOOTH).convertToFormat(_PREMULT)
    if not ctx["strict"]:
        return region.convertToFormat(QImage.Format.Format_ARGB32)
    soft = feather_mask(ctx["mask"].copy(rect), ctx["feather"])
    layer = QImage(rect.size(), _PREMULT)
    layer.fill(Qt.GlobalColor.transparent)
    p = QPainter(layer)
    p.drawImage(0, 0, region)
    p.setCompositionMode(_DEST_IN)
    p.drawImage(0, 0, soft)
    p.end()
    out = ctx["img"].convertToFormat(_PREMULT)
    p = QPainter(out)
    p.drawImage(rect.topLeft(), layer)
    p.end()
    return out.convertToFormat(QImage.Format.Format_ARGB32)


# --------------------------------------------------------------------------- #
#  Smart selection: magic wand, smart brush, box select
#  Pure Python + Qt (no numpy / onnx / extra models). All work happens on a
#  downscaled copy of the image, so a selection takes about a second or two.
# --------------------------------------------------------------------------- #

SEL_PRECISION = {"Fast": 256, "Normal": 384, "Fine": 512}


class WorkImage:
    """Small RGB copy of the image as three byte planes, for the selection algorithms."""

    def __init__(self, img: QImage, max_side: int) -> None:
        flat = flatten_on_white(img)
        W, H = flat.width(), flat.height()
        sc = min(1.0, max_side / max(W, H))
        w, h = max(8, round(W * sc)), max(8, round(H * sc))
        small = flat if (w, h) == (W, H) else flat.scaled(w, h, _IGNORE, _SMOOTH)
        small = small.convertToFormat(QImage.Format.Format_RGB32)
        ptr = small.constBits()
        ptr.setsize(small.sizeInBytes())
        data = bytes(ptr)                      # B G R x per pixel
        self.max_side = max_side
        self.w, self.h, self.n = w, h, w * h
        self.B, self.G, self.R = data[0::4], data[1::4], data[2::4]
        self.sx, self.sy = W / w, H / h        # work px -> image px


def alpha_bytes(img: QImage) -> bytes:
    """Alpha channel of an image as w*h bytes (no row padding)."""
    a = img.convertToFormat(QImage.Format.Format_Alpha8)
    w, h, bpl = a.width(), a.height(), a.bytesPerLine()
    ptr = a.constBits()
    ptr.setsize(bpl * h)
    data = bytes(ptr)
    if bpl == w:
        return data
    return b"".join(data[y * bpl: y * bpl + w] for y in range(h))


def alpha_image(data, w: int, h: int) -> QImage:
    """w*h bytes -> an Alpha8 QImage (a detached copy)."""
    stride = (w + 3) & ~3
    raw = bytes(data)
    if stride != w:
        pad = b"\0" * (stride - w)
        raw = b"".join(raw[y * w:(y + 1) * w] + pad for y in range(h))
    return QImage(raw, w, h, stride, QImage.Format.Format_Alpha8).copy()


def raster_strokes(strokes: list, w: int, h: int, sx: float, sy: float) -> bytes:
    """Rasterise scribbles [(points, radius_px_in_image)] to a w*h byte map (255 = covered)."""
    img = QImage(w, h, _PREMULT)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    white = QColor(255, 255, 255)
    for pts, r in strokes:
        width = max(2.0, r / max(sx, 1e-6))
        sp = [QPointF(pt.x() / sx, pt.y() / sy) for pt in pts]
        if len(sp) == 1:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(white)
            p.drawEllipse(sp[0], width / 2, width / 2)
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(white, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                          Qt.PenJoinStyle.RoundJoin))
            p.drawPolyline(QPolygonF(sp))
    p.end()
    return alpha_bytes(img)


def _color_centers(wi: WorkImage, idx: list, k: int = 3, cap: int = 1500) -> list:
    """Up to k representative colours of the pixels in idx (tiny k-means)."""
    if not idx:
        return []
    step = max(1, len(idx) // cap)
    pts = [(wi.R[i], wi.G[i], wi.B[i]) for i in idx[::step]]
    n = len(pts)
    mean = (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n)

    def l1(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    centers = [min(pts, key=lambda p: l1(p, mean))]
    while len(centers) < k:
        far = max(pts, key=lambda p: min(l1(p, c) for c in centers))
        if min(l1(far, c) for c in centers) < 60:
            break
        centers.append(far)
    centers = [tuple(map(float, c)) for c in centers]
    for rnd in range(9):
        sums = [[0.0, 0.0, 0.0, 0] for _ in centers]
        for p in pts:
            j = min(range(len(centers)), key=lambda q: l1(p, centers[q]))
            s = sums[j]
            s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; s[3] += 1
        if rnd in (3, 6) and len(centers) > 1:         # drop tiny clusters (blend / outlier colours)
            keep = [(s, c) for s, c in zip(sums, centers) if s[3] >= 0.06 * n]
            if keep and len(keep) < len(centers):
                centers = [c for _s, c in keep]
                continue
        centers = [(s[0] / s[3], s[1] / s[3], s[2] / s[3]) if s[3] else c
                   for s, c in zip(sums, centers)]
    return centers


def _model_cost(wi: WorkImage, centers: list) -> list:
    """Per-pixel L1 distance (0..765) to the nearest colour centre."""
    if not centers:
        return [765.0] * wi.n
    out = []
    ap = out.append
    for r, g, b in zip(wi.R, wi.G, wi.B):
        m = 765.0
        for cr, cg, cb in centers:
            d = abs(r - cr) + abs(g - cg) + abs(b - cb)
            if d < m:
                m = d
        ap(m)
    return out


def _grow(wi: WorkImage, seeds: list, mfg: list, tau: float, jump: float,
          blocked: Optional[bytearray] = None, mbg: Optional[list] = None,
          allow: Optional[bytearray] = None) -> bytearray:
    """4-connected region growing from the seeds. A neighbour joins when its colour is
    close enough to the object colours (mfg <= tau), it is not a strong colour edge away
    from the pixel we came from (L1 jump <= `jump`), it is not blocked, and (if a
    background model is given) it is not closer to the background colours than to the
    object colours."""
    w, N = wi.w, wi.n
    R, G, B = wi.R, wi.G, wi.B
    out = bytearray(N)
    stack = []
    for i in seeds:
        if not out[i]:
            out[i] = 1
            stack.append(i)
    use_bg = mbg is not None
    while stack:
        i = stack.pop()
        r0, g0, b0 = R[i], G[i], B[i]
        x = i - (i // w) * w
        for j in ((i - 1) if x > 0 else -1, (i + 1) if x < w - 1 else -1,
                  (i - w) if i >= w else -1, (i + w) if i < N - w else -1):
            if j < 0 or out[j]:
                continue
            if blocked is not None and blocked[j]:
                continue
            if allow is not None and not allow[j]:
                continue
            m = mfg[j]
            if m > tau:
                continue
            if abs(R[j] - r0) + abs(G[j] - g0) + abs(B[j] - b0) > jump:
                continue
            if use_bg and mbg[j] < m:
                continue
            out[j] = 1
            stack.append(j)
    return out


def _reach_threshold(reach: int) -> float:
    """Colour distance (L1, 0..765) beyond which a pixel is *certainly* not the object."""
    return 765.0 * (0.05 + 0.005 * max(1, min(100, reach)))


def wand_select(wi: WorkImage, seed: int, tol: int, contiguous: bool = True) -> bytearray:
    """Magic wand: pixels whose colour is within `tol` of the clicked colour."""
    w, h, N = wi.w, wi.h, wi.n
    R, G, B = wi.R, wi.G, wi.B
    sx, sy = seed % w, seed // w
    rs = gs = bs = cnt = 0
    for yy in range(max(0, sy - 1), min(h, sy + 2)):          # average a 3x3 patch
        for xx in range(max(0, sx - 1), min(w, sx + 2)):
            k = yy * w + xx
            rs += R[k]; gs += G[k]; bs += B[k]; cnt += 1
    r0, g0, b0 = rs / cnt, gs / cnt, bs / cnt
    thr = max(1, tol) * 2.2
    thr2 = thr * thr
    out = bytearray(N)
    if not contiguous:
        for i, (r, g, b) in enumerate(zip(R, G, B)):
            if (r - r0) ** 2 + (g - g0) ** 2 + (b - b0) ** 2 <= thr2:
                out[i] = 255
        return out
    seen = bytearray(N)
    seen[seed] = 1
    stack = [seed]
    while stack:
        i = stack.pop()
        dr, dg, db = R[i] - r0, G[i] - g0, B[i] - b0
        if dr * dr + dg * dg + db * db > thr2:
            continue
        out[i] = 255
        x = i - (i // w) * w
        if x > 0 and not seen[i - 1]:
            seen[i - 1] = 1; stack.append(i - 1)
        if x < w - 1 and not seen[i + 1]:
            seen[i + 1] = 1; stack.append(i + 1)
        if i >= w and not seen[i - w]:
            seen[i - w] = 1; stack.append(i - w)
        if i < N - w and not seen[i + w]:
            seen[i + w] = 1; stack.append(i + w)
    return out


def _reach_jump(reach: int) -> float:
    """Largest colour step (L1) between neighbours that region growing will walk across."""
    return 40.0 + 1.6 * max(1, min(100, reach))


def smart_select(wi: WorkImage, fg: list, bg: list, reach: int) -> bytearray:
    """Scribble segmentation. fg / bg are pixel indices the user painted over the object /
    background. The object colours come from the fg scribbles, the background colours from
    the bg scribbles (or, without any, from pixels that look nothing like the object), so a
    few strokes on the object alone are usually enough."""
    N = wi.n
    mfg = _model_cost(wi, _color_centers(wi, fg))
    tau = _reach_threshold(reach)
    top = max(mfg)
    cut = max(0.45 * top, 90.0)                                   # the colours most unlike the object
    implicit = [i for i, m in enumerate(mfg) if m >= cut] if top >= 90.0 else []
    bg_centers = _color_centers(wi, list(bg) + implicit[::max(1, len(implicit) // 800)]
                                if bg else implicit)
    mbg = _model_cost(wi, bg_centers) if bg_centers else None
    blocked = bytearray(N)
    for i in bg:
        blocked[i] = 1
    grown = _grow(wi, fg, mfg, tau, _reach_jump(reach), blocked, mbg)
    return bytearray(255 if v else 0 for v in grown)


def box_select(wi: WorkImage, box: tuple, reach: int) -> bytearray:
    """Box segmentation: everything outside the box is background. The colours in a band
    around the box describe the background; the object is whatever inside the box differs
    from them and is connected to the middle of the box."""
    w, h, N = wi.w, wi.h, wi.n
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    bw, bh = x1 - x0, y1 - y0
    inside = bytearray(N)
    for y in range(y0, y1):
        inside[y * w + x0: y * w + x1] = b"\x01" * bw
    band = max(3, int(0.06 * min(bw, bh)))
    ring = [y * w + x for y in range(max(0, y0 - band), min(h, y1 + band))
            for x in range(max(0, x0 - band), min(w, x1 + band)) if not inside[y * w + x]]
    if not ring:                                                  # box fills the picture: use its rim
        ring = [y * w + x for y in range(y0, y1) for x in range(x0, x1)
                if x < x0 + band or x >= x1 - band or y < y0 + band or y >= y1 - band]
    bg_centers = _color_centers(wi, ring, k=4)
    mbg = _model_cost(wi, bg_centers)
    spread = sorted(mbg[i] for i in ring)
    base = max(60.0, 1.8 * spread[int(0.95 * (len(spread) - 1))])
    tau_b = base * (1.5 - max(1, min(100, reach)) / 100.0)
    cx0, cx1 = x0 + int(0.3 * bw), x1 - int(0.3 * bw)
    cy0, cy1 = y0 + int(0.3 * bh), y1 - int(0.3 * bh)
    cand = [y * w + x for y in range(cy0, max(cy1, cy0 + 1)) for x in range(cx0, max(cx1, cx0 + 1))]
    seeds = [i for i in cand if mbg[i] > tau_b]
    if len(seeds) < max(10, len(cand) // 10):                     # centre looks like background
        seeds = sorted(cand, key=lambda i: -mbg[i])[:max(10, len(cand) // 10)]
    # an object pixel is any inside pixel clearly unlike the background, joined to the seeds
    unlike = bytearray(N)
    for i in range(N):
        if inside[i] and mbg[i] > tau_b:
            unlike[i] = 1
    out = bytearray(N)
    stack = []
    for i in seeds:
        out[i] = 1
        stack.append(i)
    while stack:
        i = stack.pop()
        x = i - (i // w) * w
        for j in ((i - 1) if x > 0 else -1, (i + 1) if x < w - 1 else -1,
                  (i - w) if i >= w else -1, (i + w) if i < N - w else -1):
            if j >= 0 and not out[j] and unlike[j]:
                out[j] = 1
                stack.append(j)
    return bytearray(255 if v else 0 for v in out)


def fill_small_holes(mask: bytearray, w: int, h: int, frac: float = 0.004) -> bytearray:
    """Fill enclosed background islands smaller than `frac` of the picture."""
    N = w * h
    limit = max(6, int(N * frac))
    seen = bytearray(N)

    def flood(stack: list, collect: Optional[list] = None) -> None:
        while stack:
            i = stack.pop()
            if collect is not None:
                collect.append(i)
            x = i - (i // w) * w
            if x > 0 and not mask[i - 1] and not seen[i - 1]:
                seen[i - 1] = 1; stack.append(i - 1)
            if x < w - 1 and not mask[i + 1] and not seen[i + 1]:
                seen[i + 1] = 1; stack.append(i + 1)
            if i >= w and not mask[i - w] and not seen[i - w]:
                seen[i - w] = 1; stack.append(i - w)
            if i < N - w and not mask[i + w] and not seen[i + w]:
                seen[i + w] = 1; stack.append(i + w)

    border = []
    for x in range(w):
        border += [x, (h - 1) * w + x]
    for y in range(h):
        border += [y * w, y * w + w - 1]
    st = []
    for i in border:
        if not mask[i] and not seen[i]:
            seen[i] = 1
            st.append(i)
    flood(st)
    for i in range(N):
        if not mask[i] and not seen[i]:
            seen[i] = 1
            comp: list = []
            flood([i], comp)
            if len(comp) <= limit:
                for j in comp:
                    mask[j] = 255
    return mask


def run_selection(kind: str, wi: WorkImage, params: dict) -> bytearray:
    """Run one selection job (called on a worker thread)."""
    if kind == "wand":
        m = wand_select(wi, params["seed"], params["tol"], params["contiguous"])
    elif kind == "smart":
        m = smart_select(wi, params["fg"], params["bg"], params["reach"])
    elif kind == "box":
        m = box_select(wi, params["box"], params["reach"])
    else:
        raise ValueError(f"unknown selection kind {kind}")
    return fill_small_holes(m, wi.w, wi.h)


class MaskCanvas(QWidget):
    """Image canvas with a paintable mask: brush / eraser plus wand, smart brush and box
    select, size, undo / redo, zoom, pan.

    The mask is kept as a list of operations (strokes, selections, clear, invert, grow,
    shrink) so undo/redo is cheap in memory even for large images; it is replayed onto a
    fresh mask on undo. Selection tools first show a *pending* preview (Enter accepts,
    Esc discards; starting another selection accepts the previous one).
    Left-drag paints, right-drag does the opposite, wheel zooms, Shift+wheel or [ ]
    changes the brush size, Space+drag or middle-drag pans.
    """
    changed = pyqtSignal()
    brushChanged = pyqtSignal(int)
    fileDropped = pyqtSignal(str)
    pendingChanged = pyqtSignal(bool)
    busyChanged = pyqtSignal(bool)
    message = pyqtSignal(str)

    OVERLAY = QColor(255, 40, 80)
    SELECT_TOOLS = ("wand", "smart", "box")

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)
        self.image = QImage()
        self.mask = QImage()
        self.ops: list = []
        self.redo_ops: list = []
        self.brush = 40
        self.max_brush = 400
        self.tool = "brush"
        self.opt = {"tol": 40, "reach": 50, "contiguous": True, "mode": "add", "work": 384}
        self.show_overlay = True
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._fit = True
        self._stroke: Optional[dict] = None
        self._cur: Optional[QPointF] = None
        self._pan_from: Optional[QPointF] = None
        self._space = False
        # selection session
        self.pending: Optional[dict] = None
        self._pending_img: Optional[QImage] = None
        self._sess: Optional[dict] = None
        self._sm_fg: list = []
        self._sm_bg: list = []
        self._box_rect: Optional[QRectF] = None
        self._box_drag: Optional[tuple] = None
        self._work: Optional[WorkImage] = None
        self._gen = 0
        self._busy = False
        self._tasks: list = []

    # -- state -------------------------------------------------------------
    @property
    def erase(self) -> bool:
        return self.tool == "eraser"

    @erase.setter
    def erase(self, v: bool) -> None:
        self.set_tool("eraser" if v else "brush")

    def has_image(self) -> bool:
        return not self.image.isNull()

    def can_undo(self) -> bool:
        return bool(self.ops) or self.pending is not None

    def can_redo(self) -> bool:
        return bool(self.redo_ops)

    def has_pending(self) -> bool:
        return self.pending is not None

    def is_busy(self) -> bool:
        return self._busy

    def snapshot(self) -> tuple[QImage, QImage]:
        return self.image.copy(), self.mask.copy()

    def set_image(self, img: QImage) -> None:
        self._end_session(emit=False)
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
        if max(img.width(), img.height()) > MAX_CANVAS_SIDE:
            img = img.scaled(MAX_CANVAS_SIDE, MAX_CANVAS_SIDE,
                             Qt.AspectRatioMode.KeepAspectRatio, _SMOOTH)
        self.image = img
        self._work = None
        self.ops, self.redo_ops, self._stroke = [], [], None
        self._reset_mask()
        side = max(img.width(), img.height())
        self.max_brush = max(60, side // 4)
        self.set_brush(max(6, min(self.max_brush, side // 25)))
        self.fit()
        self._update_cursor()
        self.changed.emit()
        self.pendingChanged.emit(False)

    def set_brush(self, v: int) -> None:
        self.brush = max(2, min(self.max_brush, int(v)))
        self.brushChanged.emit(self.brush)
        self.update()

    def set_tool(self, name: str) -> None:
        if name == self.tool:
            return
        self.commit_pending()
        self._end_session()
        self.tool = name
        self._update_cursor()
        self.update()

    def _reset_mask(self) -> None:
        self.mask = QImage(self.image.size(), _PREMULT)
        self.mask.fill(Qt.GlobalColor.transparent)

    # -- drawing primitives ------------------------------------------------
    def _paint_stroke(self, p: QPainter, pts: list, size: float, erase: bool) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear if erase
                             else QPainter.CompositionMode.CompositionMode_SourceOver)
        if len(pts) == 1:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self.OVERLAY)
            p.drawEllipse(pts[0], size / 2, size / 2)
        else:
            p.setPen(QPen(self.OVERLAY, size, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF(pts))

    def _dilate(self, r: int) -> None:
        r = max(1, int(r))
        src = self.mask.copy()
        n = 8 if r <= 12 else 16
        p = QPainter(self.mask)
        for rad in (float(r), r / 2.0):
            for k in range(n):
                a = 2 * math.pi * k / n
                p.drawImage(round(rad * math.cos(a)), round(rad * math.sin(a)), src)
        p.end()

    def _apply(self, op: dict) -> None:
        t = op["t"]
        if t == "stroke":
            p = QPainter(self.mask)
            self._paint_stroke(p, op["pts"], op["r"], op["erase"])
            p.end()
        elif t == "clear":
            self.mask.fill(Qt.GlobalColor.transparent)
        elif t == "invert":
            inv = QImage(self.image.size(), _PREMULT)
            inv.fill(self.OVERLAY)
            p = QPainter(inv)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            p.drawImage(0, 0, self.mask)
            p.end()
            self.mask = inv
        elif t == "sel":
            alpha = alpha_image(op["data"], op["w"], op["h"])
            layer = QImage(self.image.size(), _PREMULT)
            layer.fill(self.OVERLAY)
            p = QPainter(layer)
            p.setCompositionMode(_DEST_IN)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(QRect(0, 0, self.image.width(), self.image.height()), alpha)
            p.end()
            p = QPainter(self.mask)
            if op["mode"] == "sub":
                for _ in range(2):          # x4 gain so soft edges are removed completely
                    q2 = QPainter(layer)
                    q2.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
                    q2.drawImage(0, 0, layer.copy())
                    q2.end()
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            p.drawImage(0, 0, layer)
            p.end()
        elif t == "grow":
            self._dilate(op["r"])
        elif t == "shrink":
            self._apply({"t": "invert"})
            self._dilate(op["r"])
            self._apply({"t": "invert"})

    def _rebuild(self) -> None:
        self._reset_mask()
        for op in self.ops:
            self._apply(op)
        self.update()
        self.changed.emit()

    def _push(self, op: dict) -> None:
        self.ops.append(op)
        self.redo_ops.clear()
        self._apply(op)
        self.update()
        self.changed.emit()

    # -- public actions ----------------------------------------------------
    def undo(self) -> None:
        if self._stroke is not None:
            return
        if self.pending is not None or self._sess is not None:
            self.discard_pending()
            return
        if self.ops:
            self.redo_ops.append(self.ops.pop())
            self._rebuild()

    def redo(self) -> None:
        if self.redo_ops and self._stroke is None:
            op = self.redo_ops.pop()
            self.ops.append(op)
            self._apply(op)
            self.update()
            self.changed.emit()

    def clear_mask(self) -> None:
        self.commit_pending()
        if self.has_image() and self.ops:
            self._push({"t": "clear"})

    def invert_mask(self) -> None:
        self.commit_pending()
        if self.has_image():
            self._push({"t": "invert"})

    def grow_mask(self, px: int) -> None:
        self.commit_pending()
        if self.has_image():
            self._push({"t": "grow", "r": int(px)})

    def shrink_mask(self, px: int) -> None:
        self.commit_pending()
        if self.has_image():
            self._push({"t": "shrink", "r": int(px)})

    # -- selection sessions ------------------------------------------------
    def _selection_mode(self, invert: bool = False) -> str:
        m = self.opt["mode"]
        if invert:
            m = "sub" if m == "add" else "add"
        return m

    def _ensure_work(self) -> WorkImage:
        side = int(self.opt["work"])
        if self._work is None or self._work.max_side != side:
            self._work = WorkImage(self.image, side)
        return self._work

    def _set_busy(self, on: bool) -> None:
        if on != self._busy:
            self._busy = on
            self._update_cursor()
            self.busyChanged.emit(on)
            self.update()

    def _end_session(self, emit: bool = True) -> None:
        had = self.pending is not None
        self._gen += 1                      # drop any selection still being calculated
        self._set_busy(False)
        self.pending = None
        self._pending_img = None
        self._sess = None
        self._sm_fg, self._sm_bg = [], []
        self._box_rect = None
        self._box_drag = None
        self.update()
        if had and emit:
            self.pendingChanged.emit(False)

    def commit_pending(self) -> bool:
        """Accept the previewed selection into the mask."""
        if self.pending is None:
            self._end_session()
            return False
        pd = self.pending
        op = {"t": "sel", "mode": pd["mode"], "w": pd["w"], "h": pd["h"], "data": pd["data"]}
        self._end_session()
        self._push(op)
        return True

    def discard_pending(self) -> None:
        self._end_session()

    def reselect(self) -> None:
        """Recompute the current selection after an option (tolerance, reach...) changed."""
        if self._sess is not None and self.has_image():
            self._relaunch()

    def _relaunch(self) -> None:
        s = self._sess
        if not s:
            return
        wi = self._ensure_work()
        kind = s["kind"]
        mode = self._selection_mode(s.get("inv", False))
        if kind == "wand":
            pt = s["pt"]
            x, y = int(pt.x() / wi.sx), int(pt.y() / wi.sy)
            if not (0 <= x < wi.w and 0 <= y < wi.h):
                return
            params = {"seed": y * wi.w + x, "tol": int(self.opt["tol"]),
                      "contiguous": bool(self.opt["contiguous"])}
        elif kind == "smart":
            fgb = raster_strokes(self._sm_fg, wi.w, wi.h, wi.sx, wi.sy)
            bgb = raster_strokes(self._sm_bg, wi.w, wi.h, wi.sx, wi.sy)
            fg = [i for i, v in enumerate(fgb) if v > 100]
            bg = [i for i, v in enumerate(bgb) if v > 100]
            if not fg:
                return
            params = {"fg": fg, "bg": bg, "reach": int(self.opt["reach"])}
        else:
            r = s["rect"]
            box = (int(r.left() / wi.sx), int(r.top() / wi.sy),
                   int(math.ceil(r.right() / wi.sx)), int(math.ceil(r.bottom() / wi.sy)))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                self.message.emit("Box too small - drag a larger box.")
                return
            params = {"box": box, "reach": int(self.opt["reach"])}
        self._launch(kind, wi, params, mode)

    def _launch(self, kind: str, wi: WorkImage, params: dict, mode: str) -> None:
        self._gen += 1
        gen = self._gen
        self._set_busy(True)
        self.message.emit("Calculating selection ...")
        task = Task(lambda: run_selection(kind, wi, params))
        task.result.connect(lambda res: self._launched(res, gen, wi, mode))
        task.failed.connect(lambda msg: self._launch_failed(msg, gen))
        self._tasks.append(task)
        task.finished.connect(lambda t=task: self._tasks.remove(t) if t in self._tasks else None)
        task.start()

    def _launch_failed(self, msg: str, gen: int) -> None:
        if gen != self._gen:
            return
        self._set_busy(False)
        self.message.emit(f"Selection failed: {msg}")

    def _launched(self, res: bytearray, gen: int, wi: WorkImage, mode: str) -> None:
        if gen != self._gen:
            return                          # a newer request replaced this one
        self._set_busy(False)
        if not res.count(255):
            self.pending, self._pending_img = None, None
            self.message.emit("Nothing was selected - try a higher Reach / Tolerance or another spot.")
            self.update()
            self.pendingChanged.emit(False)
            return
        alpha = alpha_image(res, wi.w, wi.h)
        col = QColor(41, 199, 230) if mode == "add" else QColor(255, 176, 0)
        prev = QImage(wi.w, wi.h, _PREMULT)
        prev.fill(col)
        p = QPainter(prev)
        p.setCompositionMode(_DEST_IN)
        p.drawImage(0, 0, alpha)
        p.end()
        self.pending = {"mode": mode, "w": wi.w, "h": wi.h, "data": bytes(res)}
        self._pending_img = prev
        self.message.emit("Selection ready - press Enter to accept, Esc to discard.")
        self.update()
        self.pendingChanged.emit(True)

    def shutdown(self) -> None:
        for t in list(self._tasks):
            t.wait(8000)

    # -- view --------------------------------------------------------------
    def fit(self) -> None:
        if not self.has_image():
            return
        iw, ih = self.image.width(), self.image.height()
        z = min(self.width() / iw, self.height() / ih) * 0.96
        self.zoom = max(0.01, z)
        self.pan = QPointF((self.width() - iw * self.zoom) / 2, (self.height() - ih * self.zoom) / 2)
        self._fit = True
        self.update()

    def actual_size(self) -> None:
        if not self.has_image():
            return
        self.zoom = 1.0
        self.pan = QPointF((self.width() - self.image.width()) / 2,
                           (self.height() - self.image.height()) / 2)
        self._fit = False
        self.update()

    def _to_img(self, pos: QPointF) -> QPointF:
        return (pos - self.pan) / self.zoom

    def _update_cursor(self) -> None:
        if not self.has_image():
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif self._pan_from is not None:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._space:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self._busy:
            self.setCursor(Qt.CursorShape.BusyCursor)
        elif self.tool in ("wand", "box"):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.BlankCursor)     # the brush circle is the cursor

    # -- events ------------------------------------------------------------
    def event(self, e) -> bool:
        if (e.type() == QEvent.Type.ShortcutOverride and e.key() == Qt.Key.Key_Escape
                and (self.pending is not None or self._sess is not None)):
            e.accept()                      # let Esc discard the selection, not stop a job
            return True
        return super().event(e)

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#05080f"))
        if not self.has_image():
            p.setPen(QColor("#4d5b73"))
            f = QFont(self.font())
            f.setPointSize(13)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Open an image on the Mark tab\n(or drop one here)")
            p.end()
            return
        iw, ih = self.image.width(), self.image.height()
        p.save()
        p.translate(self.pan)
        p.scale(self.zoom, self.zoom)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 1.0)
        p.drawImage(0, 0, self.image)
        if self.show_overlay:
            p.setOpacity(0.55)
            p.drawImage(0, 0, self.mask)
        if self._pending_img is not None:
            p.setOpacity(0.6)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(QRectF(0, 0, iw, ih), self._pending_img)
        p.setOpacity(1.0)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        scribbles = [(self._sm_fg, QColor(61, 220, 132, 215)), (self._sm_bg, QColor(255, 82, 82, 215))]
        if self._stroke is not None and self._stroke.get("t") == "scribble":
            cur = [(self._stroke["pts"], self._stroke["r"])]
            scribbles.append((cur, QColor(255, 82, 82, 215) if self._stroke["bg"]
                              else QColor(61, 220, 132, 215)))
        for strokes, col in scribbles:
            for pts, r in strokes:
                if len(pts) == 1:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(col)
                    p.drawEllipse(pts[0], r / 2, r / 2)
                else:
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(col, r, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                  Qt.PenJoinStyle.RoundJoin))
                    p.drawPolyline(QPolygonF(pts))
        rect = self._box_rect
        if self._box_drag is not None:
            a, b, _inv = self._box_drag
            rect = QRectF(a, b).normalized()
        if rect is not None:
            pen = QPen(QColor("#29c7e6"), 1.6, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rect)
        p.restore()
        if (self._cur is not None and self._pan_from is None and not self._space
                and not self._busy and self.underMouse() and self.tool in ("brush", "eraser", "smart")):
            r = self.brush * self.zoom / 2
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(0, 0, 0, 200), 3))
            p.drawEllipse(self._cur, r, r)
            p.setPen(QPen(QColor("#ffb000") if self.erase else QColor(255, 255, 255, 235), 1.4))
            p.drawEllipse(self._cur, r, r)
        if self._busy:
            p.setPen(QColor("#29c7e6"))
            f = QFont(self.font())
            f.setPointSize(11)
            f.setBold(True)
            p.setFont(f)
            p.drawText(14, 26, "Calculating selection ...")
        p.end()

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        if self._fit:
            self.fit()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if not self.has_image():
            return
        self.setFocus()
        b = e.button()
        if b == Qt.MouseButton.MiddleButton or (b == Qt.MouseButton.LeftButton and self._space):
            self._pan_from = e.position()
            self._update_cursor()
            return
        if b not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) or self._busy:
            return
        right = b == Qt.MouseButton.RightButton
        pt = self._to_img(e.position())
        tool = self.tool
        if tool in ("brush", "eraser"):
            erase = (tool == "eraser") != right
            self._stroke = {"t": "stroke", "pts": [pt], "r": self.brush, "erase": erase}
            p = QPainter(self.mask)
            self._paint_stroke(p, [pt], self.brush, erase)
            p.end()
            self.update()
        elif tool == "wand":
            if self.pending is not None:
                self.commit_pending()
            self._sess = {"kind": "wand", "inv": right, "pt": pt}
            self._relaunch()
        elif tool == "smart":
            self._stroke = {"t": "scribble", "pts": [pt], "r": self.brush, "bg": right}
            self.update()
        elif tool == "box":
            if self.pending is not None:
                self.commit_pending()
            self._box_rect = None
            self._box_drag = (pt, pt, right)
            self.update()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        self._cur = e.position()
        if self._pan_from is not None:
            self.pan = self.pan + (e.position() - self._pan_from)
            self._pan_from = e.position()
            self._fit = False
        elif self._stroke is not None:
            pt = self._to_img(e.position())
            last = self._stroke["pts"][-1]
            if (pt - last).manhattanLength() >= 0.5:
                self._stroke["pts"].append(pt)
                if self._stroke["t"] == "stroke":
                    p = QPainter(self.mask)
                    self._paint_stroke(p, [last, pt], self._stroke["r"], self._stroke["erase"])
                    p.end()
        elif self._box_drag is not None:
            a, _b, inv = self._box_drag
            self._box_drag = (a, self._to_img(e.position()), inv)
        self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._pan_from is not None and e.button() in (Qt.MouseButton.MiddleButton,
                                                         Qt.MouseButton.LeftButton):
            self._pan_from = None
            self._update_cursor()
            return
        if self._box_drag is not None and e.button() in (Qt.MouseButton.LeftButton,
                                                         Qt.MouseButton.RightButton):
            a, b, inv = self._box_drag
            self._box_drag = None
            rect = QRectF(a, b).normalized().intersected(
                QRectF(0, 0, self.image.width(), self.image.height()))
            self._box_rect = rect
            self._sess = {"kind": "box", "inv": inv, "rect": rect}
            self._relaunch()
            self.update()
            return
        if self._stroke is not None and e.button() in (Qt.MouseButton.LeftButton,
                                                       Qt.MouseButton.RightButton):
            s, self._stroke = self._stroke, None
            if s["t"] == "scribble":
                (self._sm_bg if s["bg"] else self._sm_fg).append((s["pts"], s["r"]))
                self._sess = {"kind": "smart", "inv": False}
                self._relaunch()
                self.update()
                return
            self.ops.append(s)               # already painted live
            self.redo_ops.clear()
            self.update()
            self.changed.emit()

    def wheelEvent(self, e) -> None:  # noqa: N802
        if not self.has_image():
            return
        d = e.angleDelta()
        dy = d.y() or d.x()
        if not dy:
            return
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step = max(1, int(self.brush * 0.1))
            self.set_brush(self.brush + (step if dy > 0 else -step))
            return
        nz = max(0.02, min(40.0, self.zoom * (1.15 if dy > 0 else 1 / 1.15)))
        pos = e.position()
        self.pan = pos - (pos - self.pan) * (nz / self.zoom)
        self.zoom = nz
        self._fit = False
        self.update()

    def keyPressEvent(self, e) -> None:  # noqa: N802
        k = e.key()
        if k == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = True
            self._update_cursor()
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.pending is not None:
            self.commit_pending()
        elif k == Qt.Key.Key_Escape and (self.pending is not None or self._sess is not None):
            self.discard_pending()
        elif k == Qt.Key.Key_BracketLeft:
            self.set_brush(int(self.brush * 0.85) - 1)
        elif k == Qt.Key.Key_BracketRight:
            self.set_brush(int(self.brush * 1.18) + 1)
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = False
            self._update_cursor()
        else:
            super().keyReleaseEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self._cur = None
        self.update()

    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p and Path(p).suffix.lower() in IMAGE_EXTS:
                self.fileDropped.emit(p)
                break
        e.acceptProposedAction()


def thumb_icon(path: str, size: int = 120) -> QIcon:
    r = QImageReader(path)
    r.setAutoTransform(True)
    sz = r.size()
    if sz.isValid() and sz.width() and sz.height():
        sc = size / max(sz.width(), sz.height())
        r.setScaledSize(QSize(max(1, int(sz.width() * sc)), max(1, int(sz.height() * sc))))
    img = r.read()
    return QIcon(QPixmap.fromImage(img)) if not img.isNull() else QIcon()


# --------------------------------------------------------------------------- #
#  Dialogs
# --------------------------------------------------------------------------- #


class _DownloadDialogBase(QDialog):
    """Shared progress/cancel plumbing."""

    def __init__(self, parent, title: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 480)
        self._tasks: list[QThread] = []
        self.dl: Optional[DownloadThread] = None
        self.lay = QVBoxLayout(self)
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.msg = QLabel("")
        self.msg.setWordWrap(True)

    def _keep(self, t: QThread) -> QThread:
        self._tasks.append(t)
        t.finished.connect(lambda: self._tasks.remove(t) if t in self._tasks else None)
        return t

    def start_download(self, specs: list[DlSpec], on_ok: Callable[[list], None]) -> None:
        self.dl = DownloadThread(specs, self)
        self.dl.status.connect(self.msg.setText)
        self.dl.progress.connect(self._on_progress)
        self.dl.finished_ok.connect(on_ok)
        self.dl.failed.connect(self._on_fail)
        self._set_busy(True)
        self.dl.start()

    def _on_progress(self, done: int, total: int) -> None:
        if total:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(1000 * done / total))
            self.bar.setFormat(f"{human_size(done)} / {human_size(total)}")
        else:
            self.bar.setRange(0, 0)

    def _on_fail(self, msg: str) -> None:
        self._set_busy(False)
        self.msg.setText(msg)
        QMessageBox.warning(self, APP_NAME, msg)

    def _set_busy(self, busy: bool) -> None:  # overridden
        pass

    def reject(self) -> None:
        if self.dl and self.dl.isRunning():
            self.dl.cancel()
            self.dl.wait(3000)
        super().reject()


class HfDialog(_DownloadDialogBase):
    """Browse a Hugging Face repo and download any .gguf / .safetensors from it."""
    downloaded = pyqtSignal(list)   # list[tuple[str role, str path]]

    REPOS = ["leejet/Qwen-Image-2.1-GGUF", "unsloth/Qwen-Image-2.1-GGUF",
             "Abiray/Qwen-Image-2.1-GGUF", "AlperKTS/Qwen-Image-2.1-GGUF",
             "Haverbex/Qwen-Image-2.1-UltraFast-GGUF", "Comfy-Org/Qwen-Image-2.1",
             "Qwen/Qwen3-VL-8B-Instruct-GGUF"]

    def __init__(self, parent, models_dir: Path) -> None:
        super().__init__(parent, "Download from Hugging Face")
        self.models_dir = models_dir
        self.repo = QComboBox()
        self.repo.setEditable(True)
        self.repo.addItems(self.REPOS)
        self.list_btn = QPushButton("List files")
        top = QHBoxLayout()
        top.addWidget(QLabel("Repository:"))
        top.addWidget(self.repo, 1)
        top.addWidget(self.list_btn)
        self.files = QListWidget()
        self.files.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.role = QComboBox()
        self.role.addItems(["Auto-detect", "Diffusion model", "VAE", "Text encoder",
                            "Vision projector (mmproj)"])
        self.get_btn = QPushButton("Download selected")
        self.get_btn.setObjectName("primary")
        self.cancel_btn = QPushButton("Close")
        row = QHBoxLayout()
        row.addWidget(QLabel("Use as:"))
        row.addWidget(self.role)
        row.addStretch(1)
        row.addWidget(self.get_btn)
        row.addWidget(self.cancel_btn)
        note = QLabel("Any quant works - the runtime reads it from the file. Gated repos need "
                      "the HF_TOKEN environment variable.")
        note.setWordWrap(True)
        for w in (top,):
            self.lay.addLayout(w)
        self.lay.addWidget(self.files, 1)
        self.lay.addLayout(row)
        self.lay.addWidget(self.bar)
        self.lay.addWidget(self.msg)
        self.lay.addWidget(note)
        self.list_btn.clicked.connect(self.list_files)
        self.get_btn.clicked.connect(self.download)
        self.cancel_btn.clicked.connect(self.reject)

    def _set_busy(self, busy: bool) -> None:
        for w in (self.list_btn, self.get_btn, self.repo, self.files, self.role):
            w.setEnabled(not busy)
        if not busy:
            self.bar.setRange(0, 1)
            self.bar.setValue(0)

    def list_files(self) -> None:
        repo = self.repo.currentText().strip()
        if not repo:
            return
        self.msg.setText(f"Listing {repo} ...")
        self.files.clear()
        t = self._keep(Task(lambda: hf_list_files(repo), self))
        t.result.connect(self._fill)
        t.failed.connect(lambda m: self.msg.setText(f"Could not list repo: {m}"))
        t.start()

    def _fill(self, items: list) -> None:
        for name, size in items:
            it = QListWidgetItem(f"{name}    [{human_size(size)}]" if size else name)
            it.setData(Qt.ItemDataRole.UserRole, name)
            self.files.addItem(it)
        self.msg.setText(f"{len(items)} model file(s) found.")

    def download(self) -> None:
        repo = self.repo.currentText().strip()
        sel = self.files.selectedItems()
        if not sel:
            self.msg.setText("Select one or more files first.")
            return
        mapping = {1: "diffusion", 2: "vae", 3: "llm", 4: "mmproj"}
        folder = {"diffusion": "diffusion_models", "vae": "vae", "llm": "text_encoders",
                  "mmproj": "text_encoders", "unknown": "misc"}
        specs = []
        for it in sel:
            name = it.data(Qt.ItemDataRole.UserRole)
            role = mapping.get(self.role.currentIndex()) or classify_model_file(Path(name))
            if role == "unknown":
                role = "diffusion"
            specs.append(DlSpec(hf_url(repo, name),
                                self.models_dir / folder[role] / Path(name).name, role))
        self.start_download(specs, self._ok)

    def _ok(self, done: list) -> None:
        self._set_busy(False)
        self.msg.setText("Done.")
        self.downloaded.emit([(s.role, str(s.dest)) for s in done])


class SdCliDialog(_DownloadDialogBase):
    """Fetch a stable-diffusion.cpp release build."""
    installed = pyqtSignal(str)

    def __init__(self, parent) -> None:
        super().__init__(parent, "Get sd-cli (stable-diffusion.cpp)")
        self.assets: list[dict] = []
        self.tag = ""
        self.lst = QListWidget()
        self.lst.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.get_btn = QPushButton("Download && install selected")
        self.get_btn.setObjectName("primary")
        self.close_btn = QPushButton("Close")
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.get_btn)
        row.addWidget(self.close_btn)
        hint = QLabel("Pick the build that matches your GPU (NVIDIA RTX -> a CUDA build; AMD/"
                      "Intel -> Vulkan; otherwise AVX2 for CPU). CUDA builds may also need the "
                      "separate 'cudart' archive - select both. Qwen-Image-2.1 needs a recent "
                      "release.")
        hint.setWordWrap(True)
        self.lay.addWidget(hint)
        self.lay.addWidget(self.lst, 1)
        self.lay.addLayout(row)
        self.lay.addWidget(self.bar)
        self.lay.addWidget(self.msg)
        self.get_btn.clicked.connect(self.install)
        self.close_btn.clicked.connect(self.reject)
        self.msg.setText("Reading the latest release ...")
        t = self._keep(Task(lambda: http_json(
            "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"), self))
        t.result.connect(self._fill)
        t.failed.connect(lambda m: self.msg.setText(f"Could not reach GitHub: {m}"))
        t.start()

    def _set_busy(self, busy: bool) -> None:
        self.get_btn.setEnabled(not busy)
        self.lst.setEnabled(not busy)
        if not busy:
            self.bar.setRange(0, 1)
            self.bar.setValue(0)

    def _fill(self, rel: dict) -> None:
        self.tag = rel.get("tag_name", "latest")
        plat = {"win32": ("win",), "linux": ("linux", "ubuntu"),
                "darwin": ("mac", "osx", "darwin")}.get(sys.platform, ("",))
        assets = [a for a in rel.get("assets", []) if a["name"].lower().endswith(".zip")]
        match = [a for a in assets if any(k in a["name"].lower() for k in plat)] or assets
        self.assets = match
        for a in match:
            it = QListWidgetItem(f"{a['name']}   [{human_size(a.get('size', 0))}]")
            self.lst.addItem(it)
        for i, a in enumerate(match):          # sensible default for NVIDIA users
            n = a["name"].lower()
            if "cuda" in n or "cudart" in n:
                self.lst.item(i).setSelected(True)
        self.msg.setText(f"Release {self.tag}: {len(match)} archive(s) for this platform.")

    def install(self) -> None:
        rows = [self.lst.row(i) for i in self.lst.selectedItems()]
        if not rows:
            self.msg.setText("Select at least one archive.")
            return
        dest_dir = TOOLS_DIR / "sd-cpp" / self.tag
        specs = [DlSpec(self.assets[r]["browser_download_url"],
                        dest_dir / self.assets[r]["name"], "sd_cli", True) for r in rows]
        self.start_download(specs, lambda _d: self._ok(dest_dir))

    def _ok(self, dest_dir: Path) -> None:
        self._set_busy(False)
        exe = ""
        for nm in SD_CLI_NAMES:
            hit = next(dest_dir.rglob(nm), None)
            if hit:
                exe = str(hit)
                break
        if exe:
            self.msg.setText(f"Installed: {exe}")
            self.installed.emit(exe)
        else:
            self.msg.setText(f"Extracted to {dest_dir}, but no sd-cli executable was found "
                             "in it. Locate it manually with the '...' button.")


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #

STYLE = """
QWidget { background:#0a0e17; color:#d6dde8; font-family:'Segoe UI','Inter',sans-serif; font-size:10pt; }
QMainWindow, QDialog { background:#0a0e17; }
QGroupBox { border:1px solid #1c2a40; border-radius:4px; margin-top:14px; padding:10px 6px 6px 6px; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; color:#29c7e6; font-weight:bold; }
QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
    background:#05080f; border:1px solid #1c2a40; border-radius:3px; padding:3px;
    selection-background-color:#12506a; }
QComboBox QAbstractItemView { background:#05080f; selection-background-color:#12506a; }
QPushButton { background:#13203a; border:1px solid #24406a; border-radius:3px; padding:6px 12px; }
QPushButton:hover { border-color:#29c7e6; }
QPushButton:disabled { color:#5b6578; border-color:#1c2a40; }
QPushButton#primary { background:#0e6f86; border-color:#29c7e6; color:#fff; font-weight:bold; padding:9px 14px; }
QPushButton#primary:hover { background:#118aa6; }
QPushButton#danger { background:#6a1f1f; border-color:#c0504d; color:#fff; }
QPushButton:checked { background:#0e6f86; border-color:#29c7e6; color:#fff; }
QSlider::groove:horizontal { height:4px; background:#1c2a40; border-radius:2px; }
QSlider::handle:horizontal { background:#29c7e6; width:14px; margin:-6px 0; border-radius:7px; }
QSlider::sub-page:horizontal { background:#0e6f86; border-radius:2px; }
QTabWidget::pane { border:1px solid #1c2a40; border-radius:4px; top:-1px; }
QTabBar::tab { background:#0d1422; border:1px solid #1c2a40; padding:7px 14px; margin-right:2px;
    border-top-left-radius:4px; border-top-right-radius:4px; }
QTabBar::tab:selected { background:#13203a; color:#29c7e6; border-bottom-color:#13203a; }
QProgressBar { background:#05080f; border:1px solid #1c2a40; border-radius:3px; text-align:center; height:18px; }
QProgressBar::chunk { background:#29c7e6; }
QPlainTextEdit#log { font-family:Consolas,'Cascadia Mono',monospace; font-size:9pt; color:#7fdc9a; }
QLabel#muted { color:#7d8aa3; }
QLabel#info { color:#f0a040; }
QSplitter::handle { background:#1c2a40; }
QScrollBar:vertical { background:#05080f; width:10px; }
QScrollBar::handle:vertical { background:#24406a; border-radius:4px; min-height:24px; }
QScrollBar:horizontal { background:#05080f; height:10px; }
QScrollBar::handle:horizontal { background:#24406a; border-radius:4px; min-width:24px; }
QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()
        self.s = Settings()
        self.setWindowTitle(f"{APP_NAME}  -  Qwen-Image-2.1")
        self.resize(1480, 900)
        self.runner = Runner(self)
        self.runner.log.connect(self.append_log)
        self.runner.stage.connect(self.on_stage)
        self.runner.progress.connect(self.on_progress)
        self.runner.done.connect(self.on_done)
        self._tasks: list[QThread] = []
        self._job: Optional[JobSpec] = None
        self._work: Optional[Path] = None
        self._t0 = 0.0
        self._last_step = 0
        self._batch_idx = 0
        self._cmd: list[str] = []
        self._last_mode = "create"
        self._result_path: Optional[str] = None
        self._compare_pix: Optional[QPixmap] = None
        self._result_pix: Optional[QPixmap] = None
        self._show_canvas = True
        self._mark_ctx: Optional[dict] = None
        self._preset_busy = False
        self._family = ""

        self._build_ui()
        self._load_settings_into_ui()
        self.load_history()
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.start_job)
        QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self.start_job)
        QShortcut(QKeySequence("Esc"), self, activated=self.stop_job)
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self._paste_shortcut)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo_shortcut)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo_shortcut)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._redo_shortcut)
        self.refresh_status()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        # ---- left: tabs + sampling + run
        self.canvas = MaskCanvas()
        self.tabs = QTabWidget()
        self.tabs.addTab(scrolled(self._tab_create()), "Create")
        self.tabs.addTab(scrolled(self._tab_edit()), "Edit")
        self.tabs.addTab(scrolled(self._tab_mark()), "Mark")
        self.tabs.addTab(scrolled(self._tab_models()), "Models")
        self.tabs.addTab(scrolled(self._tab_settings()), "Settings")
        self.tabs.currentChanged.connect(self._tab_changed)

        samp = QGroupBox("Sampling")
        form = QFormLayout(samp)
        self.steps = QSpinBox(); self.steps.setRange(1, 200)
        self.cfg = QDoubleSpinBox(); self.cfg.setRange(0.0, 30.0); self.cfg.setSingleStep(0.5)
        self.cfg.setDecimals(2)
        self.seed = QSpinBox(); self.seed.setRange(-1, 2147483647)
        dice = QPushButton("random"); dice.clicked.connect(lambda: self.seed.setValue(-1))
        seed_row = QHBoxLayout(); seed_row.addWidget(self.seed, 1); seed_row.addWidget(dice)
        self.sampler = QComboBox(); self.sampler.addItems(SAMPLERS)
        self.scheduler = QComboBox(); self.scheduler.addItems(SCHEDULERS)
        self.batch = QSpinBox(); self.batch.setRange(1, 16)
        self.transparent = QCheckBox("Transparent background (RGBA prompt format, PNG/WebP)")
        self.preset = QComboBox()
        self.preset.addItems(list(SAMPLING_PRESETS) + ["Custom"])
        self.preset.currentTextChanged.connect(self._preset_chosen)
        self.steps.valueChanged.connect(self._values_changed)
        self.cfg.valueChanged.connect(self._values_changed)
        form.addRow("Preset", self.preset)
        form.addRow("Steps", self.steps)
        form.addRow("CFG scale", self.cfg)
        form.addRow("Seed (-1 = random)", seed_row)
        form.addRow("Sampler", self.sampler)
        form.addRow("Scheduler", self.scheduler)
        form.addRow("Images per run", self.batch)
        form.addRow(self.transparent)
        self.cfg_note = QLabel("")
        self.cfg_note.setObjectName("info"); self.cfg_note.setWordWrap(True)
        form.addRow(self.cfg_note)

        self.run_btn = QPushButton("Generate   (Ctrl+Enter)")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.start_job)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_job)
        self.copy_btn = QPushButton("Copy command")
        self.copy_btn.clicked.connect(self.copy_command)
        rb = QHBoxLayout(); rb.addWidget(self.run_btn, 1); rb.addWidget(self.stop_btn)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        ll.addWidget(self.tabs, 1)
        ll.addWidget(samp)
        ll.addLayout(rb)
        ll.addWidget(self.copy_btn)
        left.setMinimumWidth(410)
        left.setMaximumWidth(470)

        # ---- center: viewer + progress + log
        self.viewer = ImageViewer()
        tb = QHBoxLayout()
        self.fit_btn = QPushButton("Fit"); self.fit_btn.clicked.connect(self._fit_active)
        self.one_btn = QPushButton("100%"); self.one_btn.clicked.connect(self._actual_active)
        self.view_toggle = QPushButton("Show result")
        self.view_toggle.clicked.connect(self._toggle_center)
        self.view_toggle.setVisible(False)
        self.cmp_btn = QPushButton("Hold: show original")
        self.cmp_btn.pressed.connect(lambda: self._compare(True))
        self.cmp_btn.released.connect(lambda: self._compare(False))
        self.save_btn = QPushButton("Save as...")
        self.save_btn.clicked.connect(self.save_as)
        self.use_btn = QPushButton("Use result as input")
        self.use_btn.clicked.connect(self.use_result_as_input)
        self.folder_btn = QPushButton("Open folder")
        self.folder_btn.clicked.connect(self.open_out_dir)
        for b in (self.fit_btn, self.one_btn, self.view_toggle, self.cmp_btn):
            tb.addWidget(b)
        tb.addStretch(1)
        for b in (self.use_btn, self.save_btn, self.folder_btn):
            tb.addWidget(b)
        self.progress = QProgressBar(); self.progress.setRange(0, 1); self.progress.setValue(0)
        self.status = QLabel("Ready"); self.status.setObjectName("muted")
        self.log = QPlainTextEdit(); self.log.setObjectName("log")
        self.log.setReadOnly(True); self.log.setMaximumBlockCount(4000)
        self.log.setPlaceholderText("sd-cli output appears here ...")

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(6, 6, 6, 0)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.viewer)
        self.stack.addWidget(self.canvas)
        tl.addLayout(tb)
        tl.addWidget(self.stack, 1)
        tl.addWidget(self.progress)
        tl.addWidget(self.status)
        vs = QSplitter(Qt.Orientation.Vertical)
        vs.addWidget(top)
        vs.addWidget(self.log)
        vs.setStretchFactor(0, 5)
        vs.setStretchFactor(1, 1)
        vs.setSizes([700, 150])

        # ---- right: history
        self.hist = QListWidget()
        self.hist.setViewMode(QListView.ViewMode.IconMode)
        self.hist.setIconSize(QSize(120, 120))
        self.hist.setGridSize(QSize(136, 140))
        self.hist.setResizeMode(QListView.ResizeMode.Adjust)
        self.hist.setMovement(QListView.Movement.Static)
        self.hist.setWordWrap(True)
        self.hist.itemClicked.connect(self.history_clicked)
        self.hist.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.hist.customContextMenuRequested.connect(self.history_menu)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)
        hh = QHBoxLayout()
        lab = QLabel("HISTORY"); lab.setObjectName("muted")
        self.h_clear = QPushButton("Clear all")
        self.h_clear.setToolTip("Delete every image in the history (and its settings file) at once")
        self.h_clear.clicked.connect(self.clear_history)
        hh.addWidget(lab); hh.addStretch(1); hh.addWidget(self.h_clear)
        rl.addLayout(hh)
        rl.addWidget(self.hist, 1)
        right.setMinimumWidth(170)

        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(left)
        sp.addWidget(vs)
        sp.addWidget(right)
        sp.setStretchFactor(1, 1)
        sp.setSizes([440, 840, 220])
        self.setCentralWidget(sp)
        self.sbar = self.statusBar()

    # -- Create tab
    def _tab_create(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        g = QGroupBox("Prompt")
        gl = QVBoxLayout(g)
        self.c_prompt = QPlainTextEdit()
        self.c_prompt.setPlaceholderText("Describe the image ... (text inside the image is supported: "
                                         "put it in quotes)")
        self.c_prompt.setFixedHeight(120)
        self.c_neg = QPlainTextEdit()
        self.c_neg.setPlaceholderText("Negative prompt (optional)")
        self.c_neg.setFixedHeight(60)
        gl.addWidget(self.c_prompt)
        gl.addWidget(self.c_neg)
        lay.addWidget(g)
        sg = QGroupBox("Size")
        sl = QFormLayout(sg)
        self.c_preset = QComboBox(); self.c_preset.addItems(list(SIZE_PRESETS))
        self.c_w = QSpinBox(); self.c_w.setRange(64, 4096); self.c_w.setSingleStep(32)
        self.c_h = QSpinBox(); self.c_h.setRange(64, 4096); self.c_h.setSingleStep(32)
        self.c_preset.currentTextChanged.connect(self._preset_changed)
        sl.addRow("Preset", self.c_preset)
        sl.addRow("Width", self.c_w)
        sl.addRow("Height", self.c_h)
        hint = QLabel("Sizes must be divisible by 32 (rounded automatically).")
        hint.setObjectName("muted")
        sl.addRow(hint)
        lay.addWidget(sg)
        lay.addStretch(1)
        return w

    # -- Edit tab
    def _tab_edit(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        g = QGroupBox("Input image(s)  -  drop files here")
        gl = QVBoxLayout(g)
        self.refs = RefList()
        self.refs.filesDropped.connect(self.add_refs)
        self.refs.itemSelectionChanged.connect(self._ref_selected)
        b = QHBoxLayout()
        add = QPushButton("Add..."); add.clicked.connect(self.browse_refs)
        paste = QPushButton("Paste"); paste.clicked.connect(self.paste_ref)
        rem = QPushButton("Remove"); rem.clicked.connect(self.remove_refs)
        clr = QPushButton("Clear"); clr.clicked.connect(self.refs.clear)
        for x in (add, paste, rem, clr):
            b.addWidget(x)
        gl.addWidget(self.refs)
        gl.addLayout(b)
        n = QLabel("Several images are used in list order (e.g. 'put the dog from image 2 "
                   "into image 1'). Drag to reorder.")
        n.setWordWrap(True); n.setObjectName("muted")
        gl.addWidget(n)
        lay.addWidget(g)

        pg = QGroupBox("What should change?")
        pl = QVBoxLayout(pg)
        self.e_tpl = QComboBox()
        for label, _ in EDIT_TEMPLATES:
            self.e_tpl.addItem(label)
        self.e_tpl.currentIndexChanged.connect(self._tpl_chosen)
        self.e_prompt = QPlainTextEdit()
        self.e_prompt.setPlaceholderText("e.g. 'Remove the car in the background' / 'Add a hat to "
                                         "the man' / 'Change the sky to a stormy night'")
        self.e_prompt.setFixedHeight(110)
        self.e_neg = QPlainTextEdit()
        self.e_neg.setPlaceholderText("Negative prompt (optional)")
        self.e_neg.setFixedHeight(50)
        pl.addWidget(self.e_tpl)
        pl.addWidget(self.e_prompt)
        pl.addWidget(self.e_neg)
        lay.addWidget(pg)

        sg = QGroupBox("Output size")
        sl = QFormLayout(sg)
        self.e_mode = QComboBox(); self.e_mode.addItems(["Match input image", "Custom"])
        self.e_w = QSpinBox(); self.e_w.setRange(64, 4096); self.e_w.setSingleStep(32)
        self.e_h = QSpinBox(); self.e_h.setRange(64, 4096); self.e_h.setSingleStep(32)
        self.e_mode.currentTextChanged.connect(self._edit_mode_changed)
        sl.addRow("Mode", self.e_mode)
        sl.addRow("Width", self.e_w)
        sl.addRow("Height", self.e_h)
        self.e_info = QLabel(""); self.e_info.setObjectName("muted")
        sl.addRow(self.e_info)
        lay.addWidget(sg)
        lay.addStretch(1)
        return w

    # -- Mark tab
    def _tab_mark(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)

        g = QGroupBox("1. Image")
        gl = QVBoxLayout(g)
        b = QHBoxLayout()
        for text, fn in (("Open...", self.mark_open), ("Paste", self.mark_paste),
                         ("Use current result", self.mark_use_result)):
            btn = QPushButton(text)
            btn.clicked.connect(fn)
            b.addWidget(btn)
        gl.addLayout(b)
        self.k_info = QLabel("Open an image, or drop one onto the canvas.")
        self.k_info.setObjectName("muted"); self.k_info.setWordWrap(True)
        gl.addWidget(self.k_info)
        lay.addWidget(g)

        tg = QGroupBox("2. Mark the area")
        tl = QVBoxLayout(tg)
        self._tool_grp = QButtonGroup(self)
        self._tool_grp.setExclusive(True)
        self._tool_btns: dict = {}

        def tool_row(items) -> QHBoxLayout:
            r = QHBoxLayout()
            for key, label, tip in items:
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.setToolTip(tip)
                self._tool_grp.addButton(btn)
                self._tool_btns[key] = btn
                btn.toggled.connect(lambda on, k=key: self._tool_toggled(k, on))
                r.addWidget(btn)
            return r

        tl.addLayout(tool_row([("brush", "Brush", "Paint the area by hand"),
                               ("eraser", "Eraser", "Rub out marking by hand")]))
        lab = QLabel("Smart selection")
        lab.setObjectName("muted")
        tl.addWidget(lab)
        tl.addLayout(tool_row([("wand", "Wand", "Click to select an area of similar color"),
                               ("smart", "Smart brush", "Scribble on the object; it snaps to its edges"),
                               ("box", "Box select", "Drag a box around an object")]))
        self.k_brush_btn, self.k_erase_btn = self._tool_btns["brush"], self._tool_btns["eraser"]
        self.k_wand_btn, self.k_smart_btn = self._tool_btns["wand"], self._tool_btns["smart"]
        self.k_box_btn = self._tool_btns["box"]
        self.k_brush_btn.setChecked(True)

        self.k_selopts = QWidget()
        so = QVBoxLayout(self.k_selopts)
        so.setContentsMargins(0, 2, 0, 2)
        self.k_selmode = QComboBox(); self.k_selmode.addItems(["Add to marking", "Remove from marking"])
        self.k_prec = QComboBox(); self.k_prec.addItems(list(SEL_PRECISION))
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Mode")); r1.addWidget(self.k_selmode, 1)
        r1.addWidget(QLabel("Detail")); r1.addWidget(self.k_prec)
        so.addLayout(r1)
        self.k_tol_row = QWidget()
        r2 = QHBoxLayout(self.k_tol_row); r2.setContentsMargins(0, 0, 0, 0)
        self.k_tol = QSlider(Qt.Orientation.Horizontal); self.k_tol.setRange(1, 100)
        self.k_tol.setTracking(False)
        self.k_tol_val = QLabel("40"); self.k_tol_val.setMinimumWidth(28)
        r2.addWidget(QLabel("Tolerance")); r2.addWidget(self.k_tol, 1); r2.addWidget(self.k_tol_val)
        so.addWidget(self.k_tol_row)
        self.k_contig = QCheckBox("Contiguous only (untick: select this color everywhere)")
        so.addWidget(self.k_contig)
        self.k_reach_row = QWidget()
        r3 = QHBoxLayout(self.k_reach_row); r3.setContentsMargins(0, 0, 0, 0)
        self.k_reach = QSlider(Qt.Orientation.Horizontal); self.k_reach.setRange(1, 100)
        self.k_reach.setTracking(False)
        self.k_reach_val = QLabel("50"); self.k_reach_val.setMinimumWidth(28)
        r3.addWidget(QLabel("Reach")); r3.addWidget(self.k_reach, 1); r3.addWidget(self.k_reach_val)
        so.addWidget(self.k_reach_row)
        r4 = QHBoxLayout()
        self.k_accept = QPushButton("Accept  (Enter)"); self.k_accept.setObjectName("primary")
        self.k_discard = QPushButton("Discard  (Esc)")
        self.k_accept.setEnabled(False); self.k_discard.setEnabled(False)
        self.k_accept.clicked.connect(self.canvas.commit_pending)
        self.k_discard.clicked.connect(self.canvas.discard_pending)
        r4.addWidget(self.k_accept, 1); r4.addWidget(self.k_discard)
        so.addLayout(r4)
        self.k_sel_hint = QLabel("")
        self.k_sel_hint.setObjectName("muted"); self.k_sel_hint.setWordWrap(True)
        so.addWidget(self.k_sel_hint)
        tl.addWidget(self.k_selopts)
        self.k_tol.sliderMoved.connect(lambda v: self.k_tol_val.setText(str(v)))
        self.k_reach.sliderMoved.connect(lambda v: self.k_reach_val.setText(str(v)))
        self.k_tol.valueChanged.connect(self._sel_opts_changed)
        self.k_reach.valueChanged.connect(self._sel_opts_changed)
        self.k_contig.toggled.connect(self._sel_opts_changed)
        self.k_selmode.currentIndexChanged.connect(self._sel_opts_changed)
        self.k_prec.currentTextChanged.connect(self._sel_opts_changed)
        sz = QHBoxLayout()
        self.k_size = QSlider(Qt.Orientation.Horizontal)
        self.k_size.setRange(2, 400)
        self.k_size.setValue(40)
        self.k_size_val = QLabel("40 px")
        self.k_size_val.setMinimumWidth(52)
        sz.addWidget(QLabel("Size"))
        sz.addWidget(self.k_size, 1)
        sz.addWidget(self.k_size_val)
        tl.addLayout(sz)
        row2 = QHBoxLayout()
        self.k_undo = QPushButton("Undo"); self.k_redo = QPushButton("Redo")
        self.k_clear = QPushButton("Clear"); self.k_invert = QPushButton("Invert")
        self.k_undo.clicked.connect(self.canvas.undo)
        self.k_redo.clicked.connect(self.canvas.redo)
        self.k_clear.clicked.connect(self.canvas.clear_mask)
        self.k_invert.clicked.connect(self.canvas.invert_mask)
        for x in (self.k_undo, self.k_redo, self.k_clear, self.k_invert):
            row2.addWidget(x)
        tl.addLayout(row2)
        row3 = QHBoxLayout()
        self.k_grow = QPushButton("Grow"); self.k_shrink = QPushButton("Shrink")
        self.k_gs_px = QSpinBox(); self.k_gs_px.setRange(1, 300); self.k_gs_px.setValue(4)
        self.k_gs_px.setSuffix(" px")
        self.k_grow.clicked.connect(lambda _=False: self.canvas.grow_mask(self.k_gs_px.value()))
        self.k_shrink.clicked.connect(lambda _=False: self.canvas.shrink_mask(self.k_gs_px.value()))
        row3.addWidget(self.k_grow); row3.addWidget(self.k_shrink); row3.addWidget(self.k_gs_px)
        tl.addLayout(row3)
        self.k_overlay = QCheckBox("Show marking")
        self.k_overlay.setChecked(True)
        self.k_overlay.toggled.connect(self._overlay_toggled)
        tl.addWidget(self.k_overlay)
        tip = QLabel("Left-drag paints, right-drag does the opposite. Wheel zooms, "
                     "Shift+wheel or [ ] changes brush size, Space+drag pans, Ctrl+Z / Ctrl+Y "
                     "undo / redo. Selections show a preview first: Enter accepts, Esc discards.")
        tip.setObjectName("muted"); tip.setWordWrap(True)
        tl.addWidget(tip)
        lay.addWidget(tg)
        self.k_size.valueChanged.connect(self.canvas.set_brush)
        self.canvas.brushChanged.connect(self._brush_changed)
        self.canvas.changed.connect(self._canvas_changed)
        self.canvas.fileDropped.connect(self.load_mark_image)
        self.canvas.pendingChanged.connect(self._pending_changed)
        self.canvas.message.connect(lambda m: self.sbar.showMessage(m, 8000))
        self._canvas_changed()
        self._update_tool_panel()

        pg = QGroupBox("3. What should happen in the marked area?")
        pl = QVBoxLayout(pg)
        self.k_tpl = QComboBox()
        for label, _ in MARK_TEMPLATES:
            self.k_tpl.addItem(label)
        self.k_tpl.currentIndexChanged.connect(self._mark_tpl_chosen)
        self.k_prompt = QPlainTextEdit()
        self.k_prompt.setPlaceholderText("e.g. 'Remove it' / 'Add a red balloon' / "
                                         "'Make it look like a marble statue'")
        self.k_prompt.setFixedHeight(90)
        self.k_neg = QPlainTextEdit()
        self.k_neg.setPlaceholderText("Negative prompt (optional)")
        self.k_neg.setFixedHeight(46)
        pl.addWidget(self.k_tpl)
        pl.addWidget(self.k_prompt)
        pl.addWidget(self.k_neg)
        lay.addWidget(pg)

        og = QGroupBox("4. Options")
        of = QFormLayout(og)
        self.k_color = QComboBox(); self.k_color.addItems(list(MARK_COLORS))
        self.k_style = QComboBox(); self.k_style.addItems(MARK_STYLES)
        self.k_wrap = QCheckBox("Add 'marked area' instructions to my text")
        self.k_crop = QCheckBox("Focus mode: crop around the marking, edit, paste back")
        self.k_pad = QSpinBox(); self.k_pad.setRange(0, 2048); self.k_pad.setSingleStep(16)
        self.k_strict = QCheckBox("Keep everything outside the marking identical")
        self.k_feather = QSpinBox(); self.k_feather.setRange(0, 128)
        self.k_clean = QCheckBox("Also send the unmarked image as reference 1")
        self.k_crop.toggled.connect(self._crop_toggled)
        of.addRow("Marker color", self.k_color)
        of.addRow("Marker style", self.k_style)
        of.addRow(self.k_wrap)
        of.addRow(self.k_crop)
        of.addRow("Context around marking (px)", self.k_pad)
        of.addRow(self.k_strict)
        of.addRow("Edge feather (px)", self.k_feather)
        of.addRow(self.k_clean)
        note = QLabel("The model has no true mask input: your marking is drawn on the picture "
                      "and the model is told to change only that area. 'Keep everything outside "
                      "identical' then blends the result back through your mask. If the model "
                      "ignores the marking, try another color or style. Use Focus mode for small "
                      "objects.")
        note.setObjectName("muted"); note.setWordWrap(True)
        of.addRow(note)
        lay.addWidget(og)
        lay.addStretch(1)
        return w

    # -- Models tab
    def _tab_models(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        g = QGroupBox("Runtime")
        gl = QVBoxLayout(g)
        self.m_cli = PathRow("Executable (*)", "path to sd-cli(.exe)")
        gl.addWidget(self.m_cli)
        b = QHBoxLayout()
        get = QPushButton("Get sd-cli..."); get.clicked.connect(self.get_sd_cli)
        test = QPushButton("Test"); test.clicked.connect(self.test_sd_cli)
        b.addWidget(get); b.addWidget(test); b.addStretch(1)
        gl.addLayout(b)
        self.cli_info = QLabel(""); self.cli_info.setObjectName("muted"); self.cli_info.setWordWrap(True)
        gl.addWidget(self.cli_info)
        lay.addWidget(g)

        mg = QGroupBox("Model files  (any quant, any combination)")
        ml = QVBoxLayout(mg)
        flt = "Models (*.gguf *.safetensors);;All files (*)"
        self.m_diff = PathRow(flt, "Qwen-Image-2.1 .gguf / .safetensors")
        self.m_vae = PathRow(flt, "qwen_image_2.1_vae_bf16.safetensors")
        self.m_llm = PathRow(flt, "Qwen3-VL-8B-Instruct (GGUF or safetensors)")
        self.m_mm = PathRow(flt, "mmproj-Qwen3VL-8B-Instruct-F16.gguf (needed to EDIT with a GGUF encoder)")
        self.d_info = QLabel(""); self.d_info.setObjectName("info"); self.d_info.setWordWrap(True)
        for title, row in (("Diffusion model", self.m_diff), ("VAE", self.m_vae),
                           ("Text encoder (LLM)", self.m_llm), ("Vision projector", self.m_mm)):
            lab = QLabel(title); lab.setObjectName("muted")
            ml.addWidget(lab)
            ml.addWidget(row)
            if row is self.m_diff:
                ml.addWidget(self.d_info)
        chk_row = QHBoxLayout()
        self.m_check = QPushButton("Check files")
        self.m_check.setToolTip("Verify that every model file is complete, readable and in the right field")
        self.m_check.clicked.connect(self.check_files)
        chk_row.addWidget(self.m_check)
        chk_row.addStretch(1)
        ml.addLayout(chk_row)
        lay.addWidget(mg)

        ag = QGroupBox("Acquire")
        al = QVBoxLayout(ag)
        b1 = QPushButton("Download companions (VAE + Qwen3-VL encoder + mmproj)")
        b1.clicked.connect(self.download_companions)
        b2 = QPushButton("Download from Hugging Face...")
        b2.clicked.connect(self.open_hf)
        b3 = QPushButton("Auto-detect models in a folder...")
        b3.clicked.connect(self.autodetect_folder)
        for x in (b1, b2, b3):
            al.addWidget(x)
        self.dl_bar = QProgressBar(); self.dl_bar.setRange(0, 1)
        self.dl_msg = QLabel(""); self.dl_msg.setObjectName("muted"); self.dl_msg.setWordWrap(True)
        al.addWidget(self.dl_bar)
        al.addWidget(self.dl_msg)
        lay.addWidget(ag)
        lay.addStretch(1)

        for row, key in ((self.m_cli, "sd_cli"), (self.m_diff, "diffusion"), (self.m_vae, "vae"),
                         (self.m_llm, "llm"), (self.m_mm, "llm_vision")):
            row.btn.clicked.connect(lambda _=False, r=row, k=key: self._browse_model(r, k))
            row.changed.connect(lambda _t, k=key: self._model_field_changed(k))
        return w

    # -- Settings tab
    def _tab_settings(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        g = QGroupBox("Memory / speed")
        gl = QVBoxLayout(g)
        self.o_offload = QCheckBox("Offload weights to CPU RAM (recommended for <= 12 GB VRAM)")
        self.o_fa = QCheckBox("Flash attention for the diffusion model")
        self.o_vt = QCheckBox("VAE tiling (less VRAM when decoding large images)")
        self.o_vc = QCheckBox("Run VAE on CPU")
        self.o_cc = QCheckBox("Run text encoder on CPU")
        self.o_pc = QCheckBox("Prefix cache (faster, uses ~4 GB extra per condition)")
        for x in (self.o_offload, self.o_fa, self.o_vt, self.o_vc, self.o_cc, self.o_pc):
            gl.addWidget(x)
        f = QFormLayout()
        self.o_threads = QSpinBox(); self.o_threads.setRange(0, 128)
        self.o_threads.setSpecialValueText("auto")
        self.o_maxside = QSpinBox(); self.o_maxside.setRange(256, 4096); self.o_maxside.setSingleStep(64)
        f.addRow("CPU threads", self.o_threads)
        f.addRow("Max edit size (longest side, px)", self.o_maxside)
        gl.addLayout(f)
        lay.addWidget(g)

        g2 = QGroupBox("Advanced")
        l2 = QVBoxLayout(g2)
        self.o_verbose = QCheckBox("Verbose runtime log (-v)")
        self.o_extra = QLineEdit()
        self.o_extra.setPlaceholderText("extra sd-cli arguments, e.g.  --flow-shift 3")
        l2.addWidget(self.o_verbose)
        l2.addWidget(QLabel("Extra arguments (appended verbatim):"))
        l2.addWidget(self.o_extra)
        lay.addWidget(g2)

        g3 = QGroupBox("Folders")
        l3 = QVBoxLayout(g3)
        self.o_out = PathRow("", "output folder")
        self.o_out.btn.clicked.connect(self._browse_outdir)
        self.o_models = PathRow("", "default model download folder")
        self.o_models.btn.clicked.connect(self._browse_modelsdir)
        l3.addWidget(QLabel("Output folder:")); l3.addWidget(self.o_out)
        l3.addWidget(QLabel("Model download folder:")); l3.addWidget(self.o_models)
        lay.addWidget(g3)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------ settings
    def _load_settings_into_ui(self) -> None:
        s = self.s
        cli = s["sd_cli"] if s["sd_cli"] and Path(s["sd_cli"]).is_file() else find_sd_cli()
        self.m_cli.edit.setText(cli)
        self.m_diff.edit.setText(s["diffusion"]); self.m_vae.edit.setText(s["vae"])
        self.m_llm.edit.setText(s["llm"]); self.m_mm.edit.setText(s["llm_vision"])
        self.steps.setValue(int(s["steps"])); self.cfg.setValue(float(s["cfg"]))
        self.seed.setValue(int(s["seed"]))
        self.sampler.setCurrentText(s["sampler"]); self.scheduler.setCurrentText(s["scheduler"])
        self.batch.setValue(int(s["batch"])); self.transparent.setChecked(bool(s["transparent"]))
        self.c_preset.setCurrentText(s["size_preset"])
        self.c_w.setValue(int(s["width"])); self.c_h.setValue(int(s["height"]))
        self._preset_changed(self.c_preset.currentText())
        self.e_mode.setCurrentText(s["edit_size_mode"])
        self.e_w.setValue(int(s["width"])); self.e_h.setValue(int(s["height"]))
        self._edit_mode_changed(self.e_mode.currentText())
        self.o_offload.setChecked(bool(s["offload"])); self.o_fa.setChecked(bool(s["flash_attn"]))
        self.o_vt.setChecked(bool(s["vae_tiling"])); self.o_vc.setChecked(bool(s["vae_cpu"]))
        self.o_cc.setChecked(bool(s["clip_cpu"])); self.o_pc.setChecked(bool(s["prefix_cache"]))
        self.o_threads.setValue(int(s["threads"])); self.o_maxside.setValue(int(s["edit_max_side"]))
        self.o_verbose.setChecked(bool(s["verbose"])); self.o_extra.setText(s["extra_args"])
        self.o_out.edit.setText(s["out_dir"]); self.o_models.edit.setText(s["models_dir"])
        self.k_color.setCurrentText(s["mark_color"]); self.k_style.setCurrentText(s["mark_style"])
        self.k_wrap.setChecked(bool(s["mark_wrap"])); self.k_crop.setChecked(bool(s["mark_crop"]))
        self.k_pad.setValue(int(s["mark_pad"])); self.k_strict.setChecked(bool(s["mark_strict"]))
        self.k_feather.setValue(int(s["mark_feather"])); self.k_clean.setChecked(bool(s["mark_clean"]))
        self._crop_toggled(self.k_crop.isChecked())
        self.k_tol.setValue(int(s["sel_tol"])); self.k_reach.setValue(int(s["sel_reach"]))
        self.k_contig.setChecked(bool(s["sel_contig"]))
        self.k_prec.setCurrentText(s["sel_prec"])
        self.k_selmode.setCurrentIndex(1 if s["sel_mode"] == "Remove" else 0)
        self._sel_opts_changed()
        self._family = model_family(s["diffusion"])     # remember it; don't override saved values
        self._values_changed()
        self._refresh_model_info()

    def _save_settings_from_ui(self) -> None:
        s = self.s
        s["sd_cli"], s["diffusion"] = self.m_cli.text(), self.m_diff.text()
        s["vae"], s["llm"], s["llm_vision"] = self.m_vae.text(), self.m_llm.text(), self.m_mm.text()
        s["steps"], s["cfg"], s["seed"] = self.steps.value(), self.cfg.value(), self.seed.value()
        s["sampler"], s["scheduler"] = self.sampler.currentText(), self.scheduler.currentText()
        s["batch"], s["transparent"] = self.batch.value(), self.transparent.isChecked()
        s["size_preset"], s["width"], s["height"] = (self.c_preset.currentText(),
                                                      self.c_w.value(), self.c_h.value())
        s["edit_size_mode"] = self.e_mode.currentText()
        s["offload"], s["flash_attn"] = self.o_offload.isChecked(), self.o_fa.isChecked()
        s["vae_tiling"], s["vae_cpu"] = self.o_vt.isChecked(), self.o_vc.isChecked()
        s["clip_cpu"], s["prefix_cache"] = self.o_cc.isChecked(), self.o_pc.isChecked()
        s["threads"], s["edit_max_side"] = self.o_threads.value(), self.o_maxside.value()
        s["verbose"], s["extra_args"] = self.o_verbose.isChecked(), self.o_extra.text()
        s["mark_color"], s["mark_style"] = self.k_color.currentText(), self.k_style.currentText()
        s["mark_wrap"], s["mark_crop"] = self.k_wrap.isChecked(), self.k_crop.isChecked()
        s["mark_pad"], s["mark_strict"] = self.k_pad.value(), self.k_strict.isChecked()
        s["mark_feather"], s["mark_clean"] = self.k_feather.value(), self.k_clean.isChecked()
        s["sel_tol"], s["sel_reach"] = self.k_tol.value(), self.k_reach.value()
        s["sel_contig"], s["sel_prec"] = self.k_contig.isChecked(), self.k_prec.currentText()
        s["sel_mode"] = "Remove" if self.k_selmode.currentIndex() == 1 else "Add"
        s["out_dir"], s["models_dir"] = self.o_out.text() or str(DEFAULT_OUT_DIR), \
            self.o_models.text() or str(MODELS_DIR)
        s.save()

    def engine(self) -> EngineConfig:
        return EngineConfig(
            sd_cli=self.m_cli.text(), diffusion=self.m_diff.text(), vae=self.m_vae.text(),
            llm=self.m_llm.text(), llm_vision=self.m_mm.text(),
            offload=self.o_offload.isChecked(), flash_attn=self.o_fa.isChecked(),
            vae_tiling=self.o_vt.isChecked(), vae_cpu=self.o_vc.isChecked(),
            clip_cpu=self.o_cc.isChecked(), prefix_cache=self.o_pc.isChecked(),
            verbose=self.o_verbose.isChecked(), threads=self.o_threads.value(),
            extra_args=self.o_extra.text())

    def out_dir(self) -> Path:
        d = Path(self.o_out.text() or DEFAULT_OUT_DIR)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def models_dir(self) -> Path:
        d = Path(self.o_models.text() or MODELS_DIR)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------------------------------------------------------- small slots
    def _tab_changed(self, idx: int) -> None:
        if idx == 0:
            self._last_mode = "create"
        elif idx == 1:
            self._last_mode = "edit"
        elif idx == 2:
            self._last_mode = "mark"
            self._show_canvas = True
        self.run_btn.setText({"create": "Generate   (Ctrl+Enter)",
                              "edit": "Apply edit   (Ctrl+Enter)",
                              "mark": "Apply to marked area   (Ctrl+Enter)"}[self._last_mode])
        self._sync_center()

    # -- mark tab helpers
    def _sync_center(self) -> None:
        mark = self.tabs.currentIndex() == 2
        show_canvas = mark and self._show_canvas
        self.stack.setCurrentIndex(1 if show_canvas else 0)
        self.view_toggle.setVisible(mark)
        self.view_toggle.setText("Show result" if show_canvas else "Back to mask editor")
        self.cmp_btn.setVisible(not show_canvas)

    def _toggle_center(self) -> None:
        self._show_canvas = not self._show_canvas
        self._sync_center()

    def _fit_active(self) -> None:
        (self.canvas.fit if self.stack.currentIndex() == 1 else self.viewer.fit)()

    def _actual_active(self) -> None:
        (self.canvas.actual_size if self.stack.currentIndex() == 1 else self.viewer.actual_size)()

    def _tool_toggled(self, key: str, on: bool) -> None:
        if on:
            self.canvas.set_tool(key)
            self._update_tool_panel()

    def _update_tool_panel(self) -> None:
        if not hasattr(self, "k_sel_hint"):      # called while the panel is still being built
            return
        t = self.canvas.tool
        self.k_selopts.setVisible(t in MaskCanvas.SELECT_TOOLS)
        self.k_tol_row.setVisible(t == "wand")
        self.k_contig.setVisible(t == "wand")
        self.k_reach_row.setVisible(t in ("smart", "box"))
        self.k_sel_hint.setText(SEL_HINTS.get(t, ""))

    def _sel_opts_changed(self, *_args) -> None:
        self.k_tol_val.setText(str(self.k_tol.value()))
        self.k_reach_val.setText(str(self.k_reach.value()))
        self.canvas.opt.update(tol=self.k_tol.value(), reach=self.k_reach.value(),
                               contiguous=self.k_contig.isChecked(),
                               mode="add" if self.k_selmode.currentIndex() == 0 else "sub",
                               work=SEL_PRECISION[self.k_prec.currentText()])
        self.canvas.reselect()

    def _pending_changed(self, has: bool) -> None:
        self.k_accept.setEnabled(has)
        self.k_discard.setEnabled(has)

    def _overlay_toggled(self, on: bool) -> None:
        self.canvas.show_overlay = on
        self.canvas.update()

    def _brush_changed(self, v: int) -> None:
        self.k_size.blockSignals(True)
        self.k_size.setMaximum(self.canvas.max_brush)
        self.k_size.setValue(v)
        self.k_size.blockSignals(False)
        self.k_size_val.setText(f"{v} px")

    def _canvas_changed(self) -> None:
        self.k_undo.setEnabled(self.canvas.can_undo())
        self.k_redo.setEnabled(self.canvas.can_redo())
        self.k_clear.setEnabled(self.canvas.has_image() and bool(self.canvas.ops))
        self.k_invert.setEnabled(self.canvas.has_image())
        self.k_grow.setEnabled(self.canvas.has_image())
        self.k_shrink.setEnabled(self.canvas.has_image())

    def _crop_toggled(self, on: bool) -> None:
        self.k_pad.setEnabled(on)
        if on:
            self.k_strict.setChecked(True)      # a pasted-back crop must be composited
        self.k_strict.setEnabled(not on)

    def _mark_tpl_chosen(self, idx: int) -> None:
        text = MARK_TEMPLATES[idx][1]
        if text:
            if self.k_prompt.toPlainText().strip():
                self.k_prompt.insertPlainText("\n" + text)
            else:
                self.k_prompt.setPlainText(text)
        self.k_tpl.blockSignals(True)
        self.k_tpl.setCurrentIndex(0)
        self.k_tpl.blockSignals(False)

    def mark_open(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "Select image", self.s["last_image_browse"] or str(Path.home() / "Pictures"),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if f:
            self.s["last_image_browse"] = str(Path(f).parent)
            self.load_mark_image(f)

    def mark_paste(self) -> None:
        img = QApplication.clipboard().image()
        if img.isNull():
            QMessageBox.information(self, APP_NAME, "The clipboard does not contain an image.")
            return
        self._set_mark_image(img, "pasted image")

    def mark_use_result(self) -> None:
        if self._result_path and Path(self._result_path).is_file():
            self.load_mark_image(self._result_path)
        else:
            QMessageBox.information(self, APP_NAME, "There is no result to use yet.")

    def load_mark_image(self, path: str) -> None:
        r = QImageReader(path)
        r.setAutoTransform(True)
        img = r.read()
        if img.isNull():
            QMessageBox.warning(self, APP_NAME, f"Cannot read this image:\n{path}")
            return
        self._set_mark_image(img, Path(path).name)

    def _set_mark_image(self, img: QImage, label: str) -> None:
        ow, oh = img.width(), img.height()
        self.canvas.set_image(img)
        nw, nh = self.canvas.image.width(), self.canvas.image.height()
        self.k_info.setText(f"{label}  -  {nw} x {nh}" + (f"  (scaled down from {ow} x {oh})"
                                                          if (nw, nh) != (ow, oh) else ""))
        self.tabs.setCurrentIndex(2)
        self._show_canvas = True
        self._sync_center()

    def _undo_shortcut(self) -> None:
        if self.tabs.currentIndex() == 2 and self.stack.currentIndex() == 1:
            self.canvas.undo()

    def _redo_shortcut(self) -> None:
        if self.tabs.currentIndex() == 2 and self.stack.currentIndex() == 1:
            self.canvas.redo()

    def _mark_errors(self) -> list[str]:
        if not self.canvas.has_image():
            return ["Open an image on the Mark tab first."]
        if self.canvas.is_busy():
            return ["A selection is still being calculated - wait a moment."]
        self.canvas.commit_pending()                    # a previewed selection counts as marked
        if mask_bbox(self.canvas.mask) is None:
            return ["Paint over the area you want to change (Mark tab) before applying."]
        return []

    def _prepare_mark(self, job: JobSpec, dest: Path) -> tuple[list, dict]:
        """Render the marked picture(s) for the model and remember how to blend the result back."""
        img, mask = self.canvas.snapshot()
        bbox = mask_bbox(mask)
        if bbox is None:
            raise ValueError("Paint over the area you want to change first.")
        W, H = img.width(), img.height()
        crop = self.k_crop.isChecked()
        cap = self.o_maxside.value()
        if crop:
            pad = self.k_pad.value()
            rect = bbox.adjusted(-pad, -pad, pad, pad).intersected(QRect(0, 0, W, H))
        else:
            rect = QRect(0, 0, W, H)
        long_side = max(rect.width(), rect.height())
        target = min(max(long_side, 768), cap) if crop else min(long_side, cap)
        sc = target / long_side
        w, h = round32(rect.width() * sc), round32(rect.height() * sc)
        base = img.copy(rect).scaled(w, h, _IGNORE, _SMOOTH)
        m_reg = mask.copy(rect).scaled(w, h, _IGNORE, _SMOOTH)
        color_name, style = self.k_color.currentText(), self.k_style.currentText()
        marked = render_marker(base, m_reg, QColor(*MARK_COLORS[color_name]), style)
        clean = self.k_clean.isChecked()
        paths: list[str] = []
        for i, pic in enumerate(([base] if clean else []) + [marked], 1):
            q = dest / f"ref_{i}.png"
            if not flatten_on_white(pic).save(str(q), "PNG"):
                raise IOError(f"cannot write {q}")
            paths.append(str(q))
        job.width, job.height = w, h
        job.user_prompt = job.prompt
        if self.k_wrap.isChecked():
            job.prompt = mark_prompt(job.prompt, color_name, style, clean)
        ctx = {"img": img, "mask": mask, "rect": rect,
               "strict": True if crop else self.k_strict.isChecked(),
               "feather": self.k_feather.value(),
               "info": {"color": color_name, "style": style, "focus": crop,
                        "padding": self.k_pad.value(), "feather": self.k_feather.value(),
                        "keep_outside": True if crop else self.k_strict.isChecked(),
                        "clean_reference": clean,
                        "region": [rect.x(), rect.y(), rect.width(), rect.height()]}}
        return paths, ctx

    def _preset_changed(self, name: str) -> None:
        w, h = SIZE_PRESETS.get(name, (0, 0))
        custom = name == "Custom"
        if not custom and w:
            self.c_w.setValue(w); self.c_h.setValue(h)
        self.c_w.setEnabled(custom); self.c_h.setEnabled(custom)

    def _edit_mode_changed(self, name: str) -> None:
        custom = name == "Custom"
        self.e_w.setEnabled(custom); self.e_h.setEnabled(custom)
        self._ref_selected()

    def _tpl_chosen(self, idx: int) -> None:
        text = EDIT_TEMPLATES[idx][1]
        if text:
            if self.e_prompt.toPlainText().strip():
                self.e_prompt.insertPlainText("\n" + text)
            else:
                self.e_prompt.setPlainText(text)
        self.e_tpl.blockSignals(True)
        self.e_tpl.setCurrentIndex(0)
        self.e_tpl.blockSignals(False)

    def append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def refresh_status(self) -> None:
        cli = self.m_cli.text()
        ok = bool(cli and Path(cli).is_file())
        self.sbar.showMessage(f"sd-cli: {cli if ok else 'NOT FOUND'}   |   output: {self.out_dir()}")

    # ------------------------------------------------------- model fields
    def _browse_model(self, row: PathRow, key: str) -> None:
        start = row.text() or self.s["last_models_browse"] or str(self.models_dir())
        path, _ = QFileDialog.getOpenFileName(self, "Select file", start, row.filt or "All files (*)")
        if path:
            self.s["last_models_browse"] = str(Path(path).parent)
            row.set_text(path)

    def _model_field_changed(self, key: str) -> None:
        if key == "diffusion":
            self._auto_preset()
        self._refresh_model_info()
        self.refresh_status()

    # -- sampling presets (Standard / Turbo)
    def _apply_preset(self, name: str) -> None:
        if name not in SAMPLING_PRESETS:
            return
        steps, cfg = SAMPLING_PRESETS[name]
        self._preset_busy = True
        self.steps.setValue(steps)
        self.cfg.setValue(cfg)
        self.preset.setCurrentText(name)
        self._preset_busy = False
        self._update_cfg_note()

    def _preset_chosen(self, name: str) -> None:
        if not self._preset_busy and name != "Custom":
            self._apply_preset(name)

    def _values_changed(self, *_args) -> None:
        if self._preset_busy:
            return
        cur = (self.steps.value(), round(self.cfg.value(), 2))
        name = next((k for k, v in SAMPLING_PRESETS.items() if v == cur), "Custom")
        self._preset_busy = True
        self.preset.setCurrentText(name)
        self._preset_busy = False
        self._update_cfg_note()

    def _update_cfg_note(self) -> None:
        low = self.cfg.value() <= 1.0
        self.cfg_note.setText("CFG 1: the negative prompt is ignored (normal for Turbo models)."
                              if low else "")
        self.cfg_note.setVisible(low)

    def _auto_preset(self) -> None:
        """Switch to the Turbo / Standard preset when the loaded diffusion file changes family."""
        p = self.m_diff.text()
        fam = model_family(p)
        if fam == self._family:
            return
        self._family = fam
        if not fam:
            return
        name = "Turbo" if fam == "turbo" else "Standard"
        self._apply_preset(name)
        steps, cfg = SAMPLING_PRESETS[name]
        self.append_log(f"[preset] {Path(p).name}: using the {name} preset "
                        f"({steps} steps, CFG {cfg:g}). Change it any time in the Sampling box.")

    def _refresh_model_info(self) -> None:
        p = self.m_diff.text()
        msgs = []
        if p and Path(p).is_file():
            if p.lower().endswith(".gguf"):
                inf = read_gguf_info(p, tensors=True)
                if inf.ok:
                    desc = (f"{Path(p).name}: {inf.quant or '?'} ({inf.quant_summary}), "
                            f"{human_size(inf.size_bytes)}, {inf.tensor_count} tensors")
                    if inf.blocks:
                        desc += f", {inf.blocks} blocks"
                    msgs.append(desc)
                    if inf.blocks >= 50:
                        msgs.append("Note: this looks like the older 60-block Qwen-Image / Edit (20B) "
                                    "family. It needs ITS OWN VAE and Qwen2.5-VL encoder, not the 2.1 "
                                    "companions.")
                else:
                    msgs.append(f"Could not read GGUF header: {inf.error}")
            else:
                msgs.append(f"{Path(p).name}: {human_size(Path(p).stat().st_size)} (safetensors)")
            if "ultrafast" in Path(p).name.lower():
                msgs.append("Distilled UltraFast build: try ~8 steps.")
            if model_family(p) == "turbo":
                msgs.append("Turbo (distilled) build: 8 steps and CFG 1 are used (Turbo preset).")
        self.d_info.setText("\n".join(msgs))
        l = self.m_llm.text()
        self.cli_info.setText("")
        if l and l.lower().endswith(".gguf") and not self.m_mm.text():
            self.cli_info.setText("Tip: set the vision projector (mmproj) to use the EDIT tab.")

    def check_files(self) -> None:
        report: list[str] = []
        problems = check_model_files(self.engine(), report)
        if not report:
            QMessageBox.information(self, APP_NAME, "No model files are set yet.")
            return
        for line in report:
            self.append_log("[check] " + line)
        head = ("Problems found - re-download or move the file(s) marked PROBLEM:\n\n" if problems
                else "All model files look complete and readable.\n\n")
        QMessageBox.information(self, APP_NAME, head + "\n".join(report))

    def autodetect_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder containing your models",
                                             self.s["last_models_browse"] or str(self.models_dir()))
        if not d:
            return
        self.s["last_models_browse"] = d
        found: dict[str, list[Path]] = {}
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() not in MODEL_EXTS or not p.is_file():
                continue
            role = classify_model_file(p)
            if role == "unknown" and p.suffix.lower() == ".gguf":
                role = classify_model_file(p, read_gguf_info(p, tensors=False))
            found.setdefault(role, []).append(p)

        def best(lst: list[Path], prefs=("q4_k_m", "q4_k", "q5", "q6", "q8")):
            for pref in prefs:
                for p in lst:
                    if pref in p.name.lower():
                        return p
            return max(lst, key=lambda p: p.stat().st_mtime)

        for role, row in (("diffusion", self.m_diff), ("vae", self.m_vae), ("llm", self.m_llm),
                          ("mmproj", self.m_mm)):
            lst = found.get(role, [])
            if lst:
                if not row.text() or not Path(row.text()).is_file():
                    row.set_text(str(best(lst)))
                self.append_log(f"[detect] {role}: " + ", ".join(p.name for p in lst))
        self.append_log("[detect] Fields that already pointed to a valid file were left untouched.")

    def _apply_downloaded(self, items: list) -> None:
        rowmap = {"diffusion": self.m_diff, "vae": self.m_vae, "llm": self.m_llm, "mmproj": self.m_mm}
        for role, path in items:
            row = rowmap.get(role)
            if row and (not row.text() or not Path(row.text()).is_file() or role == "diffusion"):
                row.set_text(path)

    def download_companions(self) -> None:
        have = {"vae": self.m_vae.text(), "llm": self.m_llm.text(), "mmproj": self.m_mm.text()}
        md = self.models_dir()
        self.dl_msg.setText("Looking up files on Hugging Face ...")
        t = Task(lambda: resolve_companions(md, have), self)
        self._tasks.append(t)

        def ok(specs):
            if not specs:
                self.dl_msg.setText("Companions are already set.")
                return
            self._run_downloads(specs)

        t.result.connect(ok)
        t.failed.connect(lambda m: self.dl_msg.setText(f"Failed: {m}"))
        t.finished.connect(lambda: self._tasks.remove(t) if t in self._tasks else None)
        t.start()

    def _run_downloads(self, specs: list[DlSpec]) -> None:
        self._dl = DownloadThread(specs, self)
        self._dl.status.connect(self.dl_msg.setText)

        def prog(done, total):
            if total:
                self.dl_bar.setRange(0, 1000)
                self.dl_bar.setValue(int(1000 * done / total))
                self.dl_bar.setFormat(f"{human_size(done)} / {human_size(total)}")
            else:
                self.dl_bar.setRange(0, 0)

        self._dl.progress.connect(prog)
        self._dl.failed.connect(lambda m: (self.dl_msg.setText(m), self.dl_bar.setRange(0, 1)))

        def ok(done):
            self.dl_bar.setRange(0, 1); self.dl_bar.setValue(1)
            self.dl_msg.setText("Download complete.")
            self._apply_downloaded([(s.role, str(s.dest)) for s in done])

        self._dl.finished_ok.connect(ok)
        self._dl.start()

    def open_hf(self) -> None:
        dlg = HfDialog(self, self.models_dir())
        dlg.downloaded.connect(self._apply_downloaded)
        dlg.exec()

    def get_sd_cli(self) -> None:
        dlg = SdCliDialog(self)
        dlg.installed.connect(lambda exe: self.m_cli.set_text(exe))
        dlg.exec()

    def test_sd_cli(self) -> None:
        cli = self.m_cli.text()
        if not cli or not Path(cli).is_file():
            QMessageBox.information(self, APP_NAME, "Set the sd-cli path first.")
            return
        self.append_log(f"> {cli} --version")
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        def fin(*_):
            out = strip_ansi(bytes(proc.readAll()).decode("utf-8", "replace")).strip()
            head = "\n".join(out.splitlines()[:6]) or "(no output)"
            self.append_log(head)
            self.cli_info.setText(head.splitlines()[0] if head else "")

        proc.finished.connect(fin)
        proc.errorOccurred.connect(lambda e: self.append_log(f"Could not run sd-cli ({e})."))
        proc.start(cli, ["--version"])

    def _browse_outdir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.o_out.text())
        if d:
            self.o_out.set_text(d)
            self.load_history()

    def _browse_modelsdir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Model download folder", self.o_models.text())
        if d:
            self.o_models.set_text(d)

    # ------------------------------------------------------- reference imgs
    def browse_refs(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select image(s)", self.s["last_image_browse"] or str(Path.home() / "Pictures"),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if files:
            self.s["last_image_browse"] = str(Path(files[0]).parent)
            self.add_refs(files)

    def add_refs(self, files: list) -> None:
        for f in files:
            if f in self.refs.paths():
                continue
            icon = thumb_icon(f, 72)
            if icon.isNull():
                self.append_log(f"[warn] cannot read image: {f}")
                continue
            it = QListWidgetItem(icon, Path(f).name)
            it.setData(Qt.ItemDataRole.UserRole, f)
            it.setToolTip(f)
            self.refs.addItem(it)
        self.tabs.setCurrentIndex(1)
        if self.refs.count() and not self.refs.selectedItems():
            self.refs.setCurrentRow(0)
        self._ref_selected()

    def paste_ref(self) -> None:
        img = QApplication.clipboard().image()
        if img.isNull():
            md = QApplication.clipboard().mimeData()
            if md and md.hasUrls():
                self.add_refs([u.toLocalFile() for u in md.urls()
                               if Path(u.toLocalFile()).suffix.lower() in IMAGE_EXTS])
                return
            QMessageBox.information(self, APP_NAME, "The clipboard does not contain an image.")
            return
        CLIPBOARD_DIR.mkdir(parents=True, exist_ok=True)
        p = CLIPBOARD_DIR / f"paste_{dt.datetime.now():%Y%m%d_%H%M%S}.png"
        img.save(str(p), "PNG")
        self.add_refs([str(p)])

    def _paste_shortcut(self) -> None:
        w = QApplication.focusWidget()
        if isinstance(w, (QPlainTextEdit, QLineEdit)):
            return  # normal text paste
        if self.tabs.currentIndex() == 1:
            self.paste_ref()
        elif self.tabs.currentIndex() == 2:
            self.mark_paste()

    def remove_refs(self) -> None:
        for it in self.refs.selectedItems():
            self.refs.takeItem(self.refs.row(it))
        self._ref_selected()

    def _ref_selected(self) -> None:
        paths = self.refs.paths()
        if not paths:
            self.e_info.setText("")
            self._compare_pix = None
            return
        cur = self.refs.currentItem() or self.refs.item(0)
        p = cur.data(Qt.ItemDataRole.UserRole)
        pm = QPixmap(p)
        if not pm.isNull():
            self._compare_pix = QPixmap(paths[0])
            if not (self.runner.running()):
                self.viewer.set_image(pm)
        first = QImage(paths[0])
        if not first.isNull():
            w, h = self._edit_dims(first.width(), first.height())
            self.e_info.setText(f"Input {first.width()} x {first.height()}  ->  output {w} x {h}")

    def _edit_dims(self, iw: int, ih: int) -> tuple[int, int]:
        if self.e_mode.currentText() == "Custom":
            return round32(self.e_w.value()), round32(self.e_h.value())
        cap = self.o_maxside.value()
        sc = min(1.0, cap / max(iw, ih))
        return round32(iw * sc), round32(ih * sc)

    def use_result_as_input(self) -> None:
        if self._result_path and Path(self._result_path).is_file():
            if self.tabs.currentIndex() == 2:
                self.load_mark_image(self._result_path)
                return
            self.refs.clear()
            self.add_refs([self._result_path])
        else:
            QMessageBox.information(self, APP_NAME, "There is no result to use yet.")

    # ----------------------------------------------------------- job flow
    def collect_job(self) -> JobSpec:
        mode = self._last_mode
        j = JobSpec(mode=mode, steps=self.steps.value(), cfg=self.cfg.value(),
                    seed=self.seed.value(), sampler=self.sampler.currentText(),
                    scheduler=self.scheduler.currentText(), batch=self.batch.value(),
                    transparent=self.transparent.isChecked())
        if mode == "create":
            j.prompt, j.negative = self.c_prompt.toPlainText(), self.c_neg.toPlainText()
            j.width, j.height = round32(self.c_w.value()), round32(self.c_h.value())
        elif mode == "mark":
            j.prompt, j.negative = self.k_prompt.toPlainText(), self.k_neg.toPlainText()
        else:
            j.prompt, j.negative = self.e_prompt.toPlainText(), self.e_neg.toPlainText()
            j.refs = self.refs.paths()
        return j

    def start_job(self) -> None:
        if self.runner.running():
            return
        if self._last_mode not in ("create", "edit", "mark"):
            self._last_mode = "create"
        self._save_settings_from_ui()
        cfg, job = self.engine(), self.collect_job()
        errs = validate(cfg, job)
        if job.mode == "mark":
            errs += self._mark_errors()
        if errs:
            QMessageBox.warning(self, APP_NAME, "Cannot start yet:\n\n- " + "\n- ".join(errs))
            if any("sd-cli" in e or "model" in e.lower() or "VAE" in e or "encoder" in e for e in errs):
                self.tabs.setCurrentIndex(3)
            return
        if job.seed < 0:
            job.seed = random.randrange(0, 2 ** 31)

        out_dir = self.out_dir()
        work = out_dir / ".work" / dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (work / "refs").mkdir(parents=True, exist_ok=True)
        self._mark_ctx = None
        try:
            if job.mode == "edit":
                job.refs = self._prepare_refs(job.refs, work / "refs", job)
            elif job.mode == "mark":
                job.refs, self._mark_ctx = self._prepare_mark(job, work / "refs")
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(work, ignore_errors=True)
            QMessageBox.warning(self, APP_NAME, f"Could not prepare the input image(s):\n{e}")
            return

        self._job, self._work = job, work
        self._cmd = build_command(cfg, job, work / "result.png")
        self.append_log("")
        self.append_log(f"=== {job.mode.upper()}  seed {job.seed}  {job.width}x{job.height}  "
                        f"{job.steps} steps  cfg {job.cfg:g} ===")
        self.append_log("> " + cmdline_for_display(self._cmd))
        self._last_step, self._batch_idx = 0, 0
        self.progress.setRange(0, 0)
        self.status.setText("Starting ...")
        self._t0 = time.time()
        self.clock.start(1000)
        self._set_running(True)
        self.runner.start(self._cmd, str(work))

    def _prepare_refs(self, paths: list, dest: Path, job: JobSpec) -> list:
        """Copy refs as PNG (keeps alpha), cap the size, and derive the output size."""
        out = []
        cap = self.o_maxside.value()
        for i, p in enumerate(paths):
            img = QImage(p)
            if img.isNull():
                raise ValueError(f"cannot read {p}")
            sc = min(1.0, cap / max(img.width(), img.height()))
            w, h = round32(img.width() * sc), round32(img.height() * sc)
            if (w, h) != (img.width(), img.height()):
                img = img.scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
            if i == 0:
                if self.e_mode.currentText() == "Custom":
                    job.width, job.height = round32(self.e_w.value()), round32(self.e_h.value())
                else:
                    job.width, job.height = w, h
            q = dest / f"ref_{i + 1}.png"
            if not img.save(str(q), "PNG"):
                raise IOError(f"cannot write {q}")
            out.append(str(q))
        return out

    def stop_job(self) -> None:
        if self.runner.running():
            self.status.setText("Stopping ...")
            self.runner.cancel()

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.tabs.setEnabled(not running)
        self.canvas.setEnabled(not running)

    def _tick(self) -> None:
        if self.runner.running():
            base = self.status.text().split("   [")[0]
            self.status.setText(f"{base}   [{fmt_elapsed(time.time() - self._t0)}]")

    def on_stage(self, text: str) -> None:
        self.status.setText(text)

    def on_progress(self, cur: int, total: int) -> None:
        if not self._job or total <= 0:
            return
        if cur < self._last_step:
            self._batch_idx += 1
        self._last_step = cur
        batch = max(1, self._job.batch)
        self.progress.setRange(0, total * batch)
        self.progress.setValue(min(total * batch, self._batch_idx * total + cur))
        self.progress.setFormat(f"step {cur}/{total}" + (f"   image {self._batch_idx + 1}/{batch}"
                                                           if batch > 1 else ""))
        self.status.setText("Sampling...")

    def on_done(self, code: int, cancelled: bool, tail: str) -> None:
        self.clock.stop()
        self._set_running(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("")
        elapsed = time.time() - self._t0
        work, job = self._work, self._job
        files = []
        if work and work.exists():
            files = sorted(p for p in work.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if cancelled:
            self.status.setText("Cancelled")
            self._cleanup_work()
            return
        if code != 0 or not files:
            hint = diagnose(tail, code)
            self.status.setText(f"Failed (exit code {code})")
            self.append_log(f"[error] sd-cli exited with code {code}")
            msg = f"The run failed (exit code {code})."
            if not files and code == 0:
                msg = "sd-cli finished but produced no image."
            if hint:
                msg += "\n\n" + hint
            msg += "\n\nSee the log panel for the full runtime output."
            self._cleanup_work()
            QMessageBox.warning(self, APP_NAME, msg)
            return

        out_dir = self.out_dir()
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        saved = []
        ctx = self._mark_ctx if job.mode == "mark" else None
        for i, f in enumerate(files):
            name = f"{stamp}_{job.mode}_s{job.seed + i}" + (f"_{i + 1}" if len(files) > 1 else "")
            suffix = ".png" if ctx else f.suffix.lower()
            final = out_dir / f"{name}{suffix}"
            n = 1
            while final.exists():
                final = out_dir / f"{name}_{n}{suffix}"
                n += 1
            blended = False
            if ctx:
                try:
                    blended = composite_mark(str(f), ctx).save(str(final), "PNG")
                except Exception as ex:  # noqa: BLE001
                    self.append_log(f"[warn] could not blend result into the original: {ex}")
                if not blended:
                    final = final.with_suffix(f.suffix.lower())
                    shutil.move(str(f), str(final))
            else:
                shutil.move(str(f), str(final))
            meta = {"app": f"{APP_NAME} {APP_VERSION}", "mode": job.mode,
                    "prompt": job.user_prompt or job.prompt,
                    "sent_prompt": job.prompt if job.user_prompt else None,
                    "mark": ctx["info"] if ctx else None,
                    "negative": job.negative, "width": job.width, "height": job.height,
                    "steps": job.steps, "cfg": job.cfg, "seed": job.seed + i,
                    "sampler": job.sampler, "scheduler": job.scheduler,
                    "transparent": job.transparent,
                    "diffusion": Path(self.m_diff.text()).name,
                    "seconds": round(elapsed, 1), "command": cmdline_for_display(self._cmd)}
            final.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
            saved.append(final)
        self._cleanup_work()
        self.status.setText(f"Done in {fmt_elapsed(elapsed)}  -  {len(saved)} image(s)")
        self.append_log(f"[done] {', '.join(p.name for p in saved)}")
        for p in reversed(saved):
            self.add_history_item(p, front=True)
        self.hist.setCurrentRow(0)
        self.show_image(str(saved[0]))
        if ctx:                                   # let the user flip between before / after
            self._compare_pix = QPixmap.fromImage(ctx["img"])
            self._show_canvas = False
            self._sync_center()

    def _cleanup_work(self) -> None:
        if self._work:
            shutil.rmtree(self._work, ignore_errors=True)
            try:
                (self._work.parent).rmdir()      # remove .work when empty
            except OSError:
                pass
        self._work = None

    def copy_command(self) -> None:
        if self._cmd:
            QApplication.clipboard().setText(cmdline_for_display(self._cmd))
            self.status.setText("Command copied to the clipboard")
        else:
            self.status.setText("Nothing to copy yet - run a job first")

    # ----------------------------------------------------------- viewing
    def show_image(self, path: str) -> None:
        pm = QPixmap(path)
        if pm.isNull():
            return
        self._result_path = path
        self._result_pix = pm
        self.viewer.set_image(pm)

    def _compare(self, on: bool) -> None:
        if on and self._compare_pix and not self._compare_pix.isNull():
            self.viewer.set_image(self._compare_pix)
        elif self._result_pix:
            self.viewer.set_image(self._result_pix)

    def save_as(self) -> None:
        if not self._result_path:
            return
        src = Path(self._result_path)
        dst, _ = QFileDialog.getSaveFileName(self, "Save image", str(Path.home() / src.name),
                                             "PNG (*.png);;JPEG (*.jpg);;WebP (*.webp)")
        if dst:
            QImage(str(src)).save(dst)

    def open_out_dir(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.out_dir())))

    # ----------------------------------------------------------- history
    def load_history(self) -> None:
        self.hist.clear()
        d = self.out_dir()
        files = sorted((p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:300]
        for p in files:
            self.add_history_item(p)

    def _history_files(self) -> list[Path]:
        try:
            return [p for p in self.out_dir().iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        except OSError:
            return []

    def clear_history(self) -> None:
        if self.runner.running():
            QMessageBox.information(self, APP_NAME, "Wait for the current run to finish first.")
            return
        files = self._history_files()
        if not files:
            self.hist.clear()
            QMessageBox.information(self, APP_NAME, "The history is already empty.")
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(APP_NAME)
        box.setText(f"Delete all {len(files)} image(s) in the history?")
        box.setInformativeText(f"Folder: {self.out_dir()}\nTheir saved settings files are removed too. "
                               "Files go to the Recycle Bin when the system supports it.")
        yes = box.addButton("Delete all", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is not yes:
            return
        failed = 0
        for f in files:
            for target in (f, f.with_suffix(".json")):
                if not target.exists():
                    continue
                if QFile.moveToTrash(str(target)):
                    continue
                try:
                    target.unlink()
                except OSError:
                    if target is f:
                        failed += 1
        self.load_history()
        self.append_log(f"[history] deleted {len(files) - failed} image(s)")
        if failed:
            QMessageBox.warning(self, APP_NAME, f"{failed} file(s) could not be deleted "
                                                "(they may be open in another program).")

    def add_history_item(self, path: Path, front: bool = False) -> None:
        it = QListWidgetItem(thumb_icon(str(path), 120), path.stem[-18:])
        it.setData(Qt.ItemDataRole.UserRole, str(path))
        it.setToolTip(self._meta_tooltip(path))
        if front:
            self.hist.insertItem(0, it)
        else:
            self.hist.addItem(it)

    @staticmethod
    def _load_meta(path: Path) -> dict:
        try:
            return json.loads(path.with_suffix(".json").read_text("utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _meta_tooltip(self, path: Path) -> str:
        m = self._load_meta(path)
        if not m:
            return path.name
        return (f"{m.get('mode', '')}  seed {m.get('seed')}  {m.get('width')}x{m.get('height')}  "
                f"{m.get('steps')} steps\n{(m.get('prompt') or '')[:300]}")

    def history_clicked(self, item: QListWidgetItem) -> None:
        self.show_image(item.data(Qt.ItemDataRole.UserRole))

    def history_menu(self, pos) -> None:
        it = self.hist.itemAt(pos)
        if not it:
            return
        path = Path(it.data(Qt.ItemDataRole.UserRole))
        menu = QMenu(self)
        a_edit = menu.addAction("Use as edit input")
        a_mark = menu.addAction("Mark && edit this image")
        a_reuse = menu.addAction("Reuse settings && prompt")
        a_open = menu.addAction("Show in folder")
        a_copy = menu.addAction("Copy path")
        menu.addSeparator()
        a_del = menu.addAction("Delete...")
        act = menu.exec(self.hist.mapToGlobal(pos))
        if act == a_edit:
            self.refs.clear(); self.add_refs([str(path)])
        elif act == a_mark:
            self.load_mark_image(str(path))
        elif act == a_reuse:
            self.reuse_meta(path)
        elif act == a_open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
        elif act == a_copy:
            QApplication.clipboard().setText(str(path))
        elif act == a_del:
            if QMessageBox.question(self, APP_NAME, f"Permanently delete {path.name}?") == \
                    QMessageBox.StandardButton.Yes:
                for p in (path, path.with_suffix(".json")):
                    try:
                        p.unlink()
                    except OSError:
                        pass
                self.hist.takeItem(self.hist.row(it))

    def reuse_meta(self, path: Path) -> None:
        m = self._load_meta(path)
        if not m:
            QMessageBox.information(self, APP_NAME, "No settings were saved for this image.")
            return
        self.steps.setValue(int(m.get("steps", self.steps.value())))
        self.cfg.setValue(float(m.get("cfg", self.cfg.value())))
        self.seed.setValue(int(m.get("seed", -1)))
        self.sampler.setCurrentText(m.get("sampler", "euler"))
        self.scheduler.setCurrentText(m.get("scheduler", "default"))
        self.transparent.setChecked(bool(m.get("transparent", False)))
        if m.get("mode") == "mark":
            self.k_prompt.setPlainText(m.get("prompt", ""))
            self.k_neg.setPlainText(m.get("negative", ""))
            mk = m.get("mark") or {}
            self.k_color.setCurrentText(mk.get("color", self.k_color.currentText()))
            self.k_style.setCurrentText(mk.get("style", self.k_style.currentText()))
            self.k_crop.setChecked(bool(mk.get("focus", False)))
            self.k_feather.setValue(int(mk.get("feather", self.k_feather.value())))
            self.tabs.setCurrentIndex(2)
        elif m.get("mode") == "edit":
            self.e_prompt.setPlainText(m.get("prompt", ""))
            self.e_neg.setPlainText(m.get("negative", ""))
            self.tabs.setCurrentIndex(1)
        else:
            self.c_prompt.setPlainText(m.get("prompt", ""))
            self.c_neg.setPlainText(m.get("negative", ""))
            self.c_preset.setCurrentText("Custom")
            self.c_w.setValue(int(m.get("width", 1024)))
            self.c_h.setValue(int(m.get("height", 1024)))
            self.tabs.setCurrentIndex(0)

    # ------------------------------------------------------------ closing
    def closeEvent(self, e) -> None:  # noqa: N802
        if self.runner.running():
            if QMessageBox.question(self, APP_NAME, "A generation is still running. Stop it and quit?") \
                    != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.runner.cancel()
            self.runner.proc.waitForFinished(3000)
        self._save_settings_from_ui()
        self._cleanup_work()
        self.canvas.shutdown()
        e.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setApplicationName(APP_NAME)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
