"""
File:    app/app.py
Purpose: Part 3 of the "Build Your Own Antivirus" midterm project. A browser app where a
         user drags and drops a file and gets a verdict (Malware or Benign) from the
         trained MalConv model, together with the model's confidence.

HOW TO RUN (from the project root, the folder that contains app/ and model/)
    python app/app.py            -> serves the page locally; open the URL it prints
    python app/app.py --share    -> also prints a temporary public link (useful in Colab)

    Inside a Colab notebook you can instead run:
        from app.app import demo
        demo.launch(share=True)

WHAT IT NEEDS
    model/model.pt      the trained weights saved in Part 2 (a state_dict)
    model/config.json   {"max_len": ..., "threshold": ...}, saved in Part 2
    Python packages:    torch, numpy, gradio (see requirements.txt)

SAFETY
    An uploaded file is only READ as bytes with open(path, "rb"). It is never executed,
    never opened by another program, and never copied or saved anywhere by this code.
    Gradio keeps the upload in its own temporary folder.

The model class, the byte encoding and the PE checks below mirror part2_model.ipynb
(Sections 2, 3 and 7), so the app sees files exactly the way the model saw them in training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]          # the project root (contains model/)
MODEL_PATH = ROOT / "model" / "model.pt"
CONFIG_PATH = ROOT / "model" / "config.json"

DEFAULT_MAX_LEN = 2**20      # 1 MB, the input length used in training
DEFAULT_THRESHOLD = 0.5      # used only if config.json is missing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# The model: must match Section 3 of part2_model.ipynb exactly.
# The layer names (embedding, content, gate, output) must stay the same, because
# model.pt stores the weights under those names.
# ---------------------------------------------------------------------------

class MalConv(nn.Module):
    """MalConv baseline: byte embedding, gated 1-D convolution, global max pooling."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(257, 8, padding_idx=0)   # 256 byte values + padding
        self.content = nn.Conv1d(8, 128, kernel_size=500, stride=500)
        self.gate = nn.Conv1d(8, 128, kernel_size=500, stride=500)
        self.output = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # Pad to a whole number of 500-byte windows so the last partial window is used.
        x = nn.functional.pad(x, (0, (-x.size(1)) % 500))
        embedded = self.embedding(x).transpose(1, 2)        # (batch, 8, length)
        gated = self.content(embedded) * self.gate(embedded).sigmoid()
        pooled = gated.amax(dim=2)                           # strongest response per filter
        return self.output(pooled).squeeze(1)                # one logit per file


def tokens(raw: bytes, length: int) -> torch.Tensor:
    """Turn bytes into model input: byte b becomes token b + 1, padded with 0 up to length.

    Same encoding as tokens() in part2_model.ipynb: token 0 is reserved for padding,
    so a real zero byte (token 1) is never confused with "no byte here".
    """
    values = np.frombuffer(raw[:length], dtype=np.uint8)
    result = torch.zeros(length, dtype=torch.int32)
    result[: len(values)] = torch.from_numpy(values.astype(np.int32) + 1)
    return result


# ---------------------------------------------------------------------------
# Load the model ONCE, when the app starts (assignment section 5.1)
# ---------------------------------------------------------------------------

def load_model():
    """Return (model, max_len, threshold, problem). problem is None when all is well."""
    max_len, threshold, notes = DEFAULT_MAX_LEN, DEFAULT_THRESHOLD, []

    if CONFIG_PATH.is_file():
        config = json.loads(CONFIG_PATH.read_text())
        max_len = int(config.get("max_len", DEFAULT_MAX_LEN))
        threshold = float(config.get("threshold", DEFAULT_THRESHOLD))
    else:
        notes.append(f"{CONFIG_PATH.relative_to(ROOT)} not found, so the default "
                     f"threshold {DEFAULT_THRESHOLD} is used.")

    if not MODEL_PATH.is_file():
        return None, max_len, threshold, (
            f"Model weights not found at {MODEL_PATH.relative_to(ROOT)}. Train the model in "
            "Part 2, then put model.pt and config.json in the model/ folder."
        )

    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and "embedding.weight" not in state:
        state = state["model"]           # also accept a full training checkpoint
    model = MalConv()
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, max_len, threshold, (" ".join(notes) or None)


MODEL, MAX_LEN, THRESHOLD, STARTUP_PROBLEM = load_model()


# ---------------------------------------------------------------------------
# Checking and classifying one uploaded file
# ---------------------------------------------------------------------------

def pe_problem(stream, data: bytes) -> str | None:
    """Return a plain-language reason the file is not a Windows PE file, or None if it is.

    Same checks as predict() in part2_model.ipynb: the file must start with "MZ", and the
    position stored in bytes 60-63 must point at the "PE\\0\\0" signature.
    """
    if len(data) == 0:
        return "the file is empty."
    if len(data) < 64 or data[:2] != b"MZ":
        return "this is not a Windows executable (it does not start with the \"MZ\" signature)."

    pe_offset = int.from_bytes(data[60:64], "little")
    if pe_offset + 4 <= len(data):
        signature = data[pe_offset:pe_offset + 4]
    else:                                 # header pointer beyond what was read: look it up
        stream.seek(pe_offset)
        signature = stream.read(4)
    if signature != b"PE\x00\x00":
        return ("the file starts with \"MZ\" but has no PE header, so it is an old DOS "
                "program or a damaged file.")
    return None


def classify(path):
    """Gradio handler: returns (confidences for the Label, explanation for the Markdown)."""
    if path is None:
        return None, "Drop a Windows executable (.exe or .dll) above to scan it."
    if MODEL is None:
        return None, f"**The app is not ready:** {STARTUP_PROBLEM}"

    try:
        size = Path(path).stat().st_size
        with open(path, "rb") as stream:            # read bytes only, never execute
            data = stream.read(MAX_LEN)
            problem = pe_problem(stream, data)
    except OSError as error:
        return None, f"**Could not read the file:** {error}"

    if problem:
        return None, f"**Not scanned:** {problem}"

    with torch.inference_mode():
        x = tokens(data, MAX_LEN).unsqueeze(0).to(DEVICE)
        probability = MODEL(x).sigmoid().item()

    verdict = "Malware" if probability >= THRESHOLD else "Benign"
    details = (
        f"### Verdict: {verdict}\n\n"
        f"The model gives this file a malware probability of **{probability:.1%}**. "
        f"Files at or above **{THRESHOLD:.1%}** are classified as malware "
        "(this threshold was chosen on the validation set in Part 2)."
    )
    if size > MAX_LEN:
        details += (f"\n\nOnly the first {MAX_LEN:,} bytes of this {size:,}-byte file were "
                    "read, the same limit the model was trained with.")
    return {"Malware": probability, "Benign": 1 - probability}, details


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

if MODEL is None:
    status = f"**Model status:** not loaded. {STARTUP_PROBLEM}"
else:
    status = (f"**Model status:** loaded from model/model.pt on {DEVICE.type.upper()} · "
              f"input length {MAX_LEN:,} bytes · threshold {THRESHOLD:.3f}")
    if STARTUP_PROBLEM:
        status += f" · note: {STARTUP_PROBLEM}"

with gr.Blocks() as demo:
    gr.Markdown(
        "# MalConv antivirus\n"
        "Drop a Windows executable (.exe or .dll) below. A MalConv neural network reads "
        "its raw bytes and says whether it looks like **malware** or **benign**.\n\n"
        "The file is only read as bytes: it is never run, opened, or saved anywhere."
    )
    file_input = gr.File(label="File to scan", type="filepath")
    confidence = gr.Label(label="Model confidence", num_top_classes=2)
    explanation = gr.Markdown("Drop a Windows executable (.exe or .dll) above to scan it.")
    gr.Markdown(status)

    file_input.change(classify, inputs=file_input, outputs=[confidence, explanation])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MalConv antivirus web app.")
    parser.add_argument("--share", action="store_true",
                        help="also create a temporary public link (useful in Colab)")
    args = parser.parse_args()
    demo.launch(share=args.share)
