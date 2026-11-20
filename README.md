# Build Your Own Antivirus — MalConv PE Malware Classifier

> ⚠️ **Safety notice:** the dataset for this project contains **live Windows malware**.
> Only build it on Linux (Google Colab, the SE-122 lab, or a Linux VM). Never run, open,
> or double-click a sample. This repository contains code, a manifest of SHA-256 hashes,
> and model weights only — **no binaries**.

## Project layout

```
README.md            setup + how to run everything
requirements.txt     pinned dependencies
notebooks/           part1_data_exploration.ipynb, part2_model.ipynb
src/                 reusable Python code (data utils, model, predict)
dataset/             manifest.csv + build_dataset.py (no binaries)
model/               model.pt (trained weights)
app/                 app.py (Gradio/Streamlit)
```

## Setup

<!-- TODO: Python version, pip install -r requirements.txt -->

## Rebuild the dataset

<!-- TODO: python dataset/build_dataset.py  (Linux only) -->

## Run the app

<!-- TODO: python app/app.py -->

## Model

<!-- TODO: architecture summary, parameter count, max_len, decision threshold -->

## Test-set results

<!-- TODO -->

## AI-assistant use

- Claude (Anthropic) set up the project scaffolding: folder structure, .gitignore,
  this README outline, and the environment-setup cells in the Colab notebooks.
- ChatGPT was used to plan the VS Code + Colab workflow.
