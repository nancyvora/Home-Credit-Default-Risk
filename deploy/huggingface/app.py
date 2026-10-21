"""Hugging Face Space: the Gradio interface of the Home Credit default risk model.

The ``src`` folder beside this file is a copy of the project's serving code and saved model, made by
``scripts/build_hf_space.py``; the model's checksum is checked when it loads.

The Space runs on ZeroGPU hardware, which needs the functions behind the buttons marked with ``spaces.GPU``. The model
is LightGBM and scores on the CPU; each click only holds a GPU slot for a few seconds. ``spaces`` exists only on
Hugging Face, so elsewhere the interface runs without it.
"""
from src.app.ui import build_ui

try:
    import spaces
    wrap = spaces.GPU(duration=15)
except ImportError:
    wrap = None

demo = build_ui(wrap=wrap)

if __name__ == '__main__':
    demo.launch()
