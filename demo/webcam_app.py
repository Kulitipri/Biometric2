"""Gradio webcam demo: enroll team members, then identify in real time."""

from __future__ import annotations


def build_app():
    """Two tabs:

    1. Enroll  — upload 1 unmasked photo per identity, build gallery.
    2. Identify — webcam stream → detect → align → embed → top-1 in gallery.
    """
    # TODO: import gradio, wire to src.detector / src.embedder / src.matcher
    raise NotImplementedError


if __name__ == "__main__":
    app = build_app()
    app.launch()
