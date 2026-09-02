"""Visualization helpers for paper figures."""

from .palette import setup_style, PALETTE, save_fig
from .token_projection import render_token_projection_atlas

__all__ = [
    "setup_style",
    "PALETTE",
    "save_fig",
    "render_token_projection_atlas",
]
