# -*- coding: utf-8 -*-
"""
GRDK - GEOINT Rapid Development Kit.

GUI tooling for CUDA-optimized, model-driven image processing workflow
orchestration. Built as a set of Orange Data Mining add-on plugins on
top of the GRDL library.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-06

Modified
--------
2026-02-06
"""

__version__ = "0.1.0"
__author__ = "Claude Code (Anthropic)"


def show(data, *, geolocation=None, title=None, block=True):
    """Display image data in the GRDK viewer.

    Re-exported from ``grdk.viewers.show``.
    See :func:`grdk.viewers.show` for full documentation.
    """
    from grdk.viewers import show as _show
    return _show(data, geolocation=geolocation, title=title, block=block)


def imshow(arr, *, geolocation=None, title=None, block=True):
    """Display a numpy array in the GRDK viewer.

    Re-exported from ``grdk.viewers.imshow``.
    See :func:`grdk.viewers.imshow` for full documentation.
    """
    from grdk.viewers import imshow as _imshow
    return _imshow(arr, geolocation=geolocation, title=title, block=block)


__all__: list = ["show", "imshow"]
