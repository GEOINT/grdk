# -*- coding: utf-8 -*-
"""
OWCoRegister Widget - Co-register an image stack.

Provides controls for selecting a reference image, choosing a
co-registration algorithm (from GRDL), tuning parameters, and
viewing quality metrics. Emits a co-registered ImageStack.

Dependencies
------------
orange-widget-base

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

# Standard library
from typing import Optional

# Third-party
import numpy as np
from orangewidget import gui
from orangewidget.settings import Setting
from orangewidget.widget import OWBaseWidget, Input, Output, Msg

from AnyQt.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QTextEdit,
)

# GRDK internal
from grdk.widgets._signals import ImageStack


class OWCoRegister(OWBaseWidget):
    """Co-register images in a stack to a common pixel space.

    Uses GRDL co-registration algorithms (feature-based, affine, etc.)
    to align all images in the stack to a selected reference image.
    """

    name = "Co-Register"
    description = "Co-register image stack to a common pixel space"
    icon = "icons/coregister.svg"
    category = "GEODEV"
    priority = 30

    class Inputs:
        image_stack = Input("Image Stack", ImageStack)

    class Outputs:
        image_stack = Output("Image Stack", ImageStack)

    class Warning(OWBaseWidget.Warning):
        no_input = Msg("No image stack received.")
        single_image = Msg("Stack has only one image — nothing to register.")

    class Error(OWBaseWidget.Error):
        registration_failed = Msg("Registration failed: {}")

    # Settings
    reference_index: int = Setting(0)
    algorithm: str = Setting("feature_match_orb")
    max_features: int = Setting(5000)

    want_main_area = True

    def __init__(self) -> None:
        super().__init__()

        self._input_stack: Optional[ImageStack] = None
        self._results: list = []

        # --- Control area ---
        box = gui.vBox(self.controlArea, "Co-Registration")

        # Reference image selector
        box.layout().addWidget(QLabel("Reference Image:"))
        self._ref_combo = QComboBox(self)
        box.layout().addWidget(self._ref_combo)

        # Algorithm selector
        box.layout().addWidget(QLabel("Algorithm:"))
        self._algo_combo = QComboBox(self)
        self._algo_combo.addItem("Affine (Control Points)", "affine")
        self._algo_combo.addItem("Projective (Homography)", "projective")
        self._algo_combo.addItem("Feature Match (ORB)", "feature_match_orb")
        self._algo_combo.addItem("Feature Match (SIFT)", "feature_match_sift")
        box.layout().addWidget(self._algo_combo)

        # Run button
        btn_run = QPushButton("Run Co-Registration", self)
        btn_run.clicked.connect(self._on_run)
        box.layout().addWidget(btn_run)

        # --- Main area: quality metrics ---
        self._metrics_text = QTextEdit(self.mainArea)
        self._metrics_text.setReadOnly(True)
        self.mainArea.layout().addWidget(self._metrics_text)

    @Inputs.image_stack
    def set_image_stack(self, stack: Optional[ImageStack]) -> None:
        """Receive an image stack."""
        self._input_stack = stack
        self._ref_combo.clear()
        self._results.clear()
        self._metrics_text.clear()

        if stack is None or len(stack) == 0:
            self.Warning.no_input()
            self.Outputs.image_stack.send(None)
            return

        self.Warning.no_input.clear()

        if len(stack) == 1:
            self.Warning.single_image()

        for i, name in enumerate(stack.names):
            self._ref_combo.addItem(f"[{i}] {name}", i)

    def _on_run(self) -> None:
        """Execute co-registration."""
        if self._input_stack is None or len(self._input_stack) < 2:
            return

        self.Error.registration_failed.clear()
        self.Warning.single_image.clear()

        ref_idx = self._ref_combo.currentData()
        if ref_idx is None:
            ref_idx = 0

        algo_key = self._algo_combo.currentData()

        try:
            if algo_key == 'affine':
                from grdl.coregistration import AffineCoRegistration
                coreg = AffineCoRegistration()
            elif algo_key == 'projective':
                from grdl.coregistration import ProjectiveCoRegistration
                coreg = ProjectiveCoRegistration()
            else:
                try:
                    from grdl.coregistration import FeatureMatchCoRegistration
                except ImportError:
                    self.Error.registration_failed(
                        "OpenCV is required for feature matching. "
                        "Install with: pip install opencv-python-headless"
                    )
                    return
                method = 'orb' if 'orb' in algo_key else 'sift'
                coreg = FeatureMatchCoRegistration(
                    method=method,
                    max_features=self.max_features,
                )

            ref_reader = self._input_stack.readers[ref_idx]
            fixed = ref_reader.read_full()
            if fixed.ndim == 3:
                fixed = fixed[:, :, 0]

            results = []
            metrics_lines = [f"Reference: [{ref_idx}] {self._input_stack.names[ref_idx]}\n"]

            for i, reader in enumerate(self._input_stack.readers):
                if i == ref_idx:
                    results.append(None)
                    continue

                moving = reader.read_full()
                if moving.ndim == 3:
                    moving = moving[:, :, 0]

                result = coreg.estimate(fixed, moving)
                results.append(result)

                metrics_lines.append(
                    f"[{i}] {self._input_stack.names[i]}:\n"
                    f"  RMS: {result.residual_rms:.4f} px\n"
                    f"  Matches: {result.num_matches}\n"
                    f"  Inliers: {result.inlier_ratio:.1%}\n"
                )

            self._results = results
            self._metrics_text.setText('\n'.join(metrics_lines))

            # Emit updated stack
            output = ImageStack(
                readers=list(self._input_stack.readers),
                names=list(self._input_stack.names),
                geolocation=self._input_stack.geolocation,
                registration_results=results,
                metadata={
                    **self._input_stack.metadata,
                    'reference_index': ref_idx,
                    'algorithm': algo_key,
                },
            )
            self.Outputs.image_stack.send(output)

        except Exception as e:
            self.Error.registration_failed(str(e))
