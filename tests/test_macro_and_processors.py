# -*- coding: utf-8 -*-
"""
Tests for In-Memory Workflow Macro Recorder & GRDL Image Processing Components.

Verifies:
1. Execution of GRDL processors via grdl-runtime (DAGExecutor / WorkflowExecutor).
2. Parameter tunables (__param_specs__) for edge, spatial, speckle, fusion, and spectral transforms.
3. In-memory WorkflowMacroRecorder: step recording, execution, YAML/Python DSL serialization, and roundtrip.
4. Macro dialog and execution in GRDK viewer.
"""

from pathlib import Path
import numpy as np
import pytest
import yaml

from PyQt6.QtWidgets import QApplication

if QApplication.instance() is None:
    _app = QApplication(["--platform", "offscreen"])
else:
    _app = QApplication.instance()

# GRDL Processors
from grdl.image_processing.filters.edge import (
    HighPassFilter,
    LaplacianFilter,
    SobelFilter,
    UnsharpMaskFilter,
)
from grdl.image_processing.filters.linear import GaussianFilter, MeanFilter
from grdl.image_processing.filters.rank import MedianFilter
from grdl.image_processing.filters.sar_speckle import (
    BoxcarFilter,
    FrostFilter,
    KuanFilter,
)
from grdl.image_processing.filters.standard_lee import LeeFilter
from grdl.image_processing.fusion import (
    BroveyPanSharpenTransform,
    SimpleMeanFusionTransform,
)
from grdl.image_processing.spectral import (
    SpectralIndexTransform,
    compute_spectral_index,
)

# grdl-runtime & grdk macro
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition
from grdl_rt.execution.dag_executor import DAGExecutor
from grdk.viewers.macro_recorder import WorkflowMacroRecorder, WorkflowMacroDialog


class TestGRDLProcessorsAndTunables:
    """Validate GRDL transforms, execution, and __param_specs__ declarations."""

    def test_sobel_transform_and_tunables(self):
        f = SobelFilter(output_type="magnitude")
        assert len(SobelFilter.__param_specs__) >= 1
        arr = np.random.rand(15, 15).astype(np.float32)
        out = f.apply(arr)
        assert out.shape == arr.shape

    def test_unsharp_mask_and_tunables(self):
        f = UnsharpMaskFilter(radius=1.0, amount=1.5)
        assert len(UnsharpMaskFilter.__param_specs__) >= 2
        arr = np.random.rand(15, 15).astype(np.float32)
        out = f.apply(arr)
        assert out.shape == arr.shape

    def test_sar_speckle_filters(self):
        arr = np.random.rand(20, 20).astype(np.float32) + 0.1
        frost = FrostFilter(window_size=5, damping_factor=2.0)
        assert len(FrostFilter.__param_specs__) >= 2
        out_frost = frost.apply(arr)
        assert out_frost.shape == arr.shape

        kuan = KuanFilter(window_size=5, num_looks=1)
        assert len(KuanFilter.__param_specs__) >= 2
        out_kuan = kuan.apply(arr)
        assert out_kuan.shape == arr.shape

        box = BoxcarFilter(window_size=3)
        assert len(BoxcarFilter.__param_specs__) >= 1
        out_box = box.apply(arr)
        assert out_box.shape == arr.shape

    def test_fusion_transforms(self):
        pan = np.ones((10, 10), dtype=np.float32) * 2.0
        ms = np.ones((3, 10, 10), dtype=np.float32) * 1.0

        brovey = BroveyPanSharpenTransform(pan_weight=1.0)
        assert len(BroveyPanSharpenTransform.__param_specs__) >= 1
        fused_b = brovey.apply(ms, pan=pan)
        assert fused_b.shape == (3, 10, 10)

        mean_f = SimpleMeanFusionTransform(pan_weight=0.5)
        assert len(SimpleMeanFusionTransform.__param_specs__) >= 1
        fused_m = mean_f.apply(ms, pan=pan)
        assert fused_m.shape == (3, 10, 10)

    def test_spectral_index_transform(self):
        cube = np.random.rand(8, 20, 20).astype(np.float32) + 0.1
        spec = SpectralIndexTransform(index_type="NDVI", nir_idx=3, red_idx=2)
        assert len(SpectralIndexTransform.__param_specs__) >= 4
        ndvi = spec.apply(cube)
        assert ndvi.shape == (20, 20)
        assert np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0)


class TestWorkflowMacroRecorder:
    """Validate in-memory macro building, execution via grdl-runtime, and serialization."""

    def test_macro_step_recording(self):
        recorder = WorkflowMacroRecorder(name="Test Macro")
        assert recorder.is_empty()

        s1 = recorder.add_step(SobelFilter, params={"output_type": "magnitude"})
        assert recorder.step_count() == 1
        assert s1.id == "SobelFilter_1"
        assert s1.processor_name == "grdl.image_processing.filters.edge.SobelFilter"

        s2 = recorder.add_step(GaussianFilter, params={"sigma": 2.0})
        assert recorder.step_count() == 2
        assert s2.depends_on == ["SobelFilter_1"]

    def test_macro_execution_via_runtime(self):
        recorder = WorkflowMacroRecorder(name="Filter Chain")
        recorder.add_step(GaussianFilter, params={"sigma": 1.0})
        recorder.add_step(SobelFilter, params={"output_type": "magnitude"})

        arr = np.random.rand(30, 30).astype(np.float32)
        res = recorder.execute(arr)

        assert res.result is not None
        assert res.result.shape == (30, 30)
        assert len(res.lineage.transforms) == 2
        assert res.metrics.status == "success"

    def test_macro_serialization_and_roundtrip(self, tmp_path):
        recorder = WorkflowMacroRecorder(name="Serialization Test")
        recorder.add_step(BoxcarFilter, params={"window_size": 3})
        recorder.add_step(UnsharpMaskFilter, params={"amount": 1.2, "radius": 1.0})

        # YAML export
        yaml_str = recorder.to_yaml()
        assert "BoxcarFilter" in yaml_str
        assert "UnsharpMaskFilter" in yaml_str

        # Python DSL export
        dsl_str = recorder.to_python()
        assert "@workflow" in dsl_str
        assert "BoxcarFilter" in dsl_str

        # Save to file
        yaml_file = tmp_path / "macro.yaml"
        saved = recorder.save_to_yaml_file(yaml_file)
        assert saved.exists()

        # Load roundtrip
        new_recorder = WorkflowMacroRecorder(name="Imported")
        loaded_wf = new_recorder.load_from_yaml(yaml_file)
        assert len(loaded_wf.steps) == 2
        assert loaded_wf.steps[0].processor_name.endswith("BoxcarFilter")
        assert loaded_wf.steps[1].processor_name.endswith("UnsharpMaskFilter")

    def test_macro_dialog_creation(self):
        recorder = WorkflowMacroRecorder(name="Dialog Test")
        recorder.add_step(SobelFilter, params={"output_type": "magnitude"})

        test_arr = np.random.rand(20, 20).astype(np.float32)
        dlg = WorkflowMacroDialog(recorder, source_provider=lambda: test_arr)

        assert dlg._table.rowCount() == 1
        assert dlg._table.item(0, 0).text() == "SobelFilter_1"
