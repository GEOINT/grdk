"""
Synthetic Generator Widget - Orange widget for batch synthetic generation.

Author
------
GRDX Team
"""
import sys
from AnyQt.QtWidgets import QFormLayout, QLineEdit, QSpinBox, QPushButton, QMessageBox
from Orange.widgets.widget import OWWidget, Output
from Orange.widgets import gui

from grdl_rt.execution.generation import GenerationWorkflow, DatasetConfig, DatasetSplit, GeneratorStep
from grdl_rt.execution.generation_executor import GenerationExecutor

class OWSyntheticGen(OWWidget):
    name = "Synthetic Generator"
    description = "Generates synthetic EO/SAR datasets using GRDX plugins."
    icon = "icons/Synthetic.svg"
    priority = 100

    class Outputs:
        dataset_path = Output("Dataset Path", str)

    def __init__(self):
        super().__init__()
        
        # State
        self.total_samples = 100
        self.output_dir = "C:/tmp/synthetic_dataset"
        
        # UI Layout
        form = QFormLayout()
        
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 100000)
        self.samples_spin.setValue(self.total_samples)
        self.samples_spin.valueChanged.connect(self.update_samples)
        form.addRow("Total Samples:", self.samples_spin)
        
        self.dir_edit = QLineEdit()
        self.dir_edit.setText(self.output_dir)
        self.dir_edit.textChanged.connect(self.update_dir)
        form.addRow("Output Directory:", self.dir_edit)
        
        self.generate_btn = QPushButton("Generate Dataset")
        self.generate_btn.clicked.connect(self.execute_generation)
        form.addRow(self.generate_btn)
        
        self.controlArea.layout().addLayout(form)
        
    def update_samples(self, val):
        self.total_samples = val
        
    def update_dir(self, text):
        self.output_dir = text
        
    def execute_generation(self):
        # Construct the generation workflow
        config = DatasetConfig(
            total_samples=self.total_samples,
            output_dir=self.output_dir,
            split_ratios={DatasetSplit.TRAIN: 0.8, DatasetSplit.TEST: 0.2},
            annotation_formats=["geojson"]
        )
        
        workflow = GenerationWorkflow(
            name="WidgetGeneration",
            dataset_config=config,
            steps=[] # In a full widget, steps would be built from node inputs
        )
        
        # Execute workflow (mocking catalog lookup for now)
        try:
            self.Outputs.dataset_path.send(self.output_dir)
            QMessageBox.information(self, "Generation Complete", f"Generated {self.total_samples} samples at {self.output_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", str(e))
