import sys
import datetime
import numpy as np
import spectrauf as suf 
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QTreeWidget, 
                             QTreeWidgetItem, QLabel, QSplitter, QFileDialog, 
                             QMenu, QComboBox, QLineEdit, QDoubleSpinBox,
                             QPlainTextEdit, QCheckBox, QGroupBox, QGridLayout)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QAction, QValidator

import pyqtgraph as pg
import pyqtgraph.exporters
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

class ScientificDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimum(-1e30); self.setMaximum(1e30); self.setDecimals(10)

    def validate(self, text, pos):
        if 'e' in text.lower(): return QValidator.Acceptable, text, pos
        return super().validate(text, pos)

    def valueFromText(self, text):
        try: return float(text)
        except ValueError: return 0.0

    def textFromValue(self, value):
        if abs(value) >= 10000 or (abs(value) < 0.01 and value != 0):
            return "{:.4e}".format(value)
        return super().textFromValue(value)

class UnfoldingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraUF - Dual-View Diagnostic Suite")
        self.resize(1800, 1000)
        
        self.instances = {} 
        self.engine_map = {} 
        self.instance_metadata = {}
        self.paths = {k: None for k in ["measured", "measured_uncertainties", "a_priori", 
                                        "a_priori_uncertainties", "group_structure", "response_matrix", "target"]}

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        self.ribbon = QTabWidget()
        self.ribbon.setFixedHeight(280)
        self.setup_unfolding_tab()
        self.setup_plot_options_tab()
        self.main_layout.addWidget(self.ribbon)

        # Plotting Tabs
        self.plot_tabs = QTabWidget()
        self.plot_tabs.currentChanged.connect(self.sync_active_plot)
        
        # --- Engine 1: PyQtGraph Dual ---
        self.pqg_frame = QSplitter(Qt.Horizontal)
        self.pqg_flux = pg.PlotWidget(background='w', title="Flux (Lethargy)")
        self.pqg_measured = pg.PlotWidget(background='w', title="Measured vs Calculated")
        self.pqg_flux.addLegend(); self.pqg_measured.addLegend()
        self.pqg_frame.addWidget(self.pqg_flux); self.pqg_frame.addWidget(self.pqg_measured)
        self.plot_tabs.addTab(self.pqg_frame, "PyQtGraph")

        # --- Engine 2: Matplotlib Dual ---
        self.mpl_fig = plt.figure(figsize=(12, 5), tight_layout=True)
        self.mpl_ax_flux = self.mpl_fig.add_subplot(121)
        self.mpl_ax_meas = self.mpl_fig.add_subplot(122)
        self.mpl_canvas = FigureCanvas(self.mpl_fig)
        self.plot_tabs.addTab(self.mpl_canvas, "Matplotlib")

        # --- Engine 3: Plotly Dual ---
        self.plotly_view = QWebEngineView()
        self.plot_tabs.addTab(self.plotly_view, "Plotly")

        # Floating Buttons
        self.float_frame = QWidget(self.plot_tabs)
        float_lay = QHBoxLayout(self.float_frame)
        self.home_btn = QPushButton("🏠"); self.save_plot_btn = QPushButton("💾")
        for btn in [self.home_btn, self.save_plot_btn]:
            btn.setFixedSize(40, 40)
            btn.setStyleSheet("background: rgba(230, 230, 230, 200); border-radius: 5px; font-size: 18px;")
            float_lay.addWidget(btn)
        self.home_btn.clicked.connect(self.auto_range_plot)
        self.save_plot_btn.clicked.connect(self.save_current_plot)

        self.h_splitter = QSplitter(Qt.Horizontal)
        self.sidebar = QWidget()
        side_layout = QVBoxLayout(self.sidebar)
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["Spectra Components"])
        self.tree.itemClicked.connect(self.restore_instance_settings)
        self.root_node = QTreeWidgetItem(self.tree, ["All Instances"])
        self.root_node.setCheckState(0, Qt.Checked)
        self.tree.itemChanged.connect(self.handle_tree_checkbox)
        side_layout.addWidget(QLabel("<b>Analysis Tree</b>")); side_layout.addWidget(self.tree)
        
        self.h_splitter.addWidget(self.sidebar)
        self.h_splitter.addWidget(self.plot_tabs)
        self.h_splitter.setStretchFactor(1, 4)

        self.console = QPlainTextEdit(); self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc;")
        self.v_splitter = QSplitter(Qt.Vertical)
        self.v_splitter.addWidget(self.h_splitter); self.v_splitter.addWidget(self.console)
        self.main_layout.addWidget(self.v_splitter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.float_frame.setGeometry(self.plot_tabs.width() - 110, 35, 100, 50)

    # --- Setup Methods (Standard from before) ---
    def setup_unfolding_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)

        # --- COLUMN 1: Algorithm & Stats ---
        m_col = QGroupBox("1. Algorithm & Statistics")
        m_lay = QVBoxLayout(m_col)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Least Squares", "Entropy"])
        self.method_combo.setFixedWidth(180)
        self.method_combo.currentTextChanged.connect(self.toggle_method_ui)
        
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Instance Name...")
        self.label_input.setFixedWidth(180)

        # New: Reduced Chi Squared Input
        self.chi_box = QWidget()
        chi_lay = QHBoxLayout(self.chi_box); chi_lay.setContentsMargins(0,0,0,0)
        self.chi_val = ScientificDoubleSpinBox()
        self.chi_val.setValue(1.0) # Default to 1.0 for ideal fit
        chi_lay.addWidget(QLabel("χ²_ν:"))
        chi_lay.addWidget(self.chi_val)

        self.omega_box = QWidget()
        o_lay = QHBoxLayout(self.omega_box); o_lay.setContentsMargins(0,0,0,0)
        self.omega_val = ScientificDoubleSpinBox(); self.omega_val.setValue(1.0)
        o_lay.addWidget(QLabel("Ω:")); o_lay.addWidget(self.omega_val)
        self.omega_box.hide()

        m_lay.addWidget(QLabel("Method:")); m_lay.addWidget(self.method_combo)
        m_lay.addWidget(QLabel("Name:")); m_lay.addWidget(self.label_input)
        m_lay.addWidget(self.chi_box)   # Chi-squared sits under the name
        m_lay.addWidget(self.omega_box)
        m_lay.addStretch()
        layout.addWidget(m_col)

        # --- COLUMN 2: Data Loading ---
        f_col = QGroupBox("2. Data & Uncertainty")
        f_lay = QGridLayout(f_col); f_lay.setHorizontalSpacing(15)
        
        self.indicators = {}
        load_rows = [
            ("Measured", "measured", True, "measured_uncertainties"),
            ("A Priori", "a_priori", True, "a_priori_uncertainties"),
            ("Group Structure", "group_structure", False, None),
            ("Response Matrix", "response_matrix", False, None),
            ("Target", "target", False, None)
        ]

        for row, (lbl, d_key, has_unc, u_key) in enumerate(load_rows):
            btn = QPushButton(lbl); btn.setFixedWidth(130)
            btn.clicked.connect(lambda c, k=d_key: self.get_file(k))
            ind = QLabel("⭕"); self.indicators[d_key] = ind
            f_lay.addWidget(btn, row, 0); f_lay.addWidget(ind, row, 1)

            if has_unc:
                u_btn = QPushButton("Uncertainty")
                u_btn.setFixedWidth(120); u_btn.setMinimumHeight(25)
                # Hardened CSS to ensure text visibility
                u_btn.setStyleSheet("QPushButton { background-color: #f0f0f0; color: #000; border: 1px solid #999; font-size: 11px; }")
                u_btn.clicked.connect(lambda c, k=u_key: self.get_file(k))
                
                # Logic check: Measured uncertainty is mandatory for LS
                icon = "❌" if d_key == "measured" else "⚠️"
                u_ind = QLabel(icon); self.indicators[u_key] = u_ind
                f_lay.addWidget(u_btn, row, 2); f_lay.addWidget(u_ind, row, 3)
        
        layout.addWidget(f_col)

        # --- COLUMN 3: Execution (Circular Buttons) ---
        run_col = QVBoxLayout()
        self.run_btn = QPushButton("RUN")
        self.run_btn.setFixedSize(100, 100)
        self.run_btn.setStyleSheet("QPushButton { background-color: #0078d4; color: white; font-weight: bold; border-radius: 50px; }")
        self.run_btn.clicked.connect(self.run_calculation)
        run_col.addStretch(); run_col.addWidget(self.run_btn); run_col.addStretch()
        layout.addLayout(run_col)
        layout.addStretch(1) 
        self.ribbon.addTab(tab, "Unfolding")

    def toggle_method_ui(self, method):
        """Switches visibility between Entropy (Omega) and Least Squares (Chi-Sq) parameters."""
        is_entropy = (method == "Entropy")
        self.omega_box.setVisible(is_entropy)
        self.chi_box.setVisible(not is_entropy)

    def run_calculation(self):
        """
        Executes the unfolding. 
        Passes the user-defined reduced_chi_squared directly to the engine.
        """
        label = self.label_input.text() or f"Run_{self.root_node.childCount() + 1}"
        method = self.method_combo.currentText()
        
        # 1. Validation: Ensure required files are selected to avoid NoneType errors
        required_paths = ['measured', 'a_priori', 'group_structure', 'response_matrix']
        if method == "Least Squares":
            required_paths.append('measured_uncertainties')

        for key in required_paths:
            if not self.paths.get(key):
                self.log_command(f"<span style='color:red;'>Abort: '{key}' file is missing.</span>")
                return

        try:
            # 2. Initialize Engine
            if method == "Least Squares":
                engine = suf.leastsquares.NonlinearLeastSquareIterativeB()
            else:
                engine = suf.entropy.MaximumPotentialEntropy()
            
            # 3. Load Data into Engine
            engine.load('measured', self.paths['measured'], uncertainty=self.paths.get('measured_uncertainties'))
            engine.load('a_priori', self.paths['a_priori'], uncertainty=self.paths.get('a_priori_uncertainties'))
            engine.load('group_structure', self.paths['group_structure'])
            engine.load('response_matrix', self.paths['response_matrix'])
            if self.paths.get('target'): 
                engine.load('target', self.paths['target'])
            
            # 4. Execute with User Parameters
            # We do NOT calculate chi-squared here; we just pass your spinbox value.
            if method == "Least Squares":
                target_chi = self.chi_val.value()
                self.log_command(f"Starting Least Squares (Target χ²_ν: {target_chi})...")
                engine.run(reduced_chi_squared=target_chi)
            else:
                omega = self.omega_val.value()
                self.log_command(f"Starting Entropy (Omega: {omega})...")
                engine.run(omega=omega)

            # 5. Handle Plotting (Only if engine produced a solution)
            if engine.solution is not None:
                self.process_successful_run(engine, label)
                self.log_command(f"<b>Success:</b> {label} calculation complete.")
            else:
                self.log_command("<span style='color:orange;'>Warning: Engine finished but no solution was generated.</span>")

        except Exception as e:
            # This captures the 'NoneType' or 'Shape Mismatch' errors from the engine
            self.log_command(f"<span style='color:red;'>Engine Error: {str(e)}</span>")
            print(f"Full traceback: {e}")

    def process_successful_run(self, engine, label):
        """Standardized plotting logic after a successful run."""
        color = TAB10[self.root_node.childCount() % 10]
        parent = QTreeWidgetItem(self.root_node, [label])
        parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable)
        parent.setCheckState(0, Qt.Checked)
        
        # Save Metadata for the 'Click-to-Restore' feature
        self.instance_metadata[parent] = {
            'method': self.method_combo.currentText(),
            'name': label,
            'omega': self.omega_val.value(),
            'chi': self.chi_val.value(),
            'paths': self.paths.copy()
        }

        # Flux curves (Lethargy)
        y_sol = suf.util.convert_spectrum_to_lethargy(engine.solution, engine.group_structure)
        c_flux = self.pqg_flux.plot(engine.group_structure, y_sol, stepMode="center", 
                                    pen=pg.mkPen(color, width=2), name=f"{label} Solution")
        
        # Measured vs Recalculated Scatter
        channels = np.arange(len(engine.measured))
        c_meas = pg.ScatterPlotItem(x=channels, y=engine.measured, pen=color, symbol='o', size=8, name="Measured")
        
        # Recalculate counts: R . Phi_sol
        counts_sol = np.dot(engine.response_matrix, engine.solution)
        c_calc = pg.ScatterPlotItem(x=channels, y=counts_sol, pen='k', symbol='x', size=8, name="Re-calc")
        
        self.pqg_measured.addItem(c_meas)
        self.pqg_measured.addItem(c_calc)

        self.instances[parent] = {'curves': [c_flux, c_meas, c_calc]}
        self.engine_map[parent] = engine
        self.auto_range_plot()

    def setup_plot_options_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)

        # Scaling Group
        scale_group = QGroupBox("Axis Scaling")
        scale_lay = QVBoxLayout(scale_group)
        self.log_x_chk = QCheckBox("Log X")
        self.log_y_chk = QCheckBox("Log Y")
        scale_lay.addWidget(self.log_x_chk); scale_lay.addWidget(self.log_y_chk)
        scale_lay.addStretch()
        layout.addWidget(scale_group)

        # Range Group
        range_group = QGroupBox("Explicit Limits")
        range_lay = QGridLayout(range_group)
        self.xmin_val = ScientificDoubleSpinBox(); self.xmax_val = ScientificDoubleSpinBox()
        self.ymin_val = ScientificDoubleSpinBox(); self.ymax_val = ScientificDoubleSpinBox()
        
        for box in [self.xmin_val, self.xmax_val, self.ymin_val, self.ymax_val]:
            box.setFixedWidth(120)

        range_lay.addWidget(QLabel("X Range:"), 0, 0)
        range_lay.addWidget(self.xmin_val, 0, 1)
        range_lay.addWidget(self.xmax_val, 0, 2)
        range_lay.addWidget(QLabel("Y Range:"), 1, 0)
        range_lay.addWidget(self.ymin_val, 1, 1)
        range_lay.addWidget(self.ymax_val, 1, 2)
        layout.addWidget(range_group)

        # Apply Button (Now Circular)
        apply_col = QVBoxLayout()
        self.apply_btn = QPushButton("APPLY")
        self.apply_btn.setFixedSize(100, 100)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff7f0e; 
                color: white; 
                font-weight: bold; 
                border-radius: 50px;
                border: 2px solid #cc6600;
            }
            QPushButton:hover { background-color: #e67300; }
        """)
        self.apply_btn.clicked.connect(self.apply_plot_options)
        apply_col.addStretch(); apply_col.addWidget(self.apply_btn); apply_col.addStretch()
        layout.addLayout(apply_col)

        layout.addStretch(1)
        self.ribbon.addTab(tab, "Plot Options")

    # --- Core Logic & Dual Plotting ---
    def sync_active_plot(self):
        idx = self.plot_tabs.currentIndex()
        if idx == 1: self.render_matplotlib()
        elif idx == 2: self.render_plotly()

    def calculate_folded_counts(self, flux, matrix):
        """Calculates Recalculated Counts = Response Matrix . Flux Spectrum"""
        return np.dot(matrix, flux)

    def render_matplotlib(self):
        self.mpl_ax_flux.clear(); self.mpl_ax_meas.clear()
        
        for parent, curves in self.instances.items():
            if parent.checkState(0) != Qt.Checked: continue
            engine = self.engine_map[parent]
            gs = engine.group_structure
            matrix = engine.response_matrix
            
            # Left Plot: Flux
            y_sol = suf.util.convert_spectrum_to_lethargy(engine.solution, gs)
            self.mpl_ax_flux.step(gs, np.concatenate((y_sol, y_sol[-1:])), where='post', label=f"{parent.text(0)} Flux")
            
            # Right Plot: Scatter (Data Space)
            channels = np.arange(len(engine.measured))
            # 1. Measured
            self.mpl_ax_meas.scatter(channels, engine.measured, label="Measured", marker='o', alpha=0.6)
            # 2. Calculated (Folded)
            counts_sol = self.calculate_folded_counts(engine.solution, matrix)
            self.mpl_ax_meas.scatter(channels, counts_sol, label="Re-calc (Sol)", marker='x')
            
        self.mpl_ax_flux.set_xscale('log' if self.log_x_chk.isChecked() else 'linear')
        self.mpl_ax_flux.set_yscale('log' if self.log_y_chk.isChecked() else 'linear')
        self.mpl_ax_flux.legend(); self.mpl_ax_meas.legend()
        self.mpl_canvas.draw()

    def render_plotly(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Flux Spectrum", "Measurement Space"))
        for parent, curves in self.instances.items():
            if parent.checkState(0) != Qt.Checked: continue
            engine = self.engine_map[parent]
            channels = np.arange(len(engine.measured))
            
            # Flux
            y_sol = suf.util.convert_spectrum_to_lethargy(engine.solution, engine.group_structure)
            fig.add_trace(go.Scatter(x=engine.group_structure, y=y_sol, line_shape='hv', name=f"{parent.text(0)} Flux"), row=1, col=1)
            
            # Measured Scatter
            fig.add_trace(go.Scatter(x=channels, y=engine.measured, mode='markers', name="Measured"), row=1, col=2)
            counts_sol = self.calculate_folded_counts(engine.solution, engine.response_matrix)
            fig.add_trace(go.Scatter(x=channels, y=counts_sol, mode='markers', name="Re-calc (Sol)", marker_symbol='x'), row=1, col=2)
            
        fig.update_layout(template="plotly_white")
        self.plotly_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

    def run_calculation(self):
        label = self.label_input.text() or f"Run_{self.root_node.childCount() + 1}"
        try:
            method = self.method_combo.currentText()
            engine = suf.leastsquares.NonlinearLeastSquareIterativeB() if method == "Least Squares" else suf.entropy.MaximumPotentialEntropy()
            
            # Load Data
            engine.load('measured', self.paths['measured'])
            engine.load('a_priori', self.paths['a_priori'])
            engine.load('group_structure', self.paths['group_structure'])
            engine.load('response_matrix', self.paths['response_matrix'])
            
            engine.run(omega=self.omega_val.value()) if method == "Entropy" else engine.run()

            # --- Plotting ---
            color = TAB10[self.root_node.childCount() % 10]
            parent = QTreeWidgetItem(self.root_node, [label])
            parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable); parent.setCheckState(0, Qt.Checked)
            
            # Flux curves
            y_sol = suf.util.convert_spectrum_to_lethargy(engine.solution, engine.group_structure)
            c_flux = self.pqg_flux.plot(engine.group_structure, y_sol, stepMode="center", pen=pg.mkPen(color, width=2))
            
            # Measured scatter
            channels = np.arange(len(engine.measured))
            c_meas = pg.ScatterPlotItem(x=channels, y=engine.measured, pen=color, symbol='o', name="Measured")
            counts_sol = self.calculate_folded_counts(engine.solution, engine.response_matrix)
            c_calc = pg.ScatterPlotItem(x=channels, y=counts_sol, pen='k', symbol='x', name="Re-calc")
            self.pqg_measured.addItem(c_meas); self.pqg_measured.addItem(c_calc)

            self.instances[parent] = {'curves': [c_flux, c_meas, c_calc]}
            self.engine_map[parent] = engine
            self.instance_metadata[parent] = {'method': method, 'name': label, 'omega': self.omega_val.value(), 'paths': self.paths.copy()}
            self.auto_range_plot()
        except Exception as e: self.log_command(f"Error: {e}")

    # --- UI Helpers ---
    def get_file(self, key):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV (*.csv)")
        if path: self.paths[key] = path; self.indicators[key].setText("✅")

    def auto_range_plot(self):
        self.pqg_flux.autoRange(); self.pqg_measured.autoRange(); self.sync_active_plot()

    def apply_plot_options(self):
        lx, ly = self.log_x_chk.isChecked(), self.log_y_chk.isChecked()
        self.pqg_flux.setLogMode(x=lx, y=ly); self.sync_active_plot()

    def save_current_plot(self):
        idx = self.plot_tabs.currentIndex()
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "plot", "PNG (*.png);;PDF (*.pdf);;HTML (*.html)")
        if path:
            if idx == 1: self.mpl_fig.savefig(path)
            elif idx == 2: self.plotly_view.page().toHtml(lambda h: open(path, 'w').write(h))

    def log_command(self, msg):
        self.console.appendPlainText(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

    def handle_tree_checkbox(self, item, column):
        if item in self.instances:
            visible = item.checkState(0) == Qt.Checked
            for c in self.instances[item]['curves']: c.setVisible(visible)
        self.sync_active_plot()

    def restore_instance_settings(self, item, column):
        p = item if item in self.instance_metadata else item.parent()
        if p in self.instance_metadata:
            m = self.instance_metadata[p]
            self.method_combo.setCurrentText(m['method']); self.label_input.setText(m['name'])
            self.omega_val.setValue(m['omega']); self.paths = m['paths'].copy()

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    win = UnfoldingApp(); win.show(); sys.exit(app.exec())