from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import pathlib2
import os
from knife_edge_1 import load_positions, determine_y, derivative, gaus
from scipy.optimize import curve_fit
from sympy import Symbol
from sympy.solvers import solve

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'b')
pg.setConfigOptions(antialias=True)


class KnifeEdgeApp:

    def __init__(self):
        self.app = QtGui.QApplication([])
        self._main_window = QtGui.QMainWindow()
        self._main_window.setWindowTitle("THz Knife Edge App")
        self._main_window.resize(800, 800)
        self._central_widget = QtGui.QWidget()
        self._main_window.setCentralWidget(self._central_widget)

        self._method = 3
        self._in_mm = False
        self._x_factor = 2
        self._params = [
            {'name': 'Options', 'type': 'group', 'children': [
                {'name': 'Method', 'type': 'list', 'values': {"max": 0, "ptp": 1, "ptp2": 2, "area_td": 3},
                 'value': self._method},
                {'name': 'In mm?', 'type': 'bool', 'value': self._in_mm},
                {'name': 'X-Factor', 'type': 'float', 'value': self._x_factor},
            ]}]
        self._p = Parameter.create(name='params', type='group', children=self._params)
        self._p.sigTreeStateChanged.connect(self._tree_change)

        self._directory = "."
        self._directory_label = None
        self._file_name_example_label = None
        self._statusbar = QtGui.QStatusBar()
        self._main_window.setStatusBar(self._statusbar)
        self._split_left = None
        self._split_right = None
        self._min_t = None
        self._max_t = None
        self._lr = None
        self._plot_measurement = None
        self._plot_fit = None
        self._range_selection_plot = None
        self._result_label = None
        self._init_UI()

        self._main_window.show()

    def _tree_change(self, param, changes):
        change = None
        for param, change, data in changes:
            change = data
            path = self._p.childPath(param)
            if path is not None:
                child_name = ".".join(path)
            else:
                child_name = param.name()

        if child_name == "Options.Method":
            self._method = change
        if child_name == "Options.In mm?":
            self._in_mm = change
        if child_name == "Options.X-Factor":
            self._x_factor = change

    def _init_UI(self):
        layout = QtGui.QGridLayout()
        directory_button = QtGui.QPushButton("Select directory")
        directory_button.clicked.connect(self._file_load_clicked)
        self._directory_label = QtGui.QLabel("no directory")
        self._file_name_example_label = QtGui.QLabel("name example: ")
        self._plot_measurement = pg.PlotWidget(name='Plot1')
        self._plot_fit = pg.PlotWidget(name="Plot2")
        self._range_selection_plot = pg.PlotWidget(name="Plot3")
        t = ParameterTree()
        t.setParameters(self._p, showTop=True)
        apply_button = QtGui.QPushButton("Apply")
        apply_button.clicked.connect(self._apply)
        self._result_label = QtGui.QLabel("")

        horizontal_layout = QtGui.QHBoxLayout()
        self._split_left = QtGui.QLineEdit()
        self._split_right = QtGui.QLineEdit()
        horizontal_layout.addWidget(QtGui.QLabel("extract X"))
        horizontal_layout.addWidget(self._split_left)
        horizontal_layout.addWidget(self._split_right)
        split_widget = QtGui.QWidget()
        split_widget.setLayout(horizontal_layout)

        vertical_layout = QtGui.QVBoxLayout()
        vertical_layout.addWidget(directory_button)
        vertical_layout.addWidget(self._directory_label)
        vertical_layout.addWidget(self._file_name_example_label)
        vertical_layout.addWidget(split_widget)
        vertical_layout.addWidget(t)
        vertical_layout.addWidget(self._range_selection_plot)
        vertical_layout.addWidget(apply_button)
        vertical_layout.addWidget(self._result_label)
        options_widget = QtGui.QWidget()
        options_widget.setLayout(vertical_layout)

        layout.addWidget(options_widget, 0, 0, 2, 1)
        layout.addWidget(self._plot_measurement, 0, 1, 1, 1)
        layout.addWidget(self._plot_fit, 1, 1, 1, 1)

        self._central_widget.setLayout(layout)
        self._statusbar.showMessage("UI loaded successfully")

    def _file_load_clicked(self):
        try:
            self._directory = str(QtGui.QFileDialog.getExistingDirectory(None, "Select Directory ", self._directory))
            self._directory_label.setText(self._directory)
            self._file_name_example_label.setText("Name example: " + os.listdir(self._directory)[0])
            self._statusbar.showMessage("Directory " + self._directory + " was selected")

            os.chdir(self._directory)  # change directory
            files = os.listdir('.')
            files = sorted(files)
            data = np.loadtxt(files[-1])
            self._range_selection_plot.clear()
            self._range_selection_plot.plot(data[:,1], pen=pg.mkPen(color='b', width=1.2))
            self._range_selection_plot.setLabel(axis='left', text="Amplitude [arb. u.]")
            self._range_selection_plot.setLabel(axis='bottom', text="Index")
            self._lr = pg.LinearRegionItem([0, len(data)], bounds=[0, len(data)])
            self._lr.setZValue(-10)
            self._lr.sigRegionChanged.connect(self._region_connect)
            self._range_selection_plot.addItem(self._lr)
        except Exception as ex:
            self._statusbar.showMessage(ex)

    def _region_connect(self):
        region = self._lr.getRegion()
        self._min_t = int(region[0])
        self._max_t = int(region[1])

    def _apply(self):
        split_left = self._split_left.text()
        split_right = self._split_right.text()
        x_positions = []
        y_values = []
        if (split_left == "") | (os.listdir(self._directory)[0].find(split_left) == -1):
            self._statusbar.showMessage("Provide valid left split")
        if (split_right == "") | (os.listdir(self._directory)[0].find(split_right) == -1):
            self._statusbar.showMessage("Provide valid right split")
        try:
            x_positions = load_positions(self._directory, split_left, split_right)
        except Exception as ex:
            self._statusbar.showMessage(str(ex))
        if (not self._in_mm) & (self._x_factor != 0):
            if len(x_positions) == 0:
                self._statusbar.showMessage("Could not extract positions")
                return
            else:
                x_positions = [x * self._x_factor for x in x_positions]
        self._region_connect()
        y_values = determine_y(self._directory, (self._min_t, self._max_t), self._method)

        scatter_meas = pg.ScatterPlotItem(x_positions, y_values)
        self._plot_measurement.clear()
        self._plot_measurement.addItem(scatter_meas)
        self._plot_measurement.setLabel(axis='left', text=get_y_axis_text(self._method))
        self._plot_measurement.setLabel(axis='bottom', text="Position [mm]")

        deriv = derivative(x_positions, y_values)
        max_right_side = np.mean(y_values[-5:-1])
        min_left_side = np.mean(y_values[0:5])
        max_minus_plus_div_by_2 = (max_right_side + min_left_side) / 2
        center = x_positions[np.abs(y_values-max_minus_plus_div_by_2).argmin()]
        std = np.min(np.diff(x_positions))
        popt, pcov = curve_fit(gaus, x_positions[1:-1], deriv, p0=[max(deriv), center, std])

        self._plot_fit.clear()
        self._plot_fit.addLegend()
        self._plot_fit.plot(x_positions[1:-1], deriv/max(deriv), pen=None, symbol='o', name="Derivative")
        populated_x = np.linspace(x_positions[0], x_positions[-1], 1000)
        populated_fit = gaus(populated_x, *popt)
        self._plot_fit.plot(populated_x, populated_fit/max(deriv), pen=(255,0,0), name="Gauss fit")
        self._plot_fit.setLabel(axis='left', text="Beam profile (arb. u.)")
        self._plot_fit.setLabel(axis='bottom', text="Position [mm]")
        r2 = r2_score(deriv, gaus(x_positions[1:-1], *popt))
        r2_text = pg.TextItem("R^2 score: " + str(round(r2 * 100, 3)), color=(0, 0, 0))
        self._plot_fit.addItem(r2_text)

        self._plot_fit.plot(x_positions, np.ones_like(x_positions) * max(populated_fit)/max(deriv)/(np.e**2),
                            pen=pg.mkPen(color='m', width=1.2))
        x = Symbol('x')
        solution_e_2 = solve(- x ** 2 + 2 * x * popt[1] - popt[1] ** 2 + popt[2] ** 2, x)

        # r = sqrt(popt[1]**2-4*popt[2])
        std_err = np.sqrt(np.diag(pcov))
        summand_1 = std_err[1] * ((2 * popt[1]) / (4 * np.sqrt(popt[1] ** 2 - 4 * popt[2]))) ** 2
        summand_2 = std_err[2] * (-1 / (np.sqrt(popt[1] ** 2 - 4 * popt[2]))) ** 2
        uncertainty_r = np.sqrt(summand_1 + summand_2)
        self._result_label.setText(
            "Result:\t 1/e^2-radius:\t" + str(round((solution_e_2[1] - solution_e_2[0]) / 2, 3)) +
            " +- " + str(round(uncertainty_r, 3)) + " mm")


def r2_score(measured, fitted):
    mean_measured = np.mean(measured)
    sum_of_squares_total = np.sum((measured-mean_measured)**2)
    sum_of_squares_residual = np.sum((measured-fitted)**2)
    return 1 - sum_of_squares_residual / sum_of_squares_total


def get_y_axis_text(method: int):
    if method == 0:
        return "Maximum value"
    if method == 1:
        return "Peak-to-Peak"
    if method == 2:
        return "Peak-to-Peak squared"
    if method == 3:
        return "Intensity (area under the curve)"


if __name__ == '__main__':
    app = KnifeEdgeApp()
    app.app.exec_()