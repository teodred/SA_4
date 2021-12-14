import pprint
import sys
import os
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from matplotlib import pyplot as plt

from PyQt5.QtGui import *
from PyQt5.uic import loadUi
from logic import *


class App(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = loadUi('main.ui')
        self.ui.input_x_but.clicked.connect(self.openXfile)
        self.ui.input_y_but.clicked.connect(self.openYfile)
        self.ui.calculate.clicked.connect(self.compute)
        self.ui.show_graph.clicked.connect(self.plotGraphic)

    def openXfile(self):
        xfilename = QFileDialog.getOpenFileName(self, "Вхдний файл з Х:", "")
        xfilename = str(xfilename[0]).split('/')[-1]
        self.ui.input_x.setText(xfilename)

    def openYfile(self):
        yfilename = QFileDialog.getOpenFileName(self, "Вхдний файл з Y:", "")
        yfilename = str(yfilename[0]).split('/')[-1]
        self.ui.input_y.setText(yfilename)

    def compute(self):
        window_size = 10
        if self.ui.cheb_first.isChecked():
            polynom_type = 'chebyshev_first'
        elif self.ui.cheb_sec.isChecked():
            polynom_type = 'chebyshev_second'
        elif self.ui.lag.isChecked():
            polynom_type = 'laguerre'
        elif self.ui.ermit.isChecked():
            polynom_type = 'hermite'
        elif self.ui.lej.isChecked():
            polynom_type = 'legendre'
        else:
            polynom_type = 'error'
        

        dim_x1 = int(self.ui.x1_grade.text())
        dim_x2 = int(self.ui.x1_grade.text())
        dim_x3 = int(self.ui.x1_grade.text())

        if self.ui.add.isChecked():
            model_type = 'add'
            mult = False
        else:
            model_type = 'mult'
            mult = True

        if self.ui.normalize.isChecked():
            norm = True
        else:
            norm = False

        y_cord = int(self.ui.y_cord.text())
        polynomial_degree_values = np.array([dim_x1, dim_x2, dim_x3])

        #################
        self.output = ''

        feature_amount = 3
        feature_filenames = (self.ui.input_x.text(), self.ui.input_y.text())
        (x, feature_lengths), y = read_input_from_files(feature_filenames, self.ui.input_y.text())
        y_origin = y
        x = normalize_data(x)
        y, y_norm_values = normalize_data(y, min_max_data=True)

        x_features = split_data_by_features(x, feature_lengths)

        x_variable = x_features[y_cord - 1]
        y_variable = y[:, y_cord - 1]
        y_windowed = timeseries_to_fixed_window_array_padded(y_variable, window_size=window_size)

        A_x = create_equation_matrix(
            x_variable, polynomial_type=polynom_type,
            polynomial_degree=polynomial_degree_values[y_cord - 1])

        A_y = create_equation_matrix(
            y_windowed, polynomial_type=polynom_type,
            polynomial_degree=polynomial_degree_values[y_cord - 1])

        A_x = normalize_data(A_x)
        A_y = normalize_data(A_y)

        A = concat_equation_matrices([A_x, A_y])
        lambda_matrix = solve(A, y_variable, mult=mult)
        err = np.max(np.abs(forward(A, lambda_matrix, mult=mult) - y_variable))
        approx_values = forward(A, lambda_matrix, mult=mult)

        trustworthness = np.array(
            [1] + [calculate_trustworthness(y_variable[:ind], approx_values[:ind]) for ind in range(1, y.shape[0])])

        risk_total = np.array(
            [(0., -1), (0., -1)] + [calculate_risk(
                [y[:ind, var_ind] for var_ind in range(y.shape[1])], return_max_risk=True) for ind in
                range(2, y.shape[0])])

        risk = np.array([risk_total_el[0] for risk_total_el in risk_total])

        risk = exp_average(risk)
        trustworthness = exp_average(trustworthness)

        risk_dict = {
            -1: 'Отсутствует',
            0: 'Неустойчивое напряжение бортовой сети',
            1: 'Заканчивается топливо',
            2: 'Падает напряжение аккумуляторной батареи'
        }
        risk_source = [risk_total_el[1] if risk_total_el[0] > 0.1 else -1 for risk_total_el in risk_total]
        risk_source = [risk_dict[source] for source in risk_source]

        out_data = {
            'Напряжение бортовой сети': y_origin[:, 0],
            'Количество топлива': y_origin[:, 1],
            'Напряжение аккумуляторной батареи': y_origin[:, 2],
            'Оценка риска': risk,
            # 'Категория ситуации':,
            'Источник риска': risk_source
        }

        out_df = pd.DataFrame(out_data, index=np.arange(y.shape[0]) * 10)
        out_df.to_excel('output.xlsx', encoding='utf-8-sig')

        if not norm:
            y_variable = denormalize_data(y_variable, norm_values)
            approx_values = denormalize_data(approx_values, norm_values)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9))

        ax1.plot(approx_values, label='Прогноз')
        ax1.plot(y_variable, label='Реальні значення')
        ax1.legend()
        ax1.title.set_text('Прогнозоване і реальне значення')
        ax2.plot(risk, label='Ризик', color='b')
        ax2.plot([0.8] * len(risk), label='Критичний рівень ризику', color='r')
        ax2.plot([0.4] * len(risk), label='Загроза нестабільного стану', color='y')
        ax2.legend()
        ax2.title.set_text('Оцінка ризику')
        ax3.plot(trustworthness, label='Достовірність датчику')
        ax3.plot([0.5]*len(trustworthness), label='Критичний рівень', color='r')
        ax3.legend()
        ax3.title.set_text('Достовірність датчику')
        plt.savefig('./plot.png')
        os.startfile('output.xlsx')
    
    
    
    def plotGraphic(self):
        os.startfile('plot.png')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.ui.show()
    sys.exit(app.exec_())
