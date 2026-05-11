
import sys
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
from scipy.signal import butter, filtfilt, welch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 50  # Увеличить лимит

# ---------- Математические функции моделирования (внутри кода сохраняем латинские имена переменных)

# def generate_pn_sequence(N, rate, levels, fs):
#     samples_per_symbol = int(fs / rate)
#     seq = np.random.choice(levels, N // samples_per_symbol)
#     return np.repeat(seq, samples_per_symbol)
def generate_pn_sequence(N, rate, levels, fs):
    samples_per_symbol = fs / rate
    samples_per_symbol = int(round(samples_per_symbol))

    seq_len = int(np.ceil(N / samples_per_symbol))
    seq = np.random.choice(levels, seq_len)

    pn = np.repeat(seq, samples_per_symbol)
    return pn[:N]   # принудительно обрезаем до точного размера


def generate_sinusoid(frequency, phase, fs, N):
    t = np.arange(N) / fs
    return np.sin(2 * np.pi * frequency * t + phase)


def add_gaussian_noise(signal, std_dev, mean=0):
    noise = np.random.normal(mean, std_dev, len(signal))
    return signal + noise


def butter_lowpass_filter(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def decimate(signal, factor):
    return signal[::factor]


def limiter(signal):
    return np.where(signal > 0, 1, -1)


def calculate_power(signal):
    return np.mean(signal**2)


# def compute_psd(signal, fs, nperseg=1024):
#
#     f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
#     Pxx_db = 10 * np.log10(Pxx + 1e-20)
#     return f, Pxx_db
def compute_psd(signal, fs, nfft=2**16):
    # Zero-padding для гладкости
    Y = np.fft.fft(signal, n=nfft)
    Pxx = np.abs(Y)**2 / (fs * len(signal))

    # Только положительные частоты
    f = np.fft.fftfreq(nfft, 1/fs)
    mask = f >= 0

    Pxx_db = 10 * np.log10(Pxx[mask] + 1e-20)

    return f[mask], Pxx_db


# ---------- Виджеты для встроенных графиков
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        plt.tight_layout()


class TableInputDialog(QtWidgets.QDialog):
    def __init__(self, x_name, y_name, parent=None, saved_data=None):
        super().__init__(parent)

        self.setWindowTitle("Ввод точек")
        self.resize(500, 550)

        layout = QtWidgets.QVBoxLayout(self)

        # Создаем таблицу
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setRowCount(10)  # Начальное количество строк
        self.table.setHorizontalHeaderLabels([x_name, y_name])

        # Устанавливаем режим редактирования
        self.table.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.DoubleClicked |
                                   QtWidgets.QTableWidget.EditTrigger.EditKeyPressed)

        # Загружаем сохраненные данные, если они есть
        if saved_data is not None and len(saved_data[0]) > 0:
            x_data, y_data = saved_data
            self.table.setRowCount(len(x_data))
            for i, (x_val, y_val) in enumerate(zip(x_data, y_data)):
                self._set_table_item(i, 0, x_val)
                self._set_table_item(i, 1, y_val)

        layout.addWidget(self.table)

        # Кнопки управления таблицей (первый ряд)
        btn_control_layout1 = QtWidgets.QHBoxLayout()

        btn_add_row = QtWidgets.QPushButton("➕ Добавить строку")
        btn_add_row.clicked.connect(self._add_row)
        btn_control_layout1.addWidget(btn_add_row)

        btn_remove_row = QtWidgets.QPushButton("➖ Удалить последнюю строку")
        btn_remove_row.clicked.connect(self._remove_last_row)
        btn_control_layout1.addWidget(btn_remove_row)

        btn_clear_all = QtWidgets.QPushButton("🗑️ Очистить все")
        btn_clear_all.clicked.connect(self._clear_all)
        btn_control_layout1.addWidget(btn_clear_all)

        layout.addLayout(btn_control_layout1)

        # Кнопки сохранения и загрузки (второй ряд)
        btn_control_layout2 = QtWidgets.QHBoxLayout()

        btn_save = QtWidgets.QPushButton("💾 Сохранить точки в CSV")
        btn_save.clicked.connect(self._save_points)
        btn_control_layout2.addWidget(btn_save)


        layout.addLayout(btn_control_layout2)

        # Стандартные кнопки OK/Cancel
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _set_table_item(self, row, col, value):
        """Устанавливает значение ячейки с форматированием до 3 знаков"""
        item = QtWidgets.QTableWidgetItem(f"{value:.3f}")
        self.table.setItem(row, col, item)

    def _add_row(self):
        """Добавляет новую строку в таблицу"""
        current_row = self.table.rowCount()
        self.table.setRowCount(current_row + 1)

    def _remove_last_row(self):
        """Удаляет последнюю строку, если строк больше 1"""
        current_row = self.table.rowCount()
        if current_row > 1:
            self.table.setRowCount(current_row - 1)

    def _clear_all(self):
        """Очищает все строки, оставляя только одну пустую"""
        self.table.setRowCount(1)
        # Очищаем содержимое первой строки
        self.table.setItem(0, 0, None)
        self.table.setItem(0, 1, None)

    def _save_points(self):
        """Сохраняет точки из таблицы в CSV файл"""
        x, y = self.get_data()

        if len(x) == 0:
            QtWidgets.QMessageBox.warning(self, "Предупреждение",
                                          "Нет данных для сохранения. Заполните таблицу.")
            return

        # Получаем названия столбцов
        x_name = self.table.horizontalHeaderItem(0).text()
        y_name = self.table.horizontalHeaderItem(1).text()

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить точки", "", "CSV файлы (*.csv);;Все файлы (*.*)"
        )

        if path:
            if not path.endswith('.csv'):
                path += '.csv'

            # Сохраняем точки в CSV с округлением до 3 знаков
            data = np.column_stack((np.round(x, 3), np.round(y, 3)))
            np.savetxt(path, data, delimiter=";", fmt='%.3f',
                       header=f"{x_name};{y_name}", comments='')

            QtWidgets.QMessageBox.information(self, "Успех",
                                              f"Точки сохранены в файл:\n{path}")


    def get_data(self):
        """Возвращает данные из таблицы с округлением до 3 знаков"""
        x = []
        y = []
        rows = self.table.rowCount()

        for r in range(rows):
            item_x = self.table.item(r, 0)
            item_y = self.table.item(r, 1)

            if item_x and item_y and item_x.text() and item_y.text():
                try:
                    # Округляем до 3 знаков при считывании
                    xv = round(float(item_x.text()), 3)
                    yv = round(float(item_y.text()), 3)
                    x.append(xv)
                    y.append(yv)
                except ValueError:
                    pass

        return np.array(x), np.array(y)

# ---------- Главное окно
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Моделирование')
        self.resize(1200, 800)

        # ---------- Параметры моделирования (по умолчанию)
        self.пс_частота = 10      # pn_rate
        self.fs = 2000            # частота дискретизации
        self.длительность = 10.0  # секунды
        self.N = int(self.длительность * self.fs)
        self.частота_опорного = 200
        self.фаза = 0.0
        self.фаза_оп = 0.0
        self.шум_std = 2.0
        self.порог_решения = 0.0
        # self.дек_фактор = 200
        self.дек_фактор = int(round(self.fs / self.пс_частота)) if self.пс_частота > 0 else 1
        self.фильтр_срез = 10

        # Основной виджет и табы
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        # Создаём вкладки
        self.tab_params = QtWidgets.QWidget()
        self.tab_psp = QtWidgets.QWidget()
        self.tab_oporny = QtWidgets.QWidget()
        self.tab_modulator = QtWidgets.QWidget()
        self.tab_channel = QtWidgets.QWidget()
        self.tab_demod = QtWidgets.QWidget()
        self.tab_lpf = QtWidgets.QWidget()
        self.tab_decim = QtWidgets.QWidget()
        self.tab_decider = QtWidgets.QWidget()
        self.tab_compare = QtWidgets.QWidget()
        self.tab_eye = QtWidgets.QWidget()
        self.tab_tradeoff = QtWidgets.QWidget()

        tabs.addTab(self.tab_params, 'Параметры')
        self.tab_part1 = QtWidgets.QWidget()
        tabs.addTab(self.tab_part1, 'Часть 1 (Моделирование)')
        # tabs.addTab(self.tab_psp, '2. ПСП')
        # tabs.addTab(self.tab_oporny, '3. Опорный сигнал')
        # tabs.addTab(self.tab_modulator, '4. Модулятор')
        # tabs.addTab(self.tab_channel, '5. Канал')
        # tabs.addTab(self.tab_demod, '6. Выход перемножителя')
        # tabs.addTab(self.tab_lpf, '7. ФНЧ')
        # tabs.addTab(self.tab_decim, '8. Децимация')
        # tabs.addTab(self.tab_decider, '9. Решающее устройство')
        # tabs.addTab(self.tab_compare, '10. Сравнение')
        tabs.addTab(self.tab_eye, 'Часть 2 (Глаз-диаграмма)')
        tabs.addTab(self.tab_tradeoff, 'Часть 3 (Диаграмма обмена)')

        part1_layout = QtWidgets.QVBoxLayout(self.tab_part1)
        self.part1_tabs = QtWidgets.QTabWidget()
        part1_layout.addWidget(self.part1_tabs)

        self.part1_tabs.addTab(self.tab_psp, 'ПСП')
        self.part1_tabs.addTab(self.tab_oporny, 'Опорный сигнал')
        self.part1_tabs.addTab(self.tab_modulator, 'Модулятор')
        self.part1_tabs.addTab(self.tab_channel, 'Канал')
        self.part1_tabs.addTab(self.tab_demod, 'Выход перемножителя')
        self.part1_tabs.addTab(self.tab_lpf, 'ФНЧ')
        self.part1_tabs.addTab(self.tab_decim, 'Децимация')
        self.part1_tabs.addTab(self.tab_decider, 'Решающее устройство')
        self.part1_tabs.addTab(self.tab_compare, 'Сравнение')

        # ---------- Наполняем вкладки
        self._build_params_tab()
        self._build_psp_tab()
        self._build_oporny_tab()
        self._build_modulator_tab()
        self._build_channel_tab()
        self._build_demod_tab()
        self._build_lpf_tab()
        self._build_decim_tab()
        self._build_decider_tab()
        self._build_compare_tab()
        self._build_eye_tab()
        self._build_tradeoff_tab()

        # Сцена схемы блоков (будет размещена в параметрах)
        self._draw_block_diagram()

        # Первичная генерация сигналов
        self._generate_all_signals()
        self._update_all_plots()

    # ---------- Построение вкладки Параметры
    def _build_params_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_params)

        # Заголовок
        title = QtWidgets.QLabel("Формирование и демодуляция сигналов ФМ2")
        title_font = QtGui.QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # ---------- Группа для параметров с закруглённой рамкой
        group_box = QtWidgets.QGroupBox("Параметры моделирования")
        group_box_layout = QtWidgets.QFormLayout()
        group_box.setLayout(group_box_layout)
        title_font = QtGui.QFont()
        title_font.setPointSize(12)  # нужный размер
        title_font.setBold(True)
        group_box.setFont(title_font)  # назначаем шрифт всему QGroupBox
        layout.addWidget(group_box)

        # Стилизация рамки
        group_box.setStyleSheet("""
            QGroupBox {
                border: 2px solid gray;
                border-radius: 15px;
                margin-top: 15px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 3px;
                font-weight: bold;
                font-size: 10pt;
            }
        """)

        # ---------- Поля ввода
        spin_font = QtGui.QFont()
        spin_font.setPointSize(11)  # Размер текста в SpinBox

        label_font = QtGui.QFont()
        label_font.setPointSize(11)  # Размер текста подписей
        label_font.setBold(True)

        # Частота дискретизации
        self.input_fs = QtWidgets.QSpinBox()
        self.input_fs.setRange(100, 200000)
        self.input_fs.setValue(self.fs)
        self.input_fs.setFont(spin_font)
        lbl_fs = QtWidgets.QLabel('Частота дискретизации (fs) [Гц]:')
        lbl_fs.setFont(label_font)
        group_box_layout.addRow(lbl_fs, self.input_fs)

        # Длительность моделирования
        self.input_T = QtWidgets.QDoubleSpinBox()
        self.input_T.setRange(0.1, 3600.0)
        self.input_T.setDecimals(3)
        self.input_T.setValue(self.длительность)
        self.input_T.setFont(spin_font)
        lbl_T = QtWidgets.QLabel('Длительность моделирования T [с]:')
        lbl_T.setFont(label_font)
        group_box_layout.addRow(lbl_T, self.input_T)

        # Частота ПСП
        self.input_pn = QtWidgets.QSpinBox()
        self.input_pn.setRange(1, 1000)
        self.input_pn.setValue(self.пс_частота)
        self.input_pn.setFont(spin_font)
        lbl_pn = QtWidgets.QLabel('Частота ПСП[Гц]:')
        lbl_pn.setFont(label_font)
        group_box_layout.addRow(lbl_pn, self.input_pn)

        # Частота опорного сигнала
        self.input_freq = QtWidgets.QSpinBox()
        self.input_freq.setRange(1, 50000)
        self.input_freq.setValue(self.частота_опорного)
        self.input_freq.setFont(spin_font)
        lbl_freq = QtWidgets.QLabel('Частота опорного сигнала [Гц]:')
        lbl_freq.setFont(label_font)
        group_box_layout.addRow(lbl_freq, self.input_freq)

        # Фаза сигнала
        self.input_phase = QtWidgets.QDoubleSpinBox()
        self.input_phase.setRange(0, 360)
        self.input_phase.setValue(self.фаза)
        self.input_phase.setFont(spin_font)
        lbl_phase = QtWidgets.QLabel('Фаза сигнала:')
        lbl_phase.setFont(label_font)
        group_box_layout.addRow(lbl_phase, self.input_phase)

        # Фаза опорного
        self.input_phase_op = QtWidgets.QDoubleSpinBox()
        self.input_phase_op.setRange(0, 360)
        self.input_phase_op.setValue(self.фаза_оп)
        self.input_phase_op.setFont(spin_font)
        lbl_phase_op = QtWidgets.QLabel('Фаза опорного:')
        lbl_phase_op.setFont(label_font)
        group_box_layout.addRow(lbl_phase_op, self.input_phase_op)

        # Шум
        self.input_noise = QtWidgets.QDoubleSpinBox()
        self.input_noise.setRange(0.0, 100.0)
        self.input_noise.setDecimals(3)
        self.input_noise.setValue(self.шум_std)
        self.input_noise.setFont(spin_font)
        lbl_noise = QtWidgets.QLabel('СКО шума:')
        lbl_noise.setFont(label_font)
        group_box_layout.addRow(lbl_noise, self.input_noise)

        # Децимация
        # self.input_dec = QtWidgets.QSpinBox()
        # self.input_dec.setRange(1, 10000)
        # self.input_dec.setValue(self.дек_фактор)
        # self.input_dec.setFont(spin_font)
        # lbl_dec = QtWidgets.QLabel('Фактор децимации:')
        # lbl_dec.setFont(label_font)
        # group_box_layout.addRow(lbl_dec, self.input_dec)

        # Частота среза ФНЧ
        self.input_cut = QtWidgets.QDoubleSpinBox()
        self.input_cut.setRange(0.1, 10000)
        self.input_cut.setDecimals(3)
        self.input_cut.setValue(self.фильтр_срез)
        self.input_cut.setFont(spin_font)
        lbl_cut = QtWidgets.QLabel('Частота среза ФНЧ [Гц]:')
        lbl_cut.setFont(label_font)
        group_box_layout.addRow(lbl_cut, self.input_cut)

        # ---------- Кнопка "Применить параметры"
        btn_apply = QtWidgets.QPushButton('Применить параметры и пересчитать')
        btn_apply.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Weight.Bold))
        btn_apply.setMinimumHeight(40)
        btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        btn_apply.clicked.connect(self._on_apply_params)
        layout.addWidget(btn_apply)

        layout.addStretch()

        # ---------- Сцена для схемы блоков
        self.scene_view = QtWidgets.QGraphicsView()
        self.scene_view.setMinimumHeight(260)
        layout.addWidget(self.scene_view)

    # ---------- ПСП вкладка
    def _build_psp_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_psp)


        # графики: временная ПСП и PSD
        self.psp_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.psp_canvas)
        self.psp_toolbar = NavigationToolbar(self.psp_canvas, self)
        layout.addWidget(self.psp_toolbar)

        self.psp_psd_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.psp_psd_canvas)
        self.psp_psd_toolbar = NavigationToolbar(self.psp_psd_canvas, self)
        layout.addWidget(self.psp_psd_toolbar)

    # ---------- Опорный сигнал вкладка
    def _build_oporny_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_oporny)

        self.op_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.op_canvas)
        self.op_toolbar = NavigationToolbar(self.op_canvas, self)
        layout.addWidget(self.op_toolbar)

        self.op_psd_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.op_psd_canvas)
        self.op_psd_toolbar = NavigationToolbar(self.op_psd_canvas, self)
        layout.addWidget(self.op_psd_toolbar)

    # ---------- Модулятор
    def _build_modulator_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_modulator)

        self.mod_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.mod_canvas)
        self.mod_toolbar = NavigationToolbar(self.mod_canvas, self)
        layout.addWidget(self.mod_toolbar)

        self.mod_psd_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.mod_psd_canvas)
        self.mod_psd_toolbar = NavigationToolbar(self.mod_psd_canvas, self)
        layout.addWidget(self.mod_psd_toolbar)

    # ---------- Канал
    def _build_channel_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_channel)

        self.chan_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.chan_canvas)
        self.chan_toolbar = NavigationToolbar(self.chan_canvas, self)
        layout.addWidget(self.chan_toolbar)

        self.chan_psd_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.chan_psd_canvas)
        self.chan_psd_toolbar = NavigationToolbar(self.chan_psd_canvas, self)
        layout.addWidget(self.chan_psd_toolbar)

    # ---------- Демодулятор
    def _build_demod_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_demod)

        self.dem_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.dem_canvas)
        self.dem_toolbar = NavigationToolbar(self.dem_canvas, self)
        layout.addWidget(self.dem_toolbar)

        self.dem_psd_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.dem_psd_canvas)
        self.dem_psd_toolbar = NavigationToolbar(self.dem_psd_canvas, self)
        layout.addWidget(self.dem_psd_toolbar)

    # ---------- ФНЧ
    def _build_lpf_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_lpf)

        self.lpf_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.lpf_canvas)
        self.lpf_toolbar = NavigationToolbar(self.lpf_canvas, self)
        layout.addWidget(self.lpf_toolbar)

        self.lpf_psd_canvas = MplCanvas(self, width=8, height=3)
        layout.addWidget(self.lpf_psd_canvas)
        self.lpf_psd_toolbar = NavigationToolbar(self.lpf_psd_canvas, self)
        layout.addWidget(self.lpf_psd_toolbar)

    # ---------- Децимация
    # def _build_decim_tab(self):
    #     layout = QtWidgets.QVBoxLayout(self.tab_decim)
    #
    #     self.dec_canvas = MplCanvas(self, width=10, height=4)
    #     layout.addWidget(self.dec_canvas)
    #     self.dec_toolbar = NavigationToolbar(self.dec_canvas, self)
    #     layout.addWidget(self.dec_toolbar)
    def _build_decim_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_decim)

        # Верхний график: децимированный сигнал (как было)
        self.dec_canvas = MplCanvas(self, width=10, height=3)
        layout.addWidget(self.dec_canvas)
        self.dec_toolbar = NavigationToolbar(self.dec_canvas, self)
        layout.addWidget(self.dec_toolbar)

        # Нижний график: наложение децимированного на отфильтрованный
        self.dec_overlay_canvas = MplCanvas(self, width=10, height=3)
        layout.addWidget(self.dec_overlay_canvas)
        self.dec_overlay_toolbar = NavigationToolbar(self.dec_overlay_canvas, self)
        layout.addWidget(self.dec_overlay_toolbar)

    def _plot_decimated_overlay(self):
        ax = self.dec_overlay_canvas.ax
        ax.clear()

        # 1. Рисуем ОТФИЛЬТРОВАННЫЙ сигнал (полностью, как в табе ФНЧ)
        ax.plot(self.t, self.filtered, 'b-', alpha=0.5, label='Отфильтрованный сигнал (ФНЧ)', linewidth=0.5)

        # 2. Рисуем ДЕЦИМИРОВАННЫЙ сигнал (полностью, как в верхнем графике)
        ax.plot(self.decimated_t, self.decimated, 'r-', linewidth=1.5, label='Децимированный сигнал')
        ax.scatter(self.decimated_t, self.decimated, s=30, c='red', marker='o', zorder=5)

        ax.set_title('Полное наложение: отфильтрованный (ФНЧ) + децимированный сигналы')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Амплитуда')
        ax.legend(loc='upper right')
        ax.grid(True)

        # Добавляем информационный текст
        info_text = (f'Частота ПСП: {self.пс_частота} Гц\n'
                     f'Фактор децимации: {self.дек_фактор}\n'
                     f'Исходная частота: {self.fs} Гц\n'
                     f'После децимации: {self.fs / self.дек_фактор:.1f} Гц')

        # ax.text(0.02, 0.98, info_text,
        #         transform=ax.transAxes, fontsize=9,
        #         verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        self.dec_overlay_canvas.draw()

    # ---------- Решающее устройство
    def _build_decider_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_decider)

        self.decider_canvas = MplCanvas(self, width=10, height=4)
        layout.addWidget(self.decider_canvas)
        self.decider_toolbar = NavigationToolbar(self.decider_canvas, self)
        layout.addWidget(self.decider_toolbar)

    # ---------- Сравнение
    def _build_compare_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_compare)

        # Полотно 1 – Исходная ПСП
        self.compare_canvas_orig = MplCanvas(self, width=10, height=3)
        layout.addWidget(self.compare_canvas_orig)
        self.compare_toolbar_orig = NavigationToolbar(self.compare_canvas_orig, self)
        layout.addWidget(self.compare_toolbar_orig)

        # Полотно 2 – Восстановленная
        self.compare_canvas_rec = MplCanvas(self, width=10, height=3)
        layout.addWidget(self.compare_canvas_rec)
        self.compare_toolbar_rec = NavigationToolbar(self.compare_canvas_rec, self)
        layout.addWidget(self.compare_toolbar_rec)

        # Группа для отображения статистики ошибок
        stats_group = QtWidgets.QGroupBox("Статистика ошибок")
        stats_layout = QtWidgets.QFormLayout()
        stats_group.setLayout(stats_layout)

        # Стилизация группы
        stats_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid gray;
                border-radius: 10px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 5px;
            }
        """)

        # Поля для отображения статистики
        self.errors_label = QtWidgets.QLabel('0')
        self.errors_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
        stats_layout.addRow('Количество ошибок:', self.errors_label)

        self.ber_label = QtWidgets.QLabel('0.000000')
        self.ber_label.setStyleSheet("color: blue; font-weight: bold; font-size: 14px;")
        stats_layout.addRow('BER (Bit Error Rate):', self.ber_label)

        self.total_bits_label = QtWidgets.QLabel('0')
        self.total_bits_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        stats_layout.addRow('Всего бит:', self.total_bits_label)

        self.accuracy_label = QtWidgets.QLabel('100.00%')
        self.accuracy_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        stats_layout.addRow('Точность:', self.accuracy_label)

        layout.addWidget(stats_group)

        layout.addStretch()

    def _update_error_stats(self):
        """Обновляет отображение статистики ошибок"""
        # Получаем исходную и восстановленную ПСП
        orig = self.pn_sequence[::self.дек_фактор][:len(self.limited)]
        orig_bits = np.where(orig >= 0, 1, -1)

        # Считаем ошибки
        errors = np.sum(orig_bits != self.limited)
        total = len(self.limited)
        ber = errors / total if total > 0 else 0
        accuracy = (1 - ber) * 100

        # Обновляем метки
        self.errors_label.setText(str(errors))
        self.ber_label.setText(f'{ber:.8f}')
        self.total_bits_label.setText(str(total))
        self.accuracy_label.setText(f'{accuracy:.2f}%')

        # Меняем цвет в зависимости от количества ошибок
        if errors == 0:
            self.errors_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
            self.ber_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        else:
            self.errors_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            self.ber_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")

    # ---------- Глаз-диаграмма
    def _build_eye_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_eye)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(2)  # Уменьшаем расстояние между элементами
        layout.addLayout(controls)

        # Фаза сигнала
        phase_label = QtWidgets.QLabel('Фаза сигнала:')
        controls.addWidget(phase_label)
        self.eye_phase = QtWidgets.QDoubleSpinBox()
        self.eye_phase.setRange(0, 360)
        self.eye_phase.setValue(0)
        self.eye_phase.setMaximumWidth(80)  # Ограничиваем ширину
        controls.addWidget(self.eye_phase)

        controls.addSpacing(90)  # Небольшой отступ между группами

        # Фаза опорного
        phase_op_label = QtWidgets.QLabel('Фаза опорного:')
        controls.addWidget(phase_op_label)
        self.eye_phase_op = QtWidgets.QDoubleSpinBox()
        self.eye_phase_op.setRange(0, 360)
        self.eye_phase_op.setValue(0)
        self.eye_phase_op.setMaximumWidth(80)
        controls.addWidget(self.eye_phase_op)

        controls.addSpacing(90)

        # СКО шума
        noise_label = QtWidgets.QLabel('СКО шума:')
        controls.addWidget(noise_label)
        self.eye_noise = QtWidgets.QDoubleSpinBox()
        self.eye_noise.setRange(0.0, 5.0)
        self.eye_noise.setDecimals(3)
        self.eye_noise.setValue(0)
        self.eye_noise.setMaximumWidth(80)
        controls.addWidget(self.eye_noise)

        controls.addSpacing(90)

        # Длина реализации
        realizations_label = QtWidgets.QLabel('Длина реализации:')
        controls.addWidget(realizations_label)
        self.eye_realizations = QtWidgets.QSpinBox()
        self.eye_realizations.setRange(1, 100)
        self.eye_realizations.setValue(10)
        self.eye_realizations.setSuffix(' символов')
        self.eye_realizations.setMaximumWidth(120)
        controls.addWidget(self.eye_realizations)

        controls.addSpacing(100)

        # Кнопка
        btn_replot = QtWidgets.QPushButton('Пересчитать')
        btn_replot.clicked.connect(self._update_eye_diagram)
        controls.addWidget(btn_replot)

        controls.addStretch()  # Растяжка справа

        # Полотно для глаз-диаграммы
        self.eye_canvas = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.eye_canvas)
        self.eye_toolbar = NavigationToolbar(self.eye_canvas, self)
        layout.addWidget(self.eye_toolbar)
    # ---------- Рисуем блок-схему в QGraphicsScene
    # ---------- Рисуем блок-схему в QGraphicsScene (новая, аккуратная)
    def _draw_block_diagram(self):
        scene = QtWidgets.QGraphicsScene()
        # Уменьшил высоту сцены чтобы убрать лишнее пространство снизу
        scene.setSceneRect(0, 0, 1100, 300)
        self.scene_view.setScene(scene)


        # Жирное перо для стрелок и рамок
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black)
        pen.setWidth(3)


        # Темно-серая заливка для блоков
        brush = QtGui.QBrush(QtGui.QColor(100, 100, 100))  # Темно-серый

        # Жирный шрифт для подписей
        bold_font = QtGui.QFont()
        bold_font.setPointSize(10)
        bold_font.setBold(True)

        # параметры блоков (x, y, w, h) - уменьшил расстояние между рядами
        blocks = {
            '1-ПСП': (80, 30, 180, 70),
            '2-МОДУЛЯТОР': (330, 30, 180, 70),
            '3-КАНАЛ С АБГШ': (580, 30, 180, 70),
            '4-ПЕРЕМНОЖИТЕЛЬ': (80, 150, 180, 70),  # Поднял еще выше
            '5-ФНЧ': (330, 150, 180, 70),  # Поднял еще выше
            '6-ДЕЦИМАТОР': (580, 150, 180, 70),  # Поднял еще выше
            '7-РЕШАЮЩЕЕ\nУСТРОЙСТВО': (830, 150, 180, 70)  # Поднял еще выше
        }

        # добавляем прямоугольники и подписи
        self.block_items = {}
        for name, (x, y, w, h) in blocks.items():
            rect = scene.addRect(x, y, w, h, pen=pen, brush=brush)
            text_item = scene.addText(name)
            text_item.setDefaultTextColor(QtCore.Qt.GlobalColor.white)  # Белый текст на темном фоне
            text_item.setFont(bold_font)
            text_item_bbox = text_item.boundingRect()
            text_item.setPos(x + (w - text_item_bbox.width()) / 2, y + (h - text_item_bbox.height()) / 2)
            self.block_items[name] = (rect, (x, y, w, h))

        # функция добавить стрелку с наконечником
        def add_arrow(x1, y1, x2, y2, label=None, label_offset_x=0, label_offset_y=0):
            # Основная линия стрелки
            scene.addLine(x1, y1, x2, y2, pen)

            # Стрелочный наконечник
            angle = np.arctan2(y2 - y1, x2 - x1)
            arrow_len = 15
            left = (x2 - arrow_len * np.cos(angle - np.pi / 6), y2 - arrow_len * np.sin(angle - np.pi / 6))
            right = (x2 - arrow_len * np.cos(angle + np.pi / 6), y2 - arrow_len * np.sin(angle + np.pi / 6))
            poly = QtGui.QPolygonF([
                QtCore.QPointF(x2, y2),
                QtCore.QPointF(left[0], left[1]),
                QtCore.QPointF(right[0], right[1])
            ])
            scene.addPolygon(poly, pen=pen, brush=QtGui.QBrush(QtCore.Qt.GlobalColor.black))

            # Подпись стрелки (если есть)
            if label:
                label_item = scene.addText(label)
                label_item.setFont(bold_font)
                label_x = (x1 + x2) / 2 - 20 + label_offset_x
                label_y = (y1 + y2) / 2 - 15 + label_offset_y
                label_item.setPos(label_x, label_y)

        # ВЕРХНИЙ РЯД - горизонтальные соединения с увеличенными стрелками
        # 1-ПСП -> 2-МОДУЛЯТОР с подписью D(t) выше
        psp_x, psp_y, psp_w, psp_h = self.block_items['1-ПСП'][1]
        modulator_x, modulator_y, modulator_w, modulator_h = self.block_items['2-МОДУЛЯТОР'][1]
        add_arrow(psp_x + psp_w, psp_y + psp_h / 2, modulator_x, modulator_y + modulator_h / 2, "D(t)", 0, -20)

        # 2-МОДУЛЯТОР -> 3-КАНАЛ с подписью S_i(t) выше
        channel_x, channel_y, channel_w, channel_h = self.block_items['3-КАНАЛ С АБГШ'][1]
        add_arrow(modulator_x + modulator_w, modulator_y + modulator_h / 2, channel_x, channel_y + channel_h / 2,
                  "S_i(t)", 0, -20)

        # Локальная несущая для модулятора (входящая снизу)
        add_arrow(modulator_x + modulator_w / 2, modulator_y + modulator_h + 15,
                  modulator_x + modulator_w / 2, modulator_y + modulator_h, "-S₀·sin(ω₀t)", 40, 0)

        # Выход из КАНАЛА вправо с подписью y(t)
        add_arrow(channel_x + channel_w, channel_y + channel_h / 2, channel_x + channel_w + 60,
                  channel_y + channel_h / 2, "y(t)", 0, -20)

        # Вход в ДЕМОДУЛЯТОР слева с подписью y(t)
        demodulator_x, demodulator_y, demodulator_w, demodulator_h = self.block_items['4-ПЕРЕМНОЖИТЕЛЬ'][1]
        add_arrow(demodulator_x - 60, demodulator_y + demodulator_h / 2, demodulator_x,
                  demodulator_y + demodulator_h / 2, "y(t)", -40, -20)

        # Вход в ДЕМОДУЛЯТОР снизу с подписью S₀·sin(ω₀t) - сместил правее
        add_arrow(demodulator_x + demodulator_w / 2, demodulator_y + demodulator_h + 15,
                  demodulator_x + demodulator_w / 2, demodulator_y + demodulator_h, "S₀·sin(ω₀t)", 40, 10)

        # НИЖНИЙ РЯД - соединения между блоками
        # 4-ДЕМОДУЛЯТОР -> 5-ФНЧ
        fpf_x, fpf_y, fpf_w, fpf_h = self.block_items['5-ФНЧ'][1]
        add_arrow(demodulator_x + demodulator_w, demodulator_y + demodulator_h / 2,
                  fpf_x, fpf_y + fpf_h / 2)

        # 5-ФНЧ -> 6-ДЕЦИМАТОР
        decimator_x, decimator_y, decimator_w, decimator_h = self.block_items['6-ДЕЦИМАТОР'][1]
        add_arrow(fpf_x + fpf_w, fpf_y + fpf_h / 2,
                  decimator_x, decimator_y + decimator_h / 2)

        # 6-ДЕЦИМАТОР -> 7-РЕШАЮЩЕЕ УСТРОЙСТВО
        res_x, res_y, res_w, res_h = self.block_items['7-РЕШАЮЩЕЕ\nУСТРОЙСТВО'][1]
        add_arrow(decimator_x + decimator_w, decimator_y + decimator_h / 2,
                  res_x, res_y + res_h / 2)

        # Стрелка выхода из решающего устройства - подпись над стрелкой
        add_arrow(res_x + res_w, res_y + res_h / 2, res_x + res_w + 60, res_y + res_h / 2, "Выход", 0, -20)

        # сохраняем сцену
        self.scene = scene

    # ---------- Применение параметров
    def _on_apply_params(self):
        self.fs = int(self.input_fs.value())
        self.длительность = float(self.input_T.value())
        self.пс_частота = int(self.input_pn.value())
        self.частота_опорного = int(self.input_freq.value())
        self.фаза = float(self.input_phase.value()) * np.pi / 180.0
        self.фаза_оп = float(self.input_phase_op.value()) * np.pi / 180.0
        self.шум_std = float(self.input_noise.value())
        # self.дек_фактор = int(self.input_dec.value())
        # Рассчитываем фактор децимации автоматически
        if self.пс_частота > 0:
            # Округляем до ближайшего целого
            self.дек_фактор = int(round(self.fs / self.пс_частота))
            # Гарантируем, что фактор децимации >= 1
            self.дек_фактор = max(1, self.дек_фактор)
        else:
            self.дек_фактор = 1
        self.фильтр_срез = float(self.input_cut.value())
        self.N = int(self.длительность * self.fs)

        self._generate_all_signals()
        self._update_all_plots()
        self._update_error_stats()

    # ---------- Генерация сигналов
    def _generate_all_signals(self):
        # Временная шкала
        t = np.linspace(0, self.длительность, int(self.длительность * self.fs), endpoint=False)
        self.t = t
        N = len(t)

        # ПСП
        self.pn_sequence = generate_pn_sequence(N, self.пс_частота, [-1, 1], self.fs)

        # Синус
        self.sinusoid = generate_sinusoid(self.частота_опорного, self.фаза, self.fs, N)

        # Модуляция
        self.multiplied = self.pn_sequence * self.sinusoid

        # Шум
        self.noisy = add_gaussian_noise(self.multiplied, self.шум_std)

        # Опорный генератор
        self.reference = generate_sinusoid(self.частота_опорного, self.фаза_оп, self.fs, N)

        # Перемножитель демодуляции
        self.mixed = self.noisy * self.reference

        # Фильтрация
        self.filtered = butter_lowpass_filter(self.mixed, self.фильтр_срез, self.fs, order=5)
        # self.filtered = self.filtered - np.mean(self.filtered)

        # Децимация
        # self.decimated = decimate(self.filtered, self.дек_фактор)
        # self.decimated_t = self.t[::self.дек_фактор]
        samples_per_symbol = int(round(self.fs / self.пс_частота))
        offset = samples_per_symbol // 2

        self.decimated = self.filtered[offset::samples_per_symbol]
        self.decimated_t = self.t[offset::samples_per_symbol]

        # Решающее устройство
        self.limited = limiter(self.decimated)

        # BER
        orig = self.pn_sequence[::self.дек_фактор][:len(self.limited)]
        # Приведём значения оригинала к -1/1
        orig_bits = np.where(orig >= 0, 1, -1)
        errors = np.sum(orig_bits != self.limited)
        total = len(self.limited)
        self.ber = errors / total if total > 0 else 0.0

        # Энергия бита и Eb/N0
        signal_power = calculate_power(self.multiplied)
        bit_rate = self.пс_частота
        bit_energy = signal_power / bit_rate if bit_rate > 0 else 0
        noise_power = calculate_power(self.noisy - self.multiplied)
        bandwidth = self.fs / 2
        noise_density = noise_power / bandwidth if bandwidth > 0 else 1e-12
        if noise_density > 0:
            eb_no = 10 * np.log10(bit_energy / noise_density + 1e-30)
        else:
            eb_no = 0  # или другое значение по умолчанию
        self.eb_no = eb_no

    # ---------- Обновление всех графиков
    def _update_all_plots(self):
        self._plot_psp()
        self._plot_psp_psd()
        self._plot_oporny()
        self._plot_oporny_psd()
        self._plot_modulated()
        self._plot_modulated_psd()
        self._plot_channel()
        self._plot_channel_psd()
        self._plot_demod()
        self._plot_demod_psd()
        self._plot_lpf()
        self._plot_lpf_psd()
        self._plot_decimated()  # Старый график
        self._plot_decimated_overlay()  # Новый график с наложением
        self._plot_decider()
        self._plot_compare()
        self._update_eye_diagram()

    # ---------- Отдельные функции рисования графиков
    def _plot_psp(self):
        ax = self.psp_canvas.ax
        ax.clear()
        ax.step(self.t, self.pn_sequence, where='post')
        ax.set_title('Псевдослучайная последовательность')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Амплитуда')
        ax.grid(True)
        self.psp_canvas.draw()

    def _plot_psp_psd(self):
        ax = self.psp_psd_canvas.ax
        ax.clear()
        f, P = compute_psd(self.pn_sequence, self.fs)
        ax.plot(f, P)
        ax.set_title('Спектральная плотность мощности ПСП')
        ax.set_xlabel('Частота [Гц]')
        ax.set_ylabel('Плотность [dB]')
        ax.grid(True)
        # ax.set_xlim([-10, 100])
        # ax.set_ylim([-120, 10])
        self.psp_psd_canvas.draw()

    def _plot_oporny(self):
        ax = self.op_canvas.ax
        ax.clear()
        ax.plot(self.t, self.sinusoid)
        ax.set_title('Опорный сигнал')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Амплитуда')
        ax.grid(True)
        self.op_canvas.draw()

    def _plot_oporny_psd(self):
        ax = self.op_psd_canvas.ax
        ax.clear()
        f, P = compute_psd(self.sinusoid, self.fs)
        ax.plot(f, P)
        ax.set_title('Спектральная плотность мощности синусоида')
        ax.set_xlabel('Частота [Гц]')
        ax.set_ylabel('Плотность [dB]')
        ax.grid(True)
        self.op_psd_canvas.draw()

    def _plot_modulated(self):
        ax = self.mod_canvas.ax
        ax.clear()
        ax.plot(self.t, self.multiplied)
        ax.set_title('Модулированный сигнал')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Амплитуда')
        ax.grid(True)
        self.mod_canvas.draw()

    def _plot_modulated_psd(self):
        ax = self.mod_psd_canvas.ax
        ax.clear()
        f, P = compute_psd(self.multiplied, self.fs)
        ax.plot(f, P)
        ax.set_title('Спектральная плотность мощности модулированного сигнала')
        ax.set_xlabel('Частота [Гц]')
        ax.set_ylabel('Плотность [dB]')
        ax.grid(True)
        self.mod_psd_canvas.draw()

    def _plot_channel(self):
        ax = self.chan_canvas.ax
        ax.clear()
        ax.plot(self.t, self.noisy)
        ax.set_title('Сигнал в канале')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Амплитуда')
        ax.grid(True)
        self.chan_canvas.draw()

    def _plot_channel_psd(self):
        ax = self.chan_psd_canvas.ax
        ax.clear()
        f, P = compute_psd(self.noisy, self.fs)
        ax.plot(f, P)
        ax.set_title('Спектральная плотность мощности сигнала в канале')
        ax.set_xlabel('Частота [Гц]')
        ax.set_ylabel('Плотность [dB]')
        ax.grid(True)
        self.chan_psd_canvas.draw()

    def _plot_demod(self):
        ax = self.dem_canvas.ax
        ax.clear()
        ax.plot(self.t, self.mixed)
        ax.set_title('Выход перемножителя ')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Амплитуда')
        ax.grid(True)
        self.dem_canvas.draw()

    def _plot_demod_psd(self):
        ax = self.dem_psd_canvas.ax
        ax.clear()
        f, P = compute_psd(self.mixed, self.fs)
        ax.plot(f, P)
        ax.set_title('Спектральная плотность мощности на выходе перемножителя')
        ax.set_xlabel('Частота [Гц]')
        ax.set_ylabel('Плотность [dB]')
        ax.grid(True)
        self.dem_psd_canvas.draw()

    def _plot_lpf(self):
        ax = self.lpf_canvas.ax
        ax.clear()
        ax.plot(self.t, self.filtered)
        ax.set_title('Отфильтрованный сигнал')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Амплитуда')
        ax.grid(True)
        self.lpf_canvas.draw()

    def _plot_lpf_psd(self):
        ax = self.lpf_psd_canvas.ax
        ax.clear()
        f, P = compute_psd(self.filtered, self.fs)
        ax.plot(f, P)
        ax.set_title('Спектральная плотность мощности отфильтрованного сигнала')
        ax.set_xlabel('Частота [Гц]')
        ax.set_ylabel('Плотность [dB]')
        ax.grid(True)
        self.lpf_psd_canvas.draw()

    # def _plot_decimated(self):
    #     ax = self.dec_canvas.ax
    #     ax.clear()
    #     ax.plot(self.decimated_t, self.decimated, label='Децимированный')
    #     ax.scatter(self.decimated_t, self.decimated, s=10)
    #     ax.set_title('Децимированный сигнал')
    #     ax.set_xlabel('Время [c]')
    #     ax.set_ylabel('Амплитуда')
    #     ax.grid(True)
    #     self.dec_canvas.draw()
    def _plot_decimated(self):
        ax = self.dec_canvas.ax
        ax.clear()
        ax.plot(self.decimated_t, self.decimated, 'b-', label='Децимированный', linewidth=1)
        ax.plot(self.decimated_t, self.decimated, 'ro', markersize=4)
        ax.set_title('Децимированный сигнал')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Амплитуда')
        ax.legend()
        ax.grid(True)
        self.dec_canvas.draw()
    def _plot_decider(self):
        ax = self.decider_canvas.ax
        ax.clear()
        ax.step(self.decimated_t, self.limited, where='post')
        ax.scatter(self.decimated_t, self.limited, s=10)
        ax.set_title('После решающего устройства')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Уровень')
        ax.grid(True)
        self.decider_canvas.draw()

    def _plot_compare(self):
        # Обрежем длину
        L = min(len(self.pn_sequence[::self.дек_фактор]), len(self.limited))
        t = self.decimated_t[:L]
        orig = self.pn_sequence[::self.дек_фактор][:L]

        # ---- График 1: Исходная ПСП ----
        ax1 = self.compare_canvas_orig.ax
        ax1.clear()
        ax1.step(t, orig, where='post')
        ax1.set_title('Исходная ПСП')
        ax1.set_xlabel('Время [с]')
        ax1.set_ylabel('Уровень')
        ax1.grid(True)
        self.compare_canvas_orig.draw()

        # ---- График 2: После решающего устройства ----
        ax2 = self.compare_canvas_rec.ax
        ax2.clear()
        ax2.step(t, self.limited[:L], where='post')
        ax2.set_title('Восстановленная ПСП')
        ax2.set_xlabel('Время [с]')
        ax2.set_ylabel('Уровень')
        ax2.grid(True)
        self.compare_canvas_rec.draw()

        # Обновляем статистику ошибок
        self._update_error_stats()

    # ---------- Глаз-диаграмма (с использованием SpinBox вместо слайдеров)
    def _update_eye_diagram(self):
        # Получаем параметры из интерфейса
        phase = float(self.eye_phase.value()) * np.pi / 180.0
        phase_op = float(self.eye_phase_op.value()) * np.pi / 180.0
        noise_std = float(self.eye_noise.value())

        # Основные параметры (как в исходном коде)
        num_realizations = int(self.eye_realizations.value())
        fs = self.fs
        pn_rate = self.пс_частота
        N = self.N
        t = self.t

        # Временные параметры для сегмента (как в исходном коде)
        start_time = 0  # Начало интервала в секундах
        end_time = 2.2  # Конец интервала в секундах

        start_idx = 0
        end_idx = int(end_time * fs)

        samples_per_symbol = int(round(fs / pn_rate))
        t_symbol = np.linspace(0, 1 / pn_rate, samples_per_symbol, endpoint=False)

        ax = self.eye_canvas.ax
        ax.clear()

        # Цвета для разных реализаций (как в исходном коде)
        colors = plt.cm.viridis(np.linspace(0, 1, num_realizations))

        for realization_idx in range(num_realizations):
            # Генерация сигналов (ПОВТОРЯЕМ ЛОГИКУ ИСХОДНОГО КОДА)

            # 1. Генерация несущей с заданной фазой
            sinusoid_1 = generate_sinusoid(self.частота_опорного, phase, fs, N)

            # 2. Модуляция (умножение ПСП на несущую)
            pn_seq = generate_pn_sequence(N, pn_rate, [-1, 1], fs)
            multiplied_signal = pn_seq * sinusoid_1

            # 3. Добавление шума
            noisy_signal = add_gaussian_noise(multiplied_signal, noise_std, 0)

            # 4. Генерация опорного колебания
            reference_oscillation = generate_sinusoid(self.частота_опорного, phase_op, fs, N)

            # 5. Демодуляция (перемножение)
            mixed_signal = noisy_signal * reference_oscillation

            # 6. Фильтрация ФНЧ
            filtered_signal = butter_lowpass_filter(mixed_signal, self.фильтр_срез, fs, order=3)

            # 7. Выделение сегмента для глаз-диаграммы
            filtered_signal_segment = filtered_signal[start_idx:end_idx]

            # 8. Построение глаз-диаграммы (ОСНОВНАЯ ЧАСТЬ)
            num_segments = (len(filtered_signal_segment) - samples_per_symbol) // samples_per_symbol

            for idx, i in enumerate(range(0, len(filtered_signal_segment) - samples_per_symbol, samples_per_symbol)):
                color = plt.cm.viridis(idx / num_segments)

                ax.plot(
                    t_symbol,
                    filtered_signal_segment[i:i + samples_per_symbol],
                    color=color,
                    linewidth=0.5
                )


        # Настройка графика (как в исходном коде)
        ax.set_title('Глаз-диаграмма')
        ax.set_xlabel('Время [с]')
        ax.set_ylabel('Амплитуда')

        # Увеличение размера чисел сетки (как в исходном коде)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax.grid(True)

        self.eye_canvas.draw()

    def _build_tradeoff_tab(self):
        # Создаём вложенный QTabWidget
        self.tradeoff_tabs = QtWidgets.QTabWidget()

        # Создаём три подвкладки
        self.tradeoff_tab1 = QtWidgets.QWidget()
        self.tradeoff_tab2 = QtWidgets.QWidget()
        self.tradeoff_tab3 = QtWidgets.QWidget()

        # Добавляем подвкладки в основной виджет вкладки
        self.tradeoff_tabs.addTab(self.tradeoff_tab1, 'График 1: СКО шума → h')
        self.tradeoff_tabs.addTab(self.tradeoff_tab2, 'График 2: Δφ → h')
        self.tradeoff_tabs.addTab(self.tradeoff_tab3, 'График 3: Δφ → СКО шума')

        # Создаём макет для основной вкладки и добавляем в него вложенные вкладки
        main_layout = QtWidgets.QVBoxLayout(self.tab_tradeoff)
        main_layout.addWidget(self.tradeoff_tabs)

        # --- Заполняем первую подвкладку ---
        layout1 = QtWidgets.QVBoxLayout(self.tradeoff_tab1)

        self.trade_data1 = None

        # График 1
        self.canvas1 = MplCanvas(self, width=8, height=5)
        layout1.addWidget(self.canvas1)

        # Панель инструментов для графика 1
        self.trade_toolbar1 = NavigationToolbar(self.canvas1, self)
        layout1.addWidget(self.trade_toolbar1)

        # Кнопки для графика 1 (только задать таблицу и построить)
        btn_layout1 = QtWidgets.QHBoxLayout()

        btn_table1 = QtWidgets.QPushButton("Задать таблицу")
        btn_plot1 = QtWidgets.QPushButton("Построить график")

        btn_table1.clicked.connect(self._table_graph1)
        btn_plot1.clicked.connect(self._plot_graph1)

        btn_layout1.addWidget(btn_table1)
        btn_layout1.addWidget(btn_plot1)
        btn_layout1.addStretch()

        layout1.addLayout(btn_layout1)
        layout1.addStretch()

        # --- Заполняем вторую подвкладку ---
        layout2 = QtWidgets.QVBoxLayout(self.tradeoff_tab2)

        self.trade_data2 = None

        # График 2
        self.canvas2 = MplCanvas(self, width=8, height=5)
        layout2.addWidget(self.canvas2)

        # Панель инструментов для графика 2
        self.trade_toolbar2 = NavigationToolbar(self.canvas2, self)
        layout2.addWidget(self.trade_toolbar2)

        # Кнопки для графика 2
        btn_layout2 = QtWidgets.QHBoxLayout()

        btn_table2 = QtWidgets.QPushButton("Задать таблицу")
        btn_plot2 = QtWidgets.QPushButton("Построить график")

        btn_table2.clicked.connect(self._table_graph2)
        btn_plot2.clicked.connect(self._plot_graph2)

        btn_layout2.addWidget(btn_table2)
        btn_layout2.addWidget(btn_plot2)
        btn_layout2.addStretch()

        layout2.addLayout(btn_layout2)
        layout2.addStretch()

        # --- Заполняем третью подвкладку ---
        layout3 = QtWidgets.QVBoxLayout(self.tradeoff_tab3)

        self.trade_data3 = None

        # График 3
        self.canvas3 = MplCanvas(self, width=8, height=5)
        layout3.addWidget(self.canvas3)

        # Панель инструментов для графика 3
        self.trade_toolbar3 = NavigationToolbar(self.canvas3, self)
        layout3.addWidget(self.trade_toolbar3)

        # Кнопки для графика 3
        btn_layout3 = QtWidgets.QHBoxLayout()

        btn_table3 = QtWidgets.QPushButton("Задать таблицу")
        btn_plot3 = QtWidgets.QPushButton("Построить график")

        btn_table3.clicked.connect(self._table_graph3)
        btn_plot3.clicked.connect(self._plot_graph3)

        btn_layout3.addWidget(btn_table3)
        btn_layout3.addWidget(btn_plot3)
        btn_layout3.addStretch()

        layout3.addLayout(btn_layout3)
        layout3.addStretch()

    def _table_graph1(self):
        """Открывает диалог для графика 1 с сохранением предыдущих данных"""
        dlg = TableInputDialog("СКО шума", "h", self, self.trade_data1)
        if dlg.exec():
            self.trade_data1 = dlg.get_data()

    def _table_graph2(self):
        """Открывает диалог для графика 2 с сохранением предыдущих данных"""
        dlg = TableInputDialog("Δφ", "h", self, self.trade_data2)
        if dlg.exec():
            self.trade_data2 = dlg.get_data()

    def _table_graph3(self):
        """Открывает диалог для графика 3 с сохранением предыдущих данных"""
        dlg = TableInputDialog("Δφ", "СКО шума", self, self.trade_data3)
        if dlg.exec():
            self.trade_data3 = dlg.get_data()

    def _plot_graph1(self):
        if self.trade_data1 is None or len(self.trade_data1[0]) == 0:
            QtWidgets.QMessageBox.warning(self, "Предупреждение",
                                          "Нет данных для построения графика. Сначала задайте таблицу.")
            return

        x, y = self.trade_data1

        ax = self.canvas1.ax
        ax.clear()

        # Строим график с лейблами
        line, = ax.plot(x, y, 'o-', linewidth=2, markersize=8,
                        label='Экспериментальные точки', color='blue')

        # Добавляем подписи к точкам (значения)
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(f'({xi:.3f}, {yi:.3f})',
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Настройка осей
        ax.set_xlabel("СКО шума", fontsize=12, fontweight='bold')
        ax.set_ylabel("h", fontsize=12, fontweight='bold')

        # Добавляем заголовок
        ax.set_title('Зависимость h от СКО шума', fontsize=14, fontweight='bold')

        # Добавляем легенду
        ax.legend(loc='best', fontsize=10)

        # Добавляем сетку
        ax.grid(True, alpha=0.3)

        # Добавляем подписи минимальных и максимальных значений
        if len(x) > 0:
            max_idx = np.argmax(y)
            min_idx = np.argmin(y)

            ax.annotate(f'Max: ({x[max_idx]:.3f}, {y[max_idx]:.3f})',
                        (x[max_idx], y[max_idx]),
                        textcoords="offset points",
                        xytext=(0, -20),
                        ha='center',
                        fontsize=9,
                        color='green',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

            ax.annotate(f'Min: ({x[min_idx]:.3f}, {y[min_idx]:.3f})',
                        (x[min_idx], y[min_idx]),
                        textcoords="offset points",
                        xytext=(0, 20),
                        ha='center',
                        fontsize=9,
                        color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

        self.canvas1.draw()

    def _plot_graph2(self):
        if self.trade_data2 is None or len(self.trade_data2[0]) == 0:
            QtWidgets.QMessageBox.warning(self, "Предупреждение",
                                          "Нет данных для построения графика. Сначала задайте таблицу.")
            return

        x, y = self.trade_data2

        ax = self.canvas2.ax
        ax.clear()

        # Строим график с лейблами
        line, = ax.plot(x, y, 's-', linewidth=2, markersize=8,
                        label='Экспериментальные точки', color='red')

        # Добавляем подписи к точкам (значения)
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(f'({xi:.3f}, {yi:.3f})',
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Настройка осей
        ax.set_xlabel("Δφ (разность фаз)", fontsize=12, fontweight='bold')
        ax.set_ylabel("h", fontsize=12, fontweight='bold')

        # Добавляем заголовок
        ax.set_title('Зависимость h от разности фаз', fontsize=14, fontweight='bold')

        # Добавляем легенду
        ax.legend(loc='best', fontsize=10)

        # Добавляем сетку
        ax.grid(True, alpha=0.3)

        # Добавляем подписи минимальных и максимальных значений
        if len(x) > 0:
            max_idx = np.argmax(y)
            min_idx = np.argmin(y)

            ax.annotate(f'Max: ({x[max_idx]:.3f}, {y[max_idx]:.3f})',
                        (x[max_idx], y[max_idx]),
                        textcoords="offset points",
                        xytext=(0, -20),
                        ha='center',
                        fontsize=9,
                        color='green',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

            ax.annotate(f'Min: ({x[min_idx]:.3f}, {y[min_idx]:.3f})',
                        (x[min_idx], y[min_idx]),
                        textcoords="offset points",
                        xytext=(0, 20),
                        ha='center',
                        fontsize=9,
                        color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

        self.canvas2.draw()

    def _plot_graph3(self):
        if self.trade_data3 is None or len(self.trade_data3[0]) == 0:
            QtWidgets.QMessageBox.warning(self, "Предупреждение",
                                          "Нет данных для построения графика. Сначала задайте таблицу.")
            return

        x, y = self.trade_data3

        ax = self.canvas3.ax
        ax.clear()

        # Строим график с лейблами
        line, = ax.plot(x, y, '^-', linewidth=2, markersize=8,
                        label='Экспериментальные точки', color='green')

        # Добавляем подписи к точкам (значения)
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(f'({xi:.3f}, {yi:.3f})',
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Настройка осей
        ax.set_xlabel("Δφ (разность фаз)", fontsize=12, fontweight='bold')
        ax.set_ylabel("СКО шума", fontsize=12, fontweight='bold')

        # Добавляем заголовок
        ax.set_title('Диаграмма обмена', fontsize=14, fontweight='bold')

        # Добавляем легенду
        ax.legend(loc='best', fontsize=10)

        # Добавляем сетку
        ax.grid(True, alpha=0.3)

        # Добавляем подписи минимальных и максимальных значений
        if len(x) > 0:
            max_idx = np.argmax(y)
            min_idx = np.argmin(y)

            ax.annotate(f'Max: ({x[max_idx]:.3f}, {y[max_idx]:.3f})',
                        (x[max_idx], y[max_idx]),
                        textcoords="offset points",
                        xytext=(0, -20),
                        ha='center',
                        fontsize=9,
                        color='green',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

            ax.annotate(f'Min: ({x[min_idx]:.3f}, {y[min_idx]:.3f})',
                        (x[min_idx], y[min_idx]),
                        textcoords="offset points",
                        xytext=(0, 20),
                        ha='center',
                        fontsize=9,
                        color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

        self.canvas3.draw()


# ---------- Запуск приложения
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
