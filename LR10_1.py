
import sys
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
from scipy.signal import butter, lfilter, welch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 50  # Увеличиваем лимит


class FontDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, font_size=14, parent=None):
        super().__init__(parent)
        self.font_size = font_size

    def createEditor(self, parent, option, index):
        editor = QtWidgets.QLineEdit(parent)
        font = QtGui.QFont()
        font.setPointSize(self.font_size)
        font.setBold(False)
        editor.setFont(font)
        editor.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        return editor

    def setEditorData(self, editor, index):
        text = index.data(QtCore.Qt.ItemDataRole.EditRole)
        if text:
            editor.setText(text)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.text())

    def paint(self, painter, option, index):
        # Настраиваем шрифт для отображения
        option.font = QtGui.QFont()
        option.font.setPointSize(self.font_size)
        option.font.setBold(False)
        option.displayAlignment = QtCore.Qt.AlignmentFlag.AlignCenter
        super().paint(painter, option, index)

class TableInputDialog(QtWidgets.QDialog):
    def __init__(self, x_name, y_name, parent=None, saved_data=None):
        super().__init__(parent)

        self.setWindowTitle("Таблица значений")
        self.resize(500, 550)



        layout = QtWidgets.QVBoxLayout(self)

        # Создаем таблицу
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setRowCount(5)  # Начальное количество строк
        self.table.setHorizontalHeaderLabels([x_name, y_name])

        delegate = FontDelegate(font_size=14)
        self.table.setItemDelegate(delegate)

        # === НАСТРОЙКИ ДЛЯ ШРИФТА И ВЫРАВНИВАНИЯ ===
        # Создаем шрифт для ячеек
        # cell_font = QtGui.QFont()
        # cell_font.setPointSize(14)  # Размер шрифта
        # cell_font.setBold(False)  # Жирный или нет
        # self.table.setFont(cell_font)

        # === НАСТРОЙКИ ДЛЯ ШРИФТА ЗАГОЛОВКОВ И НУМЕРАЦИИ ===
        # Увеличиваем шрифт для заголовков столбцов (горизонтальные)
        header_font = QtGui.QFont()
        header_font.setPointSize(16)  # Увеличенный размер
        header_font.setBold(True)
        self.table.horizontalHeader().setFont(header_font)

        # Увеличиваем шрифт для нумерации строк (вертикальные заголовки)
        row_font = QtGui.QFont()
        row_font.setPointSize(16)  # Увеличенный размер
        row_font.setBold(True)
        self.table.verticalHeader().setFont(row_font)

        # Увеличиваем высоту заголовков (чтобы текст помещался)
        self.table.horizontalHeader().setMinimumHeight(30)
        self.table.verticalHeader().setDefaultSectionSize(30)  # Высота строк для нумерации

        # Устанавливаем выравнивание для всех ячеек по центру
        for row in range(self.table.rowCount()):
            for col in range(2):
                item = QtWidgets.QTableWidgetItem()
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)
        # === НОВЫЕ НАСТРОЙКИ ДЛЯ РАСТЯГИВАНИЯ ТАБЛИЦЫ ===
        # Растягивать столбцы на всю доступную ширину
        self.table.horizontalHeader().setStretchLastSection(True)  # Растянуть последний столбец
        # Или для обоих столбцов равномерно:
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

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

        # Изменяем текст кнопок
        btn_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Сохранить")
        btn_box.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("Отмена")

        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)



    def _set_table_item(self, row, col, value):
        """Устанавливает значение ячейки с форматированием до 1 знаков"""
        item = QtWidgets.QTableWidgetItem(f"{value:.1f}")
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # Выравнивание по центру
        # Устанавливаем шрифт для ячейки
        # font = QtGui.QFont()
        # font.setPointSize(16)
        # item.setFont(font)
        self.table.setItem(row, col, item)

    def _add_row(self):
        """Добавляет новую строку в таблицу"""
        current_row = self.table.rowCount()
        self.table.setRowCount(current_row + 1)

        # Настраиваем новые ячейки
        for col in range(2):
            item = QtWidgets.QTableWidgetItem()
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            # font = QtGui.QFont()
            # font.setPointSize(16)
            # item.setFont(font)
            self.table.setItem(current_row, col, item)

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

        # Получаем названия столбцов с проверкой
        x_item = self.table.horizontalHeaderItem(0)
        y_item = self.table.horizontalHeaderItem(1)

        x_name = x_item.text() if x_item else "X"
        y_name = y_item.text() if y_item else "Y"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Сохранить точки", "", "CSV файлы (*.csv);;Все файлы (*.*)"
        )

        if path:
            if not path.endswith('.csv'):
                path += '.csv'

            try:
                # Сохраняем точки в CSV
                data = np.column_stack((x, y))
                np.savetxt(path, data, delimiter=";", fmt='%.1f',
                           header=f"{x_name};{y_name}", comments='')
                QtWidgets.QMessageBox.information(self, "Успех",
                                                  f"Точки сохранены в файл:\n{path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Ошибка",
                                               f"Ошибка при сохранении:\n{str(e)}")

    def get_data(self):
        """Возвращает данные из таблицы"""
        x = []
        y = []
        rows = self.table.rowCount()

        for r in range(rows):
            item_x = self.table.item(r, 0)
            item_y = self.table.item(r, 1)

            if item_x and item_y:
                text_x = item_x.text()
                text_y = item_y.text()

                if text_x and text_y:  # Проверяем что текст не пустой
                    try:
                        text_x = text_x.replace(',', '.')
                        text_y = text_y.replace(',', '.')

                        xv = float(text_x)
                        yv = float(text_y)
                        x.append(xv)
                        y.append(yv)
                    except ValueError:
                        # Пропускаем некорректные значения
                        pass

        return np.array(x), np.array(y)
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
    y = lfilter(b, a, data)
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
        self.fig, self.ax = plt.subplots(
            figsize=(width, height),
            dpi=dpi,
            constrained_layout=True  # 🔥 ВАЖНО
        )
        super().__init__(self.fig)

# ---------- Главное окно
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Формирование и демодуляция сигналов ФМ2')
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
        self.tab_tradeoff = QtWidgets.QWidget()  # ← ДОБАВЬТЕ ЭТУ СТРОКУ

        # tabs.addTab(self.tab_params, 'Параметры')
        # self.tab_part1 = QtWidgets.QWidget()
        # tabs.addTab(self.tab_part1, 'Часть 1 (Моделирование)')
        # # tabs.addTab(self.tab_psp, '2. ПСП')
        # # tabs.addTab(self.tab_oporny, '3. Опорный сигнал')
        # # tabs.addTab(self.tab_modulator, '4. Модулятор')
        # # tabs.addTab(self.tab_channel, '5. Канал')
        # # tabs.addTab(self.tab_demod, '6. Выход перемножителя')
        # # tabs.addTab(self.tab_lpf, '7. ФНЧ')
        # # tabs.addTab(self.tab_decim, '8. Децимация')
        # # tabs.addTab(self.tab_decider, '9. Решающее устройство')
        # # tabs.addTab(self.tab_compare, '10. Сравнение')
        # tabs.addTab(self.tab_eye, 'Часть 2 (Глаз-диаграмма)')
        # tabs.addTab(self.tab_tradeoff, 'Часть 3 (Построение графиков)')
        #
        # part1_layout = QtWidgets.QVBoxLayout(self.tab_part1)
        # self.part1_tabs = QtWidgets.QTabWidget()
        # part1_layout.addWidget(self.part1_tabs)
        #
        # self.part1_tabs.addTab(self.tab_psp, 'ПСП')
        # self.part1_tabs.addTab(self.tab_oporny, 'Опорный сигнал')
        # self.part1_tabs.addTab(self.tab_modulator, 'Модулятор')
        # self.part1_tabs.addTab(self.tab_channel, 'Канал')
        # self.part1_tabs.addTab(self.tab_demod, 'Выход перемножителя')
        # self.part1_tabs.addTab(self.tab_lpf, 'ФНЧ')
        # self.part1_tabs.addTab(self.tab_decim, 'Децимация')
        # self.part1_tabs.addTab(self.tab_decider, 'Решающее устройство')
        # self.part1_tabs.addTab(self.tab_compare, 'Сравнение')
        tabs.addTab(self.tab_params, 'Параметры')
        self.tab_part1 = QtWidgets.QWidget()
        tabs.addTab(self.tab_part1, 'Часть 1 (Моделирование)')
        tabs.addTab(self.tab_eye, 'Часть 2 (Глаз-диаграмма)')
        tabs.addTab(self.tab_tradeoff, 'Часть 3 (Построение графиков)')

        # === НОВАЯ СТРУКТУРА ДЛЯ ЧАСТИ 1 ===
        part1_layout = QtWidgets.QVBoxLayout(self.tab_part1)
        self.part1_tabs = QtWidgets.QTabWidget()
        part1_layout.addWidget(self.part1_tabs)

        # 1. Модулятор (содержит ПСП, Опорный сигнал, Модулятор)
        modulator_tab = QtWidgets.QWidget()
        modulator_subtabs = QtWidgets.QTabWidget()
        modulator_layout = QtWidgets.QVBoxLayout(modulator_tab)
        modulator_layout.addWidget(modulator_subtabs)

        modulator_subtabs.addTab(self.tab_psp, 'ПСП')
        modulator_subtabs.addTab(self.tab_oporny, 'Опорный сигнал')
        modulator_subtabs.addTab(self.tab_modulator, 'Выход модулятора сигнал 2ФМ')

        # 2. Канал
        channel_tab = QtWidgets.QWidget()
        channel_layout = QtWidgets.QVBoxLayout(channel_tab)
        channel_layout.addWidget(self.tab_channel)

        # 3. Демодулятор (содержит Выход перемножителя, ФНЧ, Децимация, Решающее устройство)
        demodulator_tab = QtWidgets.QWidget()
        demodulator_subtabs = QtWidgets.QTabWidget()
        demodulator_layout = QtWidgets.QVBoxLayout(demodulator_tab)
        demodulator_layout.addWidget(demodulator_subtabs)

        demodulator_subtabs.addTab(self.tab_demod, 'Выход перемножителя демодулятора')
        demodulator_subtabs.addTab(self.tab_lpf, 'ФНЧ')
        demodulator_subtabs.addTab(self.tab_decim, 'Децимация')
        demodulator_subtabs.addTab(self.tab_decider, 'Решающее устройство')

        # 4. Сравнение (отдельная вкладка)
        compare_tab = QtWidgets.QWidget()
        compare_layout = QtWidgets.QVBoxLayout(compare_tab)
        compare_layout.addWidget(self.tab_compare)

        # Добавляем все четыре вкладки в part1_tabs
        self.part1_tabs.addTab(modulator_tab, 'Модулятор')
        self.part1_tabs.addTab(channel_tab, 'Канал')
        self.part1_tabs.addTab(demodulator_tab, 'Демодулятор')
        self.part1_tabs.addTab(compare_tab, 'Сравнение')

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

        # Прогресс-бар
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)  # Скрыт по умолчанию
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid gray;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Первичная генерация сигналов
        # self._generate_all_signals()
        # self._update_all_plots()

    def set_controls_enabled(self, enabled):
        """Блокирует/разблокирует элементы управления"""
        # Блокируем все вкладки
        self.tab_params.setEnabled(enabled)
        self.part1_tabs.setEnabled(enabled)
        self.tab_eye.setEnabled(enabled)
        self.tab_tradeoff.setEnabled(enabled)

        # Блокируем кнопки на вкладке параметров
        if hasattr(self, 'input_fs'):
            self.input_fs.setEnabled(enabled)
            self.input_T.setEnabled(enabled)
            self.input_pn.setEnabled(enabled)
            self.input_freq.setEnabled(enabled)
            self.input_phase.setEnabled(enabled)
            self.input_phase_op.setEnabled(enabled)
            self.input_noise.setEnabled(enabled)
            self.input_cut.setEnabled(enabled)

        # Находим и блокируем кнопку "Применить параметры"
        for widget in self.findChildren(QtWidgets.QPushButton):
            if widget.text() == 'Применить параметры и пересчитать':
                widget.setEnabled(enabled)

        # Блокируем кнопки на вкладке глаз-диаграммы
        if hasattr(self, 'eye_phase'):
            self.eye_phase.setEnabled(enabled)
            self.eye_phase_op.setEnabled(enabled)
            self.eye_noise.setEnabled(enabled)
            self.eye_realizations.setEnabled(enabled)

        # Находим кнопку "Пересчитать" на глаз-диаграмме
        for widget in self.findChildren(QtWidgets.QPushButton):
            if widget.text() == 'Пересчитать':
                widget.setEnabled(enabled)

        # Блокируем кнопки на вкладке tradeoff
        if hasattr(self, 'tradeoff_tabs'):
            self.tradeoff_tabs.setEnabled(enabled)
    # ---------- Построение вкладки Параметры
    def _build_params_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_params)

        # # Заголовок
        # title = QtWidgets.QLabel("Формирование и демодуляция сигналов ФМ2")
        # title_font = QtGui.QFont()
        # title_font.setPointSize(26)
        # title_font.setBold(True)
        # title.setFont(title_font)
        # title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # layout.addWidget(title)

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
                font-size: 16pt;
            }
        """)

        # ---------- Поля ввода
        spin_font = QtGui.QFont()
        spin_font.setPointSize(7)  # Размер текста в SpinBox

        label_font = QtGui.QFont()
        label_font.setPointSize(18)  # Размер текста подписей
        label_font.setBold(True)

        # Частота дискретизации
        self.input_fs = QtWidgets.QSpinBox()
        self.input_fs.setRange(100, 200000)
        self.input_fs.setValue(self.fs)
        self.input_fs.setFont(spin_font)
        lbl_fs = QtWidgets.QLabel('Частота дискретизации [Гц]:')
        lbl_fs.setFont(label_font)


        # Длительность моделирования
        self.input_T = QtWidgets.QDoubleSpinBox()
        self.input_T.setRange(0.1, 3600.0)
        self.input_T.setValue(self.длительность)
        self.input_T.setDecimals(0)
        self.input_T.setFont(spin_font)
        lbl_T = QtWidgets.QLabel('Длительность моделирования [с]:')
        lbl_T.setFont(label_font)


        # Частота ПСП
        self.input_pn = QtWidgets.QSpinBox()
        self.input_pn.setRange(1, 1000)
        self.input_pn.setValue(self.пс_частота)
        self.input_pn.setFont(spin_font)
        lbl_pn = QtWidgets.QLabel('Частота ПСП [Гц]:')
        lbl_pn.setFont(label_font)


        # Частота опорного сигнала
        self.input_freq = QtWidgets.QSpinBox()
        self.input_freq.setRange(1, 50000)
        self.input_freq.setValue(self.частота_опорного)
        self.input_freq.setFont(spin_font)
        lbl_freq = QtWidgets.QLabel('Частота немодулированной несущей [Гц]:')
        lbl_freq.setFont(label_font)


        # Фаза сигнала
        self.input_phase = QtWidgets.QDoubleSpinBox()
        self.input_phase.setRange(0, 360)
        self.input_phase.setValue(self.фаза)
        self.input_phase.setFont(spin_font)
        self.input_phase.setDecimals(0)
        lbl_phase = QtWidgets.QLabel('Фаза сигнала [градусы]:')
        lbl_phase.setFont(label_font)


        # Фаза опорного
        self.input_phase_op = QtWidgets.QDoubleSpinBox()
        self.input_phase_op.setRange(0, 360)
        self.input_phase_op.setValue(self.фаза_оп)
        self.input_phase_op.setDecimals(0)
        self.input_phase_op.setFont(spin_font)
        lbl_phase_op = QtWidgets.QLabel('Фаза немодулированной несущей [градусы]:')



        # Шум
        self.input_noise = QtWidgets.QDoubleSpinBox()
        self.input_noise.setRange(0.0, 100.0)
        self.input_noise.setDecimals(1)
        self.input_noise.setValue(self.шум_std)
        self.input_noise.setFont(spin_font)
        lbl_noise = QtWidgets.QLabel('СКО шума:')
        lbl_noise.setFont(label_font)


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
        self.input_cut.setDecimals(0)
        self.input_cut.setValue(self.фильтр_срез)
        self.input_cut.setFont(spin_font)
        lbl_cut = QtWidgets.QLabel('Частота среза ФНЧ [Гц]:')
        lbl_cut.setFont(label_font)


        group_box_layout.addRow(lbl_pn, self.input_pn)
        group_box_layout.addRow(lbl_pn, self.input_pn)
        group_box_layout.addRow(lbl_freq, self.input_freq)
        group_box_layout.addRow(lbl_fs, self.input_fs)
        group_box_layout.addRow(lbl_phase_op, self.input_phase_op)
        group_box_layout.addRow(lbl_phase, self.input_phase)
        group_box_layout.addRow(lbl_noise, self.input_noise)
        lbl_phase_op.setFont(label_font)
        group_box_layout.addRow(lbl_T, self.input_T)
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

        ax.set_title('Наложение: процесс на выходе ФНЧ + выборки процесс нв выходе ФНЧ')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Мгновенное значение')
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

        # Поле BER
        # self.ber_label = QtWidgets.QLabel('BER: -')
        # layout.addWidget(self.ber_label)
        # Группа для отображения статистики ошибок
        # stats_group = QtWidgets.QGroupBox("Статистика ошибок")
        # stats_layout = QtWidgets.QFormLayout()
        # stats_group.setLayout(stats_layout)
        #
        # # Стилизация группы
        # stats_group.setStyleSheet("""
        #             QGroupBox {
        #                 border: 2px solid gray;
        #                 border-radius: 10px;
        #                 margin-top: 10px;
        #                 font-weight: bold;
        #             }
        #             QGroupBox:title {
        #                 subcontrol-origin: margin;
        #                 subcontrol-position: top center;
        #                 padding: 5px;
        #             }
        #         """)
        #
        # # Поля для отображения статистики
        # self.errors_label = QtWidgets.QLabel('0')
        # self.errors_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
        # stats_layout.addRow('Количество ошибок:', self.errors_label)
        #
        # self.ber_label = QtWidgets.QLabel('0.000000')
        # self.ber_label.setStyleSheet("color: blue; font-weight: bold; font-size: 14px;")
        # stats_layout.addRow('BER (Bit Error Rate):', self.ber_label)
        #
        # self.total_bits_label = QtWidgets.QLabel('0')
        # self.total_bits_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        # stats_layout.addRow('Всего бит:', self.total_bits_label)
        #
        # self.accuracy_label = QtWidgets.QLabel('100.00%')
        # self.accuracy_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        # stats_layout.addRow('Точность:', self.accuracy_label)
        #
        # layout.addWidget(stats_group)
        #
        # layout.addStretch()
    # def _update_error_stats(self):
    #     """Обновляет отображение статистики ошибок"""
    #     # Получаем исходную и восстановленную ПСП
    #     orig = self.pn_sequence[::self.дек_фактор][:len(self.limited)]
    #     orig_bits = np.where(orig >= 0, 1, -1)
    #
    #     # Считаем ошибки
    #     errors = np.sum(orig_bits != self.limited)
    #     total = len(self.limited)
    #     ber = errors / total if total > 0 else 0
    #     accuracy = (1 - ber) * 100
    #
    #     # Обновляем метки
    #     self.errors_label.setText(str(errors))
    #     self.ber_label.setText(f'{ber:.8f}')
    #     self.total_bits_label.setText(str(total))
    #     self.accuracy_label.setText(f'{accuracy:.2f}%')
    #
    #     # Меняем цвет в зависимости от количества ошибок
    #     if errors == 0:
    #         self.errors_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
    #         self.ber_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
    #     else:
    #         self.errors_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
    #         self.ber_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")

    # ---------- Глаз-диаграмма
    def _build_eye_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_eye)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(2)  # Уменьшаем расстояние между элементами
        layout.addLayout(controls)

        # Фаза сигнала
        phase_label = QtWidgets.QLabel('Фаза сигнала [градусы]:')
        controls.addWidget(phase_label)
        self.eye_phase = QtWidgets.QDoubleSpinBox()
        self.eye_phase.setRange(0, 360)
        self.eye_phase.setValue(0)
        self.eye_phase.setDecimals(0)
        # self.eye_phase.setMaximumWidth(80)  # Ограничиваем ширину

        controls.addWidget(self.eye_phase)

        controls.addSpacing(40)  # Небольшой отступ между группами

        # Фаза опорного
        phase_op_label = QtWidgets.QLabel('Фаза опорного, °:')
        # controls.addWidget(phase_op_label)
        self.eye_phase_op = QtWidgets.QDoubleSpinBox()
        self.eye_phase_op.setRange(0, 360)
        self.eye_phase_op.setValue(0)
        self.eye_phase_op.setMaximumWidth(80)
        # controls.addWidget(self.eye_phase_op)

        controls.addSpacing(90)

        # СКО шума
        noise_label = QtWidgets.QLabel('СКО шума:')
        controls.addWidget(noise_label)
        self.eye_noise = QtWidgets.QDoubleSpinBox()
        self.eye_noise.setRange(0.0, 100.0)
        self.eye_noise.setDecimals(1)
        self.eye_noise.setValue(0)
        self.eye_noise.setSingleStep(0.1)
        self.eye_noise.setMaximumWidth(80)
        controls.addWidget(self.eye_noise)

        controls.addSpacing(90)

        # Длина реализации
        realizations_label = QtWidgets.QLabel('Длина реализации:')
        controls.addWidget(realizations_label)
        self.eye_realizations = QtWidgets.QSpinBox()
        self.eye_realizations.setRange(1, 1000)
        self.eye_realizations.setValue(10)
        self.eye_realizations.setSuffix(' символов')
        self.eye_realizations.setObjectName("wide_spinbox")  # 👈 ДОБАВЬТЕ ЭТУ СТРОКУ
        controls.addWidget(self.eye_realizations)

        controls.addSpacing(90)


        # Кнопка
        btn_replot = QtWidgets.QPushButton('Пересчитать')
        btn_replot.clicked.connect(self._update_eye_diagram)
        controls.addWidget(btn_replot)

        controls.addSpacing(90)

        # 👇 ДОБАВЬТЕ ЧЕКБОКС "ДЛЯ ПЕЧАТИ"
        self.print_mode_checkbox = QtWidgets.QCheckBox('Для печати (ч-б)')
        self.print_mode_checkbox.setToolTip('Переключить в черно-белый режим для печати')
        self.print_mode_checkbox.stateChanged.connect(self._toggle_print_mode)  # Сигнал при изменении
        controls.addWidget(self.print_mode_checkbox)

        controls.addStretch()  # Растяжка справа

        # Полотно для глаз-диаграммы
        self.eye_canvas = MplCanvas(self, width=10, height=6)
        layout.addWidget(self.eye_canvas)
        self.eye_toolbar = NavigationToolbar(self.eye_canvas, self)
        layout.addWidget(self.eye_toolbar)

    def _toggle_print_mode(self):
        """Переключает режим печати и перерисовывает текущую диаграмму в черно-белом режиме"""
        if self.last_eye_params is not None:
            # Блокируем интерфейс и показываем прогресс-бар
            self.set_controls_enabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Используем QTimer для асинхронной перерисовки
            QtCore.QTimer.singleShot(100, self._do_redraw_eye_diagram)

    def _do_redraw_eye_diagram(self):
        """Асинхронная перерисовка глаз-диаграммы"""
        self._redraw_eye_diagram()

    def _redraw_eye_diagram(self):
        """Перерисовывает глаз-диаграмму с текущими параметрами и учетом режима печати"""
        if self.last_eye_params is None:
            self._finish_processing()
            return

        try:
            # Получаем сохраненные параметры
            phase = self.last_eye_params['phase']
            phase_op = self.last_eye_params['phase_op']
            noise_std = self.last_eye_params['noise_std']
            num_realizations = self.last_eye_params['num_realizations']

            self.progress_bar.setValue(10)
            QtWidgets.QApplication.processEvents()

            # Проверяем режим печати
            print_mode = self.print_mode_checkbox.isChecked()

            # Основные параметры
            fs = self.fs
            pn_rate = self.пс_частота
            N = self.N

            # Временные параметры для сегмента
            start_time = 0.035
            end_time = 2.2

            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)

            samples_per_symbol = int(fs / pn_rate)
            t_symbol = np.linspace(0, 1 / pn_rate, samples_per_symbol, endpoint=False)

            ax = self.eye_canvas.ax
            ax.clear()

            self.progress_bar.setValue(20)
            QtWidgets.QApplication.processEvents()

            for realization_idx in range(num_realizations):
                # Обновляем прогресс каждые 10% реализаций
                if realization_idx % max(1, num_realizations // 10) == 0:
                    progress = 20 + int(70 * realization_idx / num_realizations)
                    self.progress_bar.setValue(progress)
                    QtWidgets.QApplication.processEvents()

                # 1. Генерация несущей с заданной фазой
                sinusoid_1 = generate_sinusoid(self.частота_опорного, phase, fs, N)

                # 2. Модуляция (умножение ПСП на несущую)
                multiplied_signal = self.pn_sequence * sinusoid_1

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

                # 8. Построение глаз-диаграммы
                for i in range(0, len(filtered_signal_segment) - samples_per_symbol, samples_per_symbol):
                    if print_mode:
                        # Режим для печати - градиент серого
                        gray_intensity = 0.3 + (i / (len(filtered_signal_segment) - samples_per_symbol)) * 0.5
                        ax.plot(t_symbol, filtered_signal_segment[i:i + samples_per_symbol],
                                color=(gray_intensity, gray_intensity, gray_intensity),
                                linewidth=0.5, alpha=0.5)
                    else:
                        # Обычный режим - цветной (viridis)
                        color = plt.cm.viridis(realization_idx / num_realizations)
                        ax.plot(t_symbol, filtered_signal_segment[i:i + samples_per_symbol],
                                color=color, linewidth=0.5)

            self.progress_bar.setValue(95)
            QtWidgets.QApplication.processEvents()

            # Настройка графика
            ax.set_title('Глаз-диаграмма')
            ax.set_xlabel('Время [с]')
            ax.set_ylabel('')
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            ax.grid(True)

            ax.axhline(y=0, color='red' if not print_mode else 'black',
                       linewidth=3, linestyle='-', alpha=0.5)

            # Убрать отступы
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.set_xlim(left=0)

            self.eye_canvas.draw()
            self.progress_bar.setValue(100)
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка",
                                           f"Ошибка при перерисовке глаз-диаграммы:\n{str(e)}")
        finally:
            # Разблокируем интерфейс после завершения
            QtCore.QTimer.singleShot(300, self._finish_processing)
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
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))  # Белый
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black)
        pen.setWidth(2)

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
            text_item.setDefaultTextColor(QtCore.Qt.GlobalColor.black)  # Белый текст на темном фоне
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
                  modulator_x + modulator_w / 2, modulator_y + modulator_h, "S₀·sin(ω₀t)", 40, 0)

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
        # Блокируем интерфейс и показываем прогресс-бар
        self.set_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Используем QTimer для имитации прогресса (или реального прогресса)
        QtCore.QTimer.singleShot(100, self._do_apply_params)

    def _do_apply_params(self):
        """Реальная обработка параметров с обновлением прогресса"""
        try:
            # Шаг 1: Чтение параметров (10%)
            self.progress_bar.setValue(10)
            QtWidgets.QApplication.processEvents()  # Обновляем UI

            self.fs = int(self.input_fs.value())
            self.длительность = float(self.input_T.value())
            self.пс_частота = int(self.input_pn.value())
            self.частота_опорного = int(self.input_freq.value())
            self.фаза = float(self.input_phase.value()) * np.pi / 180.0
            self.фаза_оп = float(self.input_phase_op.value()) * np.pi / 180.0
            self.шум_std = float(self.input_noise.value())

            # Шаг 2: Расчет децимации (20%)
            self.progress_bar.setValue(20)
            QtWidgets.QApplication.processEvents()

            if self.пс_частота > 0:
                self.дек_фактор = int(round(self.fs / self.пс_частота))
                self.дек_фактор = max(1, self.дек_фактор)
            else:
                self.дек_фактор = 1
            self.фильтр_срез = float(self.input_cut.value())
            self.N = int(self.длительность * self.fs)

            # Шаг 3: Генерация сигналов (30-80%)
            self.progress_bar.setValue(30)
            QtWidgets.QApplication.processEvents()

            self._generate_all_signals_with_progress()

            # Шаг 4: Обновление графиков (90%)
            self.progress_bar.setValue(90)
            QtWidgets.QApplication.processEvents()

            self._update_all_plots()

            # Завершение
            self.progress_bar.setValue(100)
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка при моделировании:\n{str(e)}")

        finally:
            # Разблокируем интерфейс и скрываем прогресс-бар
            QtCore.QTimer.singleShot(500, self._finish_processing)

    def _generate_all_signals_with_progress(self):
        """Генерация сигналов с обновлением прогресса"""
        # Временная шкала (30%)
        t = np.linspace(0, self.длительность, int(self.длительность * self.fs), endpoint=False)
        self.t = t
        N = len(t)
        self.progress_bar.setValue(35)
        QtWidgets.QApplication.processEvents()

        # ПСП (40%)
        self.pn_sequence = generate_pn_sequence(N, self.пс_частота, [-1, 1], self.fs)
        self.progress_bar.setValue(45)
        QtWidgets.QApplication.processEvents()

        # Синус (50%)
        self.sinusoid = generate_sinusoid(self.частота_опорного, self.фаза, self.fs, N)
        self.progress_bar.setValue(55)
        QtWidgets.QApplication.processEvents()

        # Модуляция (60%)
        self.multiplied = self.pn_sequence * self.sinusoid
        self.progress_bar.setValue(65)
        QtWidgets.QApplication.processEvents()

        # Шум (70%)
        self.noisy = add_gaussian_noise(self.multiplied, self.шум_std)
        self.progress_bar.setValue(75)
        QtWidgets.QApplication.processEvents()

        # Опорный генератор (80%)
        self.reference = generate_sinusoid(self.частота_опорного, self.фаза_оп, self.fs, N)
        self.progress_bar.setValue(85)
        QtWidgets.QApplication.processEvents()

        # Перемножитель демодуляции (90%)
        self.mixed = self.noisy * self.reference
        self.progress_bar.setValue(95)
        QtWidgets.QApplication.processEvents()

        # Фильтрация (100%)
        self.filtered = butter_lowpass_filter(self.mixed, self.фильтр_срез, self.fs, order=5)
        self.progress_bar.setValue(100)
        QtWidgets.QApplication.processEvents()

        # Децимация
        self.decimated = decimate(self.filtered, self.дек_фактор)
        self.decimated_t = self.t[::self.дек_фактор]

        # Решающее устройство
        self.limited = limiter(self.decimated)

        # BER
        orig = self.pn_sequence[::self.дек_фактор][:len(self.limited)]
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
        if bandwidth > 0 and noise_power > 0:
            noise_density = noise_power / bandwidth
            if noise_density > 1e-20 and bit_energy > 0:
                eb_no = 10 * np.log10(bit_energy / noise_density)
            else:
                eb_no = -100
        else:
            noise_density = 1e-12
            eb_no = -100
        self.eb_no = eb_no

    def _finish_processing(self):
        """Завершение обработки"""
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.set_controls_enabled(True)

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

    def _build_tradeoff_tab(self):
        # Создаём вложенный QTabWidget
        self.tradeoff_tabs = QtWidgets.QTabWidget()

        # Создаём три подвкладки
        self.tradeoff_tab1 = QtWidgets.QWidget()
        self.tradeoff_tab2 = QtWidgets.QWidget()
        self.tradeoff_tab3 = QtWidgets.QWidget()

        # Добавляем подвкладки в основной виджет вкладки
        self.tradeoff_tabs.addTab(self.tradeoff_tab1, 'График 1: h(СКО шума)')
        self.tradeoff_tabs.addTab(self.tradeoff_tab2, 'График 2: h(Δφ)')
        self.tradeoff_tabs.addTab(self.tradeoff_tab3, 'График 3: Диаграмма обмена')

        # Создаём макет для основной вкладки и добавляем в него вложенные вкладки
        main_layout = QtWidgets.QVBoxLayout(self.tab_tradeoff)
        main_layout.addWidget(self.tradeoff_tabs)

        # --- Заполняем первую подвкладку ---
        layout1 = QtWidgets.QVBoxLayout(self.tradeoff_tab1)

        self.trade_data1 = None

        # График 1
        self.canvas1 = MplCanvas(self, width=10, height=8)
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
        self.canvas2 = MplCanvas(self, width=10, height=8)
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
        self.canvas3 = MplCanvas(self, width=10, height=8)
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

    # def _plot_graph1(self):
    #     if self.trade_data1 is None or len(self.trade_data1[0]) == 0:
    #         QtWidgets.QMessageBox.warning(self, "Предупреждение",
    #                                       "Нет данных для построения графика. Сначала задайте таблицу.")
    #         return
    #
    #     # x, y = self.trade_data1
    #     x, y = self.trade_data1
    #
    #     # Сортировка по x
    #     # sorted_idx = np.argsort(x)
    #     # x = x[sorted_idx]
    #     # y = y[sorted_idx]
    #     sorted_idx = np.argsort(x)
    #     x = x[sorted_idx]
    #     y = y[sorted_idx]
    #
    #     ax = self.canvas1.ax
    #     ax.clear()
    #
    #     # Строим график с лейблами
    #     line, = ax.plot(x, y, 'o-', linewidth=2, markersize=8,
    #                     label='Экспериментальные точки', color='blue')
    #
    #     # Добавляем подписи к точкам (значения)
    #     for i, (xi, yi) in enumerate(zip(x, y)):
    #         ax.annotate(f'({xi:.3f}, {yi:.3f})',
    #                     (xi, yi),
    #                     textcoords="offset points",
    #                     xytext=(0, 10),
    #                     ha='center',
    #                     fontsize=8,
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    #
    #     # Настройка осей
    #     ax.set_xlabel("СКО шума", fontsize=12, fontweight='bold')
    #     ax.set_ylabel("h", fontsize=12, fontweight='bold')
    #
    #     # Добавляем заголовок
    #     ax.set_title('Зависимость h от СКО шума', fontsize=14, fontweight='bold')
    #
    #     # Добавляем легенду
    #     ax.legend(loc='best', fontsize=10)
    #
    #     # Добавляем сетку
    #     ax.grid(True, alpha=0.3)
    #
    #     # Добавляем подписи минимальных и максимальных значений
    #     if len(x) > 0:
    #         max_idx = np.argmax(y)
    #         min_idx = np.argmin(y)
    #
    #         ax.annotate(f'Max: ({x[max_idx]:.3f}, {y[max_idx]:.3f})',
    #                     (x[max_idx], y[max_idx]),
    #                     textcoords="offset points",
    #                     xytext=(0, -20),
    #                     ha='center',
    #                     fontsize=9,
    #                     color='green',
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    #
    #         ax.annotate(f'Min: ({x[min_idx]:.3f}, {y[min_idx]:.3f})',
    #                     (x[min_idx], y[min_idx]),
    #                     textcoords="offset points",
    #                     xytext=(0, 20),
    #                     ha='center',
    #                     fontsize=9,
    #                     color='red',
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    #
    #     self.canvas1.draw()
    #
    # def _plot_graph2(self):
    #     if self.trade_data2 is None or len(self.trade_data2[0]) == 0:
    #         QtWidgets.QMessageBox.warning(self, "Предупреждение",
    #                                       "Нет данных для построения графика. Сначала задайте таблицу.")
    #         return
    #
    #     # x, y = self.trade_data2
    #     sorted_idx = np.argsort(x)
    #     x = x[sorted_idx]
    #     y = y[sorted_idx]
    #
    #     ax = self.canvas2.ax
    #     ax.clear()
    #
    #     # Строим график с лейблами
    #     line, = ax.plot(x, y, 's-', linewidth=2, markersize=8,
    #                     label='Экспериментальные точки', color='red')
    #
    #     # Добавляем подписи к точкам (значения)
    #     for i, (xi, yi) in enumerate(zip(x, y)):
    #         ax.annotate(f'({xi:.3f}, {yi:.3f})',
    #                     (xi, yi),
    #                     textcoords="offset points",
    #                     xytext=(0, 10),
    #                     ha='center',
    #                     fontsize=8,
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    #
    #     # Настройка осей
    #     ax.set_xlabel("Δφ (разность фаз)", fontsize=12, fontweight='bold')
    #     ax.set_ylabel("h", fontsize=12, fontweight='bold')
    #
    #     # Добавляем заголовок
    #     ax.set_title('Зависимость h от разности фаз', fontsize=14, fontweight='bold')
    #
    #     # Добавляем легенду
    #     ax.legend(loc='best', fontsize=10)
    #
    #     # Добавляем сетку
    #     ax.grid(True, alpha=0.3)
    #
    #     # Добавляем подписи минимальных и максимальных значений
    #     if len(x) > 0:
    #         max_idx = np.argmax(y)
    #         min_idx = np.argmin(y)
    #
    #         ax.annotate(f'Max: ({x[max_idx]:.3f}, {y[max_idx]:.3f})',
    #                     (x[max_idx], y[max_idx]),
    #                     textcoords="offset points",
    #                     xytext=(0, -20),
    #                     ha='center',
    #                     fontsize=9,
    #                     color='green',
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    #
    #         ax.annotate(f'Min: ({x[min_idx]:.3f}, {y[min_idx]:.3f})',
    #                     (x[min_idx], y[min_idx]),
    #                     textcoords="offset points",
    #                     xytext=(0, 20),
    #                     ha='center',
    #                     fontsize=9,
    #                     color='red',
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    #
    #     self.canvas2.draw()
    #
    # def _plot_graph3(self):
    #     if self.trade_data3 is None or len(self.trade_data3[0]) == 0:
    #         QtWidgets.QMessageBox.warning(self, "Предупреждение",
    #                                       "Нет данных для построения графика. Сначала задайте таблицу.")
    #         return
    #
    #     x, y = self.trade_data3
    #
    #     ax = self.canvas3.ax
    #     ax.clear()
    #
    #     # Строим график с лейблами
    #     line, = ax.plot(x, y, '^-', linewidth=2, markersize=8,
    #                     label='Экспериментальные точки', color='green')
    #
    #     # Добавляем подписи к точкам (значения)
    #     for i, (xi, yi) in enumerate(zip(x, y)):
    #         ax.annotate(f'({xi:.3f}, {yi:.3f})',
    #                     (xi, yi),
    #                     textcoords="offset points",
    #                     xytext=(0, 10),
    #                     ha='center',
    #                     fontsize=8,
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    #
    #     # Настройка осей
    #     ax.set_xlabel("Δφ (разность фаз)", fontsize=12, fontweight='bold')
    #     ax.set_ylabel("СКО шума", fontsize=12, fontweight='bold')
    #
    #     # Добавляем заголовок
    #     ax.set_title('Диаграмма обмена', fontsize=14, fontweight='bold')
    #
    #     # Добавляем легенду
    #     ax.legend(loc='best', fontsize=10)
    #
    #     # Добавляем сетку
    #     ax.grid(True, alpha=0.3)
    #
    #     # Добавляем подписи минимальных и максимальных значений
    #     if len(x) > 0:
    #         max_idx = np.argmax(y)
    #         min_idx = np.argmin(y)
    #
    #         ax.annotate(f'Max: ({x[max_idx]:.3f}, {y[max_idx]:.3f})',
    #                     (x[max_idx], y[max_idx]),
    #                     textcoords="offset points",
    #                     xytext=(0, -20),
    #                     ha='center',
    #                     fontsize=9,
    #                     color='green',
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    #
    #         ax.annotate(f'Min: ({x[min_idx]:.3f}, {y[min_idx]:.3f})',
    #                     (x[min_idx], y[min_idx]),
    #                     textcoords="offset points",
    #                     xytext=(0, 20),
    #                     ha='center',
    #                     fontsize=9,
    #                     color='red',
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    #
    #     self.canvas3.draw()
    def _plot_graph1(self):
        if self.trade_data1 is None or len(self.trade_data1[0]) == 0:
            QtWidgets.QMessageBox.warning(self, "Предупреждение",
                                          "Нет данных для построения графика. Сначала задайте таблицу.")
            return

        x, y = self.trade_data1

        # Сортировка по x (по возрастанию)
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        ax = self.canvas1.ax
        ax.clear()

        # Строим график с лейблами
        line, = ax.plot(x_sorted, y_sorted, 'o-', linewidth=2, markersize=8,
                        label='Экспериментальные точки', color='blue')

        # Добавляем подписи к точкам (значения)
        for i, (xi, yi) in enumerate(zip(x_sorted, y_sorted)):
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

        # # Добавляем подписи минимальных и максимальных значений
        # if len(x_sorted) > 0:
        #     max_idx = np.argmax(y_sorted)
        #     min_idx = np.argmin(y_sorted)
        #
        #     ax.annotate(f'Max: ({x_sorted[max_idx]:.3f}, {y_sorted[max_idx]:.3f})',
        #                 (x_sorted[max_idx], y_sorted[max_idx]),
        #                 textcoords="offset points",
        #                 xytext=(0, -20),
        #                 ha='center',
        #                 fontsize=9,
        #                 color='green',
        #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        #
        #     ax.annotate(f'Min: ({x_sorted[min_idx]:.3f}, {y_sorted[min_idx]:.3f})',
        #                 (x_sorted[min_idx], y_sorted[min_idx]),
        #                 textcoords="offset points",
        #                 xytext=(0, 20),
        #                 ha='center',
        #                 fontsize=9,
        #                 color='red',
        #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

        self.canvas1.draw()


    def _plot_graph2(self):
        if self.trade_data2 is None or len(self.trade_data2[0]) == 0:
            QtWidgets.QMessageBox.warning(self, "Предупреждение",
                                          "Нет данных для построения графика. Сначала задайте таблицу.")
            return

        x, y = self.trade_data2

        # Сортировка по x (по возрастанию)
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        ax = self.canvas2.ax
        ax.clear()

        # Строим график с лейблами
        line, = ax.plot(x_sorted, y_sorted, 's-', linewidth=2, markersize=8,
                        label='Экспериментальные точки', color='red')

        # Добавляем подписи к точкам (значения)
        for i, (xi, yi) in enumerate(zip(x_sorted, y_sorted)):
            ax.annotate(f'({xi:.3f}, {yi:.3f})',
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Настройка осей
        ax.set_xlabel("Δφ [Градусы]", fontsize=12, fontweight='bold')
        ax.set_ylabel("h", fontsize=12, fontweight='bold')

        # Добавляем заголовок
        ax.set_title('Зависимость h от Δφ', fontsize=14, fontweight='bold')

        # Добавляем легенду
        ax.legend(loc='best', fontsize=10)

        # Добавляем сетку
        ax.grid(True, alpha=0.3)

        # # Добавляем подписи минимальных и максимальных значений
        # if len(x_sorted) > 0:
        #     max_idx = np.argmax(y_sorted)
        #     min_idx = np.argmin(y_sorted)
        #
        #     ax.annotate(f'Max: ({x_sorted[max_idx]:.3f}, {y_sorted[max_idx]:.3f})',
        #                 (x_sorted[max_idx], y_sorted[max_idx]),
        #                 textcoords="offset points",
        #                 xytext=(0, -20),
        #                 ha='center',
        #                 fontsize=9,
        #                 color='green',
        #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        #
        #     ax.annotate(f'Min: ({x_sorted[min_idx]:.3f}, {y_sorted[min_idx]:.3f})',
        #                 (x_sorted[min_idx], y_sorted[min_idx]),
        #                 textcoords="offset points",
        #                 xytext=(0, 20),
        #                 ha='center',
        #                 fontsize=9,
        #                 color='red',
        #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

        self.canvas2.draw()

    def _plot_graph3(self):
        if self.trade_data3 is None or len(self.trade_data3[0]) == 0:
            QtWidgets.QMessageBox.warning(self, "Предупреждение",
                                          "Нет данных для построения графика. Сначала задайте таблицу.")
            return

        x, y = self.trade_data3

        # Сортировка по x (по возрастанию)
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        ax = self.canvas3.ax
        ax.clear()

        # Строим график с лейблами
        line, = ax.plot(x_sorted, y_sorted, '^-', linewidth=2, markersize=8,
                        label='Экспериментальные точки', color='green')

        # Добавляем подписи к точкам (значения)
        for i, (xi, yi) in enumerate(zip(x_sorted, y_sorted)):
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

        # # Добавляем подписи минимальных и максимальных значений
        # if len(x_sorted) > 0:
        #     max_idx = np.argmax(y_sorted)
        #     min_idx = np.argmin(y_sorted)
        #
        #     ax.annotate(f'Max: ({x_sorted[max_idx]:.3f}, {y_sorted[max_idx]:.3f})',
        #                 (x_sorted[max_idx], y_sorted[max_idx]),
        #                 textcoords="offset points",
        #                 xytext=(0, -20),
        #                 ha='center',
        #                 fontsize=9,
        #                 color='green',
        #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        #
        #     ax.annotate(f'Min: ({x_sorted[min_idx]:.3f}, {y_sorted[min_idx]:.3f})',
        #                 (x_sorted[min_idx], y_sorted[min_idx]),
        #                 textcoords="offset points",
        #                 xytext=(0, 20),
        #                 ha='center',
        #                 fontsize=9,
        #                 color='red',
        #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

        self.canvas3.draw()
    # ---------- Отдельные функции рисования графиков
    def _plot_psp(self):
        ax = self.psp_canvas.ax
        ax.clear()
        ax.step(self.t, self.pn_sequence, where='post')
        ax.set_title('Информационная последовательность (ПСП)')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Уровень')
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
        ax.set_title('Немодулированная несущая')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Мгновенное значение')
        ax.grid(True)
        self.op_canvas.draw()

    def _plot_oporny_psd(self):
        ax = self.op_psd_canvas.ax
        ax.clear()
        f, P = compute_psd(self.sinusoid, self.fs)
        ax.plot(f, P)
        ax.set_title('СПМ немодулированной несущей')
        ax.set_xlabel('Частота [Гц]')
        ax.set_ylabel('Плотность [dB]')
        ax.grid(True)
        self.op_psd_canvas.draw()

    def _plot_modulated(self):
        ax = self.mod_canvas.ax
        ax.clear()
        ax.plot(self.t, self.multiplied)
        ax.set_title('Модулированный сигнал 2ФМ')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Мгновенное значение')
        ax.grid(True)
        self.mod_canvas.draw()

    def _plot_modulated_psd(self):
        ax = self.mod_psd_canvas.ax
        ax.clear()
        f, P = compute_psd(self.multiplied, self.fs)
        ax.plot(f, P)
        ax.set_title('СПМ модулированного сигнала 2ФМ')
        ax.set_xlabel('Частота [Гц]')
        ax.set_ylabel('Плотность [dB]')
        ax.grid(True)
        self.mod_psd_canvas.draw()

    def _plot_channel(self):
        ax = self.chan_canvas.ax
        ax.clear()
        ax.plot(self.t, self.noisy)
        ax.set_title('Процесс на выходе канала')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Мгновенное значение')
        ax.grid(True)
        self.chan_canvas.draw()

    def _plot_channel_psd(self):
        ax = self.chan_psd_canvas.ax
        ax.clear()
        f, P = compute_psd(self.noisy, self.fs)
        ax.plot(f, P)
        ax.set_title('СПМ процесса на выходе канала')
        ax.set_xlabel('Частота [Гц]')
        ax.set_ylabel('Плотность [dB]')
        ax.grid(True)
        self.chan_psd_canvas.draw()

    def _plot_demod(self):
        ax = self.dem_canvas.ax
        ax.clear()
        ax.plot(self.t, self.mixed)
        ax.set_title('Процесс на выходе перемножителя')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Мгновенное значение')
        ax.grid(True)
        self.dem_canvas.draw()

    def _plot_demod_psd(self):
        ax = self.dem_psd_canvas.ax
        ax.clear()
        f, P = compute_psd(self.mixed, self.fs)
        ax.plot(f, P)
        ax.set_title('СПМ процесса на выходе перемножителя')
        ax.set_xlabel('Частота [Гц]')
        ax.set_ylabel('Плотность [dB]')
        ax.grid(True)
        self.dem_psd_canvas.draw()

    def _plot_lpf(self):
        ax = self.lpf_canvas.ax
        ax.clear()
        ax.plot(self.t, self.filtered)
        ax.set_title('Процесс на выходе ФНЧ')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Мгновенное значение')
        ax.grid(True)
        self.lpf_canvas.draw()

    def _plot_lpf_psd(self):
        ax = self.lpf_psd_canvas.ax
        ax.clear()
        f, P = compute_psd(self.filtered, self.fs)
        ax.plot(f, P)
        ax.set_title('СПМ процесса на выходе ФНЧ')
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
        ax.set_title('Выборки процесса на выходе ФНЧ')
        ax.set_xlabel('Время [c]')
        ax.set_ylabel('Мгновенное значение')
        # ax.legend()
        ax.grid(True)
        self.dec_canvas.draw()
    def _plot_decider(self):
        ax = self.decider_canvas.ax
        ax.clear()
        ax.step(self.decimated_t, self.limited, where='post')
        ax.scatter(self.decimated_t, self.limited, s=10)
        ax.set_title('Выход демодулятора (решающего устройства)')
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

        # Обновляем BER
        # self.ber_label.setText(f'BER: {self.ber:.6e}   E_b/N_0: {self.eb_no:.2f} dB')
        # Обновляем статистику ошибок
        # self._update_error_stats()

    # ---------- Глаз-диаграмма (с использованием SpinBox вместо слайдеров)
    # def _update_eye_diagram(self):
    #     # Получаем параметры из интерфейса
    #     phase = float(self.eye_phase.value()) * np.pi / 180.0
    #     phase_op = float(self.eye_phase_op.value()) * np.pi / 180.0
    #     noise_std = float(self.eye_noise.value())
    #
    #     # Основные параметры (как в исходном коде)
    #     num_realizations = int(self.eye_realizations.value())
    #     fs = self.fs
    #     pn_rate = self.пс_частота
    #     N = self.N
    #     t = self.t
    #
    #     # Временные параметры для сегмента (как в исходном коде)
    #     start_time = 0.035  # Начало интервала в секундах
    #     end_time = 2.2  # Конец интервала в секундах
    #
    #     start_idx = int(start_time * fs)
    #     end_idx = int(end_time * fs)
    #
    #     samples_per_symbol = int(fs / pn_rate)
    #     t_symbol = np.linspace(0, 1 / pn_rate, samples_per_symbol, endpoint=False)
    #
    #     ax = self.eye_canvas.ax
    #     ax.clear()
    #
    #     # Цвета для разных реализаций (как в исходном коде)
    #     colors = plt.cm.viridis(np.linspace(0, 1, num_realizations))
    #
    #     for realization_idx in range(num_realizations):
    #         # Генерация сигналов (ПОВТОРЯЕМ ЛОГИКУ ИСХОДНОГО КОДА)
    #
    #         # 1. Генерация несущей с заданной фазой
    #         sinusoid_1 = generate_sinusoid(self.частота_опорного, phase, fs, N)
    #
    #         # 2. Модуляция (умножение ПСП на несущую)
    #         multiplied_signal = self.pn_sequence * sinusoid_1
    #
    #         # 3. Добавление шума
    #         noisy_signal = add_gaussian_noise(multiplied_signal, noise_std, 0)
    #
    #         # 4. Генерация опорного колебания
    #         reference_oscillation = generate_sinusoid(self.частота_опорного, phase_op, fs, N)
    #
    #         # 5. Демодуляция (перемножение)
    #         mixed_signal = noisy_signal * reference_oscillation
    #
    #         # 6. Фильтрация ФНЧ
    #         filtered_signal = butter_lowpass_filter(mixed_signal, self.фильтр_срез, fs, order=3)
    #
    #         # 7. Выделение сегмента для глаз-диаграммы
    #         filtered_signal_segment = filtered_signal[start_idx:end_idx]
    #
    #         # 8. Построение глаз-диаграммы (ОСНОВНАЯ ЧАСТЬ)
    #         for i in range(0, len(filtered_signal_segment) - samples_per_symbol, samples_per_symbol):
    #             ax.plot(t_symbol, filtered_signal_segment[i:i + samples_per_symbol],
    #                     color=colors[realization_idx], linewidth=0.5)
    #
    #     # Настройка графика (как в исходном коде)
    #     ax.set_title('Глаз-диаграмма')
    #     ax.set_xlabel('Время [с]')
    #     ax.set_ylabel('Амплитуда')
    #
    #     # Увеличение размера чисел сетки (как в исходном коде)
    #     ax.tick_params(axis='both', which='major', labelsize=14)
    #     ax.tick_params(axis='both', which='minor', labelsize=10)
    #     ax.grid(True)
    #
    #     self.eye_canvas.draw()
    def _update_eye_diagram(self):
        # Блокируем интерфейс
        self.set_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Используем QTimer для имитации прогресса
        QtCore.QTimer.singleShot(100, self._do_update_eye_diagram)

    def _do_update_eye_diagram(self):
        """Реальное обновление глаз-диаграммы"""
        try:
            # Получаем параметры
            phase = float(self.eye_phase.value()) * np.pi / 180.0
            phase_op = float(self.eye_phase_op.value()) * np.pi / 180.0
            noise_std = float(self.eye_noise.value())

            self.progress_bar.setValue(20)
            QtWidgets.QApplication.processEvents()

            self.last_eye_params = {
                'phase': phase,
                'phase_op': phase_op,
                'noise_std': noise_std,
                'num_realizations': int(self.eye_realizations.value())
            }

            num_realizations = int(self.eye_realizations.value())
            fs = self.fs
            pn_rate = self.пс_частота
            N = self.N

            start_time = 0.035
            end_time = 2.2

            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)

            samples_per_symbol = int(fs / pn_rate)
            t_symbol = np.linspace(0, 1 / pn_rate, samples_per_symbol, endpoint=False)

            ax = self.eye_canvas.ax
            ax.clear()

            print_mode = self.print_mode_checkbox.isChecked()

            for realization_idx in range(num_realizations):
                # Обновляем прогресс
                if realization_idx % max(1, num_realizations // 10) == 0:
                    progress = 20 + int(70 * realization_idx / num_realizations)
                    self.progress_bar.setValue(progress)
                    QtWidgets.QApplication.processEvents()

                sinusoid_1 = generate_sinusoid(self.частота_опорного, phase, fs, N)
                multiplied_signal = self.pn_sequence * sinusoid_1
                noisy_signal = add_gaussian_noise(multiplied_signal, noise_std, 0)
                reference_oscillation = generate_sinusoid(self.частота_опорного, phase_op, fs, N)
                mixed_signal = noisy_signal * reference_oscillation
                filtered_signal = butter_lowpass_filter(mixed_signal, self.фильтр_срез, fs, order=3)
                filtered_signal_segment = filtered_signal[start_idx:end_idx]

                for i in range(0, len(filtered_signal_segment) - samples_per_symbol, samples_per_symbol):
                    if print_mode:
                        gray_intensity = 0.3 + (i / (len(filtered_signal_segment) - samples_per_symbol)) * 0.5
                        ax.plot(t_symbol, filtered_signal_segment[i:i + samples_per_symbol],
                                color=(gray_intensity, gray_intensity, gray_intensity),
                                linewidth=0.5, alpha=0.5)
                    else:
                        color = plt.cm.viridis(realization_idx / num_realizations)
                        ax.plot(t_symbol, filtered_signal_segment[i:i + samples_per_symbol],
                                color=color, linewidth=0.5)

            self.progress_bar.setValue(95)
            QtWidgets.QApplication.processEvents()

            ax.set_title('Глаз-диаграмма')
            ax.set_xlabel('Время [с]')
            ax.set_ylabel('Амплитуда')
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            ax.grid(True)
            ax.axhline(y=0, color='red', linewidth=3, linestyle='-', alpha=0.5)
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.set_xlim(left=0)

            self.eye_canvas.draw()
            self.progress_bar.setValue(100)
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка при построении глаз-диаграммы:\n{str(e)}")

        finally:
            QtCore.QTimer.singleShot(300, self._finish_processing)


# ---------- Запуск приложения
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    # Принудительно устанавливаем светлую тему
    app.setStyle('Fusion')

    # Создаём палитру для светлой темы
    palette = QtGui.QPalette()

    # Цвета для светлой темы
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.black)
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(245, 245, 245))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.black)
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.black)
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(0, 120, 215))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtCore.Qt.GlobalColor.white)

    # Устанавливаем палитру
    app.setPalette(palette)

    # Дополнительные стили для улучшения внешнего вида
    app.setStyleSheet("""
        QToolTip {
            background-color: white;
            color: black;
            border: 1px solid gray;
        }

        /* Стили для вкладок - увеличенный шрифт и скругление */
        QTabWidget::pane {
            border: 1px solid #ccc;
            border-radius: 5px;
        }

        QTabBar::tab {
            font-size: 12px;
            font-weight: bold;
            padding: 4px 10px;
            margin: 2px;
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 8px;
        }

        QTabBar::tab:selected {
            background-color: #4CAF50;
            color: white;
            border: 1px solid #4CAF50;
        }

        QTabBar::tab:hover:!selected {
            background-color: #e0e0e0;
        }

        /* Группа параметров моделирования */
        QGroupBox {
            border: 2px solid gray;
            border-radius: 12px;
            margin-top: 20px;
            font-size: 16px;
            font-weight: bold;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 12px 0 12px;
            font-size: 16px;
            font-weight: bold;
        }

        QPushButton {
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 14px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #e0e0e0;
        }

        QPushButton:pressed {
            background-color: #d0d0d0;
        }

        /* Скругленные спинбоксы */
        QSpinBox, QDoubleSpinBox {
            background-color: white;
            border: 2px solid #aaa;
            border-radius: 8px;
            padding: 2px 6px;
            color: black;
            font-size: 15px;
            font-weight: bold;
            min-height: 12px;
            min-width: 80px;
        }

        /* Текст внутри спинбокса */
        QSpinBox::text, QDoubleSpinBox::text {
            font-size: 10px;
            font-weight: bold;
        }

        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid #4CAF50;
        }

        /* Скрываем кнопки-стрелки */
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            width: 0px;
            height: 0px;
            subcontrol-position: none;
        }

        QCheckBox {
            spacing: 8px;
            color: black;
            font-size: 14px;
            font-weight: bold;
        }

        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border: 2px solid #555555;
            border-radius: 5px;
            background-color: white;
        }

        QCheckBox::indicator:checked {
            background-color: #4CAF50;
            border: 2px solid #4CAF50;
        }

        QCheckBox::indicator:unchecked:hover {
            border: 2px solid #999999;
        }

        QCheckBox::indicator:checked:hover {
            background-color: #45a049;
            border: 2px solid #45a049;
        }

        /* Подписи к спинбоксам (QLabel) */
        QLabel {
            font-size: 14px;
            font-weight: bold;
            color: black;
        }

        QTableWidget {
            background-color: white;
            alternate-background-color: #f8f8f8;
            gridline-color: #ddd;

        }

        QHeaderView::section {
            background-color: #f0f0f0;
            padding: 6px;
            border: 1px solid #ddd;
            font-size: 16px;
            font-weight: bold;
        }

        /* Стиль только для спинбокса "Длина реализации" */
        QSpinBox#wide_spinbox {
            min-width: 120px;
            padding: 2px 8px;
        }
        
        
    """)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
