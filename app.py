import os
import sys
import csv
import glob
import datetime
import multiprocessing
import numpy as np
import Ui_untitled_ENG as Ui_win_main  
from PyQt5 import QtCore, QtGui, QtWidgets
import open3d as o3d
import time
from open3d.visualization import gui
from open3d.visualization import rendering
from util.function import (DeafaultData, communicate_Thread_callSignal, util,
                           vType)
import win32gui
import uuid
from util.showByOpen3D import showbyopen3d

PYTHON_ENV = r"D:\Anaconda\envs\pre\python.exe"     

class QSSLoader:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_qss_file(file):
        with open(file, 'r', encoding='UTF-8') as f:
            return f.read()


class SelectDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Select File or Folder")
        self.resize(200, 100)

        layout = QtWidgets.QVBoxLayout()

        self.single_file_button = QtWidgets.QPushButton("file")
        self.single_file_button.clicked.connect(self.on_single_file_clicked)
        layout.addWidget(self.single_file_button)

        self.folder_button = QtWidgets.QPushButton("files")
        self.folder_button.clicked.connect(self.on_folder_clicked)
        layout.addWidget(self.folder_button)

        self.setLayout(layout)

        self.selected_option = None

    def on_single_file_clicked(self):
        self.selected_option = 0
        self.accept()

    def on_folder_clicked(self):
        self.selected_option = 1
        self.accept()


class appmain(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        self.app = QtWidgets.QApplication(sys.argv)
        super(appmain, self).__init__()
      
        self.ui_MainWindow = Ui_win_main.Ui_MainWindow()
        self.ui_MainWindow.setupUi(self)

        self.appArgsInit()

        self.ui_MainWindowWidget_Init(self.ui_MainWindow)

    
        self.setWindowTitle('Lettuce 3D Phenotypic Analysis Software V1.0')

        # Communication
        self._queue_open3dToAppmain = multiprocessing.Queue()
        self._queue_appmainToOpen3d = multiprocessing.Queue()
        self.send = self._queue_appmainToOpen3d.put

        self.__manager = multiprocessing.Manager()
        self._data_manager = self.__manager.dict()
        self._data_manager[vType.data.geometryInfoCache] = {}   # {uuid:[filePath, modeName, modeType, modeRaw], ...}
        self._data_manager[vType.data._uuidSet] = set()
        self._data_manager[vType.data.geoUuidList] = {vType.type.mesh: [], vType.type.pointCloud: []}

        self._threadCommunicateFromOpen3d = communicate_Thread_callSignal(
            self._queue_open3dToAppmain)
        self._threadCommunicateFromOpen3d.communicate_singal.connect(
            self.connectFun_threadCommunicate_singal
        )
        self._threadCommunicateFromOpen3d.start()

        open3d_win_name = '__GXF_win__' + str(uuid.uuid4())

        self._processOpen3d = multiprocessing.Process(
            target=showbyopen3d,
            args=(self._queue_open3dToAppmain, self._queue_appmainToOpen3d, self._data_manager, open3d_win_name)
        )
        self._processOpen3d.start()

        # Open3D windows
        time.sleep(1)
        hwnd1 = win32gui.FindWindowEx(0, 0, "GLFW30", open3d_win_name)
        start = time.time()
        while hwnd1 == 0:
            time.sleep(0.01)
            hwnd1 = win32gui.FindWindowEx(0, 0, "GLFW30", open3d_win_name)
            end = time.time()
            if end - start > 5:
                break
        window = QtGui.QWindow.fromWinId(hwnd1)
        widget = QtWidgets.QWidget.createWindowContainer(window, self)
        self.window_open3d = self.ui_MainWindow.mdiArea.addSubWindow(widget, QtCore.Qt.CustomizeWindowHint)
        widget.showMaximized()

        # style
        style_file = "./lightstyle.qss"
        style_sheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(style_sheet)

        self.appStart()

    def appStart(self):
        self.show()
        sys.exit(self.app.exec_())

    def appArgsInit(self):
        self.thread_fun = None

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.time_fun)
        self.timer.start(1000)

        self.process_array = []         
        self.process_finish = True      

        self.windows_running_dialog = None

    def ui_MainWindowWidget_Init(self, ui: Ui_win_main.Ui_MainWindow):

        ui.pushButton_pre_run.clicked.connect(
            lambda: self.connectFun_pushButton_pre_run_clicked()
        )
        ui.pushButton_seg_run.clicked.connect(
            lambda: self.connectFun_pushButton_seg_run_clicked()
        )
        ui.pushButton_phe_run.clicked.connect(
            lambda: self.connectFun_pushButton_phe_run_clicked()
        )
        ui.pushButton_other_run.clicked.connect(
            lambda: self.connectFun_pushButton_other_run_clicked()
        )
        ui.pushButton_visual_model.clicked.connect(
            lambda: self.connectFun_pushButton_visual_model_clicked()
        )
        ui.pushButton_visual_table.clicked.connect(
            lambda: self.connectFun_pushButton_visual_table_clicked()
        )

       
        ui.toolButton_other_path.clicked.connect(
            lambda: self.open_folder_dialog(ui.lineEdit_other_path)
        )
        ui.toolButton_main_in_path.clicked.connect(
            lambda: self.open_folder_dialog(ui.lineEdit_main_in_path)
        )
        ui.toolButton_main_out_path.clicked.connect(
            lambda: self.open_folder_dialog(ui.lineEdit_main_out_path)
        )
        ui.toolButton_visual_path.clicked.connect(
            lambda: self.open_file_or_folder_dialog(ui.lineEdit_visual_path)
        )

    def log_print(self, info):
        current_time = datetime.datetime.now()
        self.ui_MainWindow.textEdit_log.append('[' + str(current_time)[:-7] + '] ' + info)

    def enable_all_widget(self, is_enable: bool):
        
        def get_all_widget(windows):
            select_widget = []
            for w in windows.findChildren(QtWidgets.QWidget):
                if not w.objectName():
                    continue
                if isinstance(w, (QtWidgets.QCheckBox, QtWidgets.QPushButton, QtWidgets.QLineEdit, QtWidgets.QDoubleSpinBox, QtWidgets.QComboBox)):
                    select_widget.append(w)
                select_widget.extend(get_all_widget(w))
            return select_widget

        [daw.setEnabled(is_enable) for daw in get_all_widget(self)]

    def stop_progressBar(self, num=100):
        self.ui_MainWindow.progressBar.setMaximum(100)
        self.ui_MainWindow.progressBar.setValue(num)
        self.enable_all_widget(True)
        self.process_finish = True
        self.ui_MainWindow.label_progressBar.setText('--:--:--')

    def open_folder_dialog(self, lineEdit: QtWidgets.QLineEdit):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ShowDirsOnly

        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "file", options=options)

        if folder_path:
            lineEdit.setText(folder_path)

    def open_file_dialog(self, lineEdit: QtWidgets.QLineEdit):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file_dialog.setNameFilters(["All files (*.*)"])
        if file_dialog.exec_():
            lineEdit.setText(file_dialog.selectedFiles()[0])

    def open_file_or_folder_dialog(self, lineEdit: QtWidgets.QLineEdit):
        dialog = SelectDialog()
        if dialog.exec_():
            selected_option = dialog.selected_option

            if selected_option == 0:
                self.open_file_dialog(lineEdit)
            else:
                self.open_folder_dialog(lineEdit)

    def update_progressBar_processOutInfo(self, out_info: str):
        if out_info.startswith('b\'\\r') and (out_info.endswith('it/s]\'') or out_info.endswith('s/it]\'')):
           
            end_info = out_info.split('|')[-1]  
            self.log_print(end_info)
            now_item = int(end_info.split('/')[0])                  
            all_item = int(end_info.split('/')[1].split('[')[0])    
            least_time = end_info.split('<')[1].split(',')[0]
            if '?' not in least_time:
                least_time = least_time.split(':')
                least_time = [int(lt) for lt in least_time]
                if len(least_time) == 3:
                    least_time = least_time[0]*60*60 + least_time[1]*60 + least_time[2]
                else:
                    least_time = least_time[0]*60 + least_time[1]

            self.ui_MainWindow.label_progressBar.setText(f"{now_item}/{all_item} # {least_time}s")
            self.ui_MainWindow.progressBar.setMaximum(100)
            self.ui_MainWindow.progressBar.setValue(int(now_item*100.0/all_item))
        else:
            self.log_print(out_info)

    def connectFun_pushButton_pre_run_clicked(self):
      
        in_path = self.ui_MainWindow.lineEdit_main_in_path.text()
        out_path = self.ui_MainWindow.lineEdit_main_out_path.text()

        if not os.path.isdir(in_path) or not os.path.isdir(out_path):
            self.log_print("path error...")
            return
        elif in_path == out_path:
            self.log_print("in path = out path ...")
            return

        if self.ui_MainWindow.checkBox_pre_allToCai.isChecked() and self.ui_MainWindow.doubleSpinBox_allToCai_density.isEnabled:
            allToCai_density = self.ui_MainWindow.doubleSpinBox_allToCai_density.value()

            python_env = PYTHON_ENV
            run_file_path = r"./pen_cluster.py"
            run_cmd = [run_file_path, '--in_path', in_path, '--out_path', out_path, '--allToCai_density', allToCai_density]
            self.add_process(python_env, run_cmd, ">>> ……")

    def connectFun_pushButton_seg_run_clicked(self):

        in_path = self.ui_MainWindow.lineEdit_main_in_path.text()
        out_path = self.ui_MainWindow.lineEdit_main_out_path.text()

        if not os.path.isdir(in_path) or not os.path.isdir(out_path):
            self.log_print("path error...")
            return
        elif in_path == out_path:
            self.log_print("in path = out path ...")
            return

        if self.ui_MainWindow.checkBox_seg_lettuce.isChecked():

            python_env = PYTHON_ENV
            run_file_path = './model_run.py'
            run_cmd = [run_file_path, '--in_path', in_path, '--out_path', out_path]
            self.add_process(python_env, run_cmd, ">>> ……")

        if self.ui_MainWindow.checkBox_seg_singleLeaf.isChecked():

            python_env = PYTHON_ENV
            run_file_path = './cai_to_single_leaf.py'
            run_cmd = [run_file_path, '--in_path', out_path if self.ui_MainWindow.checkBox_seg_lettuce.isChecked() else in_path, '--out_path', out_path]
            self.add_process(python_env, run_cmd, ">>> ……")

    def connectFun_pushButton_phe_run_clicked(self):

        in_path = self.ui_MainWindow.lineEdit_main_in_path.text()
        out_path = self.ui_MainWindow.lineEdit_main_out_path.text()

        if not os.path.isdir(in_path) or not os.path.isdir(out_path):
            self.log_print("path error...")
            return
        elif in_path == out_path:
            self.log_print("in path = out path ...")
            return
        
        if self.ui_MainWindow.checkBox_phe_lettuce.isChecked():

            python_env = PYTHON_ENV
            run_file_path = r".\function_phe\_QT_API_param_to_file.py"
            run_cmd = [run_file_path, '--in_path', in_path, '--out_path', out_path]
            self.add_process(python_env, run_cmd, ">>> ……")

        if self.ui_MainWindow.checkBox_phe_leaves.isChecked():

            python_env = PYTHON_ENV
            run_file_path = r".\function_phe\_QT_API_param_to_file_leaf.py"
            run_cmd = [run_file_path, '--in_path', in_path, '--out_path', out_path]
            self.add_process(python_env, run_cmd, ">>> ……")

    def connectFun_pushButton_visual_model_clicked(self):
        model_files = []
        if os.path.isfile(self.ui_MainWindow.lineEdit_visual_path.text()) and self.ui_MainWindow.lineEdit_visual_path.text()[-4:] in "*.txt *.obj *.ply":
            model_files.append(self.ui_MainWindow.lineEdit_visual_path.text())
        elif os.path.isdir(self.ui_MainWindow.lineEdit_visual_path.text()):
            model_files += glob.glob(os.path.join(self.ui_MainWindow.lineEdit_visual_path.text(), '*.txt'))
            model_files += glob.glob(os.path.join(self.ui_MainWindow.lineEdit_visual_path.text(), '*.obj'))
            model_files += glob.glob(os.path.join(self.ui_MainWindow.lineEdit_visual_path.text(), '*.ply'))
        if len(model_files):
            self.visual_model(model_files)

    def connectFun_pushButton_visual_table_clicked(self):
        csv_files = []
        if os.path.isfile(self.ui_MainWindow.lineEdit_visual_path.text()):
            csv_files.append(self.ui_MainWindow.lineEdit_visual_path.text())
        elif os.path.isdir(self.ui_MainWindow.lineEdit_visual_path.text()):
            csv_files += glob.glob(os.path.join(self.ui_MainWindow.lineEdit_visual_path.text(), '*.csv'))
        if len(csv_files):
            self.windows_running()

            datas = {}
            for file in csv_files:
                data = []
                with open(file, 'r', newline='', encoding='utf-8') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    for row in csv_reader:
                        data.append(row)
                        self.update_windows_running_prograss()
                datas[os.path.basename(file)[:-4]] = data

            self.visual_table(datas)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.thread_fun is not None and self.thread_fun.is_alive():
            self.thread_fun.terminate()
        
        if self._processOpen3d.is_alive():
            self.send([vType.all.close, None])
            self._processOpen3d.terminate()
            st = time.time()
            while self._processOpen3d.is_alive():
                if time.time()-st > 3:
                    self._processOpen3d.terminate()
                    break
        self._processOpen3d.close()

        return super().closeEvent(a0)

    def time_fun(self):
        def run_process(python_env, args):
            pro = QtCore.QProcess()

            self.log_print(f"run: [{python_env} {str(args)}]")

            args = [str(arg) for arg in args]

            pro.start(python_env, args)
            pro.readyReadStandardOutput.connect(lambda: self.log_print(str(pro.readAllStandardOutput())))
            pro.readyReadStandardError.connect(lambda: self.update_progressBar_processOutInfo(str(pro.readAllStandardError())))
            pro.finished.connect(lambda exitCode, exitStatus: self.stop_progressBar(100))

            self.ui_MainWindow.progressBar.setValue(0)
            self.enable_all_widget(False)

        if self.process_finish and len(self.process_array) != 0:
            self.ui_MainWindow.toolBox.setCurrentIndex(1)      
            pro = self.process_array[0]
            self.process_array = self.process_array[1:]

            self.process_finish = False
            self.log_print(pro[2])
            run_process(pro[0], pro[1])

    def add_process(self, python_env, args: list, info: str):
        self.process_array.append([python_env, args, info])

    def windows_running(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        self.windows_running_dialog = dialog
        dialog.setWindowTitle("Running...")
        layout = QtWidgets.QVBoxLayout(dialog)
        prograssbar = QtWidgets.QProgressBar()
        prograssbar.setObjectName('prograssbar')
        prograssbar.setMaximum(0)
        layout.addWidget(prograssbar)
        dialog.setModal(True)
        dialog.show()
        return dialog

    def update_windows_running_prograss(self, pro=0):
        prograssbar: QtWidgets.QProgressBar = self.windows_running_dialog.findChild(QtWidgets.QProgressBar, 'prograssbar')
        if pro == 0:
            prograssbar.setMaximum(0)
        else:
            prograssbar.setMaximum(100)
            prograssbar.setValue(int(pro))
        QtCore.QCoreApplication.processEvents()

    def visual_table(self, datas: dict):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Table")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(500)
        dialog.setModal(False)
        dialog.show()
        dialog.finished.connect(dialog.deleteLater)

        layout = QtWidgets.QVBoxLayout(dialog)
        tabWidget = QtWidgets.QTabWidget(dialog)
        tabWidget.setTabPosition(QtWidgets.QTabWidget.South)
        layout.addWidget(tabWidget)

        for name, data in datas.items():
            layout = QtWidgets.QVBoxLayout(tabWidget)
            table = QtWidgets.QTableWidget(tabWidget)
            table.verticalHeader().setVisible(True)
            layout.addWidget(table)
            tabWidget.addTab(table, name)

            data = np.array(data)
            table.setRowCount(data.shape[0])
            table.setColumnCount(data.shape[1])
            for row_index, row_data in enumerate(data):
                for col_index, col_data in enumerate(row_data):
                    item = QtWidgets.QTableWidgetItem(f"{col_data}")
                    table.setItem(row_index, col_index, item)
                self.update_windows_running_prograss(row_index / data.shape[0] * 100)

        if self.windows_running_dialog is not None:
            self.windows_running_dialog.close()

    @staticmethod
    def open3d_visual(model_files):
        models = []
        for file in model_files:
            try:
                if file.endswith('.txt'):
                    data = np.loadtxt(file)
                    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(data[:, :3]))
                    if np.max(data[:, 3:6]) > 1:
                        pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)
                    else:
                        pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])
                else:
                    pcd = o3d.io.read_point_cloud(file)

                models.append((os.path.basename(file)[:-4], pcd))
            except Exception as e:
                print(e)

        app = gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer("Model", 1280, 960)

        for i, geo in enumerate(models):
            vis.add_geometry(f"geometry_{i}_{geo[0]}", geo[1], rendering.MaterialRecord())

        vis.reset_camera_to_default()

        vis.show_settings = False
        vis.enable_raw_mode(True)
        app.add_window(vis)
        app.run()

    def visual_model(self, model_files: list):
        if False:      
            processOpen3d = multiprocessing.Process(
                target=appmain.open3d_visual,
                args=(model_files, )
            )
            processOpen3d.start()
        else:        
            new_uuid = []
            for file in model_files:
                objname = os.path.splitext(os.path.basename(file))[0]
                data = {}
                data[vType.type.path] = file
                data[vType.type.objType] = type
                data[vType.type.objName] = objname
                add_uuid = uuid.uuid4()
                new_uuid.append(add_uuid)
                gic = self._data_manager[vType.data.geometryInfoCache]
                gic[add_uuid] = data
                self._data_manager.update({vType.data.geometryInfoCache: gic})
            if new_uuid:
                self.send([vType.geometry.add, new_uuid])

    def connectFun_threadCommunicate_singal(self, info: list):
        vt = info[0]
        pass

if __name__ == '__main__':
    appmain()
