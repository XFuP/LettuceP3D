# import win32com.client
# import win32con
import win32gui
import enum
from PyQt5 import QtCore
import uuid

###############################################################################
# communicate_Thread Class
###############################################################################
class communicate_Thread_callSignal(QtCore.QThread):
    communicate_singal = QtCore.pyqtSignal(list)

    def __init__(self, queue) -> None:
        super(communicate_Thread_callSignal, self).__init__()
        self.queue = queue
        self.running = False

    def run(self):
        while True:
            info = self.queue.get()
            self.communicate_singal.emit(info)


class communicate_Thread_callFun(QtCore.QThread):

    def __init__(self, queue, fun) -> None:
        super(communicate_Thread_callFun, self).__init__()
        self.queue = queue
        self.running = False
        self.fun = fun

    def run(self):
        while True:
            info = self.queue.get()
            self.fun(info)


###############################################################################
# vType Class
###############################################################################
class vType():

    class all(enum.Enum):
        finish = enum.auto()
        error = enum.auto()
        close = enum.auto()

    class geometry(enum.Enum):
        add = enum.auto()
        save = enum.auto()
        show = enum.auto()
        select = enum.auto()
        sgtOut = enum.auto()

    class geometry_mesh(enum.Enum):
        add = enum.auto()
        save = enum.auto()
        show = enum.auto()
        select = enum.auto()

    class geometry_pointcloud(enum.Enum):
        add = enum.auto()
        save = enum.auto()
        show = enum.auto()
        select = enum.auto()

    class sence(enum.Enum):
        cameraReset = enum.auto()
        showAxis = enum.auto()
        showSkybox = enum.auto()
        showGroundPlane = enum.auto()
        addPointSize = enum.auto()
        dePointSize = enum.auto()
        rawMode = enum.auto()

    class type(enum.Enum):
        noneObj = enum.auto()
        objType = enum.auto()
        objName = enum.auto()
        path = enum.auto()
        raw = enum.auto()
        box = enum.auto()
        show = enum.auto()

        mesh = enum.auto()
        pointCloud = enum.auto()
        line = enum.auto()

        hasSgt = enum.auto()            # bool
        hasPhenoFile = enum.auto()      # srt: path

    class data(enum.Enum):
        geometryInfoCache = enum.auto()
        _uuidSet = enum.auto()          # 用于查询uuid重复
        geoUuidList = enum.auto()       # 用于获得不同类别的uuid

    class widgets(enum.Enum):
        progressBar = enum.auto()

    class log(enum.Enum):
        log = enum.auto()
        info = enum.auto()
        debug = enum.auto()
        warning = enum.auto()
        error = enum.auto()


###############################################################################
# deafault data
###############################################################################
class DeafaultData:
    phenotypic_table_title = {
        'P': ['品种编号', '叶片数', '麦穗数', '分蘖数', '叶面积'],
        'T': ['分蘖编号', '分蘖长度', '分蘖倾角', '方位角', '叶片数', '麦穗长', '麦穗宽', '麦穗体积', '位置 X', '位置 Y', '位置 Z'],
        'L': ['叶片序号', '叶片长度', '叶片宽度', '叶片倾角', '位置 X', '位置 Y', '位置 Z'],
    }


###############################################################################
# SetForegroundWindow Class
###############################################################################
class SetForegroundWindow():

    def __init__(self) -> None:
        self.hwnd_map = {}
        win32gui.EnumWindows(self.get_all_hwnd, 0)

    def get_all_hwnd(self, hwnd, mouse):
        if (win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd)):
            self.hwnd_map.update({hwnd: win32gui.GetWindowText(hwnd)})

    def setWin(self, name):
        for h, t in self.hwnd_map.items():
            if t:
                if t == name:
                    # h 为想要放到最前面的窗口句柄
                    # print(h)
                    win32gui.BringWindowToTop(h)

                    # shell = win32com.client.Dispatch("WScript.Shell")
                    # shell.SendKeys('%')

                    # 被其他窗口遮挡，调用后放到最前面
                    # win32gui.SetForegroundWindow(h)

                    # 解决被最小化的情况
                    # win32gui.ShowWindow(h, win32con.SW_SHOW)
                    # win32gui.UpdateWindow(h)


###############################################################################
# util Class
###############################################################################
class util():
    @staticmethod
    def rename(setName: str, allName: list):
        name = setName
        n = 0
        while name in allName:
            name = setName + '(' + str(n) + ')'
            n = n + 1
        return name
