import os
import Util.Win32API as WinAPI

from time import sleep
from Util.Memory import rwm

PROCESS_KILLED = 65892

def start(path, conf_pinball: dict):
    proc = Process(path, conf_pinball)
    return proc

class Process:
    def __init__(self, path: str, conf_pinball: dict):
        self.fullpath = path
        self.name = os.path.basename(self.fullpath)
        self.pid = None
        self.h_thread = None
        self.windowTitle = None
        self.hwnd = None
        self.killed = False
        self.start()
        # Get Pinball score memory address
        self.score_addr = WinAPI.getScoreAdderss(conf_pinball['file_name'])
        
    def load(self):
        self.hwnd = WinAPI.getForegroundWindow()
        assert self.hwnd != 0, "Process load failed."

    def start(self):
        print(self.fullpath)
        os.startfile(self.fullpath)
        self.pid = WinAPI.findPID(self.name)
        assert self.pid, f"not found '{self.name}' process."
        sleep(2)
        self.load()
        # Get Pinball process handle
        self.handle = rwm.OpenProcess(self.pid)
        self.getWindowTitle()

    def getWindowTitle(self):
        self.windowTitle = WinAPI.getWindowTitle(self.hwnd)

    def getProcessModuleAddress(self):
        return WinAPI.getProcessModuleAddress(self.handle)

    def close(self):
        pass
