import Util.Win32API as WinAPI
import os

from time import sleep

def start(path):
    proc = Process(path)
    return proc

class Process:
    def __init__(self, path):
        self.fullpath = path
        self.name = os.path.basename(self.fullpath)
        self.pid = None
        self.h_thread = None
        self.windowTitle = None
        self.hwnd = None
        self.start()
        

    def load(self):
        self.hwnd = WinAPI.getForegroundWindow()
        assert self.hwnd != 0, "Process load failed."

    def start(self):
        os.startfile(self.fullpath)
        self.pid = WinAPI.findPID(self.name)
        assert self.pid, f"not found '{self.name}' process."
        sleep(1)
        self.load()

    def getWindowTitle(self):
        self.windowTitle = WinAPI.getWindowTitle(self.hwnd)
        return self.windowTitle

    def close(self):
        pass
