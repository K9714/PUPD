import os
import win32api as wapi
import win32gui as wgui
import win32process as wproc
import win32con as wcon

def getWindowHwnd(proc):
    hwnd = wgui.FindWindow(None, proc.name)
    proc.hwnd = hwnd
    return hwnd

def getForegroundWindow():
    return wgui.GetForegroundWindow()

def setForegroundWindow(hwnd):
    print(hwnd)
    wgui.SetForegroundWindow(hwnd)

def getWindowTitle(hwnd):
    return wgui.GetWindowText(hwnd)

def findPID(proc_name):
    readline = os.popen('tasklist /FI "IMAGENAME eq '+ proc_name + '"').read()
    info = readline.split("\n")
    info = info[1:]
    if len(info) < 3:
        return False
    proc_data = info[2].split()
    img, pid, sname, session, mem, size = proc_data
    mem += size
    return int(pid)