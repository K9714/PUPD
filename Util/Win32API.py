import os
import win32api as wapi
import win32gui as wgui
import win32process as wproc
import win32con as wcon

def getWindowHwnd(proc):
    hwnd = wgui.FindWindow(None, proc.windowTitle)
    proc.hwnd = hwnd
    return hwnd

def getForegroundWindow():
    return wgui.GetForegroundWindow()

def setForegroundWindow(hwnd):
    try:
        wgui.SetForegroundWindow(hwnd)
    except Exception as e:
        pass
        #print(e)
    

def getWindowTitle(hwnd):
    return wgui.GetWindowText(hwnd)

def getWindowRect(hwnd):
    return wgui.GetWindowRect(hwnd)

def getProcessModuleAddress(handle):
    return wproc.EnumProcessModules(handle)[0]

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

def getScoreAdderss(file_name: str):
    print("PINBALL Start AOB Scan......")
    readline = os.popen(f'wmempy -n {file_name} --aob "00 ? 00 88 5A 44 00 00 ? ? ? ? 00 00 00 00 00 00 00 00 B1 01" --separator " "').read()
    info = readline.split()
    addr = int(info[-1], 16) + 0x08
    print(addr)
    print(hex(addr))
    return addr