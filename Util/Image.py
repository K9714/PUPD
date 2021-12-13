import Util.Win32API as WinAPI
import pyautogui as pag

from PIL import ImageGrab

def screenshot(proc):
    WinAPI.getWindowHwnd(proc)
    WinAPI.setForegroundWindow(proc.hwnd)
    rect = WinAPI.getWindowRect(proc.hwnd)
    image = ImageGrab.grab(rect)
    return image