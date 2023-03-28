import Util.Win32API as WinAPI
import pyautogui as pag
import numpy as np
import PyAutoMaker.image
from Util.Process import Process
from PIL import Image, ImageGrab

def screenshot(proc: Process) -> Image:
    WinAPI.getWindowHwnd(proc)
    WinAPI.setForegroundWindow(proc.hwnd)
    rect = WinAPI.getWindowRect(proc.hwnd)
    image = ImageGrab.grab(rect)
    return image

def imageSearchExByFile(src: np.ndarray, fpath: str) -> list:
    target = Image.open(fpath)
    target = np.array(target)

    return PyAutoMaker.image.imageSearchEx(src, target)

def imageSearchExByArray(src: np.ndarray, target: np.ndarray) -> list:
    return PyAutoMaker.image.imageSearchEx(src, target)
    