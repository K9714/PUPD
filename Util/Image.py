from PIL import ImageGrab
import os
import win32gui
import pyautogui as pag

def screenshot(proc):
    win32gui.SetForegroundWindow(proc.hwnd)
    rect = win32gui.GetWindowRect(proc.hwnd)
    image = ImageGrab.grab(rect)
    return image