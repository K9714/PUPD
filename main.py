"""
    Title. 
"""
import os
import pyautogui as pag

from time import sleep
from Util.Image import *
import Util.Process as Process

PINBALL_PATH = "/Game/"
PINBALL_NAME = "PINBALL.EXE"

def runPinball():
    dirname = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    proc = Process.start(dirname + PINBALL_PATH + PINBALL_NAME)
    addr = WinAPI.getScoreAdderss()
    while True:
        sleep(1)
        title = proc.getWindowTitle()
        print(f"name : {title}, addr : {addr}")
        image = screenshot(proc)
        


if __name__ == "__main__":
    runPinball()
