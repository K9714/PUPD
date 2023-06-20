import numpy as np
import pyautogui as pag
import time

from DQN.Model.Agent import Agent
from time import sleep
from Util.Memory import rwm
from Util.Image import screenshot, imageSearchExByArray
from Util.ImageScaler import *
from Util.Process import Process
from PIL import Image, ImageDraw

class Env():
    def __init__(self, base_path: str, conf: dict, proc: Process):
        self.base_path = base_path
        self.proc = proc
        self.score = 0
        self.local_min_check = False
        self.local_min_time = 0
        self.dead_reward = False
        pag.PAUSE = 0
        pag.FAILSAFE = False
        self._setup(conf)

    # Private Methods
    def _setup(self, conf: dict):
        self._load_frame_skip(conf['HParams'])
        self._load_keys(conf['Env'])
        self._load_images(conf['Training'])
        self._load_memory_address()
        self._load_ranges(conf['Training'])
        self._load_input_output(conf['Training'])
        self._load_max_coordinate(conf['Training'])

    def _load_frame_skip(self, conf_hparams: dict):
        self.frame_skip = conf_hparams['frame_skip']
        self.frame_skip_count = 0
        
    def _load_keys(self, conf_env):
        self.key_set = conf_env['key_set']
        self.key_state = [0 for _ in range(len(self.key_set))]

    def _load_images(self, conf_training: dict):
        resource_path = conf_training['resource_path']
        ball_images = conf_training['ball_images']
        dead_images = conf_training['dead_images']
        plunger_image = conf_training['plunger_image']
        mask_image = conf_training['mask_image']

        self.ball_images = []
        self.dead_images = []
        self.plunger_image = None

        for f in ball_images:
            img = Image.open(self.base_path + resource_path + f)
            temp = np.array(img)
            self.ball_images.append(temp)

        for f in dead_images:
            img = Image.open(self.base_path + resource_path + f)
            temp = np.array(img)
            self.dead_images.append(temp)

        img = Image.open(self.base_path + resource_path + plunger_image)
        temp = np.array(img)
        self.plunger_image = temp

        img = Image.open(self.base_path + resource_path + mask_image).convert('RGBA')
        self.mask_image = img

    def _load_memory_address(self):
        base = self.proc.getProcessModuleAddress()
        print(f"base : {hex(base)}")
        base += 0x2506C
        ptr1 = rwm.ReadProcessMemory(self.proc.handle, base)
        ptr2 = rwm.ReadProcessMemory(self.proc.handle, ptr1)
        print(f"base : {hex(base)}\nptr1 : {hex(ptr1)}\nptr2 : {hex(ptr2)}")
        self.x_pos_addr = ptr2
        self.y_pos_addr = ptr2 + 0x04
        print(f"x_pos_addr : {hex(self.x_pos_addr)}\ny_pos_addr : {hex(self.y_pos_addr)}")
        print(f"ADDR - PTR1 : {hex(self.proc.score_addr - ptr1)}")

    def _load_ranges(self, conf_training: dict):
        self.start_point = conf_training['start_point']
        self.warn_point = conf_training['warn_point']
        self.in_start = False
        self.in_warn = False
        self.in_special = False
        self.in_negative = False

    def _load_input_output(self, conf_training: dict):
        self.num_states = conf_training['num_states']
        self.num_actions = conf_training['num_actions']

    def _load_max_coordinate(self, conf_training: dict):
        self.max_x = conf_training['max_x']
        self.max_y = conf_training['max_y']
        self.x = 0
        self.y = 0


    def _is_in_range(self, value: int, range: list) -> bool:
        if range[0] <= value <= range[1]:
            return True
        return False

    def _in_start_point(self, x: int, y: int) -> bool:
        self.in_start = False
        if self._is_in_range(x, self.start_point[:2]) and self._is_in_range(y, self.start_point[2:]):
            self.in_start = True
            self.dead_reward = False
        return self.in_start

    def _in_warn_point(self, x: int, y: int) -> bool:
        self.in_warn = False
        if self._is_in_range(x, self.warn_point[:2]) and self._is_in_range(y, self.warn_point[2:]):
            self.in_warn = True
        return self.in_warn

    def _set_keys(self, key_state: list):
        for i, state in enumerate(key_state):
            if self.key_state[i] != state and i != 3:
                self.key_state[i] = state
                if state == 0:
                    pag.keyUp(self.key_set[i])
                elif state == 1:
                    pag.keyDown(self.key_set[i])
            elif i == 3 and state == 1:
                pag.keyDown(self.key_set[i])
                sleep(0.05)
                pag.keyUp(self.key_set[i])
            #elif i != 2 and state == 1:
            #    pag.keyDown(self.key_set[i])
            #    sleep(0.05)
            #    pag.keyUp(self.key_set[i])
                    

    # Public Methods
    def action(self, act: int):
        act = int(act)
        if act == 1:
            self._set_keys([1, -1, -1, -1, -1])
        elif act == 2:
            self._set_keys([0, -1, -1, -1, -1])
        elif act == 3:
            self._set_keys([-1, 1, -1, -1, -1])
        elif act == 4:
            self._set_keys([-1, 0, -1, -1, -1])
        elif act == 5:
            self._set_keys([-1, -1, -1, 1, -1])

        elif act == 9:
            self._set_keys([-1, -1, 1, -1, -1])
        elif act == 10:
            self._set_keys([-1, -1, 0, -1, -1])
        else:
            self._set_keys([0, 0, 0, 0, 0])
        #if act == 0:
        #    self._set_keys([1, 0, 0, 0, 0])
        #elif act == 1:
        #    self._set_keys([0, 1, 0, 0, 0])
        #elif act == 2:
        #    self._set_keys([1, 1, 0, 0, 0])
        #elif act == 3:
        #    self._set_keys([0, 0, 1, 0, 0])
        #elif act == 4:
        #    self._set_keys([0, 0, 0, 1, 0])
        #elif act == 5:
        #    self._set_keys([0, 0, 0, 0, 1])
        #else:
        #    self._set_keys([0, 0, 0, 0, 0])

    def update(self) -> np.ndarray:
        """
        state[0] : ball_x
        state[1] : ball_y
        state[2] : ball_x_diff
        state[3] : ball_y_diff
        """
        self.states = [0 for _ in range(self.num_states)]
        
        frame_origin = screenshot(self.proc)
        frame = np.array(frame_origin)

        #-----------------------------------------
        # states[0:1]
        # Retrieve ball coordinate from frame
        old_x, old_y = self.x, self.y
        self.x, self.y = 0, 0

        self.x = rwm.ReadProcessMemory(self.proc.handle, self.x_pos_addr)
        self.y = rwm.ReadProcessMemory(self.proc.handle, self.y_pos_addr)
        self._in_start_point(self.x, self.y)
        self._in_warn_point(self.x, self.y)

        self.states[0] = self.x / 360.
        self.states[1] = self.y / 410.
        self.states[2] = old_x - self.x
        self.states[3] = old_y - self.y
        self.score = rwm.ReadProcessMemory(self.proc.handle, self.proc.score_addr)
        
        """
        if self._in_start_point(self.x, self.y):
            self.states[0] = 1
        if self._in_warn_point(self.x, self.y):
            self.states[1] = 1
        
        #-----------------------------------------
        # states[2]
        # Check whether scores have changed
        old_score = self.score
        self.score = rwm.ReadProcessMemory(self.proc.handle, self.proc.score_addr)
        self.states[2] = int(old_score != self.score)

        #-----------------------------------------
        # states[3]
        # Confirmation of entry to special area
        self.states[3] = 0

        #-----------------------------------------
        # states[4:5]
        # Update ball coordinate (normalized)
        self.states[4] = self.x / self.max_x
        self.states[5] = self.y / self.max_y

        #-----------------------------------------
        # states[6]
        # Check plunger maximum
        ret = imageSearchExByArray(frame, self.plunger_image)
        if len(ret) > 0:
            self.states[6] = 1
            self.plunger_full = True
        else:
            self.plunger_full = False
        
        #-----------------------------------------
        # states[7:8]
        # Update ball velocity & gradient
        self.states[8] = 0
        if old_x == 0 or old_y == 0 or self.x == 0 or self.y == 0:
            self.states[7] = 0
        else:
            self.states[7] = abs(self.x - old_x) + abs(self.y - old_y)
            dx = (self.x - old_x)
            if dx != 0:
                self.states[8] = round((self.y - old_y) / dx, 4)
        """
        ret = imageSearchExByArray(frame, self.plunger_image)
        if len(ret) > 0:
            self.plunger_full = True
        else:
            self.plunger_full = False
        #-----------------------------------------
        # Check game ended
        self.game_end = False
        for i, dead_img in enumerate(self.dead_images):
            ret = imageSearchExByArray(frame, dead_img)
            if len(ret) == 1:
                self.game_end = True
                break

        if not self.local_min_check and self.in_warn:
            self.local_min_check = True
            self.local_min_time = time.time()
        else:
            if not self.in_warn:
                self.local_min_check = False
            elif (time.time() - self.local_min_time) > 30:
                self.game_end = True


        # ret = imageSearchExByArray(frame, self.dead_images[1])
        # if len(ret) == 1:
        #     self.game_end = True
        # ret = imageSearchExByArray(frame, self.dead_images[1])
        # if len(ret) == 1:
        #     self.game_end = True
        
        crop_image = get_crop_image(frame_origin)
        """
        flipper_area_image, _, areas = get_flipper_area(crop_image)
        for contour in areas:
            data = contour.reshape(-1, 2)
            cx = int(np.mean(data[:, 0] + 126))
            cy = int(np.mean(data[:, 1] + 344))
            if cx >= 180.0:
                self.right_flipper_x = cx / 360.
                self.right_flipper_y = cy / 410.
            else:
                self.left_flipper_x = cx / 360.
                self.left_flipper_y = cy / 410.
        self.states[4] = self.left_flipper_x
        self.states[5] = self.left_flipper_y
        self.states[6] = self.right_flipper_x
        self.states[7] = self.right_flipper_y
        """
        gray_image = get_gray_scale_image(crop_image)
        resize_image = get_resize_image(gray_image)

        self.states = np.array(resize_image)
        

    def reset(self):
        # Wait 3 Sec
        sleep(3)
        # ESC Trigger
        pag.press('enter')
        sleep(1)
        # F2 Trigger
        pag.press('f2')
        # Wait 3 sec
        sleep(3)
        self.action(-1)
        self.update()
        return self.states

    def step(self, act: list) -> tuple:
        """새로운 state 를 갱신합니다.

        Returns:
            frame (numpy.ndarray): 해당 시점의 스냅샷 프레임 이미지 데이터,
            states (list): 해당 시점의 최신 상태 갱신 값,
            reward (int): 해당 시점의 보상 값,
            game_end (bool): 게임 종료 여부
        """
        old_in_start = self.in_start
        old_in_warn = self.in_warn
        old_x = self.x
        old_y = self.y
        old_score = self.score
        old_plunger_full = self.plunger_full
        old_in_special = self.in_special
        old_in_negative = self.in_negative

        # frame skip
        #if old_x != 0 and old_y != 0:
        #    self.frame_skip_count += 1
        #else:
        #    self.frame_skip_count = 0

        #if self.frame_skip_count == self.frame_skip:
        #    self.frame_skip_count = 0
        self.action(act)

        self.update()

        # reward
        reward = 0
        """
        # if ball continue to stay at the starting point
        #if old_in_start == True and self.in_start == False:
        #    reward += 1
        #elif old_in_start == True and self.in_start == True:
        #    reward -= 1

        # if the plunger is entered at starting point
        #if self.in_start and not old_plunger_full:
        #    if act == 3:
        #        reward += 1
        # if the plunger is continuously pulled at the starting point
        if self.in_start and old_plunger_full:
            if act == 6:
                reward += 1

        if not self.in_start and act == 3:
            reward -= 1

        # if the ball is out of danger zone
        if old_in_warn == True and self.in_warn == False:
            reward += 10
        elif old_in_warn == True and self.in_warn == True:
            reward -= 1

        # if it is not a dangerous area and moves
        if not old_in_warn:
            if act <= 2 or act == 4 or act == 5:
                reward -= 1

        # movements below 5 in dangerous areas
        if old_in_warn and (act == 3 or act > 5):
            reward -= 1
            # if abs(old_x - self.x) + abs(old_y - self.y) <= 5:
            #     reward -= 1
            # else:
            #     reward += 1
        """
        """
        # if the coordinates are consistently the same
        if (self.x != 0 and self.y != 0) and old_x == self.x and old_y == self.y:
            reward -= 1
        # if the score is changed
        if old_score != self.score:
            reward += 1
        if old_in_warn == True and self.in_warn == False:
            reward += 10
        elif old_in_warn == True and self.in_warn == True:
            reward -= 1
        if old_in_negative == False and self.in_negative == True:
            reward -= 10
        if self.in_special == True:
            reward += 10
        if old_in_warn and (act == 3):
            reward -= 1

        # if y-coordinates increase
        #if (old_y != 0 and self.y != 0) and (old_y > self.y) and not self.in_start:
        #    reward += 1
        """

        # 점수가 같으면 작게 벌줌
        #if old_score == self.score:
        #    reward = -1

        if self.in_warn:
            if (self.x != 0 and self.y != 0) and old_x == self.x and old_y == self.y:
                reward -= 1

        if self.y > 400 and not self.dead_reward:
            self.dead_reward = True
            reward = -10

        # 점수가 변동되면 변동량의 0.001 배수만큼 보상
        if old_score != self.score and self.y < 390:
            reward = 0.001 * (self.score - old_score)
        

        return self.states, reward, self.game_end