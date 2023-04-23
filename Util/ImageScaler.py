import cv2
import numpy as np
from PIL import Image

def get_crop_image(src, rect=(3, 48, 360+3, 410+48)):
    ret = src.crop(rect).convert('RGB')
    return ret

def get_gray_scale_image(src):
    ret = src.convert('L')
    return ret

def get_resize_image(src, size=(180, 180)):
    ret = src.resize(size)
    return ret

def get_flipper_area(src: Image):
    crop = src.crop((126, 344, 126+115, 344+66))
    cimg = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cimg, cv2.COLOR_BGR2HSV)
    color_range = ((0, 0, 140), (40, 40, 200))
    img_mask = cv2.inRange(hsv, color_range[0], color_range[1])
    img_result = cv2.bitwise_and(cimg, cimg, mask = img_mask)
    gimg = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    moments = []
    for contour in contours:
        if len(contour) < 30:
            continue
        M = cv2.moments(contour, False)
        cX = int(M['m10'] / M['m00']) + 126
        cY = int(M['m01'] / M['m00']) + 344
        moments.append((cX, cY))
        cimg = cv2.drawContours(cimg, [contour], -1, (0, 0, 255), 2)
    rgb = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    ret = Image.new("RGB", (src.width, src.height))
    ret.paste(src, (0, 0))
    ret.paste(img, (126, 344))
    
    return ret, moments