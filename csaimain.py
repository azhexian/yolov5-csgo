

from PIL import ImageGrab
from grabscreen import grab_screen
import cv2
import win32gui
import win32con
from csmodels import loal_model

import csmodels as cs_val
import torch
from utils.augmentations import letterbox
from utils.general import (non_max_suppression,scale_coords,xyxy2xywh)
import numpy as np
##获取屏幕图片
img=ImageGrab.grab()
##获取长宽
x,y=img.width,img.height
re_x,re_y=img.width,img.height
##导入参数
device=cs_val.device
half=cs_val.half
##默认参数
augment=False
visualize=False
conf_thres=0.4
iou_thres=0.45
imgsz=640
aims=[]
names= ['ct_head','t_head','t','ct',] 
##方框顏色
fk_color=(0,255,0)
##导入模型
model =loal_model()
stride=int(model.stride)
names=model.names
##窗口名称
windowName='csgo-ai'
while True:
    ##获取屏幕图片
    img0=grab_screen(region=(0,0,x,y))
    ##图片转换
    img = letterbox(img0, imgsz, stride=stride)[0]
    #im=letterbox(img0,imgsz,stride=stride)[0]
    img = img[..., ::-1].transpose((2,0,1))  # BGR to RGB, BHWC to BCHW
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
       im = im[None]  # expand for batch dim
    pred = model(im, augment=augment)
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    #print(pred)
    for i, det in enumerate(pred):  # per imagem
        s=''
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line =(cls, *xywh)  # label format
                aim=('%g ' * len(line)).rstrip() % line
                #print(aim)
                aim=aim.split(' ')
                aims.append(aim)
        if len(aims):
            for i, det in enumerate(aims):

                ##解析數組
                labelsvalue,x_center,y_center,out_width,out_height=det
                ##計算
                x_center,out_width=re_x * float(x_center),re_x * float(out_width)
                y_center,out_height=re_y * float(y_center),re_y * float(out_height)
                ##左上角dian位
                top_left=(int(x_center -out_width/2.0),int(y_center-out_height/2.0))
                ##右下角点位
                bottom_right=(int(x_center + out_width/2.0),int(y_center + out_height/2.0))
                print(names[int(labelsvalue)])
                cv2.rectangle(img0,top_left,bottom_right,fk_color,thickness=6)
                aims=[]
            print(len(aims))
           
                    
                

    ##生成窗口
    cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
    ##窗口大小
    cv2.resizeWindow(windowName,re_x//6,re_y//6)
    cv2.imshow(windowName,img0)
    ##找到窗口
    findWindow=win32gui.FindWindow(None,windowName)
    CVRECT=cv2.getWindowImageRect(windowName)
    ##设置顶部,可移动
    win32gui.SetWindowPos(findWindow,win32con.HWND_TOPMOST,1920,1080,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        ##关闭窗口
       cv2.destroyAllWindows()
       break