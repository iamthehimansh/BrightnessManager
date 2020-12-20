import time
import screen_brightness_control as sbc
import numpy as np
import cv2
from threading import Thread
import os
person=True
person_not_timing=0
t1=time.time()
try:
    t_time=float(input("Enter Afer how much hour the Pc will Hybernate :- "))
except:
    t_time=1
#integrated webacm on my PC
camera_port = 0
#Set up the camera
camera = cv2.VideoCapture(camera_port)
finded_obj=[]
#sutup person detaction
class persondectition(Thread):
    global finded_obj
    def load(self):
        global finded_obj
        global cap
        global net
        global classNames
        global thres
        global t1
        t1=time.time()
        print(finded_obj)
        thres = 0.45 # Threshold to detect object
        cap = cv2.VideoCapture(0)
        cap.set(3,1280)
        cap.set(4,720)
        cap.set(10,70)
        classNames= []
        classFile = r'coco.names'
        with open(classFile,'rt') as f:
                classNames = f.read().rstrip('\n').split('\n')
        configPath = r'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = r'frozen_inference_graph.pb'
        
        net = cv2.dnn_DetectionModel(weightsPath,configPath)
        net.setInputSize(320,320)
        net.setInputScale(1.0/ 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        finded_obj=[]
    def run(self):
        global person
        global image
        global t1
        while True:
            retval, img = cap.read()
            classIds, confs, bbox = net.detect(img,confThreshold=thres)
            # cv2.imshow("Output",img)
            # cv2.waitKey(10000)
            #print(classIds,bbox)
            image=img
            finded_obj=[]
            if len(classIds) != 0:
                for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                    # cv2.rectangle(img,box,color=(255,255,0),thickness=2)
                    # cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    # cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
                    # cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    # cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
                    # print(classNames[classId-1].lower()+" Founded")
                    val=classNames[classId-1].lower()
                    if val=='person':
                        cv2.rectangle(img,box,color=(255,255,0),thickness=2)
                        cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
                        cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
                    finded_obj.append(val)
            print(finded_obj)
            if 'person' in finded_obj:
                person=True
                person_not_timing=0
                t1=time.time()
            else:
                person=False
                print("NoPerson")
            print(int(time.time()-t1) , 60*60*t_time)
            time.sleep(10)
            
            # cv2.imshow("Output",img)
            # cv2.waitKey()*60*60
            

def get_image():
    "return full image out of a Capture object"
    retval, image = camera.read()
    return image
zoo=persondectition()
zoo.load()
zoo.start()
while True:
    if person!=False:
        # image = get_image()
        try:
            x = 1.0 - round((np.mean(image)/256.0) * 1,2)
        except :
            continue
        #using /sys/class/backlight to change brightness file value - Hack
        #cmd = ("sudo su -c 'echo " + str(x) + " > /sys/class/backlight/acpi_video0/brightness'")
        #using xrandr using to change brightness - Tool
        #cmd = (" xrandr --output VGA-1  --brightness " + str(x))
        #status, output = commands.getstatusoutput(cmd)
        #assert status is 0
        print(x*100)
        try:
            britness=int(x*100)
            try:
                sbc.set_brightness(britness)
                #time.sleep(10)
            except sbc.ScreenBrightnessError as error:
                print(error)
        except:
            print("X is NoneType")
    else:
        if int(time.time()-t1) > 60*60*t_time :
            print("SuttingDown")
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        
    
    
