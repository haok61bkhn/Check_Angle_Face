import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glob
import cv2
from head_pose import PoseEstimator
from alignment.retinaface_pytorch.detector import Retinaface_Detector

class Checker:
    def __init__(self,img_size):
        self.ang=PoseEstimator(img_size=img_size)# img_size  = image_shape
        self.detector=Retinaface_Detector()

    def check(self,ang):
        for x in range(3):
            if(abs(ang[x])>=85): return False
        return True
    
    def detect_face(self,img):
        dets=self.detector.detect(img)# list([face, landmark])
        return dets

    def check_face(self,landmark):
        angle=self.ang.face_orientation(landmark)
        print(angle)
        return self.check(angle)

if __name__=="__main__":
    cam=cv2.VideoCapture("/dev/video2")
    _,frame=cam.read()
    X=Checker(img_size=(frame.shape[0],frame.shape[1]))
    while frame is not None:
        dets=X.detect_face(frame)
        for det in dets:

            print(X.check_face(det[1]))
        
        cv2.imshow("image",frame)
        if cv2.waitKey(1) == ord('q'):
         break
        _,frame = cam.read()
