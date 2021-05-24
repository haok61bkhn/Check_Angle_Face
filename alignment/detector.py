import os
import sys
import numpy as np
import cv2 
import torch
from torch.autograd import Variable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from retinaface_pytorch.retinaface import load_retinaface_mbnet, RetinaFace_MobileNet
from retinaface_pytorch.utils import RetinaFace_Utils
from retinaface_pytorch.align_trans import get_reference_facial_points, warp_and_crop_face # align mat giua vao 5 diem

class Retinaface_Detector(object):
    def __init__(self, device = None, thresh = 0.6, scales = [480, 640]):
        self.target_size = scales[0]
        self.max_size = scales[1]
        self.threshold = thresh

        if device:
            # assert device in
            self.device = device
        else:
            self.device = torch.device("cpu")
        self.model = RetinaFace_MobileNet()
        self.model = self.model.to(self.device)
        checkpoint = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'retinaface_pytorch/checkpoint.pth'))
        self.model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        # self.model = load_retinaface_mbnet('retinaface_pytorch/mnet.25').to(device)
        
        self.model.eval()
        self.pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.pixel_scale = float(1.0)
        self.refrence = get_reference_facial_points(default_square= True)
        self.utils = RetinaFace_Utils()
    
    def img_process(self, img):
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > self.max_size:
            im_scale = float(self.max_size) / float(im_size_max)
        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # im = im.astype(np.float32)

        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
        for i in range(3):
            im_tensor[0, i, :, :] = (im[:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i])/self.pixel_stds[2 - i]
        return im_tensor, im_scale

    def align(self, img, limit = None, min_face_size=None, thresholds = None, nms_thresholds=None):
        boxes, faces = self.align_multi(img)
        return boxes[0], faces[0]

    def align_multi(self, img, limit = None, min_face_size=None, thresholds = None, nms_thresholds=None):
        img = np.array(img)
        # img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        im, im_scale = self.img_process(img)
        im = torch.from_numpy(im)
        im_tensor = Variable(im).to(self.device)
        output = self.model(im_tensor)
        boxes, landmarks = self.utils.detect(im, output, self.threshold, im_scale)          
        if len(boxes) ==0:
            return [], []
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for i, landmark in enumerate(landmarks):
            boxes[i] = boxes[i].astype(np.int)
            landmark = landmark.astype(np.int)
            warped_face, _ = warp_and_crop_face(img, landmark, self.refrence, crop_size=(112,112))
            faces.append(Image.fromarray(warped_face))
        return boxes, faces

    def detect(self, img):
        results = []
        im, im_scale = self.img_process(img)
        im = torch.from_numpy(im)
        im_tensor = Variable(im).to(self.device)
        output = self.model(im_tensor)
        faces, landmarks = self.utils.detect(im, output, self.threshold, im_scale)
        
        if faces is None or landmarks is None:
            return results
        
        for face, landmark in zip(faces, landmarks):
            face = face.astype(np.int)
            landmark = landmark.astype(np.int)
            results.append([face, landmark])
        
        return results

import time
if __name__ == '__main__':
    font = cv2.FONT_HERSHEY_PLAIN
    reti = Retinaface_Detector()
    c = 0
    for path in os.listdir("in"):
        c += 1
        img = cv2.imread("in/testFace.jpg")
        t = time.time()
        
        boxes, faces = reti.align_multi(img)
        t2 = time.time()
        print(t2 - t)
        i = 0
        for box, face in zip(boxes, faces):
            box = box.astype(int)
            i += 1
            img_face = img[box[1] : box[3], box[0] : box[2], :]
            cv2.imwrite("face_out1/a%d.jpg" % i, img_face)
            face.save("face_out2/a%d.jpg"%i)
    
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 234, 123), 2)
            # cv2.putText(img, "face :" + " " + str(round(score, 2)), (box[0] + 30, box[1] - 10), font, 2, (0, 255, 0), 2)

        cv2.imwrite("result%d.jpg"%c, img)
            
        break
