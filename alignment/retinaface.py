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
from retinaface_pytorch.align_trans import get_reference_facial_points, warp_and_crop_face
from torchvision import transforms as trans

def sort_list(list1): 
    z = [list1.index(x) for x in sorted(list1, reverse = True)] 
    return z 
class Retinaface_Detector(object):
    def __init__(self, device = None, thresh = 0.5, atrib = True,  scales = [480, 640]):
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

        self.model.eval()
        self.pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.pixel_scale = float(1.0)
        self.refrence = get_reference_facial_points(default_square= True)
        self.utils = RetinaFace_Utils()

    def align(self, img, limit = None, min_face_size=None, thresholds = None, nms_thresholds=None):
        dict_output = self.align_multi(img)
        if len(dict_output['faces']) > 0:
            return dict_output["bboxs"][0], dict_output['faces'][0]
        return None, None

    def align_multi(self, img, limit = None, min_face_size=None, thresholds = None, nms_thresholds=None):
        faces = []
        faces_2_tensor = []
        genders = []
        ages = []
        dict_result = {}

        img = np.array(img)
        im, im_scale = self.img_process(img)
        im = torch.from_numpy(im)
        im_tensor = Variable(im).to(self.device)
        output = self.model(im_tensor)
        sort = True
        boxes, landmarks = self.utils.detect(im, output, self.threshold, im_scale)
        if len(boxes) ==0:
            return [], []     
        if limit: 
            boxes, landmarks = boxes[:limit], landmarks[:limit]

        boxes = boxes.astype(np.int)
        landmarks = landmarks.astype(np.int)
        face_area = (boxes[:,2] - boxes[:,0])*(boxes[:,3]-boxes[:, 1])
        if len(boxes) > 0 and sort:
            face_area = (boxes[:,2] - boxes[:,0])*(boxes[:,3]-boxes[:, 1])
            indexs = np.argsort(face_area)[::-1]
            boxes = boxes[indexs]
            landmarks = landmarks[indexs]
            for i, landmark in enumerate(landmarks):
                warped_face, face_img_tranform = warp_and_crop_face(img, landmark, self.refrence, crop_size=(112,112))
                face = Image.fromarray(warped_face)
                faces.append(face)
                faces_2_tensor.append(torch.FloatTensor(face_img_tranform).to(self.device).unsqueeze(0))
        num_face = len(boxes)
        dict_result["num_face"] = num_face
        dict_result["bboxs"] = boxes
        dict_result["faces"] = faces
        dict_result["faces_to_tensor"] = faces_2_tensor
        dict_result["landmarks"] = landmarks
        return dict_result

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


    def age_gender_pred(self, img):
        weigh = np.linspace(1, self.age_cls_unit, self.age_cls_unit)
        input = self.transform_gen_age(img).unsqueeze(0).cuda()
        gender_out, age_out = self.model_gen_age(input)
        gen_prob, gen_pred = torch.max(gender_out, 1)
        gen_pred = gen_pred.cpu().data.numpy()[0]
        age_probs = age_out.cpu().data.numpy()
        age_probs.resize((self.age_cls_unit,))
        # expectation and variance
        # age_prob, age_pred_a = torch.max(age_out, 1)
        
        age_pred = sum(age_probs * weigh)
        age_var = np.square(np.mean(age_probs * np.square(weigh - age_pred)))
        return gen_pred, age_pred
import time

def test_camera():
    # Loading image
    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0
    while True:
        _, frame = cap.read()
        frame_id += 1
        if frame_id % 2 == 0 :
            continue
        frame = cv2.flip(frame, 1)
        dict_result = reti.align_multi(frame)
        bboxs, faces, landmarks = dict_result["bboxs"], dict_result["faces"], dict_result["landmarks"]

        for box, landmark in zip(bboxs, landmarks):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 234, 123), 2)
            landmark = landmark.reshape(10, 1)
            cv2.circle(frame,(landmark[0],landmark[1]),radius=1,color=(0,0,255),thickness=2)
            cv2.circle(frame,(landmark[2],landmark[3]),radius=1,color=(0,255,0),thickness=2)
            cv2.circle(frame,(landmark[4],landmark[5]),radius=1,color=(255,0,0),thickness=2)
            cv2.circle(frame,(landmark[6],landmark[7]),radius=1,color=(0,255,255),thickness=2)
            cv2.circle(frame,(landmark[8],landmark[9]),radius=1,color=(255,255,0),thickness=2)
    
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (50, 50), font, 4, (123, 255, 0), 3)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    font = cv2.FONT_HERSHEY_PLAIN
    reti = Retinaface_Detector()
    # c = 0
    # for path in os.listdir("in"):
    #     c += 1
    #     img = cv2.imread("testFace.jpg")
    #     t = time.time()
        
    #     dict_result = reti.align_multi(img)
    #     t2 = time.time()
    #     print(t2 - t)
    #     bboxs, faces, landmarks = dict_result["bboxs"], dict_result["faces"], dict_result["landmarks"]
    #     i = 0
    #     for box, landmark, face in zip(bboxs, landmarks, faces):
    #         i += 1
    #         img_face = img[box[1] : box[3], box[0] : box[2], :]
    #         cv2.imwrite("face_out1/a%d.jpg" % i, img_face)
    #         face.save("face_out2/a%d.jpg"%i)
    #         landmark = landmark.reshape(10, 1)
    
    #         cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 234, 123), 2)
    #         cv2.circle(img,(landmark[0],landmark[1]),radius=1,color=(0,0,255),thickness=2)
    #         cv2.circle(img,(landmark[2],landmark[3]),radius=1,color=(0,255,0),thickness=2)
    #         cv2.circle(img,(landmark[4],landmark[5]),radius=1,color=(255,0,0),thickness=2)
    #         cv2.circle(img,(landmark[6],landmark[7]),radius=1,color=(0,255,255),thickness=2)
    #         cv2.circle(img,(landmark[8],landmark[9]),radius=1,color=(255,255,0),thickness=2)
    #         # cv2.putText(img, "face :" + " " + str(round(score, 2)), (box[0] + 30, box[1] - 10), font, 2, (0, 255, 0), 2)

    #     cv2.imwrite("result%d.jpg"%c, img)
            
    #     break
    test_camera()