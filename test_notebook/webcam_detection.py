import onnxruntime as ort
import random
import numpy as np
import cv2
import time
import torch
import torchvision
import argparse

import argparse

import cv2
import numpy as np
import torch
import os
from iresnet import iresnet50
from pathlib import Path

class ort_v5:
    def __init__(self, img_path, onnx_model, conf_thres, iou_thres, img_size, classes, webcam=False):
        self.webcam = webcam
        self.img_path= img_path
        self.onnx_model=onnx_model
        self.conf_thres=conf_thres
        self.iou_thres =iou_thres
        self.img_size=img_size
        # self.cuda= cuda
        self.names= classes
        self.net=None
        self.ort_session=None

    def __call__(self):
        #image preprocessing
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
        # providers = ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(self.onnx_model, providers=providers)
        self.net = iresnet50(False)
        self.net.load_state_dict(torch.load('backbone.pth',map_location=torch.device('cpu')))
        self.net.eval()
        if self.webcam:
          vid = cv2.VideoCapture(0)
          cnt = 0
          while True:
            ret, frame = vid.read()
            cnt += 1
            if (cnt%1 != 0):
                continue
            t_1 = time.time()
            output = self.detect_img(frame)
            t_2 = time.time()
            print(t_2 - t_1)
            cv2.imshow('Face Detection', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
          vid.release()
          cv2.destroyAllWindows()         
        else:
          image_or= cv2.imread(self.img_path)
          self.detect_img(image_or)

    def detect_img(self, image_or):
        
        # print(image_or)
        image, ratio, dwdh = self.letterbox(image_or, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        # print(im.shape)

        #onnxruntime session
        session= self.ort_session
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        # print('input-output names:',inname,outname)
        inp = {inname[0]:im}

        # ONNXRuntime inference
        t1 = time.time()
        outputs = session.run(outname, inp)[0]
        print('outputs type', type(outputs))
        t2 = time.time()
        output= torch.from_numpy(outputs)
        # output = torch.tensor(np.array(outputs, dtype=np.float64))

        # temp = [[o[0], o[1], o[2], o[3], o[-1], 0] for o in output_temp[0]]
        # t=output_temp[0].resize_(np.asarray(temp).shape)
        # t = [t]
        # output = torch.stack(t)
        # output[0] = torch.FloatTensor(temp)        

        out =self.non_max_suppression_face(output, self.conf_thres, self.iou_thres)[0]
        
        print(type(out))
        print(out.shape)
        print(out)
        # print('yolov5 ONNXRuntime Inference Time:', t2-t1)
        img=self.result(image_or,ratio, dwdh, out)
        if self.webcam:
            return img
        else:
            cv2.imwrite('./result.jpg', img)
        # print('result', img.shape)
        # cv2.imshow('result',img)
        # cv2.waitKey(0)
 
    # Display results
    def result(self,img,ratio, dwdh, out):
        names= self.class_name()
        colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}  
        for i,(x0,y0,x1,y1, score) in enumerate(out[:,0:5]):
            box = np.array([x0,y0,x1,y1])
            
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            # cls_id = int(cls_id)
            score = round(float(score),3)
            name = names[0]
            color = colors[name]
            name += ' '+str(score)
            # if(np.array(box).any()<0) or (box[2]>img.shape[0]) or (box[3]>img.shape[1]):
                # continue
            cropped_box=img[box[1]:box[3],box[0]:box[2]]
            if(cropped_box.shape[0]==0 or cropped_box.shape[1]==0):
                continue
            # cropped_box = cv2.cvtColor(cropped_box, cv2.COLOR_BGR2RGB)
            # print(cropped_box)
            if(type(cropped_box) == type(None)):
                pass
            else:
                cropped_box= cv2.resize(cropped_box, (112, 112))
            # print(cropp)
            cropped_box = np.transpose(cropped_box, (2, 0, 1))
            cropped_box = torch.from_numpy(cropped_box).unsqueeze(0).float()
            cropped_box.div_(255).sub_(0.5).div_(0.5)
            print(cropped_box)
            import os
            ts=os.listdir('database_tensor')
            lc=0
            w=[]
            feat = self.net(cropped_box).detach().numpy()
            for i in range (len(ts)):
                tmp=np.load('database_tensor'+'/'+ts[i])
                w.append(tmp)
                # distance = torch.cdist(feat,tmp)
                distance = np.linalg.norm(feat-tmp)
                # if distance<=torch.cdist(feat, tmp):
                if distance<=np.linalg.norm(feat - w[lc]):
                    lc=i                     
            cv2.rectangle(img,box[:2],box[2:],color,2)
            cv2.putText(img,ts[lc].replace('.npy',''),(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2) 
            # cv2.putText(img,'person'.replace('.npy',''),(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
        return img
 
    def box_iou(self,box1, box2, eps=1e-7):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def box_iou(box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
                torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        # iou = inter / (area1 + area2 - inter)
        return inter / (area1[:, None] + area2 - inter)


    def non_max_suppression_face(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
        """Performs Non-Maximum Suppression (NMS) on inference results
        Returns:
            detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """

        nc = prediction.shape[2] - 15  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 15), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 15] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, landmarks, cls)
            if multi_label:
                i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 15, None], x[i, 5:15] ,j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 15:].max(1, keepdim=True)
                x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Batched NMS
            c = x[:, 15:16] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            #if i.shape[0] > max_det:  # limit detections
            #    i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        return output

    

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    # Read classes.txt 
    def class_name(self):
        classes=[]
        file= open(self.names,'r')
        while True:
          name=file.readline().strip('\n')
          classes.append(name)
          if not name:
            break
        return classes

    def letterbox(self, im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape= self.img_size
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    # Initialize ONNXRuntime session   
    # def ort_session(self):
    #     providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
    #     # providers = ['CPUExecutionProvider']
    #     session = ort.InferenceSession(self.onnx_model, providers=providers)
    #     # print(session.get_providers())
    #     return session

image = './/yolov5-face//data/images/test.jpg'
# weights = './/yolov5-face//yolov5m-face.onnx'
weights = './yolov5m-face.onnx'
conf = 0.7
iou_thres = 0.5
img_size = 640
classes_txt = './/yolov5-face//classes.txt'

ORT= ort_v5(image, weights, conf, iou_thres, (img_size, img_size), classes=classes_txt, webcam=True)
ORT()