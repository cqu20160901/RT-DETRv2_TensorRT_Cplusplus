#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import math
import onnxruntime as ort
import time

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

ClassNum = len(CLASSES)

ObjThresh = 0.6
input_imgH = 640
input_imgW = 640
max_num = 300


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def precess_image(img_src, resize_w, resize_h):
    image = cv2.resize(img_src, (resize_w, resize_h))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255
    return image


def postprocess(pred_results):

    output = []
    for i in range(len(pred_results)):
        output.append(pred_results[i].reshape((-1)))

    pred_logits = output[0]
    pred_boxes = output[1]

    predBoxs = []
    for i in range(max_num):
        softmaxmax = 0
        softmaxindex = 0

        for c in range(ClassNum):
            if c == 0:
                softmaxmax = pred_logits[i * ClassNum + c]
                softmaxindex = c
            else:
                if softmaxmax < pred_logits[i * ClassNum + c]:
                    softmaxmax = pred_logits[i * ClassNum + c]
                    softmaxindex = c

        if softmaxmax > ObjThresh:
            cx, cy, w, h = pred_boxes[i * 4 + 0], pred_boxes[i * 4 + 1], pred_boxes[i * 4 + 2], pred_boxes[i * 4 + 3]
            box = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]

            rect = DetectBox(softmaxindex, softmaxmax, box[0], box[1], box[2], box[3])
            predBoxs.append(rect)

    return predBoxs


def detect(img_path):
    orig = cv2.imread(img_path)
    img_h, img_w = orig.shape[:2]
    image = precess_image(orig, input_imgW, input_imgH)

    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)

    ort_session = ort.InferenceSession('./rtdetrv2_r18vd.onnx')
    pred_results = (ort_session.run(None, {'images': image}))

    for i in range(len(pred_results)):
        print(pred_results[i].shape)

    predbox = postprocess(pred_results)

    print('obj num is :', len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin * img_w + 0.5)
        ymin = int(predbox[i].ymin * img_h + 0.5)
        xmax = int(predbox[i].xmax * img_w + 0.5)
        ymax = int(predbox[i].ymax * img_h + 0.5)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(orig, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        ptext = (xmin, ymin + 15)
        title = str(CLASSES[classId]) + ":%.2f" % score
        cv2.putText(orig, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./test_onnx_result.jpg', orig)


if __name__ == '__main__':
    print('This is main ....')
    img_path = './test.jpg'
    detect(img_path)