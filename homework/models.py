#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : models.py
# @Time     : 2023/10/23 18:24
# @Project  : homework


import cv2
import numpy as np
import copy
import xml.etree.ElementTree as ET


def test_image(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_iou(box1, box2):
    """
     边框以左上为原点
     box:[x1,y2,x2,y2],依次为左上右下坐标
     """
    h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    iou = inter / union
    return iou


def loss_area(area_rate):
    if area_rate < 0:
        return 0
    elif area_rate > 2:
        return 0
    else:
        return area_rate * (2 - area_rate)


class MyImageObject(object):
    """
    MyImageObject: 自定义图像对象，用于封装常用操作
    """
    def __init__(self, image_path, annotation_path, cfg):
        self.cfg = cfg
        self.image_data = cv2.imread(image_path)
        self.annotation_list = self.parse_xml_annotation_to_bbox_list(annotation_path)
        self.object_list, self.output = self.get_object_list()

    def show(self, annotation=False):
        """
        :argument: 在屏幕直接显示图片，并等待鼠标点击
        :param: 是否显示标注框信息，默认不显示
        """
        if not annotation:
            cv2.imshow('demo', self.image_data)
        else:
            image_data = copy.deepcopy(self.image_data)
            for (x1, y1, x2, y2) in self.annotation_list:
                cv2.rectangle(image_data, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow('demo', image_data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def parse_xml_annotation_to_bbox_list(xml_file_path):
        """
        :param: VOC标准的XML格式的目标检测数据标注文件路径
        :return: 包含全部标注外接矩形BBOX(x1, y1, x2, y2)的列表
        """
        bbox_list = []
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            xml_box = obj.find('bndbox')
            x_min = int(xml_box.find('xmin').text)
            y_min = int(xml_box.find('ymin').text)
            x_max = int(xml_box.find('xmax').text)
            y_max = int(xml_box.find('ymax').text)
            bbox_list.append([x_min, y_min, x_max, y_max])
        return bbox_list

    def get_object_list(self):
        """
        :return: 包含全部算法处理得到的目标外接矩形BBOX(x1, y1, x2, y2)的列表
        """
        object_list = []

        image_data_initial = copy.deepcopy(self.image_data)
        image_data = copy.deepcopy(image_data_initial)
        max_height, max_width = image_data.shape[:2]
        # 泛洪处理
        if self.cfg['is_floodfill']:
            h, w = image_data.shape[:2]
            mask = np.zeros((h+2, w+2), dtype=np.uint8)
            cv2.floodFill(image_data, mask, (w-2, h-2), (255, 255, 255), (3, 3, 3),
                          (3, 3, 3), 8)
        # test_image('floodfill', image_data)

        # 转化成灰度图
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        # test_image('gray', gray)

        # 高斯滤波（模糊处理）
        if self.cfg['is_blur']:
            blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0, sigmaY=0)
        else:
            blur = gray
        # test_image("blur", blur)

        # 二值化处理
        if self.cfg['is_thresh']:
            ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            thresh = blur
        # test_image("thresh", thresh)

        # 边缘提取
        canny = cv2.Canny(thresh, 100, 190)
        # 消除较小的噪音边缘
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            bounding_rect = cv2.boundingRect(contour)
            if area <= 10 or (bounding_rect[2] <= 50 and bounding_rect[3] <= 50):
                cv_contours.append(contour)
            else:
                continue
        cv2.fillPoly(canny, cv_contours, (0, 0, 0))
        # test_image("canny", canny)

        # 闭运算去除内部噪声
        if self.cfg['is_close']:
            kernel = np.ones((9, 9), np.uint8)
            canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=1)
        # test_image("canny", canny)

        # 在提取出的轮廓图像中找出圆
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=65, param1=80, param2=22.5, minRadius=40, maxRadius=150)
        circles = circles[0, :, :]
        circles = np.uint16(np.around(circles))

        # 在终端打印出硬币个数
        # print("图像中的硬币共有:", len(circles), "个")

        # 对轮廓显示的box剪裁后重新判定是不是圆
        for circle in circles[:]:
            circle = list(map(int, circle))
            cropped_image = blur[
                            max(circle[1] - circle[2] - 5, 0): min(circle[1] + circle[2] + 5, max_height),
                            max(circle[0] - circle[2] - 5, 0): min(circle[0] + circle[2] + 5, max_width)
                            ]
            h, w = cropped_image.shape[:2]  # 剪裁后长宽
            # 泛洪处理
            if self.cfg['is_floodfill']:
                mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                cv2.floodFill(cropped_image, mask, (3, 3), (0, 0, 0), (5, 5, 5),
                              (5, 5, 5), 8)
            cropped_canny = cv2.Canny(cropped_image, 100, 200)
            cropped_circles = cv2.HoughCircles(
                cropped_canny, cv2.HOUGH_GRADIENT, dp=1, minDist=min(h, w) / 4,
                param1=80, param2=25, minRadius=int(min(h, w) / 4), maxRadius=int(min(h, w) / 2) + 1
            )
            if cropped_circles is not None:
                cropped_circles = cropped_circles[0, :, :]
                if len(cropped_circles) >= 1:
                    object_list.append(
                        [circle[0] - circle[2], circle[1] - circle[2], circle[0] + circle[2], circle[1] + circle[2]]
                    )

        # 绘制出轮廓
        for i in object_list:
            if self.cfg['out_shape'] == 'Rect':
                cv2.rectangle(image_data_initial, (i[0], i[1], i[2] - i[0], i[3] - i[1]), (0, 255, 0), 2)
            elif self.cfg['out_shape'] == 'Circle':
                cv2.circle(
                    image_data_initial, (int((i[0] + i[2]) / 2), int((i[1] + i[3]) / 2)), int((i[2] - i[0]) / 2),
                    (0, 255, 0), 2
                )
        # last test
        # test_image("output", image_data_initial)
        return object_list, image_data_initial

    def calculate_metrics(self):
        """
        :return: 衡量算法处理结果的性能指标
        """
        box_area = 0
        real_area = 0
        match_list_pre = self.match_box(self.object_list)
        # 获取一对一匹配表
        match_list_final = []
        for annotation in self.annotation_list:
            # 顺带计算面积和，为之后性能计算做准备
            real_area = real_area + abs(annotation[2] - annotation[0]) * abs(annotation[3] - annotation[1])

            iou_max = 0
            index_match = -1
            for match_pre in match_list_pre:
                if match_pre[1] == self.annotation_list.index(annotation) and match_pre[2] > iou_max:
                    iou_max = match_pre[2]
                    index_match = match_pre[0]
            if index_match != -1:
                match_list_final.append([index_match, self.annotation_list.index(annotation), iou_max])
        # 算法性能指标计算
        score = 0.
        score_max = float(len(self.annotation_list))
        for match_final in match_list_final:
            score = score + match_final[2]
        for box in self.object_list:
            box_area = box_area + abs(box[2] - box[0]) * abs(box[3] - box[1])
        if len(self.object_list) <= len(self.annotation_list):
            accuracy = score / score_max
        else:
            accuracy = ((score / score_max) *
                        (1 + self.cfg['over_rate'] * loss_area(box_area / real_area)) / (1 + self.cfg['over_rate']))
        return accuracy

    def match_box(self, boxes):
        """
        :param boxes: 待检测的预测对象列表
        :return: 匹配列表，具体为 <index_box, index_matched_annotation, iou>
        """
        box_match_list = []
        for box in boxes:
            iou_max = 0
            index = 0
            for annotation in self.annotation_list:
                iou_temp = get_iou(box, annotation)
                if iou_temp > iou_max:
                    iou_max = iou_temp
                    index = self.annotation_list.index(annotation)
            box_match_list.append([boxes.index(box), index, iou_max])
        return box_match_list
