#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : demo.py
# @Time     : 2023/10/23 18:18
# @Project  : homework
import cv2

from models import MyImageObject
import json
import os
import datetime

if __name__ == "__main__":
    #获取参数数据
    cfg_file_path = './database/config.json'
    with open(cfg_file_path, 'r') as f:
        cfg = json.load(f)

    #准备数据
    with open(os.path.join(cfg['database_root'], 'src_data.txt'), 'r') as ft:
        images = ft.readlines()

    #设置模型并输出性能和结果
    score = 0.  # 总的准确率
    time_now = str(datetime.datetime.now().replace(microsecond=0)).split(' ')
    logs_path = cfg['logs_root'] + '/' + time_now[0] + '-' + time_now[1].replace(':', '-')
    if not os.path.exists(logs_path + '/pictures'):
        os.makedirs(logs_path + '/pictures', exist_ok=True)
    with open(logs_path + '/accuracy.txt', 'w', encoding="utf-8") as f_log:
        for index in range(10):
            image = MyImageObject(
                image_path=images[index].split(';')[1],
                annotation_path=images[index].split(';')[0],
                cfg=cfg
            )
            score = score + image.calculate_metrics()
            f_log.write(f'{index}\tAccuracy: {round(image.calculate_metrics(), 7)}\n')
            image_out = image.output
            cv2.putText(image_out, f'{index} Accuracy: {round(image.calculate_metrics(), 7)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(logs_path + f'/pictures/{index}.png', image_out)
        f_log.write(f'Accuracy_average: {round((score / 9), 6)}')
    f_log.close()
    os.rename(logs_path + '/accuracy.txt', logs_path + f'/accuracy_{round((score / 9), 2)}.txt')
