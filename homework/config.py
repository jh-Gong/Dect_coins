"""
说明: 统一调整和保存重要参数
注意: 完成调整后记得运行保存为文件, 否则不会起作用
"""

import json


if __name__ == "__main__":
    cfg = {
        # 是否显示标注框,显示设置为True,不显示设置为False
        'is_annotation': True,
        # 数据库路径
        'database_root': './database',
        # 数据集路径
        'datasets_path': './database/datasets',
        # 日志根目录
        'logs_root': './logs',
        # 是否泛洪处理,进行操作为True,不进行为False
        'is_floodfill': True,
        # 是否高斯滤波,进行操作为True,不进行为False
        'is_blur': True,
        # 是否二值化处理,进行操作为True,不进行为False
        'is_thresh': False,
        # 是否进行闭运算降噪,进行操作为True,不进行为False
        'is_close': True,
        # 性能检测多余对象影响参数
        'over_rate': 1,
        # 输出对象box以什么形式画出: [Circle, Rect]
        'out_shape': 'Circle',
    }
    with open("./database/config.json", 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Save config file down!')
