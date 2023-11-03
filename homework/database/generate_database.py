import os
import json


def generate_database(datasets_path):
    src_list = os.listdir('./datasets')
    annotations_path = os.path.join('./datasets', src_list[0]).replace('\\', '/')  # 0: annotations
    images_path = os.path.join('./datasets', src_list[1]).replace('\\', '/')  # 1: images
    input_path = os.path.join('./datasets', src_list[2]).replace('\\', '/')  # 2: input

    with open('src_data.txt', 'w') as ft:
        file_num = 0
        images = os.listdir(images_path)
        for annotation in os.listdir(annotations_path):
            if annotation.endswith('.xml'):
                ft.write(os.path.join(datasets_path, src_list[0], annotation).replace('\\', '/') + ';' +
                         os.path.join(datasets_path, src_list[1],  images[file_num]).replace('\\', '/') +
                         ';\n')
                file_num = file_num + 1
    ft.close()
    print(f"Generate src_data.txt down. Total File Num: {file_num}")

    with open('input.txt', 'w') as f_in:
        input_num = 0
        for input_image in os.listdir(input_path):
            if input_image.endswith('.png'):
                f_in.write(os.path.join(datasets_path, src_list[2], input_image).replace('\\', '/') + ';\n')
                input_num = input_num + 1
    f_in.close()
    print(f"Generate input.txt down. Total File Num: {input_num}")


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    generate_database(cfg['datasets_path'])
