import os
import glob
import json
import cv2
import tqdm
import argparse


def image2json(dataset_path):
    image_file_list = sorted(glob.glob(os.path.join(dataset_path, 'images-blur', '*')))
    dataset = []
    for image_file in tqdm.tqdm(image_file_list):
        image = cv2.imread(image_file)
        height, width = image.shape[:2]
        data = {
            'filename': os.path.basename(image_file),
            'height': height,
            'width': width,
        }
        dataset.append(data)

    json_file = os.path.join(dataset_path, 'test.json')
    with open(json_file, 'w') as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--dataset_path', default='dataset', type=str, help='dataset path')
    parser.add_argument('-d', '--dataset_name', type=str, help='dataset name')
    opts = parser.parse_args()
    print(opts)

    dataset_path = os.path.join(opts.dataset_path, opts.dataset_name)
    image2json(dataset_path)
