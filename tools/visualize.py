import os
from glob import glob
from pathlib import Path
import copy
import cv2
import argparse
import numpy as np
import torch
from openseg.utils import load_config_file
from openseg.models import build_model
from openseg.dataset import build_pipeline


def main():
    np.random.seed(100)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='path to config file')
    parser.add_argument('-w', '--weight', type=str, help='path to weight file')
    parser.add_argument('-f', '--files', type=str, help='image path or dir')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='select device')
    parser.add_argument('-e', '--ecport', type=str, default=None, help='output path')
    parser.add_argument('-s', '--show', type=bool, action='store_true', help='show mask image')
    args = parser.parse_args()

    config, text = load_config_file(args.config)

    model, pipeline = build_model_and_pipeline(config=config, weight=args.weight, device=args.device)
    num_classes = model.num_classes
    classes_color = np.random.randint(low=0, high=255, size=(num_classes, 3), dtype=np.int32)

    files = load_image_files(args.files)

    for image_file in files:
        file_name = Path(image_file).name
        result = {'image_path': image_file}
        result = pipeline(results=result)
        meta = copy.deepcopy(result['meta'])

        image = result['image'].to(args.device)
        with torch.inference_mode():
            logits = model.forward_inference(image)

        predict = pipeline.transforms['TestTimeAugment'].post_process(
            logits=logits, augmented_results=result['meta']
        )

        predict = np.argmax(predict, axis=2)
        segment = classes_color[predict]
        
        image = result['image'].squeeze(0).numpy().transpose(1, 2, 0) * 255
        image = cv2.resize(image, dsize=(meta['original_shape']))
        image = image.astype(np.uint8)
        segment = segment.astype(np.uint8)
        segment_image = np.hstack((segment, image))

        if args.show:
            cv2.imshow('segment', segment_image)
            cv2.waitKey(0)
        
        if args.ecport is not None:
            export_path = os.path.join(args.ecport, file_name + '.png')
            cv2.imwrite(export_path, segment)



def build_model_and_pipeline(config, weight, device):
    model = build_model(config=config['model'], pretrained_weight=weight).to(device)
    pipeline = build_pipeline(config=config['test_pipeline'])
    return model, pipeline


def load_image_files(path):
    if Path(path).is_file():
        return [path]
    else:
        if path[-1] == '/':
            path = path[:-1]
        return glob(pathname=path + '/' + '*')
    

if __name__ == '__main__':
    main()

