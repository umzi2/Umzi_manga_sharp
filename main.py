from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import json
import os
from time import time as ttime
from tqdm import tqdm
import cv2
import numpy as np
import argparse


def process_image(filename):
    image_path = os.path.abspath(os.path.join(image_folder, filename))
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден. Пропуск файла.")
        return
    valid_extensions = ['.jpg', '.jpeg', '.png']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"Файл {image_path} не является изображением. Пропуск файла.")
        return
    umzi = cv2.imread(image_path)
    if umzi is None:
        print(f"Не удалось загрузить изображение {image_path}. Пропуск файла.")
        return
    gray = cv2.cvtColor(umzi, cv2.COLOR_RGB2GRAY)

    image = np.clip(((gray.astype(np.float32) - configs.get("low_input")) / (
            configs.get("high_input") - configs.get("low_input"))) ** configs.get("gamma") * (
                            configs.get("high_output") - configs.get("low_output")) + configs.get("low_output"), 0,
                    255).astype(np.uint8)
    if configs.get("diapason_black") != -1:
        _, black_mask = cv2.threshold(image, configs.get("diapason_black"), 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(black_mask, (3, 3), 0)

        output2 = np.clip(((blur.astype(np.float32) - 253) / (254 - 253)) ** 1.0 * (255 - 0) + 0, 0, 255).astype(
            np.uint8)
        image = cv2.bitwise_and(image, image, mask=output2)

    if configs.get("cenny"):
        edges = cv2.Canny(image, 750, 800, apertureSize=3, L2gradient=True)
        inverted_edges = cv2.bitwise_not(edges)

        image = cv2.bitwise_and(image, image, mask=inverted_edges)

    if configs.get("diapason_white") != -1:
        gray2 = cv2.medianBlur(gray, 3)
        mask2 = cv2.inRange(gray2, max(255 - configs.get("diapason_white"), 0),
                            min(255 + configs.get("diapason_white"), 255))
        image = cv2.add(image, mask2)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)


def process_image_with_progress(filename):
    process_image(filename)
    pbar.update(1)  # Обновление прогресс-бара


def parse_args():
    parser = argparse.ArgumentParser(description="Image processing script")
    parser.add_argument("--input_folder", type=str, default='INPUT',
                        help="Path to the input image folder")
    parser.add_argument("--output_folder", type=str, default='SHARP',
                        help="Path to the output image folder")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_folder = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    with open('config.json') as f:
        configs = json.load(f)

    image_files = [f for f in os.listdir(image_folder) if
                   os.path.isfile(os.path.join(image_folder, f)) and not f.startswith('.')]
    num_images = len(image_files)
    num_cores = os.cpu_count() if os.cpu_count() is not None else 1

    t0 = ttime()
    with tqdm(total=num_images, desc="Processing images") as pbar:
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(process_image_with_progress, filename) for filename in image_files]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    t1 = ttime()

    print("Time spent:", t1 - t0)
