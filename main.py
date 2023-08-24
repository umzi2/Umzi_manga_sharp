import cv2
import os
import numpy as np
import concurrent.futures
import configparser
from time import time as ttime
config_file_path = "config.ini"
t0 = ttime()
default_config = {
    "low_input": "4",
    "high_input": "255",
    "low_output": "0",
    "high_output": "255",
    "gamma": "1.0",
    "diapason_black": "14",
    "diapason_white": "10"
}
def read_config():
    if os.path.exists(config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        diapason_white = int(config.get("Settings", "diapason_white"))
        low_input = int(config.get("Settings", "low_input"))
        high_input = int(config.get("Settings", "high_input"))
        low_output = int(config.get("Settings", "low_output"))
        high_output = int(config.get("Settings", "high_output"))
        gamma = float(config.get("Settings", "gamma"))
        diapason = int(config.get("Settings", "diapason_black"))
    else:
        diapason_white = int(default_config["diapason_white"])
        low_input = int(default_config["low_input"])
        high_input = int(default_config["high_input"])
        low_output = int(default_config["low_output"])
        high_output = int(default_config["high_output"])
        gamma = float(default_config["gamma"])
        diapason = int(default_config["diapason_black"])
        config = configparser.ConfigParser()
        config["Settings"] = default_config
        with open(config_file_path, "w") as configfile:
            config.write(configfile)
    return  diapason_white, low_input, high_input, low_output, high_output, gamma, diapason
image_folder = 'INPUT'
output_folder = 'SHARP'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
diapason_white, low_input, high_input, low_output, high_output, gamma, diapason = read_config()
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
    gray2 = cv2.medianBlur(gray, 3)
    mask2 = cv2.inRange(gray2, max(255 - diapason_white, 0), min(255 + diapason_white, 255))
    image = np.clip(((gray.astype(np.float32) - low_input) / (high_input - low_input)) ** gamma * (high_output - low_output) + low_output, 0, 255).astype(np.uint8)
    _, black_mask = cv2.threshold(image, diapason, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(black_mask, (3, 3), 0)
    output2 = np.clip(((blur.astype(np.float32) - 253) / (254 - 253)) ** 1.0 * (255 - 0) + 0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(image, 750, 800, apertureSize=3, L2gradient=True)
    inverted_edges = cv2.bitwise_not(edges)
    sharpened_image = cv2.bitwise_and(image, image, mask=output2)
    result = cv2.bitwise_and(sharpened_image, sharpened_image, mask=inverted_edges)
    final_result = cv2.add(result, mask2)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, final_result)
    print(f"Результат сохранён{output_path}")
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and not f.startswith('.')]
num_images = len(image_files)
num_cores = os.cpu_count() if os.cpu_count() is not None else 1
num_workers = min(num_images, num_cores)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(process_image, image_files)
t1 = ttime()
print("time_spent", t1 - t0)