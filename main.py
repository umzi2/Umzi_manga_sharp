from utils.sharp import Sharp
import cv2
import numpy as np
import argparse
import os
import json
from tqdm.contrib.concurrent import process_map


class Start:
    def __init__(self):
        self.in_folder = ""
        self.out_folder = str
        self.sharp = Sharp

    def __arg_parse(self) -> None:
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('-i', '--input', type=str,
                            help='Input_folder')
        parser.add_argument('-o', '--output', type=str,
                            help='Output_folder')

        args = parser.parse_args()
        in_folder = args.input
        out_folder = args.output
        if not in_folder:
            in_folder = "INPUT"
        if not out_folder:
            out_folder = "OUTPUT"
        self.in_folder = in_folder
        self.out_folder = out_folder
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        if not os.path.exists(in_folder):
            os.makedirs(in_folder)
            raise print("no in folder")

    def __json_parse(self) -> None:
        with open("config.json", "r") as f:
            json_config = json.load(f)
        if list(json_config.keys()) != ['low_input', 'high_input', "gamma", 'diapason_white', 'cenny']:
            raise print('Not correct config')
        diapason_white = json_config["diapason_white"]
        low_input = json_config["low_input"]
        high_input = json_config["high_input"]
        gamma = json_config["gamma"]
        cenny = json_config["cenny"]
        try:
            self.sharp = Sharp(diapason_white, low_input, high_input, gamma, cenny)
        except RuntimeError as e:
            raise print(f"incorrect data type {e}")
        pass

    def sharp_img(self, img_name):
        try:
            folder = f"{self.in_folder}/{img_name}"
            basename = ".".join(img_name.split(".")[:-1])
            img = cv2.imread(folder)

            if img is None:
                return print(f"{img_name}, not supported")
            array = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            out_image = self.sharp.run(array) * 255
            cv2.imwrite(f'{self.out_folder}/{basename}.png', out_image)
        except RuntimeError as e:
            print(e)

    def start_process(self) -> None:
        self.__arg_parse()
        self.__json_parse()
        list_files = [
            file
            for file in os.listdir(self.in_folder)
            if os.path.isfile(os.path.join(self.in_folder, file))
        ]
        process_map(self.sharp_img, list_files)


if __name__ == "__main__":
    s = Start()
    s.start_process()
