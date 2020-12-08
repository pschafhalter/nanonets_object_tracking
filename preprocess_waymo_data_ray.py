from absl import app, flags
from tqdm import tqdm
import cv2
import glob
import numpy as np
import os
import pickle
import ray
import re

FLAGS = flags.FLAGS

flags.DEFINE_string("waymo_path", None, "Path to the Waymo dataset")
flags.DEFINE_string("out_path", None, "Path to the output dataset")

MIN_AREA = 1500 # The minimum area of an acceptable bounding box.
# The proportion of samples used for training in the train/test split
TRAIN_PROPORTION = 0.8


@ray.remote(num_cpus=1)
def process_pickle_file(filename, waymo_path, out_path):
    training_dir = os.path.basename(os.path.dirname(filename))
    training_num = re.findall("\d+", training_dir)[0]
    scenario_name = os.path.basename(filename)
    scenario_num = re.findall("\d+", scenario_name)[0]

    with open(filename, "rb") as f:
        scenario_data = pickle.load(f)

    for img_num, annotated_img in enumerate(scenario_data):
        img = annotated_img["center_camera_feed"]
        for obstacle in annotated_img["obstacles"]:
            obstacle_id = obstacle["id"]
            label = obstacle["label"]
            x_min, x_max, y_min, y_max = list(map(int, obstacle["bbox"]))
            # Check that the area covered by the bounding box
            # isn't too small.
            area = (x_max - x_min) * (y_max - y_min)
            if area < MIN_AREA:
                print((f"Filtered training{training_num}, s{scenario_num}"))
                print(f"   image:       {img_num}")
                print(f"   obstacle id: {obstacle_id}")
                print(f"   label:       {label}")
                print(f"   area:        {area}")

            crop = img[y_min:y_max, x_min:x_max, :]

            obstacle_dir = (f"training{training_num}_s{scenario_num}"
                f"_{obstacle_id}_{label}")
            if not os.path.exists(f"{out_path}/train/{obstacle_dir}"):
                os.mkdir(f"{out_path}/train/{obstacle_dir}")
                os.mkdir(f"{out_path}/test/{obstacle_dir}")
            # Train/test split
            if np.random.random() < TRAIN_PROPORTION:
                crop_path = f"{out_path}/train/{obstacle_dir}/{img_num}.png"
                cv2.imwrite(crop_path, crop)
            else:
                crop_path = f"{out_path}/test/{obstacle_dir}/{img_num}.png"
                cv2.imwrite(crop_path, crop)

def preprocess_waymo_data(waymo_path, out_path):
    """Load from Eyal's pickle files.
    
    Args:
        waymo_path: the path to the root of the waymo dataset.
        out_path: the output path of the tracking dataset.
    """
    # Make train and test folders
    os.makedirs(f"{out_path}/train")
    os.makedirs(f"{out_path}/test")

    ray.init(_redis_password="5241590074719971")
    files = glob.glob(f"{waymo_path}/training_*/*.pl")
    tasks = [process_pickle_file.remote(filename, waymo_path, out_path)
            for filename in files]
    while tasks:
        msg = f"* {len(files) - len(tasks)}/{len(files)} scenarios processed *"
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))
        ready, tasks = ray.wait(tasks)

    print("********")
    print("* Done *")
    print("********")
    ray.shutdown()


def main(args):
    print(f"loading from {FLAGS.waymo_path}")
    print(f"writing to {FLAGS.out_path}")
    preprocess_waymo_data(FLAGS.waymo_path, FLAGS.out_path)


if __name__ == "__main__":
    app.run(main)

