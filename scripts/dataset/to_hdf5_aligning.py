import os
import pickle
from glob import glob

import cv2
import numpy as np
import h5py
from tqdm import tqdm


task_name = "aligning"


hdf = h5py.File(f"{task_name}.hdf5", "w")


dataset_dir = f"environments/dataset/data/{task_name}"


data_dir_files = sorted(glob(os.path.join(dataset_dir, "train_files*.pkl")))
data_dir_files.append(os.path.join(dataset_dir, "eval_files.pkl"))
print("Data files:")
for data_dir_file in data_dir_files:
    print(f"  - {os.path.basename(data_dir_file).replace('.pkl', '')}")


file_group = hdf.create_group("data_files")
for data_dir_file in data_dir_files:
    key = os.path.basename(data_dir_file).replace(".pkl", "")
    env_files = np.load(data_dir_file, allow_pickle=True)
    ds = file_group.create_dataset(key, shape=len(env_files), dtype=h5py.string_dtype())
    ds[:] = env_files


context_files = [
    os.path.join(dataset_dir, "train_contexts.pkl"),
    os.path.join(dataset_dir, "test_contexts.pkl"),
]


context_group = hdf.create_group("contexts")
for context_file in context_files:
    key = os.path.basename(context_file).replace(".pkl", "")
    contexts = [
        np.concatenate(c, dtype=np.float32).reshape(2, -1)
        for c in np.load(context_file, allow_pickle=True)
    ]
    contexts = np.stack(contexts, dtype=np.float32)
    ds = context_group.create_dataset(
        key, shape=contexts.shape, dtype="float32", data=contexts
    )


data_group = hdf.create_group("data")
state_files = sorted(glob(os.path.join(dataset_dir, "all_data/state/*.pkl")))
image_dirs = sorted(glob(os.path.join(dataset_dir, "all_data/images/*")))

print(f"Number of episodes = {len(state_files)}")
print("Camera names:")
for image_dir in image_dirs:
    print(f"  - {os.path.basename(image_dir)}")


for state_file in tqdm(state_files):
    env_name = os.path.basename(state_file).replace(".pkl", "")
    image_dir_dict = {
        os.path.basename(d): os.path.join(d, env_name) for d in image_dirs
    }
    # print(state_file, image_dir_dict)

    env_group = data_group.create_group(env_name)

    state_group = env_group.create_group("state")
    with open(state_file, "rb") as f:
        state_dict = pickle.load(f)
        for key, child in state_dict.items():
            if key == "context":
                context = np.concatenate(state_dict["context"], dtype=np.float32)
                state_group.create_dataset(
                    key, shape=context.shape, dtype=context.dtype, data=context
                )
            else:
                child_group = state_group.create_group(key)
                for child_key, value in state_dict[key].items():
                    child_group.create_dataset(
                        child_key, shape=value.shape, dtype=value.dtype, data=value
                    )

    image_group = env_group.create_group("images")
    for cam_name, dirname in image_dir_dict.items():
        img_files = glob(os.path.join(dirname, "*.jpg"))
        img_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
        images = np.stack([cv2.imread(img_file) for img_file in img_files])
        image_group.create_dataset(
            cam_name, shape=images.shape, dtype=images.dtype, data=images
        )

hdf.close()