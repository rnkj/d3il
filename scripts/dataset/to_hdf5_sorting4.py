import os
import pickle
from glob import glob

import cv2
import numpy as np
import h5py
from tqdm import tqdm


task_name = "sorting"
dataset_dir = f"environments/dataset/data/{task_name}"

# for num_boxes in (2, 4, 6):
for num_boxes in (4,):
    print(f"Num. boxes = {num_boxes}")
    hdf = h5py.File(f"{task_name}_{num_boxes}_boxes.hdf5", "w")
    file_group = hdf.create_group("data_files")
    context_group = hdf.create_group("contexts")

    data_dir_files = [
        os.path.join(dataset_dir, f"{num_boxes}_boxes_train_files.pkl"),
        os.path.join(dataset_dir, f"{num_boxes}_boxes_eval_files.pkl"),
    ]
    for data_dir_file in data_dir_files:
        key = os.path.basename(data_dir_file).replace(".pkl", "")
        env_files = np.load(data_dir_file, allow_pickle=True)
        ds = file_group.create_dataset(
            key, shape=len(env_files), dtype=h5py.string_dtype()
        )
        ds[:] = env_files

    modes = np.load(
        os.path.join(dataset_dir, f"{num_boxes}_modes.pkl"), allow_pickle=True
    )
    mode_prob = np.load(
        os.path.join(dataset_dir, f"{num_boxes}_mode_prob.pkl"), allow_pickle=True
    )
    for key in modes.keys():
        hdf.attrs[f"{num_boxes}_mode_metadata"] = key
        hdf.attrs[f"{num_boxes}_modes"] = modes[key]
        hdf.attrs[f"{num_boxes}_mode_prob"] = mode_prob[key]

    context_files = [
        os.path.join(dataset_dir, f"{num_boxes}_train_contexts.pkl"),
        os.path.join(dataset_dir, f"{num_boxes}_test_contexts.pkl"),
    ]
    for context_file in context_files:
        key = os.path.basename(context_file).replace(".pkl", "")
        contexts = np.load(context_file, allow_pickle=True)
        for i in range(len(contexts)):
            cont = np.stack([np.concatenate(c, dtype=np.float32) for c in contexts[i]])
            contexts[i] = cont
        contexts = np.stack(contexts, dtype=np.float32)
        ds = context_group.create_dataset(
            key, shape=contexts.shape, dtype="float32", data=contexts
        )

    data_group = hdf.create_group("data")
    data_dir = f"{num_boxes}_boxes"
    state_files = sorted(glob(os.path.join(dataset_dir, data_dir, "state/*.pkl")))
    image_dirs = sorted(glob(os.path.join(dataset_dir, data_dir, "images/*")))

    print(f"Number of episodes = {len(state_files)}")
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
                    context = np.stack(
                        [
                            np.concatenate(c, dtype=np.float32)
                            for c in state_dict["context"]
                        ]
                    )
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