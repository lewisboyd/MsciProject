#!/usr/bin/env python

import os

import cv2
import numpy as np
import torch

from model import ResNet10


def get_img(base_dir, id):
    """Index dataset to get image."""
    img_name = base_dir + "img" + str(id) + ".png"
    img = cv2.imread(img_name)
    # Convert to float and rescale to [0,1]
    img = img.astype(np.float32) / 255
    return img


if __name__ == '__main__':
    # Load and save paths
    STATES_TENSOR = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/image_data/states")
    IMG_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/image_data/images/")
    RET_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/image_data/retina_images/")
    STATE_DICT_IMG = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/sr/state_dicts/net_50")
    STATE_DICT_RET = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/sr_retina/state_dicts/net_50")
    OUT_FOLDER_RET = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/examples/sr/")
    OUT_FOLDER_IMG = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/examples/sr_retina/")
    OUT_FOLDER_GROUND = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/examples/ground/")

    # Index of images to process
    INDICES = np.arange(0, 1000, 10)
    STATES = torch.load(STATES_TENSOR)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get object locations using normal images
    srnet = ResNet10(2).to(device).eval()
    srnet.load_state_dict(torch.load(STATE_DICT_IMG))
    srnet_img_locs = []
    for index in INDICES:
        img = get_img(IMG_FOLDER, index)
        img = cv2.resize(img, None, fx=0.7, fy=0.7,
                         interpolation=cv2.INTER_AREA)
        img = torch.tensor(img, dtype=torch.float,
                           device=device).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            loc = srnet(img)
        srnet_img_locs.append(loc.cpu().numpy()[0])

    # Get object locations using retina images
    srnet.load_state_dict(torch.load(STATE_DICT_RET))
    srnet_retina_locs = []
    for index in INDICES:
        img = get_img(RET_FOLDER, index)
        img = cv2.resize(img, None, fx=0.7, fy=0.7,
                         interpolation=cv2.INTER_AREA)
        img = torch.tensor(img, dtype=torch.float,
                           device=device).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            loc = srnet(img)
        srnet_retina_locs.append(loc.cpu().numpy()[0])

    # Get ground truth locations
    ground_truths = []
    for index in INDICES:
        ground_truths.append(STATES[index].numpy())

    # Draw circles at predicted/ground locations and save images
    for i, index in enumerate(INDICES):
        # Rescale from [-1, 1] to [0, 2]
        ground_truths[i] = ground_truths[i] + 1
        srnet_img_locs[i] = srnet_img_locs[i] + 1
        srnet_retina_locs[i] = srnet_retina_locs[i] + 1

        # Multiply by half img height and width to get pixel location
        ground_truths[i][0] = ground_truths[i][0] * 234
        ground_truths[i][1] = ground_truths[i][1] * 123
        srnet_img_locs[i][0] = srnet_img_locs[i][0] * 234
        srnet_img_locs[i][1] = srnet_img_locs[i][1] * 123
        srnet_retina_locs[i][0] = srnet_retina_locs[i][0] * 234
        srnet_retina_locs[i][1] = srnet_retina_locs[i][1] * 123

        # Draw circle at ground truth location and save
        img = get_img(IMG_FOLDER, index)
        loc = (ground_truths[i][0], ground_truths[i][1])
        img = cv2.circle(img, loc, 5, (0, 255, 0), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(OUT_FOLDER_GROUND + "img" + str(index) + ".png", img * 255)

        # Draw circle at location predicted using normal images and save
        img = get_img(IMG_FOLDER, index)
        loc = (srnet_img_locs[i][0], srnet_img_locs[i][1])
        cv2.circle(img, loc, 5, (0, 255, 0), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(OUT_FOLDER_IMG + "img" + str(index) + ".png", img * 255)

        # Draw circle at location predicted using retina images and save
        img = get_img(IMG_FOLDER, index)
        loc = (srnet_retina_locs[i][0], srnet_retina_locs[i][1])
        cv2.circle(img, loc, 5, (0, 255, 0), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(OUT_FOLDER_RET + "img" + str(index) + ".png", img * 255)

    # Create folders if not already exist
    if not os.path.isdir(OUT_FOLDER_IMG):
        os.makedirs(OUT_FOLDER_IMG)
    if not os.path.isdir(OUT_FOLDER_RET):
        os.makedirs(OUT_FOLDER_RET)
    if not os.path.isdir(OUT_FOLDER_GROUND):
        os.makedirs(OUT_FOLDER_GROUND)

    # Save pixel locations
    np.save(OUT_FOLDER_IMG + "pixel_locations", srnet_img_locs)
    np.save(OUT_FOLDER_RET + "pixel_locations", srnet_retina_locs)
    np.save(OUT_FOLDER_GROUND + "pixel_locations", ground_truths)
