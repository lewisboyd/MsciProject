import cPickle as pickle
import cv2
import numpy as np
import torch
import torchvision.transforms as T

import cortex
import cortex_cuda
import retina_cuda
from environment import Retina


class PendulumPreprocessor:
    """Preprocessor for low state Pendulum environment."""

    def __call__(self, obs):
        """Minmax observation and return in a tensor."""
        obs[2] = obs[2] / 8
        return torch.tensor(obs, dtype=torch.float)


class BaxterPreprocessor:
    """Preprocessor for low state Baxter environment."""

    def __call__(self, obs):
        """Return observation in a tensor."""
        # The baxter environment may return None when episode finishes
        if None in obs:
            return torch.tensor([0, 0, 0, 0], dtype=torch.float)
        return torch.tensor(obs, dtype=torch.float)


class BaxterImagePreprocessor:
    """Class to preprocess Baxter env observation before use by DDPG agent."""

    def __init__(self, srnet):
        """Initialise.

        Args:
            srnet (object): Network to process image input.

        """
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.srnet = srnet.to(self.device)
        self.srnet = self.srnet.eval()

    def __call__(self, obs):
        """Find object in image then returns complete state tensor."""
        img = obs['img']
        state = torch.tensor(obs['state'], dtype=torch.float)
        img = cv2.resize(img, (468, 246),
                         interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, None, fx=0.7, fy=0.7,
                         interpolation=cv2.INTER_AREA)
        # Convert to float and rescale to [0,1]
        img = img.astype(np.float32) / 255
        img = torch.tensor(img, dtype=torch.float,
                           device=self.device).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            sr = self.srnet(img).cpu()
        state = torch.cat((sr[0], state))
        return state


class BaxterRetinaPreprocessor:
    """Class to preprocess Baxter env observation before use by DDPG agent."""

    def __init__(self, srnet, visualise=False, display_centre=False):
        """Initialise.

        Args:
            srnet (object): Network to process image input.
            visualise (bool): If true displays retina image until keypress
            display_centre (bool): If true displays image with predicted centre
                                   until keypress

        """
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.srnet = srnet.to(self.device)
        self.retina = Retina(1280, 800)
        self.visualise = visualise
        if visualise:
            cv2.namedWindow("Retina", cv2.WINDOW_NORMAL)
        self.display_centre = display_centre
        if display_centre:
            cv2.namedWindow("Centre", cv2.WINDOW_NORMAL)

    def __call__(self, obs):
        """Find object in cortical image then returns complete state tensor."""
        img = obs['img']
        state = torch.tensor(obs['state'], dtype=torch.float)
        ret_img = self.retina.sample(img)

        # Display retina image
        if self.visualise:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("Retina", ret_img)

        # Preprocess and run through srnet
        ret_img = cv2.resize(ret_img, None, fx=0.7, fy=0.7,
                             interpolation=cv2.INTER_AREA)
        ret_img = img.astype(np.float32) / 255
        ret_img = torch.tensor(img, dtype=torch.float,
                               device=self.device).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            obj_loc = self.srnet(img).cpu()

        # Display predicted centre on image
        if self.display_centre:
            loc = obj_loc[0] + 1
            loc[0] = loc[0] * (img.shape[1] / 2)
            loc[1] = loc[1] * (img.shape[0] / 2)
            img = cv2.circle(img, loc, 5, (0, 255, 0), -1)
            if not self.visualise:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("Centre", img)

        # Display images
        if self.visualise or self.display_centre:
            cv2.waitKey(0)

        # Concatenate state and return
        state = torch.cat((obj_loc[0], state))
        return state


class ImagePreprocessor:

    def __init__(self, num_images):
        self.num_images = num_images
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((64, 64)),
            T.ToTensor()
        ])

    def __call__(self, obs):
        obs_processed = torch.ones((self.num_images, 64, 64))
        for i in range(self.num_images):
            obs_processed[i] = self.transform(obs[i])
        return obs_processed


class RetinaPreprocessor:

    class Retina:
        """GPU Retina to sample images."""

        def __init__(self, height, width):
            """Initialise GPU retina and cortex.

            Args:
                height (int): height of image
                width (int): width of image
            """
            # Load in data
            retina_path = '/home/lewis/Downloads/RetinaCUDA-master/Retinas'
            with open(retina_path + '/ret50k_loc.pkl', 'rb') as handle:
                loc50k = pickle.load(handle)
            with open(retina_path + '/ret50k_coeff.pkl', 'rb') as handle:
                coeff50k = pickle.load(handle)

            # Create retina and cortex
            L, R = cortex.LRsplit(loc50k)
            L_loc, R_loc = cortex.cort_map(L, R)
            L_loc, R_loc, G, cort_size = cortex.cort_prepare(L_loc, R_loc)
            self.ret = retina_cuda.create_retina(
                loc50k, coeff50k, (width, height), (int(
                    height / 2), int(width / 2))
            )
            self.cort = cortex_cuda.create_cortex_from_fields_and_locs(
                L, R, L_loc, R_loc, cort_size, gauss100=G, rgb=False
            )

        def sample(self, image):
            """Generate cortical image."""
            v_c = self.ret.sample(image)
            l_c = self.cort.cort_image_left(v_c)
            r_c = self.cort.cort_image_right(v_c)
            return np.concatenate((np.rot90(l_c), np.rot90(r_c, k=3)), axis=1)

    def __init__(self, num_images, img_height, img_width):
        self.num_images = num_images
        self.ret = Retina(img_height, img_width)

    def __call__(self, obs):
        obs_processed = torch.ones((self.num_images, 64, 64)).to(self.device)
        for i in range(self.num_images):
            # Convert image to grayscale
            img = obs[i]
            img = T.functional.to_pil_image(img)
            img = T.functional.to_grayscale(img)
            img = T.functional.to_tensor(img)

            # Generate cortical image
            cortical_img = self.ret.sample(img.squeeze().numpy())
            cortical_img = np.expand_dims(cortical_img, 2)

            # Resize image
            cortical_img = T.functional.to_pil_image(cortical_img)
            cortical_img = T.functional.resize(cortical_img, (64, 64))
            cortical_img = T.functional.to_tensor(cortical_img)
            obs_processed[i] = cortical_img
        return obs
