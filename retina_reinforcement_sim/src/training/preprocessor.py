import cPickle as pickle
import numpy as np
import torch
import torchvision.transforms as T

import cortex
import cortex_cuda
import retina_cuda


class Preprocessor:

    def __call__(self, obs):
        return torch.tensor(obs, dtype=torch.float)


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
