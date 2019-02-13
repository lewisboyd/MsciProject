import cPickle as pickle

import cortex
import cortex_cuda
import retina_cuda
from ddpg_cnn import DdpgCnn

class DdpgRetina(DdpgCnn):
    """DDPG agent with built in retina."""

    def __init__(self, memory_capacity, batch_size, noise_function, init_noise,
                 final_noise, exploration_len, num_images, num_actions,
                 reward_scale, image_size):
        """Initialise agent and retina.

        Args:
            image_size (int tuple): (height, width) of observation images
        """
        DdpgCnn.__init__(memory_capacity, batch_size, noise_function,
                         init_noise, final_noise, exploration_len, num_images,
                         num_actions, reward_scale)

        self.num_images = num_images
        self.ret = Retina(*image_size)

        # Create transform function to prepare image for retina
        self.pre_retina_transform = T.Compose([
                                               T.ToPILImage(),
                                               T.Grayscale(),
                                               T.ToTensor()
        ])

        # Create transform to resize cortical image
        self.post_retina_transform = T.Compose([
                                                T.ToPILImage(),
                                                T.Resize((64,64)),
                                                T.ToTensor()
        ])

    def interpet(self, obs):
        state_tensor = torch.ones((self.num_images, 64, 64)).to(self.device)
        for i in range(self.num_images):
            cortical_img = self.ret(self.pre_retina_transform(obs[i]))
            state_tensor[i] = self.post_retina_transform(cortical_img)
        return state_tensor

class Retina:
    """GPU Retina to sample images."""

    def __init__(self, height, width):
        """Initialise GPU retina and cortex.

        Args:
            height (int): height of image
            width (int): width of image"""
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
            loc50k, coeff50k, (width, height), (int(height / 2), int(width / 2))
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
