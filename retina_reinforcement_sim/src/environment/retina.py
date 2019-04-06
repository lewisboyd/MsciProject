import cPickle as pickle
import numpy as np

import cortex
import cortex_cuda
import retina_cuda


class Retina:
    """GPU Retina to sample images."""

    def __init__(self, width, height, rgb=True):
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
        if rgb:
            self.ret = retina_cuda.create_retina(
                loc50k, coeff50k, (height, width, 3), (int(
                    width / 2), int(height / 2))
            )
            self.cort = cortex_cuda.create_cortex_from_fields_and_locs(
                L, R, L_loc, R_loc, cort_size, gauss100=G, rgb=True
            )
        else:
            self.ret = retina_cuda.create_retina(
                loc50k, coeff50k, (height, width), (int(
                    width / 2), int(height / 2))
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
