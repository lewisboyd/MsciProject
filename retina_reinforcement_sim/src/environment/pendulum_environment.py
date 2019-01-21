import gym
import numpy as np
import cPickle as pickle
import cortex
import cortex_cuda
import retina_cuda

class Environment:

    def __init__(self):
        self.env = gym.make('Pendulum-v0')
        self.retina = Retina()

    def reset(self):
        """Resets the environment returning the new initial state."""
        self.env.reset()
        img1 = self.retina.sample(env.render(mode='rgb_array'))
        env.step([0])
        img2 = self.retina.sample(env.render(mode='rgb_array'))
        env.step([0])
        img3 = self.retina.sample(env.render(mode='rgb_array'))
        return np.stack((img1, img2, img3))

    def step(self, action):
        """Executes the action returning the new state and reward."""
        action = action * 2
        _, reward1, _, _ = env.step(action)
        img1 = self.retina.sample(env.render(mode='rgb_array'))
        _, reward2, _, _ = env.step(action)
        img2 = self.retina.sample(env.render(mode='rgb_array'))
        _, reward3, _, _ = env.step(action)
        img3 = self.retina.sample(env.render(mode='rgb_array'))
        new_state = np.stack((img1, img2, img3))
        reward = reward1 + reward2 + reward3
        return new_state, reward


class Retina:

    def __init__(self):
        self.ret, self.cort = self._create_retina_and_cortex()

    def sample(image):
        v_c = self.ret.sample(image)
        l_c = self.cort.cort_image_left(v_c)
        r_c = self.cort.cort_image_right(v_c)
        return np.concatenate((np.rot90(l_c), np.rot90(r_c, k=3)), axis=1)

    def _create_retina_and_cortex():
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
        ret = retina_cuda.create_retina(loc50k, coeff50k,
                                        (500, 500, 3), (250, 250))
        cort = cortex_cuda.create_cortex_from_fields_and_locs(
            L, R, L_loc, R_loc, cort_size, gauss100=G, rgb=True
        )

        # Return the retina and cortex
        return ret, cort
