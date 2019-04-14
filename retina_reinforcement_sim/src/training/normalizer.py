import torch


class Normalizer():
    def __init__(self, num_inputs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else
                                   "cpu")
        self.n = torch.zeros(num_inputs).to(self.device)
        self.mean = torch.zeros(num_inputs).to(self.device)
        self.mean_diff = torch.zeros(num_inputs).to(self.device)
        self.var = torch.zeros(num_inputs).to(self.device)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean)/obs_std

    def save(self, path):
        torch.save(self.n, path + "n")
        torch.save(self.mean, path + "mean")
        torch.save(self.mean_diff, path + "mean_diff")
        torch.save(self.var, path + "var")

    def load(self, path):
        self.n = torch.load(path + "n")
        self.mean = torch.load(path + "mean")
        self.mean_diff = torch.load(path + "mean_diff")
        self.var = torch.load(path + "var")


class FeatureNormalizer():
    def __init__(self, num_inputs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else
                                   "cpu")
        self.num_inputs = num_inputs
        self.n = torch.zeros(num_inputs).to(self.device)
        self.mean = torch.zeros(num_inputs).to(self.device)
        self.mean_diff = torch.zeros(num_inputs).to(self.device)
        self.var = torch.zeros(num_inputs).to(self.device)

    def observe(self, x):
        if len(x.size()) == 1:
            x = x[-self.num_inputs:]
        else:
            x = x[:, -self.num_inputs:]
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * \
            (x - self.mean)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        if len(inputs.size()) == 1:
            x = inputs[-self.num_inputs:]
            return torch.cat((inputs[:-self.num_inputs],
                              (x - self.mean) / obs_std))
        else:
            x = inputs[:, -self.num_inputs:]
            return torch.cat((inputs[:, :-self.num_inputs],
                              (x - self.mean) / obs_std), dim=1)

    def save(self, path):
        torch.save(self.n, path + "n")
        torch.save(self.mean, path + "mean")
        torch.save(self.mean_diff, path + "mean_diff")
        torch.save(self.var, path + "var")

    def load(self, path):
        self.n = torch.load(path + "n")
        self.mean = torch.load(path + "mean")
        self.mean_diff = torch.load(path + "mean_diff")
        self.var = torch.load(path + "var")
