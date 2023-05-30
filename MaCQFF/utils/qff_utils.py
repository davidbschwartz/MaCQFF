import torch
import gpytorch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

from utils.embedding import *

import argparse
import yaml


workspace = "MaCQFF"

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="smcc_MaCQFF")  # params: smcc_MacOpt_GP or smcc_MaCQFF

parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=0)
args = parser.parse_args()

with open(workspace + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
#print(params)


class QFFModel():

    def __init__(self, train_X, train_Y) -> None:

        # noise
        self.noise = params["agent"]["Fx_noise"]

        # kernel attributes to conform to BoTorch's SingleTaskGP
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.likelihood = GaussianLikelihood()
        self.likelihood.noise = torch.tensor([self.noise])
        self.Fx_lengthscale = params["agent"]["Fx_lengthscale"]
        self.covar_module.base_kernel.lengthscale = self.Fx_lengthscale # only for some kernels (Matérn)
        
        # QFF embedding to approximate the kernel (RBF kernel with 100 basis functions)
        self.emb = MaternEmbedding(gamma=self.Fx_lengthscale, nu=2.5, m=80, d=2, kernel='modified_matern') # adapt nu?

        # mean and covariance of Fourier feature mapping
        self.training_embedding = self.emb.embed(train_X.double())
        self.sigma_t = torch.matmul(torch.transpose(self.training_embedding, 0, 1), self.training_embedding) + self.noise * torch.eye(self.emb.m)
        self.nu_t = torch.matmul(torch.matmul(torch.inverse(self.sigma_t), torch.transpose(self.training_embedding, 0, 1)), train_Y.double())

    
    def update(self, X):

        # mean and covariance of approximated posterior
        mu_Fx = torch.matmul(self.evaluation_embedding.transpose(0,1), self.nu_t)
        covar_matrix_Fx = self.noise * torch.matmul(self.evaluation_embedding.transpose(0,1), torch.matmul(torch.inverse(self.sigma_t), self.evaluation_embedding))

        # multi-variate Gaussian (perturb matrix to ensure pos. definiteness)
        self.mvn = MultivariateNormal(mu_Fx[:,0], covar_matrix_Fx + 1e-9 * torch.eye(X.shape[0]))

        return self.mvn


    def posterior(self, X) -> GPyTorchPosterior:

        self.evaluation_embedding = self.emb.embed(X.double()).transpose(0,1)

        approximated_posterior = self.update(X)

        ret = GPyTorchPosterior(approximated_posterior)

        return ret
