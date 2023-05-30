import torch
import gpytorch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

from embedding import *

import time


class QFFModel():

    def __init__(self, train_X, train_Y) -> None:

        # noise
        self.noise = 1e-3 # Gaussian white noise incurred to the observations

        # kernel attributes to conform to BoTorch's SingleTaskGP
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.likelihood = GaussianLikelihood()
        self.likelihood.noise = torch.tensor([self.noise])
        self.Fx_lengthscale = 0.4
        self.covar_module.base_kernel.lengthscale = self.Fx_lengthscale # only for some kernels (Matérn)
        
        # QFF embedding to approximate the kernel (RBF kernel with 100 basis functions)
        self.emb = MaternEmbedding(gamma=0.4, nu=2.5, m=80, d=2, kernel='modified_matern')

        # mean and covariance of Fourier feature mapping
        self.training_embedding = self.emb.embed(train_X)
        self.sigma_t = torch.matmul(torch.transpose(self.training_embedding, 0, 1), self.training_embedding) + self.noise * torch.eye(self.emb.m)
        self.nu_t = torch.matmul(torch.matmul(torch.inverse(self.sigma_t), torch.transpose(self.training_embedding, 0, 1)), train_Y)

    
    def update(self, X):

        # mean and covariance of approximated posterior
        mu_Fx = torch.matmul(self.evaluation_embedding.transpose(0,1), self.nu_t)
        covar_matrix_Fx = self.noise * torch.matmul(self.evaluation_embedding.transpose(0,1), torch.matmul(torch.inverse(self.sigma_t), self.evaluation_embedding))

        # multi-variate Gaussian (perturb matrix to ensure pos. definiteness)
        self.mvn = MultivariateNormal(mu_Fx[:,0], covar_matrix_Fx + 1e-12 * torch.eye(X.shape[0]))

        return self.mvn


    def posterior(self, X) -> GPyTorchPosterior:

        self.evaluation_embedding = self.emb.embed(X).transpose(0,1)

        approximated_posterior = self.update(X)

        ret = GPyTorchPosterior(approximated_posterior)

        return ret
