import numpy as np
import matplotlib.pyplot as plt

import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll

import time

# model of the objective (single task GP or QFF approximation): (un)comment one
#from botorch.models import SingleTaskGP as GPModel
from utils import QFFModel as GPModel


class Environment():

    def __init__(self, domain) -> None:
        
        # domain
        self.domain = domain

        # ground truth "black-box" function
        true_model = SingleTaskGP(torch.tensor([0.]).double().reshape(-1,1), torch.tensor([0.]).double().reshape(-1,1))
        true_model.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        true_model.likelihood.noise = torch.tensor([1e-2])
        true_model_lengthscale = 0.5
        true_model.covar_module.base_kernel.lengthscale = true_model_lengthscale
        true_func = true_model.posterior(self.domain).sample(torch.Size([1])).reshape(-1,1)

        self.objective = true_func[:,0]

        self.max_objective = torch.max(self.objective).reshape(-1,1)


class BOpt():

    def __init__(self, domain, objective) -> None:

        # domain
        self.domain = domain

        # objective
        self.objective = objective

        # prior mean and variance
        self.mu_Fx = torch.zeros_like(self.domain).reshape(-1,1) # prior mean to compute the location of the 2nd point
        self.sigma2_Fx = 1 * torch.ones_like(self.domain).reshape(-1,1) # prior variance to compute the location of the 2nd point

        # noise
        self.noise = 1e-2 # Gaussian white noise incurred to the observations

        # training data and observations
        self.train_X = torch.tensor([0.]).double().reshape(-1,1) # training features
        self.Fx_train_Y = torch.tensor([self.objective[200]]).double().reshape(-1, 1) # training observations

        # model of the objective
        self.Fx_model = GPModel(self.train_X, self.Fx_train_Y)
        self.Fx_model.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.Fx_model.likelihood.noise = torch.tensor([self.noise])
        self.Fx_lengthscale = 0.5
        self.Fx_model.covar_module.base_kernel.lengthscale = self.Fx_lengthscale # only for some kernels (Matérn)

        # UCB
        self.beta = torch.tensor([4.])
        self.epsilon = 1e-1 # stopping condition for instantaneous regret

        
    def learn(self):

        # compute a candidate value and concatenate
        newX = self.domain[torch.argmax(self.mu_Fx + self.beta * torch.sqrt(self.sigma2_Fx))].double().reshape(-1, 1)
        self.train_X = torch.cat([self.train_X, newX]).reshape(-1, 1)

        # sample the objective and concatenate
        newY = self.objective[torch.argmax(self.mu_Fx + self.beta * torch.sqrt(self.sigma2_Fx))].double().reshape(-1,1)
        self.Fx_train_Y = torch.cat([self.Fx_train_Y, newY]).reshape(-1, 1)

        # specify the kernel hyperparams (only if the underlying model is BoTorch's SingleTaskGP)
        self.Fx_model = GPModel(self.train_X, self.Fx_train_Y)
        self.Fx_model.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.Fx_model.likelihood.noise = torch.tensor([1e-2])
        self.Fx_lengthscale = 0.5
        self.Fx_model.covar_module.base_kernel.lengthscale = self.Fx_lengthscale
        
        # optimize the hyperparams using Empirical Bayes
        #mll = ExactMarginalLogLikelihood(self.Fx_model.likelihood, self.Fx_model) # evidence
        #fit_gpytorch_mll(mll) # fit mll (to optimize kernel hyperparams)

        # perform a Bayesian update (fit and predict)
        self.mu_Fx = self.Fx_model.posterior(self.domain).mean # posterior mean
        self.sigma2_Fx = self.Fx_model.posterior(self.domain).variance # posterior variance


    def plot(self):
        
        plt.plot(self.domain.detach().numpy(), self.objective.detach().numpy(), label=r"$f(x)$", linestyle="dotted")
        plt.scatter(self.train_X.detach().numpy(), self.Fx_train_Y.detach().numpy(), label="Observations")
        plt.plot(self.domain.detach().numpy(), self.mu_Fx.detach().numpy(), label="Mean prediction")
        plt.fill_between(
            self.domain.ravel(),
            self.mu_Fx.detach().numpy().ravel() - self.beta.detach().numpy().ravel() * np.sqrt(self.sigma2_Fx.detach().numpy().ravel()),
            self.mu_Fx.detach().numpy().ravel() + self.beta.detach().numpy().ravel() * np.sqrt(self.sigma2_Fx.detach().numpy().ravel()),
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.ylim([-5.0, 5.0])
        plt.legend()
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        _ = plt.title("BO Algorithm")
        #plt.savefig(str(i)+".png")
        #plt.close()
        plt.show()


def plot_time(T, cpu_time, iter_list):

    iterations = np.arange(1, T+1, 1) # include the 0th point?

    trimmed_cpu_time = np.trim_zeros(cpu_time, trim='b')
    trimmed_iter_list = np.trim_zeros(iter_list, trim='b')

    if len(trimmed_cpu_time > 0):

        gpucb_line, = plt.plot(iterations, trimmed_cpu_time[1:]/trimmed_iter_list[1:], '--')
        gpucb_line.set_label('BO Algorithm')

        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("CPU time in seconds")
        _ = plt.title("CPU time of each iteration")
        plt.show()


def plot_cum_time(T, cum_cpu_time, iter_list):

    iterations = np.arange(1, T+1, 1) # include the 0th point?

    trimmed_cum_cpu_time = np.trim_zeros(cum_cpu_time, trim='b')
    trimmed_iter_list = np.trim_zeros(iter_list, trim='b')

    gpucb_line, = plt.plot(iterations, trimmed_cum_cpu_time[1:]/trimmed_iter_list[1:], '--')
    gpucb_line.set_label('BO Algorithm')

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("CPU time in seconds")
    _ = plt.title("Cumulative CPU time")
    plt.show()


def plot_cum_regret(T, CR, iter_list):

    iterations = np.arange(1, T+1, 1) # include the 0th point?

    trimmed_CR = np.trim_zeros(CR, trim='b')
    trimmed_iter_list = np.trim_zeros(iter_list, trim='b')

    if len(trimmed_CR > 0):

        gpucb_line, = plt.plot(iterations, trimmed_CR[1:]/trimmed_iter_list[1:], '--')
        gpucb_line.set_label('BO Algorithm')

        plt.legend()
        plt.xlabel("$N$")
        plt.ylabel("$CR/N$")
        _ = plt.title("Cumulative regret per number of iterations")
        plt.show()
        

def main():

    # number of experiments
    num_experiments = 1
    largest_num_samples = 10000 # assumption: T <= 10000

    # metrics for evaluation and comparison
    cpu_runtimes = np.zeros(largest_num_samples)
    cum_regret_list = np.zeros(largest_num_samples)
    cum_cpu_runtimes = np.zeros(largest_num_samples)
    T = []
    iter_list = np.zeros(largest_num_samples)

    for i in range(num_experiments):

        print("Current iteration:", i)

        # init problem
        domain = torch.tensor(np.linspace(-1, 1, 401))
        env = Environment(domain)
        agent = BOpt(domain,env.objective)
        static_optimum = env.max_objective

        num_samples = 0
        inst_regret = np.inf
        cum_regret = 0
        cum_time = 0

        # GP-UCB learning
        while inst_regret > agent.epsilon:
            start_time = time.process_time()
            agent.learn()
            end_time = time.process_time()
            num_samples += 1
            inst_time = end_time - start_time
            cum_time += inst_time
            cpu_runtimes[num_samples] += inst_time
            cum_cpu_runtimes[num_samples] += cum_time
            inst_regret = torch.abs(static_optimum - agent.Fx_train_Y[-1]).detach().numpy().item()
            cum_regret += inst_regret
            cum_regret_list[num_samples] += cum_regret/num_samples
            #agent.plot() # uncomment to plot at every iteration
            iter_list[num_samples] += 1
    
        T.append(num_samples)

        print('BO Algorithm:\n')
        print('Number of samples T required:', num_samples)
        print('Intantaneous regret:', inst_regret)
        print('CPU execution time:', sum(cpu_runtimes), 'seconds\n')

        # agent.plot()

    # plot the cumulative regret over time and the CPU time
    plot_cum_regret(max(T), cum_regret_list, iter_list)
    plot_time(max(T), cpu_runtimes, iter_list)
    plot_cum_time(max(T), cum_cpu_runtimes, iter_list)


if __name__ == "__main__":
    main()
