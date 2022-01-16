import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.autograd.functional as AGF
from pytorch_lightning.callbacks import EarlyStopping
import math
import torch.linalg as linalg
from pytorch_lightning import loggers as pl_loggers
from torchdyn.core import ODEProblem
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

# in this work we parameterize the neural network as $C=K(q-zeta)$ for some function $K$
# and use neural network to learn K
# the advantage is that we do nto need to minimze the cost \| \partial C / \partial x J\|, which save a lot of computation


class TrainDataset(Dataset):
    def __init__(self, q_range, p_range, zeta_range):

        grid_q, grid_p, grid_zeta = torch.meshgrid(
            q_range, p_range, zeta_range)

        self.q_data = grid_q.reshape(-1,)
        self.p_data = grid_p.reshape(-1,)
        self.zeta_data = grid_zeta.reshape(-1,)

    def __len__(self):
        return len(self.q_data)

    def __getitem__(self, idx):

        return self.q_data[idx], self.p_data[idx], self.zeta_data[idx]


class casimir_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x


class controller_hamiltonian_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x


class neural_energy_casimir(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.casimir = casimir_net()
        self.controller_hamiltonian = controller_hamiltonian_net()
        self.controller_equilibrium = torch.nn.Parameter(
            torch.randn(1)[0], requires_grad=True)

    def forward(self, x):
        return 0

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(
            self.casimir.parameters(), lr=1e-4)
        optimizer2 = torch.optim.Adam(
            self.controller_hamiltonian.parameters(), lr=1e-4)
        optimizer3 = torch.optim.Adam(
            [self.controller_equilibrium], lr=1e-4)

        return optimizer1, optimizer2, optimizer3

    def training_step(self, train_batch, batch_idx, optimizer_idx):

        q_star = torch.tensor(math.pi/4, requires_grad=True,
                              dtype=torch.float32).to(self.device)
        p_star = torch.tensor(0.0, requires_grad=True,
                              dtype=torch.float32).to(self.device)

        def lyapunov_fcn(input):
            q = input[0]
            p = input[1]
            zeta = input[2]
            return 1/2*p**2+(1-torch.cos(q))+self.casimir((q-zeta).reshape(1))+self.controller_hamiltonian(zeta.reshape(1))

        equilibrium_grad = AGF.jacobian(
            lyapunov_fcn, torch.stack((q_star, p_star, self.controller_equilibrium.to(self.device))), create_graph=True)

        equilibrium_assignment = torch.norm(equilibrium_grad)

        # ! I can require the hessian to be greater than a*I, to improve the positiveness of hessian
        equilibrium_hessian = AGF.hessian(
            lyapunov_fcn, torch.stack((q_star, p_star, self.controller_equilibrium.to(self.device))), create_graph=True)-0.5*torch.eye(3).to(self.device)

        minimum_condition = F.relu(-torch.min(
            torch.real(linalg.eigvals(equilibrium_hessian))))

        loss = equilibrium_assignment + minimum_condition

        self.log('train_loss', loss)
        self.log('equilibrium_gradient(q)', equilibrium_grad[0, 0])
        self.log('equilibrium_gradient(p)', equilibrium_grad[0, 1])
        self.log('equilibrium_gradient(zeta)', equilibrium_grad[0, 2])
        self.log('equilibrium_assignment', equilibrium_assignment)
        self.log('minimum_condition', minimum_condition)
        self.log('controller_equilibrium', self.controller_equilibrium)
        return loss

# data


if __name__ == '__main__':

    # in the parameterization of Casimir case, we do not need any trian data, however, since the pytorch_lightning requires a
    # train dataloader, I still create a small data set here and pass to the trainer
    training_data = TrainDataset(
        torch.linspace(math.pi/4-2, math.pi/4+2, 1), torch.linspace(-2, 2, 1), torch.linspace(math.sqrt(2)/2+math.pi/4-2, math.sqrt(2)/2+math.pi/4+2, 1))

    train_dataloader = DataLoader(
        training_data, batch_size=1, num_workers=30, persistent_workers=True)

    tb_logger = pl_loggers.TensorBoardLogger(
        "logs", default_hp_metric=False)
    # model
    model = neural_energy_casimir()
    # training

    early_stopping = EarlyStopping(
        monitor="train_loss",
        stopping_threshold=0.0001,
        # divergence_threshold=100,
        min_delta=0.0,
        patience=90000,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(gpus=None, num_nodes=1,
                         callbacks=[], max_epochs=2000, logger=tb_logger, log_every_n_steps=50)

    trainer.fit(model, train_dataloader)

    trainer.save_checkpoint("example_pendulum.ckpt")

    new_model = neural_energy_casimir.load_from_checkpoint(
        checkpoint_path="example_pendulum.ckpt")

    print(new_model.controller_equilibrium)

# python3 -m tensorboard.main --logdir=logs

    # the following odefunc represents the closed-loop system under the neural energy casimir controller.

    class odefunc(nn. Module):
        def __init__(self, energy_casimir_model):
            super().__init__()
            self.model = energy_casimir_model

        def lyapunov_fcn(self, input):
            q = input[0]
            p = input[1]
            zeta = input[2]
            return 1/2*p**2+(1-torch.cos(q))+self.model.casimir((q-zeta).reshape(1))+self.model.controller_hamiltonian(zeta.reshape(1))

        def total_hamiltonian(self, input):
            q = input[0]
            p = input[1]
            zeta = input[2]
            return 1/2*p**2+(1-torch.cos(q))+self.model.controller_hamiltonian(zeta.reshape(1))

        def forward(self, t, x):

            return torch.matmul(torch.tensor(
                [[0.0, 1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
                AGF.jacobian(self.total_hamiltonian, x).T).reshape(-1,) \
                - torch.matmul(torch.diag(torch.tensor([0.0, 5.0, 6.0])),
                               torch.cat((torch.zeros(1),
                                          AGF.jacobian(self.lyapunov_fcn, x).reshape(-1,)[1:])))

    system = ODEProblem(
        odefunc(new_model), sensitivity='adjoint', solver='dopri5')

    t_span = torch.linspace(0, 30, 90)

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.plot(t_span, math.pi/4*torch.ones(t_span.size()), '--')
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel(r'$q(t)$')
    # ax1.set_ylim((0, 2))

    ax2.plot(t_span, 0*torch.ones(t_span.size()), '--')
    ax2.tick_params(labelbottom=False)
    ax2.set_ylabel(r'$p(t)$')
    # ax2.set_ylim((-1, 1))

    ax3.plot(t_span, new_model.controller_equilibrium.detach()
             * torch.ones(t_span.size()), '--')
    ax3.set_ylabel(r'$\xi(t)$')
    # ax3.set_ylim((0, 3))
    ax3.set_xlabel(r'$t$')

    num_trajectory = 10
    for i in range(num_trajectory):
        print("plot trajectory ", i)
        init_state = torch.tensor(
            [math.pi/4, 0, new_model.controller_equilibrium.detach()])+0.5*torch.ones(3)-1*torch.rand(3)
        # init_state = 20*torch.randn(3)

        _, trajectory = system.odeint(init_state, t_span)

        trajectory = trajectory.detach().cpu()

        ax1.plot(t_span, trajectory[:, 0])
        ax2.plot(t_span, trajectory[:, 1])
        ax3.plot(t_span, trajectory[:, 2])

    # plt.show()
    plt.savefig('pendulum_system_response.pdf')

    # calculate the distance between \bar{z} and z_star
    # z_bar = torch.zeros(3)
    z_bar = trajectory[-1, :]
    print('z_bar is', z_bar)

    z_star = torch.tensor([math.pi/4, 0, new_model.controller_equilibrium])
    print('z_star', z_star)

    print('norm(z_bar-z_star)', torch.norm(z_bar-z_star))

# plot to understand the behavioral of Lyapunov function around the equilibrium
    f = odefunc(new_model)

    q_star = torch.tensor(math.pi/4)
    p_star = torch.tensor(0.0)
    controller_equilibrium = new_model.controller_equilibrium

    q = torch.linspace(0, 2, 40)
    lyapunov_val = torch.zeros(q.size())
    for i in range(len(q)):
        lyapunov_val[i] = f.lyapunov_fcn(
            torch.stack((q[i], p_star, controller_equilibrium)))

    fig1 = plt.figure()
    plt.plot(q, lyapunov_val.detach().numpy())
    plt.plot(q_star, f.lyapunov_fcn(
        torch.stack((q_star, p_star, controller_equilibrium))).detach().numpy(), '*')
    plt.xlabel(r'$q$')
    plt.ylabel(r'$V_{\theta}(q, 0, \xi^*)$')
    # plt.show()
    plt.savefig('pendulum_lyapunov_fcn.pdf')

# plot the 3D figure for the lyapunov function for a fixed controller_equilibrium

    fig2 = plt.figure()

    q_star = torch.tensor(math.pi/4)
    p_star = torch.tensor(0.0)
    controller_equilibrium = new_model.controller_equilibrium

    q = torch.linspace(-2, 3, 40)
    p = torch.linspace(-2, 2, 40)

    grid_q, grid_p = torch.meshgrid(q, p)

    q_data = grid_q.reshape(-1,)
    p_data = grid_p.reshape(-1,)

    lyapunov_val = torch.zeros(q_data.size())
    for i in range(len(q_data)):
        lyapunov_val[i] = f.lyapunov_fcn(
            torch.stack((q_data[i], p_data[i], controller_equilibrium)))
    grid_lyapunov_val = lyapunov_val.reshape_as(grid_q)

    ax = plt.axes(projection='3d')
    ax.plot_surface(grid_q.detach().numpy(), grid_p.detach().numpy(), grid_lyapunov_val.detach().numpy(), rstride=1, cstride=1,
                    cmap='plasma', edgecolor='none')
    # ax.set_title('surface')
    plt.xlabel(r'$q$')
    plt.ylabel(r'$p$')
    # plt.zlabel(r'$V_{\theta}(q, p, \xi^*)$')

    # below code plot the projection

    ax.set(xlim=(-2, 3), ylim=(-2, 2), zlim=(-3, 4),
           xlabel=r'$q$', ylabel=r'$p$', zlabel=r'$V_{\theta}(q, p, \xi^*)$')

    ax.contour(grid_q.detach().numpy(), grid_p.detach().numpy(), grid_lyapunov_val.detach().numpy(),
               levels=20, offset=-3, cmap='plasma')

    # below code plot the desired equilibrium and the achieved equilibrium

    q_star_achieved = z_star[0]
    p_star_achieved = z_star[1]
    plt.plot(q_star, p_star, -3, 'o')
    plt.plot(q_star_achieved, p_star_achieved, -3, '+')

    plt.show()
    plt.savefig('pendulum_lyapunov_fcn3d.pdf')
