import torch

from .. import basic_balancer
from .. import balancers


@balancers.register("amtl")
class AlignedMTLBalancer(basic_balancer.BasicBalancer):
    def __init__(self, scale_mode='min', scale_decoder_grad=False, **kwargs):
        super().__init__(**kwargs)
        self.scale_decoder_grad = scale_decoder_grad
        self.scale_mode = scale_mode
        print('AMGDA balancer scale mode:', self.scale_mode)

    def step(self, losses,
             shared_params,
             task_specific_params=None,
             shared_representation=None,
             last_shared_layer_params=None
        ):
        grads = self.get_G_wrt_shared(losses, shared_params, update_decoder_grads=True)
        grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0), self.scale_mode)
        grad, weights = grads[0].sum(-1), weights.sum(-1)

        if self.compute_stats:
            self.compute_metrics(grads[0])

        self.set_shared_grad(shared_params, grad)
        if self.scale_decoder_grad is True:
            self.apply_decoder_scaling(task_specific_params, weights)

        self.set_losses({task_id: losses[task_id] * weights[i] for i, task_id in enumerate(losses)})

        total_loss = 0.
        for key, value in self.losses.items():
            total_loss += value

        return total_loss


class ProcrustesSolver:
    @staticmethod
    def apply(grads, scale_mode='min'):
        assert (
            len(grads.shape) == 3
        ), f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable"

        with torch.no_grad():
            cov_grad_matrix_e = torch.matmul(grads.permute(0, 2, 1), grads)
            cov_grad_matrix_e = cov_grad_matrix_e.mean(0)

            singulars, basis = torch.symeig(cov_grad_matrix_e, eigenvectors=True)
            tol = (
                torch.max(singulars)
                * max(cov_grad_matrix_e.shape[-2:])
                * torch.finfo().eps
            )
            rank = sum(singulars > tol)

            order = torch.argsort(singulars, dim=-1, descending=True)
            singulars, basis = singulars[order][:rank], basis[:, order][:, :rank]

            if scale_mode == 'min':
                weights = basis * torch.sqrt(singulars[-1]).view(1, -1)
            elif scale_mode == 'median':
                weights = basis * torch.sqrt(torch.median(singulars)).view(1, -1)
            elif scale_mode == 'rmse':
                weights = basis * torch.sqrt(singulars.mean())

            weights = weights / torch.sqrt(singulars).view(1, -1)
            weights = torch.matmul(weights, basis.T)
            grads = torch.matmul(grads, weights.unsqueeze(0))

            return grads, weights, singulars

