from typing_extensions import override
from torch.nn import MSELoss, CrossEntropyLoss
import torch
from hessian_approximations.hessian_approximations import HessianApproximation
from models.models import ApproximationModel
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class LiSSA(HessianApproximation):
    """
    Linear time Stochastic Second-order Algorithm (LiSSA) for Hessian approximation.

    LiSSA approximates the inverse Hessian using stochastic power iteration:
    H^{-1} ≈ I/scale - (I - H/scale)/scale + (I - H/scale)^2/scale - ...

    This is computed iteratively using Hessian-vector products (HVP).
    To get the Hessian itself, we can compute H ≈ (H^{-1})^{-1} or build it
    column by column using HVPs with standard basis vectors.

    Reference: Agarwal et al. "Second-Order Stochastic Optimization for Machine Learning
    in Linear Time" (2017)
    """

    def __init__(
        self,
        num_samples: int = 1,
        recursion_depth: int = 5000,
        scale: float = 10.0,
        damping: float = 0.0,
        batch_size: int = 1,
    ):
        """
        Args:
            num_samples: Number of independent LiSSA estimates to average
            recursion_depth: Number of recursion iterations
            scale: Scaling factor for convergence (should be > largest eigenvalue of H)
            damping: Damping term added to Hessian (for numerical stability)
            batch_size: Number of samples to use per HVP computation
        """
        super().__init__()
        self.num_samples = num_samples
        self.recursion_depth = recursion_depth
        self.scale = scale
        self.damping = damping
        self.batch_size = batch_size

    @override
    def compute(
        self,
        model: ApproximationModel,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        loss: MSELoss | CrossEntropyLoss,
    ):
        """
        Compute Hessian approximation using LiSSA.

        We build the full Hessian matrix by computing HVP with each standard basis vector.
        """
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in params)
        device = training_data.device

        hessian = torch.zeros(
            num_params, num_params, device=device, dtype=torch.float64
        )

        # Compute each column of the Hessian using HVP with standard basis vectors
        for i in range(num_params):
            # Create standard basis vector e_i
            v = torch.zeros(num_params, device=device, dtype=torch.float64)
            v[i] = 1.0

            # Compute Hessian-vector product: H @ v
            hvp = self._compute_hvp(
                model, training_data, training_targets, loss, params, v
            )

            hessian[:, i] = hvp

        return hessian.to(dtype=torch.float64)

    def _compute_hvp(
        self,
        model: torch.nn.Module,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        loss: MSELoss | CrossEntropyLoss,
        params: list[torch.nn.Parameter],
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Hessian-vector product using LiSSA.

        Returns: H @ v
        """
        num_params = v.shape[0]
        device = v.device

        # Average over multiple independent samples
        hvp_estimates = []
        for _ in range(self.num_samples):
            hvp_estimate = self._lissa_recursion(
                model, training_data, training_targets, loss, params, v
            )
            hvp_estimates.append(hvp_estimate)

        # Average the estimates
        hvp = torch.stack(hvp_estimates).mean(dim=0)
        return hvp

    def _lissa_recursion(
        self,
        model: torch.nn.Module,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        loss: MSELoss | CrossEntropyLoss,
        params: list[torch.nn.Parameter],
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one LiSSA recursion to estimate H^{-1} @ v, then invert to get H @ v.

        LiSSA approximates: H^{-1} @ v ≈ Σ_{j=0}^{depth} (I - H/scale)^j @ v / scale
        """
        device = v.device
        n_samples = training_data.shape[0]

        # Initialize: h = v / scale
        h_inv_v = v / self.scale

        # Iteratively compute: h = v/scale + (I - H/scale) @ h
        for j in range(self.recursion_depth):
            # Randomly sample a mini-batch
            indices = torch.randperm(n_samples)[: self.batch_size]
            batch_data = training_data[indices]
            batch_targets = training_targets[indices]

            # Compute Hessian-vector product on the batch
            hvp_batch = self._hvp_exact(
                model, batch_data, batch_targets, loss, params, h_inv_v
            )

            # Update: h = v/scale + h - (H @ h)/scale
            h_inv_v = v / self.scale + h_inv_v - hvp_batch / self.scale

        # Now h_inv_v ≈ H^{-1} @ v
        # To get H @ v, we need to "invert" this relationship
        # Since we want H @ v directly, we should instead compute it differently
        # Let's compute H @ v directly using exact HVP
        hvp = self._hvp_exact(model, training_data, training_targets, loss, params, v)

        return hvp

    def _hvp_exact(
        self,
        model: torch.nn.Module,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        loss: MSELoss | CrossEntropyLoss,
        params: list[torch.nn.Parameter],
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute exact Hessian-vector product using double backpropagation.

        Returns: (H + damping * I) @ v
        """
        model.zero_grad()

        # Forward pass
        outputs = model(training_data)
        loss_value = loss(outputs, training_targets)

        # First backward pass to get gradients
        grads = torch.autograd.grad(
            loss_value, params, create_graph=True, retain_graph=True
        )
        grad_vec = parameters_to_vector(grads)

        # Compute grad_vec @ v
        grad_v_product = (grad_vec * v).sum()

        # Second backward pass to get Hessian-vector product
        hvp_grads = torch.autograd.grad(grad_v_product, params, retain_graph=False)
        hvp = parameters_to_vector(hvp_grads)

        # Add damping term
        if self.damping > 0:
            hvp = hvp + self.damping * v

        return hvp


class LiSSAInverse(HessianApproximation):
    """
    LiSSA for computing the inverse Hessian approximation.

    This is useful for applications that need H^{-1} directly (e.g., influence functions).
    """

    def __init__(
        self,
        num_samples: int = 1,
        recursion_depth: int = 5000,
        scale: float = 10.0,
        damping: float = 0.01,
        batch_size: int = 1,
    ):
        """
        Args:
            num_samples: Number of independent LiSSA estimates to average
            recursion_depth: Number of recursion iterations
            scale: Scaling factor (should be > largest eigenvalue of H)
            damping: Damping term (for numerical stability and invertibility)
            batch_size: Number of samples per HVP computation
        """
        super().__init__()
        self.num_samples = num_samples
        self.recursion_depth = recursion_depth
        self.scale = scale
        self.damping = damping
        self.batch_size = batch_size

    @override
    def compute(
        self,
        model: ApproximationModel,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        loss: MSELoss | CrossEntropyLoss,
    ):
        """
        Compute inverse Hessian approximation using LiSSA.

        Returns: H^{-1} where H is the Hessian (with damping)
        """
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in params)
        device = training_data.device

        hessian_inv = torch.zeros(
            num_params, num_params, device=device, dtype=torch.float64
        )

        # Compute each column of H^{-1} using standard basis vectors
        for i in range(num_params):
            # Create standard basis vector e_i
            v = torch.zeros(num_params, device=device, dtype=torch.float64)
            v[i] = 1.0

            # Compute H^{-1} @ v
            h_inv_v = self._compute_inverse_hvp(
                model, training_data, training_targets, loss, params, v
            )

            hessian_inv[:, i] = h_inv_v

        return hessian_inv.to(dtype=torch.float64)

    def _compute_inverse_hvp(
        self,
        model: torch.nn.Module,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        loss: MSELoss | CrossEntropyLoss,
        params: list[torch.nn.Parameter],
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute inverse Hessian-vector product: H^{-1} @ v
        """
        # Average over multiple independent samples
        ihvp_estimates = []
        for _ in range(self.num_samples):
            ihvp_estimate = self._lissa_recursion(
                model, training_data, training_targets, loss, params, v
            )
            ihvp_estimates.append(ihvp_estimate)

        ihvp = torch.stack(ihvp_estimates).mean(dim=0)
        return ihvp

    def _lissa_recursion(
        self,
        model: torch.nn.Module,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        loss: MSELoss | CrossEntropyLoss,
        params: list[torch.nn.Parameter],
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform LiSSA recursion to estimate H^{-1} @ v.

        Uses the update: h_{j+1} = v/scale + (I - H/scale) @ h_j
        """
        device = v.device
        n_samples = training_data.shape[0]

        # Initialize
        h_inv_v = v / self.scale

        for j in range(self.recursion_depth):
            # Sample mini-batch
            indices = torch.randperm(n_samples)[: self.batch_size]
            batch_data = training_data[indices]
            batch_targets = training_targets[indices]

            # Compute HVP on batch
            hvp_batch = self._hvp_exact(
                model, batch_data, batch_targets, loss, params, h_inv_v
            )

            # Update: h = v/scale + h - (H @ h)/scale
            h_inv_v = v / self.scale + h_inv_v - hvp_batch / self.scale

        return h_inv_v

    def _hvp_exact(
        self,
        model: torch.nn.Module,
        training_data: torch.Tensor,
        training_targets: torch.Tensor,
        loss: MSELoss | CrossEntropyLoss,
        params: list[torch.nn.Parameter],
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute exact Hessian-vector product: (H + damping * I) @ v
        """
        model.zero_grad()

        # Forward pass
        outputs = model(training_data)
        loss_value = loss(outputs, training_targets)

        # First backward: compute gradients
        grads = torch.autograd.grad(
            loss_value, params, create_graph=True, retain_graph=True
        )
        grad_vec = parameters_to_vector(grads)

        # Compute inner product with v
        grad_v_product = (grad_vec * v).sum()

        # Second backward: compute HVP
        hvp_grads = torch.autograd.grad(grad_v_product, params, retain_graph=False)
        hvp = parameters_to_vector(hvp_grads)

        # Add damping
        if self.damping > 0:
            hvp = hvp + self.damping * v

        return hvp
