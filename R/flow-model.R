#' @title Flow-matching model wrapping a velocity MLP
#'
#' @description Wraps an [mlp_diffusion()] denoiser as a velocity field
#'   `v(x, t)` for flow matching. The wrapper has no preconditioning (unlike
#'   the EDM [precond()] used by [diffputer_model()]); the underlying MLP
#'   directly outputs a velocity that transports `x_0 ~ N(0, I)` at `t = 0` to
#'   the data distribution at `t = 1`.
#'
#'   On `forward(x)` the module computes the per-element flow-matching loss
#'   used by [diffputer_trainer()] when `method = "flow"`.
#'
#' @param denoise_fn The underlying MLP denoiser, e.g. an [mlp_diffusion()]
#'   instance. Reused unchanged from the EDM path.
#'
#' @return A `torch::nn_module()` generator.
#' @references Lipman, Y. et al. "Flow Matching for Generative Modeling."
#'   ICLR 2023.
#' @export
flow_model <- torch::nn_module(
  classname = "FlowModel",
  initialize = function(denoise_fn) {
    self$denoise_fn_F <- denoise_fn
    # Mirror the public attributes that `precond()` exposes so samplers can
    # treat the two model wrappers interchangeably.
    self$sigma_min <- 0
    self$sigma_max <- 1
  },
  forward = function(x) {
    loss <- flow_loss(self, x)
    loss$mean(dim = -1L)$mean()
  },
  velocity = function(x, t) {
    # `flow_velocity()` calls this; centralised so external callers can
    # subclass `flow_model` and override the forward pass cheaply.
    self$denoise_fn_F(x, t)
  }
)

#' @title Velocity field of a flow-matching model
#'
#' @description Convenience accessor: returns `v_theta(x, t)` for a [flow_model()]
#'   wrapper. `t` may be a scalar or a length-`B` 1-D tensor.
#'
#' @param net A [flow_model()] instance.
#' @param x A torch tensor of shape `(B, D)`.
#' @param t Time tensor; shape `()`, `(1,)`, or `(B,)`. Values in `[0, 1]`.
#'
#' @return Velocity tensor of shape `(B, D)`.
#' @export
flow_velocity <- function(net, x, t) {
  if (!inherits(t, "torch_tensor")) {
    t <- torch::torch_tensor(t, dtype = torch::torch_float(), device = x$device)
  }
  if (t$dim() == 0L) {
    t <- t$expand(x$shape[1])
  } else if (t$shape[1] == 1L && x$shape[1] != 1L) {
    t <- t$expand(x$shape[1])
  }
  net$velocity(x, t)
}
