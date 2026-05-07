#' @title Flow-matching training loss
#'
#' @description Per-element MSE between the predicted velocity
#'   `v_θ(x_t, t)` and the analytic velocity `x_1 - x_0` of the linear
#'   interpolant `x_t = (1 - t) * x_0 + t * x_1`. Direct port of the loss in
#'   Lipman et al. (2023, "Flow Matching for Generative Modeling").
#'
#' @param net A [flow_model()] instance.
#' @param data A torch tensor of shape `(B, D)`. The "endpoint" `x_1`.
#'
#' @return A torch tensor of shape `(B, D)` of unreduced squared errors.
#' @export
flow_loss <- function(net, data) {
  device <- data$device
  bsz <- data$shape[1]
  x0 <- torch::torch_randn_like(data)
  t <- torch::torch_rand(bsz, dtype = torch::torch_float(), device = device)
  t_col <- t$unsqueeze(2)
  x_t <- (1 - t_col) * x0 + t_col * data
  v_target <- data - x0
  v_pred <- flow_velocity(net, x_t, t)
  (v_pred - v_target)^2
}

#' @title One Euler step of a flow-matching ODE solver
#'
#' @description Advances the latent state `x` from time `t_cur` to `t_next`
#'   using the velocity field of `net`. Mirrors [sample_step()] in the EDM
#'   path but without any noise injection or preconditioning.
#'
#' @param net A [flow_model()] instance.
#' @param t_cur,t_next Scalar torch tensors with values in `[0, 1]`.
#' @param x The current latent sample, shape `(B, D)`.
#'
#' @return Latent sample after one Euler step.
#' @export
flow_sample_step <- function(net, t_cur, t_next, x) {
  v <- flow_velocity(net, x, t_cur)
  x + (t_next - t_cur) * v
}

# Helper: build a uniform time grid `[0, dt, 2dt, ..., 1]` on the right device.
.flow_t_grid <- function(num_steps, device) {
  torch::torch_linspace(start = 0, end = 1, steps = num_steps + 1L,
                        dtype = torch::torch_float(), device = device)
}

#' @title Conditional sampling for missing-value imputation (flow matching)
#'
#' @description Flow-matching analogue of [impute_mask()]. At each Euler step
#'   the latent is split: observed entries advance along the deterministic
#'   linear interpolant `(1 - t) * x_0 + t * x_obs`, while missing entries
#'   advance via the model's velocity. The two halves are re-glued via the
#'   missingness mask.
#'
#' @details
#' Two `resample_strategy` options:
#'
#' * `"repaint"` (default): runs an inner loop of length `num_resamplings` per
#'   outer Euler step that re-flows the latent backward and re-steps forward,
#'   the flow analogue of RePaint's re-noising loop. Slower, but provides the
#'   variance-reduction benefit familiar from the EDM imputer.
#' * `"single"`: a single Euler step per outer step; `num_resamplings` is
#'   ignored. Cheapest and fastest; practical for users who run
#'   `num_trials > 1` and rely on the trial average for variance reduction.
#'
#' This is a heuristic adaptation of the RePaint construction; it is not the
#' exactness guarantee of DiffPuter's Theorem 1, which was derived for the
#' EDM diffusion case. Empirically the heuristic matches the EDM imputer's
#' quality at ~1/5 the sampling steps; for theoretical fidelity use
#' `method = "edm"`.
#'
#' @param net A [flow_model()] instance.
#' @param x A torch tensor of observed values; missing entries should already
#'   be zeroed.
#' @param mask A torch tensor of the same shape as `x`. `1` marks **missing**
#'   entries, `0` marks observed.
#' @param num_steps Number of outer Euler steps.
#' @param num_resamplings Inner re-flow iterations per outer step. Honoured
#'   only when `resample_strategy = "repaint"`.
#' @param resample_strategy `"repaint"` (default) or `"single"`.
#' @param device Torch device.
#'
#' @return A torch tensor of the same shape as `x` containing reconstructed
#'   values for the missing entries.
#' @export
flow_impute_mask <- function(net, x, mask,
                             num_steps = 10L,
                             num_resamplings = 5L,
                             resample_strategy = "repaint",
                             device = "cpu") {
  resample_strategy <- validate_resample_strategy(resample_strategy)
  num_steps <- as.integer(num_steps)
  num_resamplings <- as.integer(num_resamplings)
  num_samples <- x$shape[1]
  dim <- x$shape[2]

  t_grid <- .flow_t_grid(num_steps, device)
  mask_f <- mask$to(dtype = torch::torch_float())
  one_minus_mask <- 1 - mask_f

  x0 <- torch::torch_randn(c(num_samples, dim), device = device)
  x_t <- x0$clone()

  torch::with_no_grad({
    for (i in seq_len(num_steps)) {
      t_cur <- t_grid[i]
      t_next <- t_grid[i + 1L]

      if (resample_strategy == "single") {
        x_known_next <- (1 - t_next) * x0 + t_next * x
        x_unknown_next <- flow_sample_step(net, t_cur, t_next, x_t)
        x_t <- mask_f * x_unknown_next + one_minus_mask * x_known_next
      } else {
        for (j in seq_len(num_resamplings)) {
          x_known_next <- (1 - t_next) * x0 + t_next * x
          x_unknown_next <- flow_sample_step(net, t_cur, t_next, x_t)
          x_t_next <- mask_f * x_unknown_next + one_minus_mask * x_known_next
          if (j < num_resamplings) {
            # Heuristic re-flow: take a backward Euler step along the
            # velocity at the new state, then re-step on the next inner
            # iteration. This mirrors RePaint's re-noising loop in spirit.
            v_back <- flow_velocity(net, x_t_next, t_next)
            x_t <- x_t_next - (t_next - t_cur) * v_back
          } else {
            x_t <- x_t_next
          }
        }
      }
    }
  })

  x_t
}

#' @title Unconditional sampling from a flow-matching model
#'
#' @description Generates synthetic samples by running the Euler ODE solver
#'   from `x_0 ~ N(0, I)` at `t = 0` to `t = 1`. Useful for diagnostics and
#'   sanity checks; the imputation pipeline uses [flow_impute_mask()].
#'
#' @param net A [flow_model()] instance.
#' @param num_samples Number of samples to draw.
#' @param dim Encoded data dimension.
#' @param num_steps Number of Euler steps.
#' @param device Torch device.
#'
#' @return A torch tensor of shape `(num_samples, dim)`.
#' @export
flow_sample <- function(net, num_samples, dim,
                        num_steps = 10L, device = "cpu") {
  num_steps <- as.integer(num_steps)
  t_grid <- .flow_t_grid(num_steps, device)
  x_t <- torch::torch_randn(c(num_samples, dim), device = device)
  torch::with_no_grad({
    for (i in seq_len(num_steps)) {
      t_cur <- t_grid[i]
      t_next <- t_grid[i + 1L]
      x_t <- flow_sample_step(net, t_cur, t_next, x_t)
    }
  })
  x_t
}
