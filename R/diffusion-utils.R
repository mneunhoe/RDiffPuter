#' @title EDM training loss
#'
#' @description Weighted mean-squared denoising loss with log-normal noise
#'   schedule, as introduced by Karras et al. ("Elucidating the Design Space of
#'   Diffusion-Based Generative Models", 2022). Direct port of `EDMLoss` from
#'   `DiffPuter/diffusion_utils.py`.
#'
#' @param net A preconditioned denoiser, e.g. `precond()` instance.
#' @param data Input batch as a torch tensor of shape `(B, D)`.
#' @param P_mean,P_std Mean and standard deviation of the log-noise distribution.
#' @param sigma_data Expected standard deviation of the data.
#'
#' @return A torch tensor of unreduced per-element loss with shape `(B, D)`.
#' @export
edm_loss <- function(net, data, P_mean = -1.2, P_std = 1.2, sigma_data = 0.5) {
  rnd_normal <- torch::torch_randn(data$shape[1], device = data$device)
  sigma <- (rnd_normal * P_std + P_mean)$exp()
  weight <- (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
  noise <- torch::torch_randn_like(data) * sigma$unsqueeze(2)
  d_yn <- net(data + noise, sigma)
  weight$unsqueeze(2) * (d_yn - data)^2
}

# EDM sampler hyperparameters (mirroring DiffPuter/diffusion_utils.py).
.SIGMA_MIN <- 0.002
.SIGMA_MAX <- 80
.RHO <- 7
.S_CHURN <- 1
.S_MIN <- 0
.S_MAX <- Inf
.S_NOISE <- 1

#' @title Single Heun-style EDM sampling step
#'
#' @description Performs one outer denoising step with the second-order Heun
#'   correction. Direct port of `sample_step()` from
#'   `DiffPuter/diffusion_utils.py`.
#'
#' @param net Preconditioned denoiser (`precond()` instance).
#' @param num_steps Total number of outer steps in the schedule.
#' @param i Current step index, 0-based to match the reference implementation.
#' @param t_cur,t_next Current and next noise levels, scalar torch tensors.
#' @param x_next The current latent sample.
#'
#' @return The latent sample after one denoising step.
#' @export
sample_step <- function(net, num_steps, i, t_cur, t_next, x_next) {
  x_cur <- x_next
  t_cur_val <- as.numeric(t_cur$cpu())
  in_range <- (.S_MIN <= t_cur_val) && (t_cur_val <= .S_MAX)
  gamma <- if (in_range) min(.S_CHURN / num_steps, sqrt(2) - 1) else 0
  t_hat <- net$round_sigma(t_cur + gamma * t_cur)
  noise <- .S_NOISE * torch::torch_randn_like(x_cur)
  x_hat <- x_cur + (t_hat^2 - t_cur^2)$sqrt() * noise
  denoised <- net(x_hat, t_hat)$to(dtype = torch::torch_float())
  d_cur <- (x_hat - denoised) / t_hat
  x_next <- x_hat + (t_next - t_hat) * d_cur
  if (i < num_steps - 1L) {
    denoised <- net(x_next, t_next)$to(dtype = torch::torch_float())
    d_prime <- (x_next - denoised) / t_next
    x_next <- x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
  }
  x_next
}

# Construct the rho-schedule of noise levels.
.edm_t_steps <- function(num_steps, sigma_min, sigma_max, device) {
  step_indices <- torch::torch_arange(
    start = 0, end = num_steps - 1L, step = 1L,
    dtype = torch::torch_float(), device = device
  )
  t_steps <- (sigma_max^(1 / .RHO) +
                step_indices / (num_steps - 1L) *
                  (sigma_min^(1 / .RHO) - sigma_max^(1 / .RHO)))^.RHO
  torch::torch_cat(list(t_steps, torch::torch_zeros(1L, device = device)))
}

#' @title Conditional sampling for missing-value imputation
#'
#' @description RePaint-style sampler. At each outer denoising step the latent
#'   is split: known features are re-noised from the observed values while
#'   unknown features advance through one Heun EDM step. The two halves are
#'   re-glued via the missingness mask. An inner resampling loop of length
#'   `num_resamplings` (N=20 in the reference paper) helps match the marginal
#'   to the data distribution. Direct port of `impute_mask()` from
#'   `DiffPuter/diffusion_utils.py`.
#'
#' @param net A preconditioned denoiser.
#' @param x A torch tensor of observed values; missing entries should already
#'   be set to zero.
#' @param mask A torch tensor of the same shape as `x`. `1` marks **missing**
#'   entries, `0` marks observed (matching the reference implementation).
#' @param num_steps Number of outer denoising steps.
#' @param num_resamplings Number of inner re-noising iterations per outer step.
#' @param device Torch device.
#'
#' @return A torch tensor of the same shape as `x` containing reconstructed
#'   values for the missing entries (observed entries are not preserved here;
#'   the caller is expected to overlay the original observed values).
#' @export
impute_mask <- function(net, x, mask, num_steps = 50L,
                        num_resamplings = 20L, device = "cpu") {
  num_steps <- as.integer(num_steps)
  num_resamplings <- as.integer(num_resamplings)
  num_samples <- x$shape[1]
  dim <- x$shape[2]

  sigma_min <- max(.SIGMA_MIN, net$sigma_min)
  sigma_max <- min(.SIGMA_MAX, net$sigma_max)
  t_steps <- .edm_t_steps(num_steps, sigma_min, sigma_max, device)

  mask_f <- mask$to(dtype = torch::torch_float())
  one_minus_mask <- 1 - mask_f

  x_t <- torch::torch_randn(c(num_samples, dim), device = device) * t_steps[1]

  torch::with_no_grad({
    for (i in seq_len(num_steps)) {
      t_cur <- t_steps[i]
      t_next <- t_steps[i + 1L]
      if (i < num_steps) {
        for (j in seq_len(num_resamplings)) {
          n_prev <- torch::torch_randn_like(x_t) * t_next
          x_known_t_prev <- x + n_prev
          x_unknown_t_prev <- sample_step(net, num_steps, i - 1L,
                                          t_cur, t_next, x_t)
          x_t_prev <- x_known_t_prev * one_minus_mask +
            x_unknown_t_prev * mask_f
          if (j == num_resamplings) {
            x_t <- x_t_prev
          } else {
            n <- torch::torch_randn_like(x_t) *
              (t_cur^2 - t_next^2)$sqrt()
            x_t <- x_t_prev + n
          }
        }
      } else {
        n_prev <- torch::torch_randn_like(x_t) * t_next
        x_known_t_prev <- x + n_prev
        x_unknown_t_prev <- sample_step(net, num_steps, i - 1L,
                                        t_cur, t_next, x_t)
        x_t <- x_known_t_prev * one_minus_mask +
          x_unknown_t_prev * mask_f
      }
    }
  })

  x_t
}

#' @title Unconditional sampling from the diffusion model
#'
#' @description Draws synthetic samples by running the full Heun EDM sampler
#'   from a Gaussian prior. Useful for diagnostics and sanity checks; the
#'   imputation pipeline itself uses `impute_mask()`.
#'
#' @param net A preconditioned denoiser.
#' @param num_samples Number of samples to generate.
#' @param dim Encoded data dimension.
#' @param num_steps Number of outer denoising steps.
#' @param device Torch device.
#'
#' @return A torch tensor of shape `(num_samples, dim)`.
#' @export
diffusion_sample <- function(net, num_samples, dim,
                             num_steps = 50L, device = "cpu") {
  num_steps <- as.integer(num_steps)
  sigma_min <- max(.SIGMA_MIN, net$sigma_min)
  sigma_max <- min(.SIGMA_MAX, net$sigma_max)
  t_steps <- .edm_t_steps(num_steps, sigma_min, sigma_max, device)
  x_next <- torch::torch_randn(c(num_samples, dim), device = device) * t_steps[1]
  torch::with_no_grad({
    for (i in seq_len(num_steps)) {
      t_cur <- t_steps[i]
      t_next <- t_steps[i + 1L]
      x_next <- sample_step(net, num_steps, i - 1L, t_cur, t_next, x_next)
    }
  })
  x_next
}
