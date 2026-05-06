#' @title Sinusoidal positional embedding for diffusion timesteps
#'
#' @description Maps a 1-D tensor of noise levels to a `num_channels`-wide
#'   sinusoidal positional embedding, used as the time-conditioning signal of
#'   the MLP denoiser. Direct port of `PositionalEmbedding` from
#'   `DiffPuter/model.py`.
#'
#' @param num_channels Embedding width. Must be even.
#' @param max_positions Maximum number of distinct positions encoded.
#' @param endpoint Whether to include the endpoint in the frequency grid.
#'
#' @return A `torch::nn_module()` generator.
#' @export
positional_embedding <- torch::nn_module(
  classname = "PositionalEmbedding",
  initialize = function(num_channels, max_positions = 10000, endpoint = FALSE) {
    self$num_channels <- as.integer(num_channels)
    self$max_positions <- max_positions
    self$endpoint <- endpoint
  },
  forward = function(x) {
    half <- as.integer(self$num_channels %/% 2)
    freqs <- torch::torch_arange(
      start = 0, end = half - 1L, step = 1L,
      dtype = torch::torch_float(), device = x$device
    )
    denom <- half - if (self$endpoint) 1 else 0
    freqs <- freqs / denom
    freqs <- (1 / self$max_positions) ^ freqs
    out <- torch::torch_outer(x$to(dtype = freqs$dtype), freqs)
    torch::torch_cat(list(out$cos(), out$sin()), dim = 2L)
  }
)

#' @title MLP denoiser used by the diffusion model
#'
#' @description Fully connected denoiser with sinusoidal time conditioning.
#'   Translated from `MLPDiffusion` in `DiffPuter/model.py`. Network depth and
#'   activations match the reference implementation: a `nn_linear` projection
#'   into a `dim_t`-wide latent, addition of the time embedding, then four
#'   `Linear -> SiLU` blocks with widths `dim_t * 2 -> dim_t * 2 -> dim_t ->
#'   d_in`.
#'
#' @param d_in Input (and output) dimension, i.e. encoded data width.
#' @param dim_t Hidden width.
#'
#' @return A `torch::nn_module()` generator.
#' @export
mlp_diffusion <- torch::nn_module(
  classname = "MLPDiffusion",
  initialize = function(d_in, dim_t = 512L) {
    self$dim_t <- as.integer(dim_t)
    self$proj <- torch::nn_linear(d_in, dim_t)
    self$mlp <- torch::nn_sequential(
      torch::nn_linear(dim_t, dim_t * 2L),
      torch::nn_silu(),
      torch::nn_linear(dim_t * 2L, dim_t * 2L),
      torch::nn_silu(),
      torch::nn_linear(dim_t * 2L, dim_t),
      torch::nn_silu(),
      torch::nn_linear(dim_t, d_in)
    )
    self$map_noise <- positional_embedding(num_channels = dim_t)
    self$time_embed <- torch::nn_sequential(
      torch::nn_linear(dim_t, dim_t),
      torch::nn_silu(),
      torch::nn_linear(dim_t, dim_t)
    )
  },
  forward = function(x, noise_labels) {
    emb <- self$map_noise(noise_labels)
    half <- as.integer(emb$shape[2] %/% 2L)
    emb <- torch::torch_cat(
      list(emb$narrow(2L, half + 1L, half), emb$narrow(2L, 1L, half)),
      dim = 2L
    )
    emb <- self$time_embed(emb)
    h <- self$proj(x) + emb
    self$mlp(h)
  }
)

#' @title EDM preconditioning wrapper
#'
#' @description Applies the EDM preconditioning coefficients
#'   (`c_skip`, `c_out`, `c_in`, `c_noise`) around a denoiser network. Direct
#'   port of `Precond` from `DiffPuter/model.py`. Inputs and outputs are
#'   coerced to float32 for numerical consistency with the reference
#'   implementation.
#'
#' @param denoise_fn The underlying denoiser, e.g. an `mlp_diffusion()`
#'   instance.
#' @param sigma_min,sigma_max,sigma_data Standard EDM hyperparameters.
#'
#' @return A `torch::nn_module()` generator.
#' @export
precond <- torch::nn_module(
  classname = "Precond",
  initialize = function(denoise_fn,
                        sigma_min = 0,
                        sigma_max = Inf,
                        sigma_data = 0.5) {
    self$denoise_fn_F <- denoise_fn
    self$sigma_min <- sigma_min
    self$sigma_max <- sigma_max
    self$sigma_data <- sigma_data
  },
  forward = function(x, sigma) {
    x <- x$to(dtype = torch::torch_float())
    sigma <- sigma$to(dtype = torch::torch_float())$reshape(c(-1L, 1L))
    c_skip <- self$sigma_data^2 / (sigma^2 + self$sigma_data^2)
    c_out <- sigma * self$sigma_data / (sigma^2 + self$sigma_data^2)$sqrt()
    c_in <- 1 / (self$sigma_data^2 + sigma^2)$sqrt()
    c_noise <- sigma$log() / 4
    x_in <- c_in * x
    f_x <- self$denoise_fn_F(x_in, c_noise$flatten())
    c_skip * x + c_out * f_x
  },
  round_sigma = function(sigma) {
    if (inherits(sigma, "torch_tensor")) sigma else torch::torch_tensor(sigma)
  }
)

#' @title Diffusion model wrapper combining preconditioned denoiser and EDM loss
#'
#' @description Convenience module returning the EDM training loss for an input
#'   batch. Mirrors `Model` from `DiffPuter/model.py`.
#'
#' @param denoise_fn The denoiser network.
#' @param P_mean,P_std,sigma_data EDM noise-distribution hyperparameters.
#'
#' @return A `torch::nn_module()` generator with a `forward(x)` returning the
#'   scalar loss.
#' @export
diffputer_model <- torch::nn_module(
  classname = "DiffPuterModel",
  initialize = function(denoise_fn,
                        P_mean = -1.2,
                        P_std = 1.2,
                        sigma_data = 0.5) {
    self$denoise_fn_D <- precond(denoise_fn, sigma_data = sigma_data)
    self$P_mean <- P_mean
    self$P_std <- P_std
    self$sigma_data <- sigma_data
  },
  forward = function(x) {
    loss <- edm_loss(self$denoise_fn_D, x, self$P_mean, self$P_std, self$sigma_data)
    loss$mean(dim = -1L)$mean()
  }
)
