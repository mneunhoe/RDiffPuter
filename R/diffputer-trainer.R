#' @title Train a diffusion denoiser on a (possibly imputed) tabular matrix
#'
#' @description Runs one M-step of the DiffPuter EM loop: fits an EDM diffusion
#'   model to `x` using Adam with a `ReduceLROnPlateau` schedule and best-loss
#'   checkpointing. Mirrors the training loop in `DiffPuter/main.py`. The
#'   returned object has class `trained_RDiffPuter` and is suitable input for
#'   [impute()].
#'
#' @param x Numeric matrix or torch tensor of shape `(n, d)`. Missing entries
#'   should already be filled in (e.g. with `0` for the first EM iteration, or
#'   with the previous iteration's reconstructions).
#' @param method Generative model. `"edm"` (default) trains an EDM diffusion
#'   denoiser as in the reference DiffPuter; `"flow"` trains a flow-matching
#'   velocity field with linear interpolant noise-to-data paths. Flow models
#'   typically need fewer sampling steps at imputation time.
#' @param hidden_dim Hidden width of the MLP denoiser. Defaults to 1024 to
#'   match the reference implementation; set lower (e.g. 256) for small data.
#' @param epochs Maximum number of training epochs.
#' @param batch_size Mini-batch size.
#' @param base_lr Adam learning rate.
#' @param patience Early-stopping patience: number of epochs without
#'   improvement on the best loss before stopping.
#' @param scheduler_factor Multiplicative factor for `ReduceLROnPlateau`.
#' @param scheduler_patience Epoch patience for the learning-rate scheduler.
#' @param device One of `"cpu"`, `"cuda"`, or `"mps"`. Falls back to `"cpu"`
#'   with a warning if the requested accelerator is unavailable.
#' @param seed Optional integer seed for R and torch RNGs.
#' @param checkpoint_path Optional file path to save the best model
#'   `state_dict()` to disk. If `NULL`, the best state is kept in memory only.
#' @param P_mean,P_std,sigma_data EDM noise distribution parameters. Ignored
#'   when `method = "flow"`.
#' @param verbose If `TRUE`, displays a progress bar with the running loss.
#'
#' @return A list with class `trained_RDiffPuter` containing:
#' \describe{
#'   \item{model}{The fitted `diffputer_model` (with best weights loaded).}
#'   \item{denoiser}{Convenience handle for `model$denoise_fn_D`.}
#'   \item{optimizer}{The Adam optimizer.}
#'   \item{losses}{Vector of per-epoch mean training losses.}
#'   \item{best_loss}{The lowest loss achieved.}
#'   \item{settings}{List of hyperparameters used.}
#' }
#'
#' @references Zhang, H. et al. "DiffPuter: Empowering Diffusion Models for
#'   Missing Data Imputation." ICLR 2025.
#' @export
#' @examples
#' \dontrun{
#'   data <- data.frame(a = rnorm(200), b = runif(200))
#'   transformer <- data_transformer$new()
#'   transformer$fit(data)
#'   transformed <- transformer$transform(data)
#'   trained <- diffputer_trainer(transformed$x, hidden_dim = 64,
#'                                epochs = 50, batch_size = 64)
#' }
diffputer_trainer <- function(x,
                              method = "edm",
                              hidden_dim = 1024L,
                              epochs = 10000L,
                              batch_size = 4096L,
                              base_lr = 1e-4,
                              patience = 500L,
                              scheduler_factor = 0.9,
                              scheduler_patience = 50L,
                              device = "cpu",
                              seed = NULL,
                              checkpoint_path = NULL,
                              P_mean = -1.2,
                              P_std = 1.2,
                              sigma_data = 0.5,
                              verbose = TRUE) {
  method <- validate_method(method)
  validate_positive_integer(hidden_dim, "hidden_dim")
  validate_positive_integer(epochs, "epochs")
  validate_positive_integer(batch_size, "batch_size")
  validate_positive_number(base_lr, "base_lr")
  validate_positive_integer(patience, "patience")
  device <- resolve_device(device)
  set_seeds(seed)

  if (inherits(x, "torch_tensor")) {
    x_tensor <- x$to(dtype = torch::torch_float())$to(device = "cpu")
  } else {
    if (!is.matrix(x)) x <- as.matrix(x)
    x_tensor <- torch::torch_tensor(x, dtype = torch::torch_float())
  }
  n <- x_tensor$shape[1]
  d <- x_tensor$shape[2]
  if (n == 0L) stop("x has zero rows", call. = FALSE)
  effective_batch <- min(as.integer(batch_size), n)
  steps_per_epoch <- max(1L, n %/% effective_batch)

  denoiser <- mlp_diffusion(d_in = d, dim_t = as.integer(hidden_dim))
  model <- if (method == "flow") {
    flow_model(denoise_fn = denoiser)$to(device = device)
  } else {
    diffputer_model(
      denoise_fn = denoiser,
      P_mean = P_mean, P_std = P_std, sigma_data = sigma_data
    )$to(device = device)
  }
  optimizer <- torch::optim_adam(model$parameters, lr = base_lr, weight_decay = 0)
  scheduler <- torch::lr_reduce_on_plateau(
    optimizer, mode = "min",
    factor = scheduler_factor, patience = scheduler_patience
  )

  best_loss <- Inf
  best_state <- NULL
  epochs_without_improvement <- 0L
  losses <- numeric(0)

  if (verbose) cli::cli_progress_bar("Training diffusion model", total = epochs)
  on.exit(if (verbose) cli::cli_progress_done(), add = TRUE)

  for (epoch in seq_len(epochs)) {
    perm <- sample.int(n)
    epoch_loss <- 0
    seen <- 0L
    for (s in seq_len(steps_per_epoch)) {
      from <- (s - 1L) * effective_batch + 1L
      to <- min(s * effective_batch, n)
      idx <- perm[from:to]
      batch <- x_tensor[idx, , drop = FALSE]$to(device = device)
      loss <- model(batch)
      optimizer$zero_grad()
      loss$backward()
      optimizer$step()
      bs <- to - from + 1L
      epoch_loss <- epoch_loss + as.numeric(loss$detach()$cpu()) * bs
      seen <- seen + bs
    }
    epoch_loss <- epoch_loss / seen
    losses <- c(losses, epoch_loss)
    scheduler$step(epoch_loss)

    if (epoch_loss < best_loss) {
      best_loss <- epoch_loss
      epochs_without_improvement <- 0L
      best_state <- lapply(model$state_dict(), function(t) t$clone())
      if (!is.null(checkpoint_path)) {
        torch::torch_save(model$state_dict(), checkpoint_path)
      }
    } else {
      epochs_without_improvement <- epochs_without_improvement + 1L
    }
    if (verbose) {
      cli::cli_progress_update(
        status = sprintf("epoch %d  loss=%.4f  best=%.4f",
                         epoch, epoch_loss, best_loss)
      )
    }
    if (epochs_without_improvement >= as.integer(patience)) {
      if (verbose) cli::cli_alert_info(
        sprintf("Early stopping at epoch %d (no improvement for %d epochs)",
                epoch, patience)
      )
      break
    }
  }

  if (!is.null(best_state)) model$load_state_dict(best_state)

  # Both wrappers expose the velocity / denoiser through a uniform handle
  # so downstream samplers don't have to branch on the method.
  net_handle <- if (method == "flow") model else model$denoise_fn_D

  out <- list(
    model = model,
    net = net_handle,
    denoiser = if (method == "edm") model$denoise_fn_D else NULL,
    optimizer = optimizer,
    losses = losses,
    best_loss = best_loss,
    settings = list(
      method = method,
      hidden_dim = hidden_dim,
      epochs = epochs,
      batch_size = batch_size,
      base_lr = base_lr,
      patience = patience,
      device = device,
      checkpoint_path = checkpoint_path,
      P_mean = P_mean,
      P_std = P_std,
      sigma_data = sigma_data,
      d_in = d
    )
  )
  class(out) <- "trained_RDiffPuter"
  out
}

#' @export
print.trained_RDiffPuter <- function(x, ...) {
  cat("<trained_RDiffPuter>\n")
  cat(sprintf("  method: %s, d_in: %d, hidden_dim: %d, device: %s\n",
              x$settings$method, x$settings$d_in,
              x$settings$hidden_dim, x$settings$device))
  cat(sprintf("  epochs trained: %d, best loss: %.4f\n",
              length(x$losses), x$best_loss))
  invisible(x)
}
