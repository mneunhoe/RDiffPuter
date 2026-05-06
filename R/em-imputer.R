#' @title Diffusion-based EM imputation for arbitrary tabular data
#'
#' @description High-level entry point. Given a data frame with `NA` entries,
#'   fits a diffusion model in an Expectation-Maximisation loop and returns the
#'   imputed data frame in the original schema. Continuous and categorical
#'   columns are handled automatically. Runs on CPU, CUDA, or Apple Silicon
#'   (MPS) via the `torch` package.
#'
#' @details
#' Each iteration alternates an **M-step** (training a diffusion denoiser on
#' the current best estimate of the data, with missing entries filled by the
#' previous iteration's reconstruction) with an **E-step** (drawing
#' `num_trials` RePaint-style samples and averaging them). Iteration 0 fills
#' missing entries with `0`, matching the reference implementation. Use the
#' lower-level [diffputer_trainer()] / [impute()] pair for finer control.
#'
#' @param data A data frame whose missing entries are encoded as `NA`.
#'   Continuous columns must be numeric; categorical columns are auto-detected
#'   from `factor`, `character`, or `logical` types.
#' @param max_iter Number of EM iterations.
#' @param epochs Maximum training epochs per M-step.
#' @param patience Early-stopping patience (epochs without loss improvement).
#' @param hidden_dim Hidden width of the MLP denoiser.
#' @param batch_size Mini-batch size during training.
#' @param base_lr Adam learning rate.
#' @param num_steps Outer denoising steps per E-step trial.
#' @param num_trials Independent reconstruction trials averaged in each E-step.
#' @param num_resamplings Inner RePaint iterations per outer step.
#' @param discrete_columns Optional character vector of column names to treat as
#'   discrete; otherwise auto-detected.
#' @param encoding Categorical encoding, `"binary"` (default, DiffPuter-faithful)
#'   or `"onehot"`.
#' @param device One of `"cpu"`, `"cuda"`, `"mps"`. Falls back to `"cpu"` with
#'   a warning if unavailable.
#' @param seed Optional integer RNG seed.
#' @param checkpoint_path Optional path; if set, the best per-iteration model
#'   state is saved here.
#' @param validation_data Optional held-out data frame in the same schema.
#'   When supplied, MAE / RMSE on its missing entries are tracked per
#'   iteration. (Held-out NAs in `validation_data` mark cells whose ground
#'   truth in `data` is observed.)
#' @param verbose If `TRUE`, prints a per-iteration summary.
#'
#' @return A list with class `RDiffPuter_em` containing:
#' \describe{
#'   \item{imputed}{Final imputed data frame.}
#'   \item{model}{Last-iteration `trained_RDiffPuter`.}
#'   \item{transformer}{The fitted `data_transformer`.}
#'   \item{history}{Per-iteration list with `loss`, `mae`, `rmse`.}
#' }
#'
#' @references Zhang, H. et al. "DiffPuter: Empowering Diffusion Models for
#'   Missing Data Imputation." ICLR 2025.
#' @export
#' @examples
#' \dontrun{
#'   set.seed(1)
#'   data <- data.frame(a = rnorm(200), b = runif(200, 1, 5),
#'                      c = factor(sample(letters[1:3], 200, replace = TRUE)))
#'   data$a[sample.int(200, 30)] <- NA
#'   data$c[sample.int(200, 20)] <- NA
#'   result <- diffputer_em(data, max_iter = 2, epochs = 100,
#'                          hidden_dim = 64, num_trials = 4,
#'                          num_steps = 10, num_resamplings = 5,
#'                          batch_size = 64, patience = 20)
#'   head(result$imputed)
#' }
diffputer_em <- function(data,
                         max_iter = 10L,
                         epochs = 10000L,
                         patience = 500L,
                         hidden_dim = 1024L,
                         batch_size = 4096L,
                         base_lr = 1e-4,
                         num_steps = 50L,
                         num_trials = 20L,
                         num_resamplings = 20L,
                         discrete_columns = NULL,
                         encoding = "binary",
                         device = "cpu",
                         seed = NULL,
                         checkpoint_path = NULL,
                         validation_data = NULL,
                         verbose = TRUE) {
  validate_data_frame_with_na(data, "data", require_na = TRUE)
  validate_positive_integer(max_iter, "max_iter")
  validate_positive_integer(num_trials, "num_trials")
  validate_positive_integer(num_steps, "num_steps")
  validate_positive_integer(num_resamplings, "num_resamplings")
  encoding <- validate_encoding(encoding)
  device <- resolve_device(device)
  set_seeds(seed)

  transformer <- data_transformer$new()
  transformer$fit(data, discrete_columns = discrete_columns, encoding = encoding)
  encoded <- transformer$transform(data)
  x_clean <- encoded$x
  mask <- encoded$mask
  d <- ncol(x_clean)
  current_x <- x_clean
  current_x[mask] <- 0  # iteration-0 initialisation

  history <- list(loss = numeric(0), mae = numeric(0), rmse = numeric(0))
  trained <- NULL
  imputed_df <- NULL

  for (it in seq_len(as.integer(max_iter))) {
    if (verbose) cli::cli_h2(sprintf("EM iteration %d / %d", it, max_iter))

    iter_ckpt <- if (!is.null(checkpoint_path)) {
      sprintf("%s-iter%02d.pt", checkpoint_path, it)
    } else NULL

    trained <- diffputer_trainer(
      x = current_x,
      hidden_dim = hidden_dim,
      epochs = epochs,
      batch_size = batch_size,
      base_lr = base_lr,
      patience = patience,
      device = device,
      seed = NULL,
      checkpoint_path = iter_ckpt,
      verbose = verbose
    )
    history$loss <- c(history$loss, trained$best_loss)

    imputed_df <- impute.trained_RDiffPuter(
      trained, data, transformer,
      num_trials = num_trials,
      num_steps = num_steps,
      num_resamplings = num_resamplings,
      device = device
    )

    encoded_iter <- transformer$transform(imputed_df)
    current_x <- encoded_iter$x
    current_x[!mask] <- x_clean[!mask]

    if (!is.null(validation_data)) {
      err <- .compute_validation_error(
        imputed_df, data, validation_data, transformer
      )
      history$mae <- c(history$mae, err$mae)
      history$rmse <- c(history$rmse, err$rmse)
      if (verbose) cli::cli_alert_info(sprintf(
        "iter %d: loss=%.4f  MAE=%.4f  RMSE=%.4f",
        it, trained$best_loss, err$mae, err$rmse
      ))
    } else if (verbose) {
      cli::cli_alert_info(sprintf("iter %d: loss=%.4f", it, trained$best_loss))
    }
  }

  out <- list(
    imputed = imputed_df,
    model = trained,
    transformer = transformer,
    history = history,
    settings = list(
      max_iter = max_iter, epochs = epochs, patience = patience,
      hidden_dim = hidden_dim, batch_size = batch_size, base_lr = base_lr,
      num_steps = num_steps, num_trials = num_trials,
      num_resamplings = num_resamplings, encoding = encoding, device = device
    )
  )
  class(out) <- "RDiffPuter_em"
  out
}

# Compute MAE / RMSE on validation_data: cells that are NA in data but observed
# in validation_data are scored.
.compute_validation_error <- function(imputed_df, data_with_na,
                                      validation_data, transformer) {
  ground_truth <- transformer$transform(validation_data)$x
  imputed_x <- transformer$transform(imputed_df)$x
  mask <- transformer$transform(data_with_na)$mask
  diff <- (imputed_x - ground_truth)[mask]
  if (length(diff) == 0L) return(list(mae = NA_real_, rmse = NA_real_))
  list(mae = mean(abs(diff)), rmse = sqrt(mean(diff^2)))
}

#' @export
print.RDiffPuter_em <- function(x, ...) {
  cat("<RDiffPuter_em>\n")
  cat(sprintf("  iterations: %d, encoded_dim: %d\n",
              length(x$history$loss), x$transformer$encoded_dim))
  cat(sprintf("  final training loss: %.4f\n",
              utils::tail(x$history$loss, 1L)))
  if (length(x$history$mae) > 0L) {
    cat(sprintf("  final validation MAE: %.4f, RMSE: %.4f\n",
                utils::tail(x$history$mae, 1L),
                utils::tail(x$history$rmse, 1L)))
  }
  invisible(x)
}
