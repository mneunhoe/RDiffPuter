#' @title Impute missing values with a trained diffusion model
#'
#' @description S3 method that runs the E-step of DiffPuter on a data frame
#'   containing `NA`s. Encodes the data via `transformer`, performs
#'   `num_trials` independent RePaint-style sampling passes, averages them, and
#'   decodes the result back into the original schema.
#'
#' @param object A `trained_RDiffPuter` object as returned by
#'   [diffputer_trainer()] or [diffputer_em()].
#' @param data A data frame whose schema matches the one used to fit
#'   `transformer`. May contain `NA`s in any subset of columns.
#' @param transformer A fitted [data_transformer] (R6 instance).
#' @param num_trials Number of independent reconstruction trials whose mean is
#'   used as the imputation. Higher values reduce sampling variance.
#' @param num_steps Number of outer denoising steps per trial.
#' @param num_resamplings Inner RePaint iterations per outer step. For
#'   flow-trained models with `resample_strategy = "single"` this is ignored.
#' @param resample_strategy For flow-trained models: `"repaint"` (default,
#'   inner re-flow loop) or `"single"` (one Euler step per outer step). Has
#'   no effect when the model was trained with `method = "edm"`.
#' @param device Device to run sampling on. Defaults to whatever the model was
#'   trained on.
#' @param ... Currently unused.
#'
#' @return A data frame with the same column names and types as `data`, with
#'   `NA` entries replaced by sampled imputations and observed entries left
#'   unchanged.
#' @export
impute <- function(object, data, transformer, ...) {
  UseMethod("impute")
}

#' @rdname impute
#' @export
impute.trained_RDiffPuter <- function(object,
                                      data,
                                      transformer,
                                      num_trials = NULL,
                                      num_steps = NULL,
                                      num_resamplings = NULL,
                                      resample_strategy = "repaint",
                                      device = NULL,
                                      ...) {
  if (!inherits(transformer, "data_transformer")) {
    stop("transformer must be a fitted data_transformer", call. = FALSE)
  }
  method <- object$settings$method %||% "edm"
  resample_strategy <- validate_resample_strategy(resample_strategy)
  defaults <- .impute_defaults(method)
  num_trials <- num_trials %||% defaults$num_trials
  num_steps <- num_steps %||% defaults$num_steps
  num_resamplings <- num_resamplings %||% defaults$num_resamplings
  validate_positive_integer(num_trials, "num_trials")
  validate_positive_integer(num_steps, "num_steps")
  validate_positive_integer(num_resamplings, "num_resamplings")
  if (is.null(device)) device <- object$settings$device
  device <- resolve_device(device)

  encoded <- transformer$transform(data)
  x_obs <- encoded$x
  mask <- encoded$mask
  if (!any(mask)) return(as.data.frame(data, stringsAsFactors = FALSE))

  x_t <- torch::torch_tensor(x_obs, dtype = torch::torch_float())$to(device = device)
  mask_t <- torch::torch_tensor(mask + 0, dtype = torch::torch_float())$to(device = device)

  net <- (object$net %||% object$denoiser)$to(device = device)
  net$eval()
  reconstructions <- torch::torch_zeros_like(x_t)
  for (trial in seq_len(as.integer(num_trials))) {
    rec <- if (method == "flow") {
      flow_impute_mask(net, x_t, mask_t,
                       num_steps = as.integer(num_steps),
                       num_resamplings = as.integer(num_resamplings),
                       resample_strategy = resample_strategy,
                       device = device)
    } else {
      impute_mask(net, x_t, mask_t,
                  num_steps = as.integer(num_steps),
                  num_resamplings = as.integer(num_resamplings),
                  device = device)
    }
    reconstructions <- reconstructions + rec
  }
  reconstructions <- reconstructions / as.integer(num_trials)

  filled <- reconstructions * mask_t + x_t * (1 - mask_t)
  filled_mat <- as.matrix(torch::as_array(filled$cpu()))

  imputed_df <- transformer$inverse_transform(filled_mat)
  if (!is.data.frame(data)) data <- as.data.frame(data, stringsAsFactors = FALSE)

  for (col in names(imputed_df)) {
    orig <- data[[col]]
    new_col <- imputed_df[[col]]
    na_idx <- is.na(orig)
    if (is.factor(orig)) {
      out <- as.character(orig)
      out[na_idx] <- as.character(new_col)[na_idx]
      imputed_df[[col]] <- factor(out, levels = levels(orig))
    } else if (is.character(orig)) {
      out <- orig
      out[na_idx] <- as.character(new_col)[na_idx]
      imputed_df[[col]] <- out
    } else if (is.logical(orig)) {
      out <- orig
      out[na_idx] <- as.logical(as.numeric(new_col)[na_idx] >= 0.5)
      imputed_df[[col]] <- out
    } else {
      out <- orig
      out[na_idx] <- as.numeric(new_col)[na_idx]
      imputed_df[[col]] <- out
    }
  }
  imputed_df
}
