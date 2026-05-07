#' @title Multiple imputation with a trained DiffPuter model
#'
#' @description Runs the E-step `m` times with fresh random noise and returns
#'   `m` independently completed data sets. By default each draw uses
#'   `num_trials = 1`, so between-draw variability reflects the model's
#'   posterior uncertainty in the spirit of Rubin's multiple imputation. The
#'   returned object is a list of data frames with class `"mi_imputations"`;
#'   it can be passed straight to `mice::pool()` after fitting an analysis
#'   model on each element.
#'
#' @details
#' For pooled inference with the `mice` ecosystem, do something like:
#' \preformatted{
#'   mi <- multi_impute(em_result, data, m = 10)
#'   fits <- lapply(mi, function(d) lm(y ~ x1 + x2, data = d))
#'   summary(mice::pool(fits))
#' }
#'
#' @param object A `trained_RDiffPuter` (from [diffputer_trainer()]) or an
#'   `RDiffPuter_em` (from [diffputer_em()]).
#' @param data The data frame whose `NA` entries should be imputed. Required
#'   for both methods (the `RDiffPuter_em` object does not retain the original
#'   `NA`-containing input).
#' @param transformer A fitted [data_transformer]. Optional when `object` is
#'   an `RDiffPuter_em` (the transformer is taken from the object).
#' @param m Number of completed data sets to produce.
#' @param num_trials Within-draw averaging. `1` (default) gives Rubin-style
#'   posterior draws; values above 1 reduce within-draw variance but also
#'   shrink between-draw variance, which can hurt pooled standard errors.
#' @param num_steps,num_resamplings,resample_strategy Forwarded to [impute()].
#'   Defaults follow the model's `method`.
#' @param device Device on which to run sampling.
#' @param seed Optional integer seed; controls the sequence of `m` draws.
#' @param ... Currently unused.
#'
#' @return A list of `m` data frames, with class `mi_imputations` and
#'   attributes `m` and `method`.
#' @export
#' @examples
#' \dontrun{
#'   set.seed(1)
#'   data <- data.frame(x = rnorm(200), y = rnorm(200))
#'   data$x[sample.int(200, 30)] <- NA
#'   em <- diffputer_em(data, max_iter = 2, epochs = 100, hidden_dim = 64,
#'                      batch_size = 32, patience = 30)
#'   mi <- multi_impute(em, data, m = 5)
#'   fits <- lapply(mi, function(d) lm(y ~ x, data = d))
#'   summary(mice::pool(fits))
#' }
multi_impute <- function(object, data, transformer = NULL, m = 5L, ...) {
  UseMethod("multi_impute")
}

#' @rdname multi_impute
#' @export
multi_impute.trained_RDiffPuter <- function(object,
                                            data,
                                            transformer,
                                            m = 5L,
                                            num_trials = 1L,
                                            num_steps = NULL,
                                            num_resamplings = NULL,
                                            resample_strategy = "repaint",
                                            device = NULL,
                                            seed = NULL,
                                            ...) {
  validate_positive_integer(m, "m")
  if (!inherits(transformer, "data_transformer")) {
    stop("transformer must be a fitted data_transformer", call. = FALSE)
  }
  set_seeds(seed)
  m_int <- as.integer(m)
  out <- vector("list", m_int)
  for (k in seq_len(m_int)) {
    out[[k]] <- impute.trained_RDiffPuter(
      object, data, transformer,
      num_trials = num_trials,
      num_steps = num_steps,
      num_resamplings = num_resamplings,
      resample_strategy = resample_strategy,
      device = device
    )
  }
  attr(out, "m") <- m_int
  attr(out, "method") <- object$settings$method %||% "edm"
  attr(out, "n_rows") <- nrow(data)
  attr(out, "n_cols") <- ncol(data)
  class(out) <- c("mi_imputations", "list")
  out
}

#' @rdname multi_impute
#' @export
multi_impute.RDiffPuter_em <- function(object,
                                       data,
                                       transformer = NULL,
                                       m = 5L,
                                       ...) {
  transformer <- transformer %||% object$transformer
  multi_impute.trained_RDiffPuter(object$model, data, transformer,
                                  m = m, ...)
}

#' @export
print.mi_imputations <- function(x, ...) {
  cat(sprintf(
    "<mi_imputations: %d completed data sets, %d x %d (method = %s)>\n",
    attr(x, "m"), attr(x, "n_rows"), attr(x, "n_cols"),
    attr(x, "method")
  ))
  invisible(x)
}

#' @title Train a DiffPuter model and produce multiple imputations
#'
#' @description One-call wrapper that runs the EM loop via [diffputer_em()]
#'   and then draws `m` completed data sets with [multi_impute()]. Returns a
#'   single object containing the trained model, the transformer, the EM
#'   training history, and the list of `m` completions.
#'
#' @inheritParams diffputer_em
#' @param m Number of completed data sets to produce after training.
#' @param mi_num_trials Within-draw averaging for the MI step. `1` (default)
#'   gives Rubin-style posterior draws.
#' @param mi_num_steps,mi_num_resamplings Sampling budgets used at MI time.
#'   When `NULL`, defaults follow the model's `method`.
#'
#' @return A list with class `RDiffPuter_em_mi`:
#' \describe{
#'   \item{imputations}{An `mi_imputations` list of length `m`.}
#'   \item{model}{The trained `trained_RDiffPuter`.}
#'   \item{transformer}{The fitted `data_transformer`.}
#'   \item{history}{Per-iteration loss / validation metrics.}
#'   \item{settings}{The combined training and MI settings.}
#' }
#' @export
#' @examples
#' \dontrun{
#'   set.seed(1)
#'   data <- data.frame(x = rnorm(200), y = rnorm(200),
#'                      g = factor(sample(letters[1:3], 200, replace = TRUE)))
#'   data$x[1:30] <- NA
#'   data$g[40:55] <- NA
#'   res <- diffputer_em_mi(data, m = 10, max_iter = 2, epochs = 200,
#'                          hidden_dim = 64, batch_size = 32, patience = 50)
#'   length(res$imputations)   # 10
#'   head(res$imputations[[1]])
#' }
diffputer_em_mi <- function(data,
                            m = 5L,
                            method = "edm",
                            max_iter = 10L,
                            epochs = 10000L,
                            patience = 500L,
                            hidden_dim = 1024L,
                            batch_size = 4096L,
                            base_lr = 1e-4,
                            num_steps = NULL,
                            num_trials = NULL,
                            num_resamplings = NULL,
                            resample_strategy = "repaint",
                            discrete_columns = NULL,
                            encoding = "binary",
                            device = "cpu",
                            seed = NULL,
                            checkpoint_path = NULL,
                            validation_data = NULL,
                            verbose = TRUE,
                            mi_num_trials = 1L,
                            mi_num_steps = NULL,
                            mi_num_resamplings = NULL) {
  validate_positive_integer(m, "m")
  em <- diffputer_em(
    data, method = method,
    max_iter = max_iter, epochs = epochs, patience = patience,
    hidden_dim = hidden_dim, batch_size = batch_size, base_lr = base_lr,
    num_steps = num_steps, num_trials = num_trials,
    num_resamplings = num_resamplings,
    resample_strategy = resample_strategy,
    discrete_columns = discrete_columns, encoding = encoding,
    device = device, seed = seed,
    checkpoint_path = checkpoint_path,
    validation_data = validation_data, verbose = verbose
  )
  imps <- multi_impute(
    em, data = data, m = m,
    num_trials = mi_num_trials,
    num_steps = mi_num_steps,
    num_resamplings = mi_num_resamplings,
    resample_strategy = resample_strategy,
    device = device,
    seed = NULL
  )
  out <- list(
    imputations = imps,
    model = em$model,
    transformer = em$transformer,
    history = em$history,
    settings = c(em$settings, list(
      m = as.integer(m),
      mi_num_trials = mi_num_trials,
      mi_num_steps = mi_num_steps,
      mi_num_resamplings = mi_num_resamplings
    ))
  )
  class(out) <- "RDiffPuter_em_mi"
  out
}

#' @export
print.RDiffPuter_em_mi <- function(x, ...) {
  cat("<RDiffPuter_em_mi>\n")
  cat(sprintf("  method: %s, iterations: %d, encoded_dim: %d\n",
              x$settings$method %||% "edm",
              length(x$history$loss), x$transformer$encoded_dim))
  cat(sprintf("  m = %d completed data sets\n", x$settings$m))
  cat(sprintf("  final training loss: %.4f\n",
              utils::tail(x$history$loss, 1L)))
  invisible(x)
}
