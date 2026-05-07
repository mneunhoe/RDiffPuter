#' @title Shared argument validators for RDiffPuter entry points
#'
#' @description Internal helpers that throw informative errors for invalid user
#'   inputs. Each function is small and self-explanatory; the goal is to keep
#'   `diffputer_trainer()`, `diffputer_em()`, and `impute()` free of repetitive
#'   `if/stop` boilerplate.
#'
#' @name input-validation
#' @keywords internal
NULL

#' @rdname input-validation
validate_positive_integer <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || x <= 0 || x != as.integer(x)) {
    stop(sprintf("%s must be a positive integer", name), call. = FALSE)
  }
  invisible(as.integer(x))
}

#' @rdname input-validation
validate_positive_number <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || x <= 0) {
    stop(sprintf("%s must be a positive number", name), call. = FALSE)
  }
  invisible(x)
}

#' @rdname input-validation
validate_data_frame_with_na <- function(data, name = "data", require_na = FALSE) {
  if (!is.data.frame(data) && !is.matrix(data)) {
    stop(sprintf("%s must be a data.frame or matrix", name), call. = FALSE)
  }
  if (nrow(data) == 0L) {
    stop(sprintf("%s cannot be empty (zero rows)", name), call. = FALSE)
  }
  if (ncol(data) == 0L) {
    stop(sprintf("%s must have at least one column", name), call. = FALSE)
  }
  if (require_na && !any(is.na(data))) {
    warning(sprintf("%s contains no NA values; nothing will be imputed", name))
  }
  invisible(data)
}

#' @rdname input-validation
validate_encoding <- function(encoding) {
  encoding <- match.arg(encoding, c("binary", "onehot"))
  encoding
}

#' @rdname input-validation
validate_method <- function(method) {
  method <- match.arg(method, c("edm", "flow"))
  method
}

#' @rdname input-validation
validate_resample_strategy <- function(strategy) {
  strategy <- match.arg(strategy, c("repaint", "single"))
  strategy
}
