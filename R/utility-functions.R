#' @title Resolve a torch device string with fallback
#'
#' @description Validate a user-supplied device string and silently fall back to
#'   `"cpu"` (with a warning) when CUDA or MPS is requested but unavailable. This
#'   mirrors the pattern used by the `RGAN` package so that calls succeed on any
#'   machine.
#'
#' @param device One of `"cpu"`, `"cuda"`, or `"mps"`.
#'
#' @return The resolved device string. Always one of `"cpu"`, `"cuda"`, `"mps"`.
#' @export
#' @examples
#' \dontrun{
#' resolve_device("cuda")  # -> "cuda" if available, else "cpu" with warning
#' }
resolve_device <- function(device = "cpu") {
  device <- match.arg(device, c("cpu", "cuda", "mps"))
  if (device == "cuda" && !torch::cuda_is_available()) {
    warning("CUDA device requested but not available. Falling back to CPU.")
    device <- "cpu"
  }
  if (device == "mps" && !torch::backends_mps_is_available()) {
    warning("MPS device requested but not available. Falling back to CPU.")
    device <- "cpu"
  }
  device
}

#' @title Set R and torch random seeds together
#'
#' @description Seeds R's RNG and torch's RNG in a single call so that runs are
#'   reproducible across the package's data preparation and model training
#'   stages. Note that CUDA reductions are not bit-deterministic by default;
#'   numerical results may still differ slightly between GPU runs.
#'
#' @param seed Integer seed. If `NULL`, no seeds are set.
#'
#' @return The seed (invisibly), or `NULL` if no seed was set.
#' @export
set_seeds <- function(seed = NULL) {
  if (is.null(seed)) return(invisible(NULL))
  set.seed(seed)
  torch::torch_manual_seed(seed)
  invisible(seed)
}

#' @title Column-wise mean and standard deviation under a missingness mask
#'
#' @description Computes per-column mean and standard deviation while ignoring
#'   missing entries, mirroring `mean_std()` from the reference Python
#'   implementation. The mask uses `TRUE` to mark missing entries (matching
#'   `is.na()` semantics in R).
#'
#' @param data A numeric matrix.
#' @param mask A logical matrix of the same shape; `TRUE` = missing.
#'
#' @return A list with elements `mean` and `sd` (both length `ncol(data)`).
#'   Columns with zero observed values produce mean `0` and sd `1`.
#' @export
masked_mean_std <- function(data, mask) {
  if (!is.matrix(data)) data <- as.matrix(data)
  if (!is.matrix(mask)) mask <- as.matrix(mask)
  stopifnot(identical(dim(data), dim(mask)))
  observed <- !mask
  obs_count <- colSums(observed)
  safe_count <- ifelse(obs_count == 0, 1, obs_count)
  data_zero <- data
  data_zero[mask] <- 0
  col_mean <- colSums(data_zero) / safe_count
  centered <- sweep(data_zero, 2, col_mean, "-")
  centered[mask] <- 0
  col_var <- colSums(centered^2) / safe_count
  col_sd <- sqrt(col_var)
  col_sd[col_sd == 0 | obs_count == 0] <- 1
  list(mean = col_mean, sd = col_sd)
}

#' @title Detect categorical columns in a data frame
#'
#' @description Returns the names of columns that are factors, characters, or
#'   logicals, treating those as categorical. Numeric and integer columns are
#'   considered continuous. Used by `data_transformer$fit()` when
#'   `discrete_columns = NULL`.
#'
#' @param data A data frame.
#'
#' @return A character vector of column names.
#' @keywords internal
detect_discrete_columns <- function(data) {
  if (!is.data.frame(data)) {
    stop("data must be a data.frame to auto-detect discrete columns")
  }
  is_discrete <- vapply(data, function(col) {
    is.factor(col) || is.character(col) || is.logical(col)
  }, logical(1))
  names(data)[is_discrete]
}

#' @title Default sampling budgets per generative method
#'
#' @description Internal helper. Returns the recommended `num_trials`,
#'   `num_steps`, and `num_resamplings` for the EDM and flow imputers when the
#'   user does not override them. Flow defaults are roughly five times
#'   cheaper, reflecting the straighter ODE trajectories.
#'
#' @param method `"edm"` or `"flow"`.
#' @return A named list.
#' @keywords internal
.impute_defaults <- function(method) {
  if (identical(method, "flow")) {
    list(num_trials = 10L, num_steps = 10L, num_resamplings = 5L)
  } else {
    list(num_trials = 20L, num_steps = 50L, num_resamplings = 20L)
  }
}

#' @title NULL-coalescing operator
#'
#' @description Returns `lhs` if it is non-`NULL`, otherwise `rhs`. Lifted from
#'   common R idioms to keep argument-defaulting code readable.
#'
#' @param lhs Left-hand side.
#' @param rhs Right-hand side.
#' @return One of the operands.
#' @keywords internal
#' @name grapes-or-or-grapes
`%||%` <- function(lhs, rhs) if (is.null(lhs)) rhs else lhs

#' @title Replicate a column-wise mask to match a wider encoded dimension
#'
#' @description Given a per-original-column mask and a vector giving the encoded
#'   width of each original column (e.g. number of bits for binary encoding or
#'   number of levels for one-hot), replicate each mask column across its
#'   encoded width. This is the R analogue of `extend_mask()` from
#'   `DiffPuter/dataset.py`.
#'
#' @param mask A logical matrix `n x p` where `TRUE` marks missing.
#' @param widths Integer vector of length `p`; the encoded width of each
#'   original column.
#'
#' @return A logical matrix with `n` rows and `sum(widths)` columns.
#' @keywords internal
extend_mask <- function(mask, widths) {
  if (!is.matrix(mask)) mask <- as.matrix(mask)
  if (length(widths) != ncol(mask)) {
    stop("length(widths) must equal ncol(mask)")
  }
  if (ncol(mask) == 0) {
    return(matrix(FALSE, nrow = nrow(mask), ncol = 0))
  }
  out <- matrix(FALSE, nrow = nrow(mask), ncol = sum(widths))
  start <- 1L
  for (j in seq_along(widths)) {
    w <- widths[j]
    if (w > 0) {
      out[, start:(start + w - 1L)] <- mask[, j]
      start <- start + w
    }
  }
  out
}
