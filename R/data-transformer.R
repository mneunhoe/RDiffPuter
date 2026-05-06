#' @title data_transformer R6 class
#'
#' @description Preprocesses an arbitrary tabular data set for the diffusion
#'   imputer. Handles continuous (numeric) and discrete (factor / character /
#'   logical) columns, builds the missingness mask from `NA`s, and inverts the
#'   transform after imputation. Mirrors the `data_transformer` design from the
#'   `RGAN` package.
#'
#' @details
#' The encoded representation is `(numeric_columns, categorical_columns)` along
#' the column axis. Numeric columns are standardised with the formula
#' `(x - mean) / sd / 2` (matching the reference DiffPuter implementation), with
#' mean and sd computed ignoring `NA`. Categorical columns are encoded according
#' to `encoding`:
#'
#' * `"binary"` (default): a column with `K` levels becomes
#'   `ceil(log2(max(K, 2)))` 0/1 columns whose bit pattern indexes the level.
#'   Mirrors `DiffPuter/dataset.py`.
#' * `"onehot"`: each level becomes its own indicator column.
#'
#' The missingness mask is replicated bit-wise (or level-wise) so that a missing
#' entry blocks all bits / levels of its original column.
#'
#' Object methods:
#' * `$fit(data, discrete_columns = NULL, encoding = "binary")` learns the
#'   transformation parameters.
#' * `$transform(data)` returns a list with components `x` (encoded numeric
#'   matrix), `mask` (encoded logical matrix; `TRUE` = missing), and
#'   `encoded_widths`.
#' * `$inverse_transform(x_encoded)` returns a data frame in the original
#'   schema, decoding bit-strings or one-hot blocks back to factor levels and
#'   un-standardising numeric columns.
#' * `$build_mask(data)` returns the encoded missingness mask alone.
#'
#' @return An R6 generator object.
#' @importFrom R6 R6Class
#' @export
data_transformer <- R6::R6Class(
  classname = "data_transformer",
  public = list(
    #' @field column_names Original column names in the input order.
    column_names = NULL,
    #' @field column_types Per-column type label, `"numeric"` or `"categorical"`.
    column_types = NULL,
    #' @field num_idx Integer indices of numeric columns in the original data.
    num_idx = NULL,
    #' @field cat_idx Integer indices of categorical columns in the original
    #'   data.
    cat_idx = NULL,
    #' @field num_mean Per-numeric-column mean (NA-aware).
    num_mean = NULL,
    #' @field num_sd Per-numeric-column standard deviation (NA-aware).
    num_sd = NULL,
    #' @field cat_levels List of character vectors; one entry per categorical
    #'   column giving the level vocabulary in encoding order.
    cat_levels = NULL,
    #' @field cat_widths Integer vector; encoded width of each categorical
    #'   column under the chosen encoding.
    cat_widths = NULL,
    #' @field encoding `"binary"` or `"onehot"`.
    encoding = NULL,
    #' @field encoded_dim Total number of columns in the encoded matrix.
    encoded_dim = NULL,
    #' @field fitted Has `$fit()` been called?
    fitted = FALSE,

    #' @description Fit the transformer to a data frame.
    #' @param data A data frame (preferred) or matrix.
    #' @param discrete_columns Character vector of column names to treat as
    #'   discrete. If `NULL`, factor / character / logical columns are detected
    #'   automatically.
    #' @param encoding `"binary"` or `"onehot"`.
    fit = function(data, discrete_columns = NULL, encoding = "binary") {
      encoding <- validate_encoding(encoding)
      data <- private$.coerce_input(data)
      self$column_names <- names(data)
      if (is.null(discrete_columns)) {
        discrete_columns <- detect_discrete_columns(data)
      } else {
        unknown <- setdiff(discrete_columns, self$column_names)
        if (length(unknown) > 0L) {
          stop(sprintf(
            "discrete_columns names not in data: %s",
            paste(unknown, collapse = ", ")
          ), call. = FALSE)
        }
      }
      types <- ifelse(self$column_names %in% discrete_columns,
                      "categorical", "numeric")
      self$column_types <- types
      self$num_idx <- which(types == "numeric")
      self$cat_idx <- which(types == "categorical")
      self$encoding <- encoding

      if (length(self$num_idx) > 0L) {
        num_mat <- vapply(self$num_idx, function(j) {
          col <- data[[j]]
          if (is.logical(col)) col <- as.numeric(col)
          if (!is.numeric(col)) {
            stop(sprintf(
              "Column '%s' is not numeric but was not declared discrete",
              self$column_names[j]
            ), call. = FALSE)
          }
          as.numeric(col)
        }, numeric(nrow(data)))
        if (!is.matrix(num_mat)) num_mat <- matrix(num_mat, nrow = nrow(data))
        ms <- masked_mean_std(num_mat, is.na(num_mat))
        self$num_mean <- ms$mean
        self$num_sd <- ms$sd
      } else {
        self$num_mean <- numeric(0)
        self$num_sd <- numeric(0)
      }

      if (length(self$cat_idx) > 0L) {
        self$cat_levels <- lapply(self$cat_idx, function(j) {
          col <- data[[j]]
          if (is.factor(col)) {
            levs <- levels(col)
          } else {
            levs <- unique(stats::na.omit(as.character(col)))
          }
          if (length(levs) < 2L) {
            stop(sprintf(
              "Column '%s' has fewer than two observed levels; remove it before fitting",
              self$column_names[j]
            ), call. = FALSE)
          }
          as.character(levs)
        })
        self$cat_widths <- vapply(self$cat_levels, function(levs) {
          k <- length(levs)
          if (encoding == "binary") {
            max(1L, as.integer(ceiling(log2(k))))
          } else {
            as.integer(k)
          }
        }, integer(1))
      } else {
        self$cat_levels <- list()
        self$cat_widths <- integer(0)
      }

      self$encoded_dim <- length(self$num_idx) + sum(self$cat_widths)
      self$fitted <- TRUE
      invisible(self)
    },

    #' @description Encode a data frame to the numeric matrix form expected by
    #'   the diffusion model.
    #' @param data A data frame with the same schema used at `$fit()`.
    transform = function(data) {
      private$.require_fitted()
      data <- private$.coerce_input(data)
      private$.check_schema(data)
      n <- nrow(data)

      num_part <- matrix(0, nrow = n, ncol = length(self$num_idx))
      num_mask <- matrix(FALSE, nrow = n, ncol = length(self$num_idx))
      if (length(self$num_idx) > 0L) {
        for (k in seq_along(self$num_idx)) {
          j <- self$num_idx[k]
          col <- data[[j]]
          if (is.logical(col)) col <- as.numeric(col)
          col_num <- as.numeric(col)
          na_idx <- is.na(col_num)
          col_num[na_idx] <- 0
          standardised <- (col_num - self$num_mean[k]) / self$num_sd[k] / 2
          standardised[na_idx] <- 0
          num_part[, k] <- standardised
          num_mask[, k] <- na_idx
        }
      }

      cat_blocks <- vector("list", length(self$cat_idx))
      cat_mask_cols <- matrix(FALSE, nrow = n, ncol = length(self$cat_idx))
      if (length(self$cat_idx) > 0L) {
        for (k in seq_along(self$cat_idx)) {
          j <- self$cat_idx[k]
          col <- as.character(data[[j]])
          na_idx <- is.na(col) | !(col %in% self$cat_levels[[k]])
          cat_mask_cols[, k] <- na_idx
          col[na_idx] <- self$cat_levels[[k]][1L]
          idx <- match(col, self$cat_levels[[k]])
          width <- self$cat_widths[k]
          if (self$encoding == "binary") {
            block <- private$.indices_to_bits(idx - 1L, width)
          } else {
            block <- matrix(0, nrow = n, ncol = width)
            block[cbind(seq_len(n), idx)] <- 1
          }
          block[na_idx, ] <- 0
          cat_blocks[[k]] <- block
        }
      }

      if (length(cat_blocks) > 0L) {
        cat_part <- do.call(cbind, cat_blocks)
        cat_mask <- extend_mask(cat_mask_cols, self$cat_widths)
      } else {
        cat_part <- matrix(0, nrow = n, ncol = 0)
        cat_mask <- matrix(FALSE, nrow = n, ncol = 0)
      }

      x <- cbind(num_part, cat_part)
      mask <- cbind(num_mask, cat_mask)
      list(
        x = x,
        mask = mask,
        encoded_widths = c(rep(1L, length(self$num_idx)), self$cat_widths)
      )
    },

    #' @description Build the missingness mask only.
    #' @param data A data frame with the same schema used at `$fit()`.
    build_mask = function(data) {
      self$transform(data)$mask
    },

    #' @description Decode an encoded numeric matrix back into the original
    #'   schema. Numeric columns are un-standardised; categorical columns are
    #'   recovered by thresholding bits (binary) or `argmax` (one-hot).
    #' @param x_encoded An `n x encoded_dim` numeric matrix or torch tensor.
    inverse_transform = function(x_encoded) {
      private$.require_fitted()
      if (inherits(x_encoded, "torch_tensor")) {
        x_encoded <- as.matrix(torch::as_array(x_encoded$cpu()))
      }
      if (!is.matrix(x_encoded)) x_encoded <- as.matrix(x_encoded)
      if (ncol(x_encoded) != self$encoded_dim) {
        stop(sprintf(
          "x_encoded has %d cols, expected %d",
          ncol(x_encoded), self$encoded_dim
        ), call. = FALSE)
      }
      n <- nrow(x_encoded)
      out <- vector("list", length(self$column_names))
      names(out) <- self$column_names

      n_num <- length(self$num_idx)
      if (n_num > 0L) {
        for (k in seq_len(n_num)) {
          standardised <- x_encoded[, k]
          col <- standardised * 2 * self$num_sd[k] + self$num_mean[k]
          out[[self$num_idx[k]]] <- col
        }
      }

      if (length(self$cat_idx) > 0L) {
        offset <- n_num
        for (k in seq_along(self$cat_idx)) {
          width <- self$cat_widths[k]
          block <- x_encoded[, (offset + 1L):(offset + width), drop = FALSE]
          offset <- offset + width
          levs <- self$cat_levels[[k]]
          if (self$encoding == "binary") {
            bits <- (block > 0.5) + 0L
            idx <- private$.bits_to_indices(bits)
            idx <- pmin(pmax(idx, 0L), length(levs) - 1L)
          } else {
            idx <- max.col(block, ties.method = "first") - 1L
          }
          out[[self$cat_idx[k]]] <- factor(levs[idx + 1L], levels = levs)
        }
      }

      as.data.frame(out, stringsAsFactors = FALSE)
    }
  ),
  private = list(
    .coerce_input = function(data) {
      if (is.matrix(data)) data <- as.data.frame(data)
      if (!is.data.frame(data)) {
        stop("data must be a data.frame or matrix", call. = FALSE)
      }
      data
    },
    .require_fitted = function() {
      if (!isTRUE(self$fitted)) {
        stop("data_transformer has not been fitted; call $fit() first", call. = FALSE)
      }
    },
    .check_schema = function(data) {
      if (!identical(names(data), self$column_names)) {
        stop("data column names do not match those used at $fit()", call. = FALSE)
      }
    },
    .indices_to_bits = function(idx, width) {
      n <- length(idx)
      bits <- matrix(0, nrow = n, ncol = width)
      for (b in seq_len(width)) {
        shift <- width - b
        bits[, b] <- bitwAnd(bitwShiftR(idx, shift), 1L)
      }
      bits
    },
    .bits_to_indices = function(bits) {
      width <- ncol(bits)
      idx <- integer(nrow(bits))
      for (b in seq_len(width)) {
        shift <- width - b
        idx <- idx + bitwShiftL(as.integer(bits[, b]), shift)
      }
      idx
    }
  )
)
