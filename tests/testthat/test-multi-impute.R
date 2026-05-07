make_toy <- function(n = 60, seed = 1) {
  set.seed(seed)
  df <- data.frame(
    x = rnorm(n),
    y = rnorm(n),
    g = factor(sample(letters[1:3], n, replace = TRUE))
  )
  df$x[sample.int(n, 8)] <- NA
  df$g[sample.int(n, 6)] <- NA
  df
}

test_that("multi_impute returns m completed data sets in original schema", {
  skip_if_no_torch()
  df <- make_toy()
  em <- diffputer_em(df, max_iter = 1, epochs = 20, hidden_dim = 16,
                     batch_size = 32, patience = 1000,
                     num_trials = 1, num_steps = 4, num_resamplings = 2,
                     verbose = FALSE)
  mi <- multi_impute(em, data = df, m = 5, seed = 1,
                     num_steps = 4, num_resamplings = 2)
  expect_s3_class(mi, "mi_imputations")
  expect_length(mi, 5)
  expect_true(all(vapply(mi, is.data.frame, logical(1))))
  expect_true(all(vapply(mi, function(d) identical(names(d), names(df)), logical(1))))
  expect_false(any(vapply(mi, function(d) any(is.na(d)), logical(1))))
})

test_that("multi_impute draws differ on missing entries but agree on observed", {
  skip_if_no_torch()
  df <- make_toy()
  em <- diffputer_em(df, max_iter = 1, epochs = 20, hidden_dim = 16,
                     batch_size = 32, patience = 1000,
                     num_trials = 1, num_steps = 4, num_resamplings = 2,
                     verbose = FALSE)
  mi <- multi_impute(em, data = df, m = 6, seed = 1,
                     num_steps = 4, num_resamplings = 2)

  na_idx <- which(is.na(df$x))
  obs_idx <- which(!is.na(df$x))

  # Observed values must be preserved across all draws
  for (k in seq_along(mi)) {
    expect_equal(mi[[k]]$x[obs_idx], df$x[obs_idx])
  }
  # At least one missing entry should differ across draws
  vals <- vapply(mi, function(d) d$x[na_idx[1]], numeric(1))
  expect_gt(length(unique(round(vals, 6))), 1)
})

test_that("multi_impute is reproducible given a seed", {
  skip_if_no_torch()
  df <- make_toy()
  em <- diffputer_em(df, max_iter = 1, epochs = 20, hidden_dim = 16,
                     batch_size = 32, patience = 1000,
                     num_trials = 1, num_steps = 4, num_resamplings = 2,
                     verbose = FALSE)
  a <- multi_impute(em, data = df, m = 3, seed = 42,
                    num_steps = 4, num_resamplings = 2)
  b <- multi_impute(em, data = df, m = 3, seed = 42,
                    num_steps = 4, num_resamplings = 2)
  for (k in seq_along(a)) {
    expect_equal(a[[k]]$x, b[[k]]$x)
  }
})

test_that("diffputer_em_mi returns an RDiffPuter_em_mi with m imputations", {
  skip_if_no_torch()
  df <- make_toy()
  res <- diffputer_em_mi(df, m = 4,
                         max_iter = 1, epochs = 20, hidden_dim = 16,
                         batch_size = 32, patience = 1000,
                         mi_num_steps = 4, mi_num_resamplings = 2,
                         verbose = FALSE)
  expect_s3_class(res, "RDiffPuter_em_mi")
  expect_s3_class(res$imputations, "mi_imputations")
  expect_length(res$imputations, 4)
  expect_equal(res$settings$m, 4)
  expect_s3_class(res$model, "trained_RDiffPuter")
})

test_that("multi_impute validates m and rejects non-data_transformer", {
  skip_if_no_torch()
  df <- make_toy()
  em <- diffputer_em(df, max_iter = 1, epochs = 10, hidden_dim = 8,
                     batch_size = 32, patience = 1000,
                     num_trials = 1, num_steps = 3, num_resamplings = 1,
                     verbose = FALSE)
  expect_error(multi_impute(em, data = df, m = 0))
  expect_error(multi_impute(em, data = df, m = 1.5))
  expect_error(
    multi_impute.trained_RDiffPuter(em$model, df, transformer = "nope", m = 2),
    "data_transformer"
  )
})
