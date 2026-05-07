for (m in c("edm", "flow")) {
  test_that(sprintf("diffputer_em (method=%s) returns a data frame with no NAs", m), {
    skip_if_no_torch()
    set.seed(123)
    n <- 80
    df <- data.frame(
      a = rnorm(n),
      b = runif(n, 1, 5),
      c = factor(sample(letters[1:3], n, replace = TRUE))
    )
    df$a[sample.int(n, 10)] <- NA
    df$c[sample.int(n, 8)] <- NA

    result <- diffputer_em(
      df, method = m,
      max_iter = 1, epochs = 30, hidden_dim = 32,
      batch_size = 32, patience = 1000, num_trials = 2,
      num_steps = 5, num_resamplings = 3, verbose = FALSE
    )
    expect_s3_class(result, "RDiffPuter_em")
    expect_equal(names(result$imputed), names(df))
    expect_equal(nrow(result$imputed), nrow(df))
    expect_false(any(is.na(result$imputed)))
    expect_true(is.factor(result$imputed$c))
    expect_equal(result$settings$method, m)
    # Observed entries must be preserved exactly.
    observed <- !is.na(df$a)
    expect_equal(result$imputed$a[observed], df$a[observed])
  })
}

test_that("diffputer_em flow + resample_strategy='single' completes", {
  skip_if_no_torch()
  set.seed(4)
  df <- data.frame(a = rnorm(60), b = runif(60))
  df$a[1:8] <- NA
  result <- diffputer_em(
    df, method = "flow", resample_strategy = "single",
    max_iter = 1, epochs = 20, hidden_dim = 16,
    batch_size = 32, patience = 1000, num_trials = 2,
    num_steps = 4, num_resamplings = 1, verbose = FALSE
  )
  expect_false(any(is.na(result$imputed)))
  expect_equal(result$settings$resample_strategy, "single")
})

test_that("diffputer_em warns when data has no NAs", {
  skip_if_no_torch()
  df <- data.frame(a = rnorm(10), b = rnorm(10))
  expect_warning(
    diffputer_em(df, max_iter = 1, epochs = 5, hidden_dim = 16,
                 batch_size = 8, patience = 100, num_trials = 1,
                 num_steps = 3, num_resamplings = 1, verbose = FALSE),
    "no NA"
  )
})

test_that("diffputer_trainer + impute round-trip preserves observed entries", {
  skip_if_no_torch()
  set.seed(7)
  df <- data.frame(a = rnorm(60), b = runif(60))
  df_obs <- df
  df_obs$a[1:5] <- NA

  transformer <- data_transformer$new()
  transformer$fit(df_obs)
  encoded <- transformer$transform(df_obs)
  trained <- diffputer_trainer(
    encoded$x, hidden_dim = 16, epochs = 20,
    batch_size = 16, patience = 1000, verbose = FALSE
  )
  imputed <- impute(trained, df_obs, transformer,
                    num_trials = 2, num_steps = 4, num_resamplings = 2)
  expect_equal(imputed$a[6:60], df$a[6:60], tolerance = 1e-5)
})
