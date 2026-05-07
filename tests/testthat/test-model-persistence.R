test_that("save_diffputer + load_diffputer round-trip a trained model", {
  skip_if_no_torch()
  set.seed(2)
  df <- data.frame(a = rnorm(50), b = runif(50))
  df$a[1:5] <- NA
  transformer <- data_transformer$new()
  transformer$fit(df)
  encoded <- transformer$transform(df)
  trained <- diffputer_trainer(encoded$x, hidden_dim = 16, epochs = 8,
                               batch_size = 16, patience = 100,
                               verbose = FALSE)

  tmp <- tempfile()
  paths <- save_diffputer(trained, tmp, transformer = transformer)
  expect_true(all(file.exists(paths)))

  restored <- load_diffputer(tmp)
  expect_s3_class(restored, "trained_RDiffPuter")
  # Restored model produces same forward output for identical input
  torch::torch_manual_seed(0)
  x <- torch::torch_randn(c(4L, ncol(encoded$x)))
  noise <- torch::torch_rand(4L) + 0.1
  o1 <- trained$denoiser$denoise_fn_F(x, noise)
  o2 <- restored$model$denoise_fn_D$denoise_fn_F(x, noise)
  expect_lt(as.numeric((o1 - o2)$abs()$max()$cpu()), 1e-5)

  unlink(paths)
})

test_that("save_diffputer + load_diffputer round-trip a flow-trained model", {
  skip_if_no_torch()
  set.seed(3)
  df <- data.frame(a = rnorm(40), b = runif(40))
  df$a[1:5] <- NA
  transformer <- data_transformer$new()
  transformer$fit(df)
  encoded <- transformer$transform(df)
  trained <- diffputer_trainer(
    encoded$x, method = "flow", hidden_dim = 16, epochs = 8,
    batch_size = 16, patience = 100, verbose = FALSE
  )
  expect_equal(trained$settings$method, "flow")

  tmp <- tempfile()
  paths <- save_diffputer(trained, tmp, transformer = transformer)
  restored <- load_diffputer(tmp)
  expect_s3_class(restored, "trained_RDiffPuter")
  expect_equal(restored$settings$method, "flow")

  # Restored velocity matches original on the same input.
  torch::torch_manual_seed(0)
  x <- torch::torch_randn(c(4L, ncol(encoded$x)))
  t <- torch::torch_rand(4L)
  v1 <- trained$model$velocity(x, t)
  v2 <- restored$model$velocity(x, t)
  expect_lt(as.numeric((v1 - v2)$abs()$max()$cpu()), 1e-5)
  unlink(paths)
})
