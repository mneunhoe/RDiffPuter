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
