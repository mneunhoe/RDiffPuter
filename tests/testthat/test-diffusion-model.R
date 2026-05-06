test_that("mlp_diffusion forward and backward run", {
  skip_if_no_torch()
  torch::torch_manual_seed(0)
  d_in <- 5L
  net <- mlp_diffusion(d_in = d_in, dim_t = 16L)
  x <- torch::torch_randn(c(4L, d_in))
  noise <- torch::torch_rand(4L) + 0.1
  out <- net(x, noise)
  expect_equal(dim(out), c(4L, d_in))
})

test_that("diffputer_model produces a finite scalar loss", {
  skip_if_no_torch()
  torch::torch_manual_seed(0)
  d_in <- 5L
  denoiser <- mlp_diffusion(d_in = d_in, dim_t = 16L)
  model <- diffputer_model(denoise_fn = denoiser)
  x <- torch::torch_randn(c(8L, d_in))
  loss <- model(x)
  expect_equal(loss$dim(), 0L)
  expect_true(is.finite(as.numeric(loss$cpu())))
})

test_that("impute_mask returns a tensor of the right shape and is finite", {
  skip_if_no_torch()
  torch::torch_manual_seed(0)
  d_in <- 4L
  n <- 8L
  denoiser <- mlp_diffusion(d_in = d_in, dim_t = 16L)
  model <- diffputer_model(denoise_fn = denoiser)
  x <- torch::torch_randn(c(n, d_in))
  mask <- torch::torch_tensor(matrix(rbinom(n * d_in, 1, 0.3), n, d_in),
                              dtype = torch::torch_float())
  rec <- impute_mask(model$denoise_fn_D, x * (1 - mask), mask,
                     num_steps = 4L, num_resamplings = 2L)
  expect_equal(dim(rec), c(n, d_in))
  expect_true(torch::torch_isfinite(rec)$all()$item())
})
