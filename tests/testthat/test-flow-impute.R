test_that("flow_impute_mask returns finite tensor of right shape (repaint)", {
  skip_if_no_torch()
  torch::torch_manual_seed(0)
  d_in <- 4L; n <- 8L
  fm <- flow_model(denoise_fn = mlp_diffusion(d_in = d_in, dim_t = 16L))
  x <- torch::torch_randn(c(n, d_in))
  mask <- torch::torch_tensor(matrix(rbinom(n * d_in, 1, 0.3), n, d_in),
                              dtype = torch::torch_float())
  rec <- flow_impute_mask(fm, x * (1 - mask), mask,
                          num_steps = 4L, num_resamplings = 2L,
                          resample_strategy = "repaint")
  expect_equal(dim(rec), c(n, d_in))
  expect_true(torch::torch_isfinite(rec)$all()$item())
})

test_that("flow_impute_mask 'single' strategy ignores num_resamplings", {
  skip_if_no_torch()
  torch::torch_manual_seed(0)
  d_in <- 4L; n <- 6L
  fm <- flow_model(denoise_fn = mlp_diffusion(d_in = d_in, dim_t = 16L))
  x <- torch::torch_randn(c(n, d_in))
  mask <- torch::torch_tensor(matrix(rbinom(n * d_in, 1, 0.3), n, d_in),
                              dtype = torch::torch_float())

  torch::torch_manual_seed(7)
  r1 <- flow_impute_mask(fm, x * (1 - mask), mask,
                         num_steps = 4L, num_resamplings = 1L,
                         resample_strategy = "single")
  torch::torch_manual_seed(7)
  r2 <- flow_impute_mask(fm, x * (1 - mask), mask,
                         num_steps = 4L, num_resamplings = 99L,
                         resample_strategy = "single")
  diff <- (r1 - r2)$abs()$max()
  expect_lt(as.numeric(diff$cpu()), 1e-6)
})

test_that("validate_resample_strategy rejects unknown values", {
  expect_equal(validate_resample_strategy("repaint"), "repaint")
  expect_equal(validate_resample_strategy("single"), "single")
  expect_error(validate_resample_strategy("noisy"))
})

test_that("validate_method accepts edm and flow only", {
  expect_equal(validate_method("edm"), "edm")
  expect_equal(validate_method("flow"), "flow")
  expect_error(validate_method("ode"))
})
