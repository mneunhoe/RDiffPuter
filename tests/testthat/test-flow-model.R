test_that("flow_model forward returns finite scalar loss and gradients flow", {
  skip_if_no_torch()
  torch::torch_manual_seed(0)
  d_in <- 5L
  denoiser <- mlp_diffusion(d_in = d_in, dim_t = 16L)
  fm <- flow_model(denoise_fn = denoiser)

  x <- torch::torch_randn(c(8L, d_in))
  loss <- fm(x)
  expect_equal(loss$dim(), 0L)
  expect_true(is.finite(as.numeric(loss$cpu())))

  loss$backward()
  # First MLP weight should have a non-zero gradient.
  grad <- denoiser$proj$weight$grad
  expect_false(is.null(grad))
  expect_gt(as.numeric(grad$abs()$sum()$cpu()), 0)
})

test_that("flow_loss matches the analytic formula", {
  skip_if_no_torch()
  torch::torch_manual_seed(2)
  d_in <- 4L
  denoiser <- mlp_diffusion(d_in = d_in, dim_t = 16L)
  fm <- flow_model(denoise_fn = denoiser)

  x <- torch::torch_randn(c(6L, d_in))
  # Re-seed so flow_loss draws the same noise + t inside.
  torch::torch_manual_seed(99)
  loss_per_elem <- flow_loss(fm, x)
  expect_equal(dim(loss_per_elem), c(6L, d_in))

  # Recompute manually with the same seed.
  torch::torch_manual_seed(99)
  x0 <- torch::torch_randn_like(x)
  t  <- torch::torch_rand(6L)
  xt <- (1 - t$unsqueeze(2)) * x0 + t$unsqueeze(2) * x
  v_pred <- fm$velocity(xt, t)
  expected <- (v_pred - (x - x0))^2
  diff <- (loss_per_elem - expected)$abs()$max()
  expect_lt(as.numeric(diff$cpu()), 1e-5)
})
