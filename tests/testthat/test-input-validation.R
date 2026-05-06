test_that("validate_positive_integer rejects bad input", {
  expect_error(validate_positive_integer(0, "x"), "positive integer")
  expect_error(validate_positive_integer(-1, "x"), "positive integer")
  expect_error(validate_positive_integer(1.5, "x"), "positive integer")
  expect_error(validate_positive_integer("a", "x"), "positive integer")
  expect_silent(validate_positive_integer(7L, "x"))
})

test_that("validate_positive_number accepts decimals, rejects non-positives", {
  expect_silent(validate_positive_number(0.0001, "lr"))
  expect_error(validate_positive_number(0, "lr"), "positive number")
  expect_error(validate_positive_number(-1, "lr"), "positive number")
})

test_that("validate_data_frame_with_na rejects empty", {
  expect_error(validate_data_frame_with_na(data.frame()), "zero rows")
  expect_error(validate_data_frame_with_na(matrix(numeric(), 0, 0)), "zero rows")
})

test_that("validate_encoding accepts only known modes", {
  expect_equal(validate_encoding("binary"), "binary")
  expect_equal(validate_encoding("onehot"), "onehot")
  expect_error(validate_encoding("ordinal"))
})

test_that("resolve_device falls back when accelerator unavailable", {
  if (!torch::cuda_is_available()) {
    expect_warning(d <- resolve_device("cuda"), "CUDA")
    expect_equal(d, "cpu")
  }
  if (!torch::backends_mps_is_available()) {
    expect_warning(d <- resolve_device("mps"), "MPS")
    expect_equal(d, "cpu")
  }
  expect_equal(resolve_device("cpu"), "cpu")
  expect_error(resolve_device("tpu"))
})
