skip_if_no_torch <- function() {
  testthat::skip_if_not_installed("torch")
  if (!isTRUE(torch::torch_is_installed())) {
    testthat::skip("torch backend not installed")
  }
}
