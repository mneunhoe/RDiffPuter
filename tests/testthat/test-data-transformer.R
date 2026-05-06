make_mixed_df <- function(n = 50, seed = 1) {
  set.seed(seed)
  data.frame(
    a = rnorm(n),
    b = runif(n, 1, 5),
    c = factor(sample(letters[1:4], n, replace = TRUE)),
    d = sample(c("x", "y", "z", "w", "v"), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

test_that("data_transformer round-trips numeric and categorical (binary)", {
  df <- make_mixed_df()
  t <- data_transformer$new()
  t$fit(df, encoding = "binary")
  expect_equal(t$encoded_dim, 2L + 2L + 3L)  # 2 num, ceil(log2(4))=2, ceil(log2(5))=3
  rt <- t$inverse_transform(t$transform(df)$x)
  expect_equal(rt$a, df$a, tolerance = 1e-6)
  expect_equal(rt$b, df$b, tolerance = 1e-6)
  expect_equal(as.character(rt$c), as.character(df$c))
  expect_equal(as.character(rt$d), as.character(df$d))
})

test_that("data_transformer round-trips categorical (onehot)", {
  df <- make_mixed_df()
  t <- data_transformer$new()
  t$fit(df, encoding = "onehot")
  expect_equal(t$encoded_dim, 2L + 4L + 5L)
  rt <- t$inverse_transform(t$transform(df)$x)
  expect_equal(as.character(rt$c), as.character(df$c))
  expect_equal(as.character(rt$d), as.character(df$d))
})

test_that("build_mask reports NAs at the encoded width", {
  df <- make_mixed_df()
  df$a[c(1, 5)] <- NA
  df$c[c(2, 3)] <- NA
  t <- data_transformer$new()
  t$fit(df, encoding = "binary")
  m <- t$build_mask(df)
  expect_equal(dim(m), c(nrow(df), t$encoded_dim))
  # Column 'a' is the first numeric: 2 NAs.
  expect_equal(sum(m[, 1]), 2)
  # Column 'b' has none.
  expect_equal(sum(m[, 2]), 0)
  # Column 'c' has 2 NAs replicated across its 2 bit-cols.
  expect_equal(sum(m[, 3:4]), 4)
})

test_that("data_transformer rejects single-level factor", {
  df <- data.frame(a = rnorm(20), b = factor(rep("x", 20)))
  t <- data_transformer$new()
  expect_error(t$fit(df), "fewer than two")
})
