#' @title Save a trained DiffPuter model and its data transformer
#'
#' @description Persists the components needed to resume imputation in a later
#'   R session. The diffusion model `state_dict` is saved via
#'   `torch::torch_save()` (`.pt` file), the fitted transformer and bookkeeping
#'   metadata via `saveRDS()` (`.rds` file). Files share a common stem.
#'
#' @param object A `trained_RDiffPuter` (from [diffputer_trainer()]) or an
#'   `RDiffPuter_em` (from [diffputer_em()]).
#' @param path Path stem (e.g. `"models/run1"`); the function appends
#'   `_state.pt` and `_meta.rds`.
#' @param transformer Optional `data_transformer` to save alongside. Required
#'   if `object` is a `trained_RDiffPuter`; ignored for `RDiffPuter_em` (which
#'   already carries its transformer).
#'
#' @return The character vector `c(state_path, meta_path)` invisibly.
#' @export
save_diffputer <- function(object, path, transformer = NULL) {
  state_path <- paste0(path, "_state.pt")
  meta_path <- paste0(path, "_meta.rds")

  if (inherits(object, "RDiffPuter_em")) {
    trained <- object$model
    transformer <- object$transformer
    meta <- list(
      kind = "RDiffPuter_em",
      settings = object$settings,
      trained_settings = trained$settings,
      transformer = transformer,
      history = object$history
    )
  } else if (inherits(object, "trained_RDiffPuter")) {
    if (is.null(transformer)) {
      stop("transformer must be supplied when saving a trained_RDiffPuter",
           call. = FALSE)
    }
    trained <- object
    meta <- list(
      kind = "trained_RDiffPuter",
      trained_settings = trained$settings,
      transformer = transformer
    )
  } else {
    stop("object must be of class trained_RDiffPuter or RDiffPuter_em",
         call. = FALSE)
  }

  dir <- dirname(path)
  if (nzchar(dir) && !dir.exists(dir)) dir.create(dir, recursive = TRUE)
  torch::torch_save(trained$model$state_dict(), state_path)
  saveRDS(meta, meta_path)
  invisible(c(state_path, meta_path))
}

#' @title Load a previously saved DiffPuter model
#'
#' @description Inverse of [save_diffputer()]. Reconstructs the diffusion model
#'   architecture from the saved settings, restores the weights, and returns an
#'   object of the original class.
#'
#' @param path Path stem used at save time.
#' @param device Device on which to instantiate the restored model.
#'
#' @return Either a `trained_RDiffPuter` or an `RDiffPuter_em` (matching the
#'   saved class). Both carry an attached `transformer` so [impute()] is ready
#'   to call.
#' @export
load_diffputer <- function(path, device = "cpu") {
  state_path <- paste0(path, "_state.pt")
  meta_path <- paste0(path, "_meta.rds")
  if (!file.exists(state_path)) stop("missing ", state_path, call. = FALSE)
  if (!file.exists(meta_path)) stop("missing ", meta_path, call. = FALSE)
  device <- resolve_device(device)
  meta <- readRDS(meta_path)

  s <- meta$trained_settings
  denoiser <- mlp_diffusion(d_in = s$d_in, dim_t = s$hidden_dim)
  model <- diffputer_model(
    denoise_fn = denoiser,
    P_mean = s$P_mean, P_std = s$P_std, sigma_data = s$sigma_data
  )$to(device = device)
  state <- torch::torch_load(state_path)
  model$load_state_dict(state)

  trained <- list(
    model = model,
    denoiser = model$denoise_fn_D,
    optimizer = NULL,
    losses = numeric(0),
    best_loss = NA_real_,
    settings = utils::modifyList(s, list(device = device))
  )
  class(trained) <- "trained_RDiffPuter"

  if (meta$kind == "RDiffPuter_em") {
    out <- list(
      imputed = NULL,
      model = trained,
      transformer = meta$transformer,
      history = meta$history,
      settings = meta$settings
    )
    class(out) <- "RDiffPuter_em"
    out
  } else {
    attr(trained, "transformer") <- meta$transformer
    trained
  }
}
