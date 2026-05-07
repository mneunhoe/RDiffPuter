# RDiffPuter

An R port of [DiffPuter](https://github.com/hengruizhang98/DiffPuter)
(Zhang et al., *DiffPuter: Empowering Diffusion Models for Missing Data
Imputation*, ICLR 2025). Imputes missing values in arbitrary tabular data
using an EDM diffusion model trained inside an EM loop, with full GPU
support via the [`torch`](https://torch.mlverse.org/) R package.

## Features

- Accepts any `data.frame` with `NA` entries — no manual mask file needed.
- Handles continuous, factor / character, and logical columns automatically.
- Two categorical encodings: `"binary"` (DiffPuter-faithful, smaller encoded
  dim) and `"onehot"` (more interpretable).
- Two generative backbones: `method = "edm"` (default, paper-faithful) or
  `method = "flow"` (flow matching with linear interpolant — typically
  needs ~5–10 sampling steps where EDM uses 50). See
  `vignette("flow-matching")` for the trade-off.
- Runs on CPU, CUDA, or Apple Silicon (MPS) with automatic fallback.
- Two-layer API: a high-level `diffputer_em()` for one-call imputation and
  the lower-level `diffputer_trainer()` + `impute()` pair for power users.

## Installation

```r
# 1. Install the torch backend (one-time):
install.packages("torch")
torch::install_torch()

# 2. Install RDiffPuter:
remotes::install_github("mneunhoe/RDiffPuter")
```

## Quickstart

```r
library(RDiffPuter)
library(torch)

set.seed(1)
data <- data.frame(
  age      = rnorm(200, 40, 10),
  income   = exp(rnorm(200, 10, 0.5)),
  region   = factor(sample(c("N", "S", "E", "W"), 200, replace = TRUE))
)
data$age[1:20]    <- NA
data$income[5:15] <- NA
data$region[40:55] <- NA

device <- if (cuda_is_available()) "cuda" else
          if (torch::backends_mps_is_available()) "mps" else "cpu"

result <- diffputer_em(
  data,
  max_iter   = 3,
  epochs     = 200,
  hidden_dim = 128,
  num_trials = 8,
  num_steps  = 20,
  num_resamplings = 5,
  device     = device
)
head(result$imputed)
```

## API at a glance

| Function | Purpose |
|---|---|
| `diffputer_em()` | High-level: data.frame in, imputed data.frame out (full EM loop). |
| `data_transformer$new()` | Preprocess: detect types, encode, build mask. |
| `diffputer_trainer()` | Train one diffusion denoiser (single M-step). |
| `impute()` | Run the E-step on new data with a trained model. |
| `save_diffputer()` / `load_diffputer()` | Persist a trained model + transformer. |

See `vignette("getting-started")` and `vignette("mixed-types")` for worked
examples.

## License

MIT © 2026 Marcel Neunhoeffer. Algorithm by Zhang et al. (ICLR 2025).
