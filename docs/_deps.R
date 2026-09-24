# R dependencies for the benchmark docs. Kept in a plain R file so `renv`
# discovers and pins them in renv.lock; the Quarto document loads only some of
# these directly (knitr/rmarkdown are required by Quarto's knitr engine).
library(knitr)
library(rmarkdown)
library(reticulate)
library(qte)
library(RhpcBLASctl)
library(pbapply)
