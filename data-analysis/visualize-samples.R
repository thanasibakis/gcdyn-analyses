#!/usr/bin/env Rscript

library(tidyverse)
library(glue)

out_path <- commandArgs(trailingOnly = TRUE)[1]
posterior_samples <- read_csv(file.path(out_path, "samples-posterior.csv")) |>
    mutate(treeset = "Data")

prior_densities <- list(
    `φ[1]` = \(x) dlnorm(x, 0.5, 0.75),
    `φ[2]` = \(x) dlnorm(x, 0.5, 0.75),
    `φ[3]` = \(x) dnorm(x, 0, sqrt(2)),
    `φ[4]` = \(x) dlnorm(x, -0.5, 1.2),
    μ      = \(x) dlnorm(x, 0, 0.5),
    δ      = \(x) dlnorm(x, 0, 0.5)
)

prior_quantiles <- list(
    `φ[1]` = \(x) qlnorm(x, 0.5, 0.75),
    `φ[2]` = \(x) qlnorm(x, 0.5, 0.75),
    `φ[3]` = \(x) qnorm(x, 0, sqrt(2)),
    `φ[4]` = \(x) qlnorm(x, -0.5, 1.2),
    μ      = \(x) qlnorm(x, 0, 0.5),
    δ      = \(x) qlnorm(x, 0, 0.5)
)

# Setup for sigmoid plotting

X <- seq(-3, 3, 0.05)
TYPE_SPACE <- c(-2.4270176906430416, -1.4399117849363843, -0.6588015552361666, -0.13202968692343608, 0.08165101396850624, 0.7981793588605735, 1.3526378568771724, 2.1758707012574643)

sigmoid <- function(x, φ1, φ2, φ3, φ4) φ1 / (1 + exp(-φ2 * (x - φ3))) + φ4

plot_sigmoid <- function(quantiles) {
    ggplot(quantiles, aes(x)) +
        facet_wrap(vars(treeset)) +
        geom_ribbon(
            aes(ymin = q05, ymax = q95),
            alpha = 0.15,
            fill = "dodgerblue4"
        ) +
        geom_ribbon(
            aes(ymin = q10, ymax = q90),
            alpha = 0.15,
            fill = "dodgerblue4"
        ) +
        geom_ribbon(
            aes(ymin = q20, ymax = q80),
            alpha = 0.15,
            fill = "dodgerblue4"
        ) +
        geom_ribbon(
            aes(ymin = q30, ymax = q70),
            alpha = 0.15,
            fill = "dodgerblue4"
        ) +
        geom_ribbon(
            aes(ymin = q40, ymax = q60),
            alpha = 0.15,
            fill = "dodgerblue4"
        ) +
        geom_vline(xintercept = TYPE_SPACE, linewidth = 1, linetype = "dashed") +
        scale_color_manual(values = "black") +
        expand_limits(y = c(0, 2)) +
        theme_bw(base_size = 16) +
        theme(legend.position = "bottom", legend.title = element_blank())
}

# Plot sigmoid posteriors

posterior_λ_quantiles <-
    posterior_samples |>
    select(treeset, starts_with("φ")) |>
    expand_grid(x = X) |>
    mutate(λ = sigmoid(x, `φ[1]`, `φ[2]`, `φ[3]`, `φ[4]`)) |>
    summarise(
        q05 = quantile(λ, 0.05),
        q10 = quantile(λ, 0.1),
        q20 = quantile(λ, 0.2),
        q30 = quantile(λ, 0.3),
        q40 = quantile(λ, 0.4),
        q60 = quantile(λ, 0.6),
        q70 = quantile(λ, 0.7),
        q80 = quantile(λ, 0.8),
        q90 = quantile(λ, 0.9),
        q95 = quantile(λ, 0.95),
        .by = c(treeset, x)
    )

prior_λ_quantiles <-
    map(
        names(prior_densities),
        \(param) tibble(
            Parameter = param,
            q05 = prior_quantiles[[param]](0.05),
            q10 = prior_quantiles[[param]](0.1),
            q20 = prior_quantiles[[param]](0.2),
            q30 = prior_quantiles[[param]](0.3),
            q40 = prior_quantiles[[param]](0.4),
            q60 = prior_quantiles[[param]](0.6),
            q70 = prior_quantiles[[param]](0.7),
            q80 = prior_quantiles[[param]](0.8),
            q90 = prior_quantiles[[param]](0.9),
            q95 = prior_quantiles[[param]](0.95),
        )
    ) |>
    list_rbind() |>
    pivot_longer(
        c(starts_with("q")),
        names_to = "Quantile",
        values_to = "Value"
    ) |>
    pivot_wider(
        names_from = Parameter,
        values_from = Value
    ) |>
    expand_grid(x = X) |>
    mutate(λ = sigmoid(x, `φ[1]`, `φ[2]`, `φ[3]`, `φ[4]`)) |>
    select(x, λ, Quantile) |>
    pivot_wider(names_from = Quantile, values_from = λ) |>
    mutate(treeset = "Prior")

ggsave(
    file.path(out_path, "posterior-sigmoids.png"),
    plot_sigmoid(
        bind_rows(prior_λ_quantiles, posterior_λ_quantiles)
    ) + labs(
        title = "Birth rate posteriors",
        y = expression(lambda(x))
    ),
    width = 15,
    height = 10,
    dpi = 300
)

ggsave(
    file.path(out_path, "posterior-sigmoids-no-prior.png"),
    plot_sigmoid(posterior_λ_quantiles) + labs(
        title = "Birth rate posteriors",
        y = expression(lambda(x))
    ),
    width = 15,
    height = 10,
    dpi = 300
)

# Plot net rate and rate ratio posteriors

posterior_net_rate_quantiles <-
    posterior_samples |>
    mutate(`φ[4]` = `φ[4]` - μ) |>
    select(treeset, starts_with("φ")) |>
    expand_grid(x = X) |>
    mutate(λ = sigmoid(x, `φ[1]`, `φ[2]`, `φ[3]`, `φ[4]`)) |>
    summarise(
        q05 = quantile(λ, 0.05),
        q10 = quantile(λ, 0.1),
        q20 = quantile(λ, 0.2),
        q30 = quantile(λ, 0.3),
        q40 = quantile(λ, 0.4),
        q60 = quantile(λ, 0.6),
        q70 = quantile(λ, 0.7),
        q80 = quantile(λ, 0.8),
        q90 = quantile(λ, 0.9),
        q95 = quantile(λ, 0.95),
        .by = c(treeset, x)
    )

posterior_rate_ratio_quantiles <-
    posterior_samples |>
    mutate(
        `φ[1]` = `φ[1]` / μ,
        `φ[4]` = `φ[4]` / μ
    ) |>
    select(treeset, starts_with("φ")) |>
    expand_grid(x = X) |>
    mutate(λ = sigmoid(x, `φ[1]`, `φ[2]`, `φ[3]`, `φ[4]`)) |>
    summarise(
        q05 = quantile(λ, 0.05),
        q10 = quantile(λ, 0.1),
        q20 = quantile(λ, 0.2),
        q30 = quantile(λ, 0.3),
        q40 = quantile(λ, 0.4),
        q60 = quantile(λ, 0.6),
        q70 = quantile(λ, 0.7),
        q80 = quantile(λ, 0.8),
        q90 = quantile(λ, 0.9),
        q95 = quantile(λ, 0.95),
        .by = c(treeset, x)
    )

ggsave(
    file.path(out_path, "posterior-net-rates.png"),
    plot_sigmoid(
        posterior_net_rate_quantiles
    ) + labs(
        title = "Net rate posteriors",
        y = expression(lambda(x) - mu)
    ),
    width = 15,
    height = 10,
    dpi = 300
)

ggsave(
    file.path(out_path, "posterior-rate-ratios.png"),
    plot_sigmoid(
        posterior_rate_ratio_quantiles
    ) + labs(
        title = "Rate ratio posteriors",
        y = expression(lambda(x) / mu)
    ),
    width = 15,
    height = 10,
    dpi = 300
)

# Plot histograms

for (parameter in names(prior_densities)) {
    p <- ggplot() +
        facet_grid(rows = vars(treeset)) +
        stat_function(
            aes(fill = "Prior"),
            fun = prior_densities[[parameter]],
            geom = "area",
        ) +
        geom_histogram(
            aes(Sample, after_stat(density), fill = "Posterior"),
            data = posterior_samples |> rename(Sample = parameter) |> mutate(Parameter = parameter),
            alpha = 0.5
        ) +
        scale_fill_manual(
            values = c("Prior" = "grey", "Posterior" = "dodgerblue4")
        ) +
        theme_bw(base_size = 16) +
        theme(legend.position = "bottom", legend.title = element_blank()) +
        labs(title = "Posterior histogram", x = parameter)

    ggsave(
        file.path(out_path, glue("posterior-{parameter}.png")),
        p,
        width = 4,
        height = 8,
        dpi = 300
    )
}
