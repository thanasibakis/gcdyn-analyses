#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(tidyverse)
    library(glue)
})

dir.create("out/", showWarnings = FALSE)

# Setup values

priors <- list(
    "φ[1]" = list(
        density = \(x) dlnorm(x, 0.5, 0.75),
        quantile = \(x) qlnorm(x, 0.5, 0.75)
    ),
    "φ[2]" = list(
        density = \(x) dlnorm(x, 0.5, 0.75),
        quantile = \(x) qlnorm(x, 0.5, 0.75)
    ),
    "φ[3]" = list(
        density = \(x) dnorm(x, 0, sqrt(2)),
        quantile = \(x) qnorm(x, 0, sqrt(2))
    ),
    "φ[4]" = list(
        density = \(x) dlnorm(x, -0.5, 1.2),
        quantile = \(x) qlnorm(x, -0.5, 1.2)
    ),
    "μ" = list(
        density = \(x) dlnorm(x, 0, 0.5),
        quantile = \(x) qlnorm(x, 0, 0.5)
    ),
    "δ" = list(
        density = \(x) dlnorm(x, 0, 0.5),
        quantile = \(x) qlnorm(x, 0, 0.5)
    )
)

truth <- list(
    "φ[1]" = 6.547622203826904,
    "φ[2]" = 1.4588056802749634,
    "φ[3]" = 1.9493433237075806,
    "φ[4]" = 0,
    "μ" = 1
)

X <- seq(-3, 3, 0.05)
TYPE_SPACE <- c(-2.4270176906430416, -1.4399117849363843, -0.6588015552361666, -0.13202968692343608, 0.08165101396850624, 0.7981793588605735, 1.3526378568771724, 2.1758707012574643)

# Load posterior samples and compute posterior sigmoid quantiles
posterior_samples <- read_csv("out/posterior-samples.csv")

sigmoid <- function(x, φ1, φ2, φ3, φ4) φ1 / (1 + exp(-φ2 * (x - φ3))) + φ4

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
    ) |>
    mutate("Quantity" = "$\\lambda(x)$")

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
    ) |>
    mutate("Quantity" = "$\\lambda(x) - \\mu$")

posterior_quantiles <- bind_rows(posterior_λ_quantiles, posterior_net_rate_quantiles) |>
    mutate(
        Quantity = factor(Quantity, levels = c("$\\lambda(x)$", "$\\lambda(x) - \\mu$")),
    )

# Plot posterior sigmoids for one treeset

p <- posterior_quantiles |>
    filter(treeset == 1) |>
    ggplot(aes(x)) +
    facet_grid(cols = vars(Quantity)) +
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
    stat_function(
        aes(color = "Truth"),
        fun = sigmoid,
        args = list(
            φ1 = truth[["φ[1]"]],
            φ2 = truth[["φ[2]"]],
            φ3 = truth[["φ[3]"]],
            φ4 = truth[["φ[4]"]]
        ),
        linewidth = 1,
        data = posterior_quantiles |> filter(Quantity == "$\\lambda(x)$")
    ) +
    geom_vline(xintercept = TYPE_SPACE, linewidth = 1, linetype = "dashed") +
    scale_color_manual(values = "black") +
    theme_bw(base_size = 12) +
    theme(
        legend.position = "bottom",
        legend.title = element_blank(),
        axis.title.y = element_blank()
    )

ggsave(
    "out/posterior-sigmoids.png",
    p,
    width = 13,
    height = 6
)

# Plot posterior sigmoids for all treesets

p <- posterior_quantiles |>
    ggplot(aes(x)) +
    facet_grid(
        rows = vars(Quantity),
        cols = vars(treeset),
        switch = "y",
        labeller = labeller(
            treeset = \(t) case_when(
                t == "Merged" ~ t,
                TRUE ~ glue("Treeset {t}")
            )
        )
    ) +
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
    stat_function(
        aes(color = "Truth"),
        fun = sigmoid,
        args = list(
            φ1 = truth[["φ[1]"]],
            φ2 = truth[["φ[2]"]],
            φ3 = truth[["φ[3]"]],
            φ4 = truth[["φ[4]"]]
        ),
        linewidth = 1,
        data = posterior_quantiles |> filter(Quantity == "$\\lambda(x)$")
    ) +
    geom_vline(xintercept = TYPE_SPACE, linewidth = 0.5, linetype = "dashed") +
    scale_color_manual(values = "black") +
    theme_bw(base_size = 12) +
    theme(
        legend.position = "bottom",
        legend.title = element_blank(),
        axis.title.y = element_blank(),
        strip.placement = "outside",
        strip.background.y = element_blank(),
        strip.switch.pad.grid = unit(0, "pt")
    )

ggsave(
    "out/posterior-sigmoids-all.png",
    p,
    width = 13,
    height = 6
)


# Plot posterior histograms and traceplots

posterior_samples <- posterior_samples |>
    pivot_longer(c(starts_with("φ"), μ, δ), names_to = "Parameter", values_to = "Sample")

p <- posterior_samples |>
    ggplot() +
    facet_grid(
        rows = vars(treeset),
        cols = vars(Parameter),
        labeller = labeller(
            Parameter = \(param) case_when(
                str_starts(param, "φ") ~ glue("$\\phi_{str_sub(param, 3, 3)}$"),
                param == "μ" ~ "$\\mu$",
                param == "δ" ~ "$\\delta$"
            ),
            treeset = \(t) case_when(
                t == "Merged" ~ t,
                TRUE ~ glue("Treeset {t}")
            )
        )
    ) +
    geom_histogram(
        aes(Sample, after_stat(density), alpha = "Posterior"),
        fill = "dodgerblue4"
    ) +
    scale_alpha_manual(
        values = c(
            "Prior" = 0.3,
            "Posterior" = 0.8
        ),
        breaks = c("Prior", "Posterior")
    ) +
    theme_bw(base_size = 17) +
    theme(
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.spacing.x = unit(0.6, "cm"),
        legend.key.spacing.x = unit(1, "cm"),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank()
    ) +
    labs(x = "Sample", y = "Density")

for (parameter in names(priors)) {
    p <- p +
        stat_function(
            aes(alpha = "Prior"),
            fun = priors[[parameter]]$density,
            geom = "area",
            fill = "dodgerblue4",
            data = tibble(Parameter = parameter),
            show.legend = parameter == "μ" # so that the alpha isn't overlayed for every parameter
        )
}

ggsave(
    "out/posterior-histograms.png",
    p,
    width = 10,
    height = 8
)

p <- posterior_samples |>
    ggplot() +
    facet_grid(
        rows = vars(treeset),
        cols = vars(Parameter),
        labeller = labeller(
            Parameter = \(param) case_when(
                str_starts(param, "φ") ~ glue("$\\phi_{str_sub(param, 3, 3)}$"),
                param == "μ" ~ "$\\mu$",
                param == "δ" ~ "$\\delta$"
            ),
            treeset = \(t) case_when(
                t == "Merged" ~ t,
                TRUE ~ glue("Treeset {t}")
            )
        ),
        scales = "free_y"
    ) +
    geom_line(aes(iteration, Sample), color = "dodgerblue4") +
    scale_x_continuous(breaks = c(500, 1000, 1500)) +
    theme_bw(base_size = 13) +
    theme(
        legend.position = "bottom",
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
    ) +
    labs(x = "Iteration", y = "Sample")

ggsave(
    "out/posterior-traceplots.png",
    p,
    width = 9,
    height = 7
)
