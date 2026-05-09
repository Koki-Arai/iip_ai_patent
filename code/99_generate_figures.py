"""
99_generate_figures.py
=======================

Generates the three figures used in the paper:
- Figure 1: Core AI (G06N) and broad AI shares of total applications, 2010–2020
- Figure 2: Event-study coefficients for grant and forward citations
- Figure 3: Inverted-U relationship between AI density and follow-on innovation

Run after the other scripts have produced the relevant CSVs in results/tables/.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import TABLES_DIR, FIGURES_DIR, PLOT_NAVY, PLOT_GRAY

# Common style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ----------------------------------------------------------------------
# Figure 1: AI shares 2010–2020
# ----------------------------------------------------------------------
def figure_1():
    # Data points reported in Section 4.1 of the paper
    years_core = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    core_ai_pct = [0.028, 0.030, 0.035, 0.045, 0.058, 0.073,
                   0.106, 0.173, 0.283, 0.392, 0.540]

    years_broad = [2010, 2014, 2016, 2018, 2020]
    broad_ai_pct = [10.47, 10.18, 9.67, 10.23, 11.46]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(years_broad, broad_ai_pct, color=PLOT_NAVY, linewidth=2,
             marker='o', markersize=7,
             markerfacecolor=PLOT_NAVY, markeredgecolor='white')
    ax1.set_ylabel("Broad AI share (%)")
    ax1.set_ylim(8, 13)
    ax1.set_title("Panel A: Broad AI measure (G06N + G06F + G06Q + G06T + G10L + H04N)",
                  loc='left', fontsize=11)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

    ax2.plot(years_core, core_ai_pct, color=PLOT_NAVY, linewidth=2,
             marker='o', markersize=7,
             markerfacecolor=PLOT_NAVY, markeredgecolor='white')
    ax2.axvline(x=2018, color=PLOT_GRAY, linewidth=1, linestyle='--', alpha=0.7)
    ax2.text(2018.1, 0.45, "JPO AI guideline\n(Mar 2018)",
             fontsize=9, color=PLOT_GRAY)
    ax2.set_xlabel("Application year")
    ax2.set_ylabel("Core AI (G06N) share (%)")
    ax2.set_ylim(0, 0.6)
    ax2.set_title("Panel B: Core AI measure (G06N only)", loc='left', fontsize=11)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "Figure1_AI_shares.png", dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[99_generate_figures] Figure 1 saved.")

# ----------------------------------------------------------------------
# Figure 2: Event study (grant + forward_cites)
# ----------------------------------------------------------------------
def figure_2():
    df = pd.read_csv(TABLES_DIR / "event_studies_master.csv")
    df_grant = df[(df["outcome"] == "grant") &
                  (df["control_group"] == "all")].sort_values("year")
    df_fwd = df[(df["outcome"] == "log1p_forward_cites") &
                (df["control_group"] == "all")].sort_values("year")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for ax, sub, title in [
        (ax1, df_grant, "Panel A: Grant probability (relative to 2014)"),
        (ax2, df_fwd,   "Panel B: log(1 + forward citations) (relative to 2014)"),
    ]:
        ax.errorbar(sub["year"], sub["estimate"],
                    yerr=[sub["estimate"] - sub["ci_lo"],
                          sub["ci_hi"] - sub["estimate"]],
                    fmt='o-', color=PLOT_NAVY, ecolor=PLOT_NAVY,
                    linewidth=1.5, markersize=6,
                    markerfacecolor=PLOT_NAVY, markeredgecolor='white',
                    capsize=3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=2017.5, color=PLOT_GRAY, linewidth=1,
                   linestyle='--', alpha=0.7)
        ax.set_title(title, loc='left', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_ylabel(r"AI$_j$ × year coefficient")
    ax2.set_xlabel("Application year")
    ax2.set_xticks(range(2010, 2019))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "Figure2_event_study.png", dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[99_generate_figures] Figure 2 saved.")

# ----------------------------------------------------------------------
# Figure 3: Inverted-U
# ----------------------------------------------------------------------
def figure_3():
    # Read coefficients from Table 10
    t10 = pd.read_csv(TABLES_DIR / "table10_subgroup_density.csv")
    b1 = t10.loc[t10["specification"].str.contains("density$", regex=True) &
                 t10["specification"].str.contains("quadratic"),
                 "coef"].iloc[0]
    b2 = t10.loc[t10["specification"].str.contains("density²"), "coef"].iloc[0]

    density_star = -b1 / (2 * b2)
    x = np.linspace(0, 1, 200)
    y = b1 * x + b2 * x ** 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color=PLOT_NAVY, linewidth=2.5)

    ax.axvline(x=density_star, color=PLOT_GRAY, linewidth=1.2,
               linestyle='--', alpha=0.8)
    peak_y = b1 * density_star + b2 * density_star ** 2
    ax.plot(density_star, peak_y, 'o', markersize=10, color=PLOT_NAVY,
            markerfacecolor='white', markeredgewidth=2, zorder=5)
    ax.annotate(f"Turning point\ndensity* ≈ {density_star:.3f}",
                xy=(density_star, peak_y),
                xytext=(density_star + 0.12, peak_y + 0.04),
                fontsize=10,
                arrowprops=dict(arrowstyle='-', color=PLOT_GRAY, lw=0.8))

    ax.fill_between(x, 0, y, where=(x <= density_star),
                    alpha=0.10, color=PLOT_NAVY)
    ax.fill_between(x, 0, y, where=(x > density_star),
                    alpha=0.10, color=PLOT_GRAY)

    ax.text(0.18, 0.06, "Prospect-like spillover regime\n(Kitch 1977)",
            fontsize=10, ha='center', style='italic')
    ax.text(0.72, 0.06, "Anti-commons-like regime\n(Heller & Eisenberg 1998)",
            fontsize=10, ha='center', style='italic')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel("AI patent density (G06N share in subgroup-year)")
    ax.set_ylabel(r"$\Delta \log(1+$FollowOn$)$ (partial effect)")
    ax.set_title(r"Predicted effect: $\beta_1 \cdot$ density $+ \beta_2 \cdot$ density$^2$",
                 loc='left', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 0.35)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "Figure3_inverted_U.png", dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[99_generate_figures] Figure 3 saved.")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    figure_1()
    figure_2()
    figure_3()
    print("[99_generate_figures] All figures generated.")

if __name__ == "__main__":
    main()
