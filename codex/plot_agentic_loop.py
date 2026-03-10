from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RESULTS_PATH = Path(__file__).with_name("results.tsv")
OUTPUT_PATH = Path(__file__).with_name("agentic_loop_speedup.png")


@dataclass
class ResultRow:
    phase: str
    order: int
    experiment: str
    description: str
    converged: str
    avg_step: int | None
    run_details: str
    status: str
    lesson: str

    @property
    def success_rate(self) -> float:
        numerator, denominator = self.converged.split("/")
        return int(numerator) / int(denominator)

    @property
    def short_label(self) -> str:
        if self.experiment.startswith("baseline_"):
            return "B1" if self.experiment.endswith("v1") else "B2"
        if self.experiment.startswith("exp"):
            return self.experiment.split("_", 1)[0].replace("exp", "")
        return self.experiment


def load_results(path: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    phase = "Phase 1"
    order = 0

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "Phase 2" in line:
                phase = "Phase 2"
                order = 0
            continue
        if line.startswith("experiment\t"):
            continue

        parts = raw_line.split("\t")
        if len(parts) != 7:
            raise ValueError(f"Unexpected row format: {raw_line}")

        experiment, description, converged, avg_step, run_details, status, lesson = parts
        rows.append(
            ResultRow(
                phase=phase,
                order=order,
                experiment=experiment,
                description=description,
                converged=converged,
                avg_step=None if avg_step == "DNF" else int(avg_step),
                run_details=run_details,
                status=status,
                lesson=lesson,
            )
        )
        order += 1

    return rows


def phase_summary(rows: list[ResultRow]) -> tuple[ResultRow, ResultRow, float]:
    numeric_rows = [row for row in rows if row.avg_step is not None]
    keep_rows = [row for row in numeric_rows if row.status == "keep"]
    if not keep_rows:
        raise ValueError(f"No kept converged rows found for {rows[0].phase}")
    first_keep = keep_rows[0]
    best_keep = min(keep_rows, key=lambda row: row.avg_step)
    return first_keep, best_keep, first_keep.avg_step / best_keep.avg_step


def plot_phase(ax: plt.Axes, rows: list[ResultRow], phase_note: str) -> None:
    numeric_rows = [row for row in rows if row.avg_step is not None]
    dnf_rows = [row for row in rows if row.avg_step is None]
    first_keep, best_keep, speedup = phase_summary(rows)

    x_numeric = [row.order for row in numeric_rows]
    y_numeric = [row.avg_step for row in numeric_rows]
    colors = ["#0f8a5f" if row.status == "keep" else "#d94841" for row in numeric_rows]

    ax.set_facecolor("#fcfcf8")
    ax.scatter(
        x_numeric,
        y_numeric,
        s=54,
        c=colors,
        edgecolors="#102018",
        linewidths=0.8,
        alpha=0.98,
        zorder=3,
    )

    y_min = min(y_numeric)
    y_max = max(y_numeric)
    y_span = max(y_max - y_min, 1)
    dnf_y = y_max + y_span * 0.12

    if dnf_rows:
        ax.scatter(
            [row.order for row in dnf_rows],
            [dnf_y for _ in dnf_rows],
            marker="x",
            s=34,
            c="#8d877d",
            linewidths=1.0,
            zorder=2,
        )

    running_best: int | None = None
    line_x: list[int] = []
    line_y: list[int] = []
    for row in rows:
        if row.status == "keep" and row.avg_step is not None:
            running_best = row.avg_step if running_best is None else min(running_best, row.avg_step)
        if running_best is not None:
            line_x.append(row.order)
            line_y.append(running_best)

    ax.plot(line_x, line_y, color="#1d2733", linewidth=2.8, zorder=4)
    ax.scatter(
        [first_keep.order, best_keep.order],
        [first_keep.avg_step, best_keep.avg_step],
        s=[100, 100],
        facecolors="none",
        edgecolors="#1d2733",
        linewidths=1.4,
        zorder=5,
    )

    ax.set_title(rows[0].phase, fontsize=20, fontweight="bold", loc="left", pad=30, color="#1f1f1f")
    ax.text(0.0, 1.035, phase_note, transform=ax.transAxes, fontsize=11, color="#5b554d", va="bottom")
    ax.text(
        0.995,
        1.035,
        f"{speedup:.2f}x faster   {first_keep.avg_step}->{best_keep.avg_step}",
        transform=ax.transAxes,
        fontsize=11.5,
        ha="right",
        va="bottom",
        fontweight="bold",
        color="#17212b",
        bbox={"boxstyle": "round,pad=0.28", "fc": "#fffaf0", "ec": "#d7cdbf", "alpha": 0.98},
    )

    ax.set_ylabel("Avg steps to converge", fontsize=14, fontweight="bold")
    ax.set_xticks([row.order for row in rows])
    ax.set_xticklabels([row.short_label for row in rows], fontsize=12)
    ax.set_xlim(-0.6, rows[-1].order + 0.8)
    ax.set_ylim(y_min - y_span * 0.06, dnf_y + y_span * 0.06)
    ax.grid(axis="y", color="#dfd7ca", linewidth=1.0, alpha=0.95)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", pad=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9d9589")
    ax.spines["bottom"].set_color("#9d9589")


def main() -> None:
    rows = load_results(RESULTS_PATH)
    phase_1_rows = [row for row in rows if row.phase == "Phase 1"]
    phase_2_rows = [row for row in rows if row.phase == "Phase 2"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#f4efe7",
            "axes.facecolor": "#fcfcf8",
            "font.weight": "medium",
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    plot_phase(
        axes[0],
        phase_1_rows,
        "1 ep/checkpoint, threshold 450, 3 runs",
    )
    plot_phase(
        axes[1],
        phase_2_rows,
        "5 eps/checkpoint, threshold 475, 5 runs",
    )
    axes[1].set_xlabel("Experiment order", fontsize=14, fontweight="bold", labelpad=14)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Kept", markerfacecolor="#0f8a5f",
               markeredgecolor="#102018", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Reverted", markerfacecolor="#d94841",
               markeredgecolor="#102018", markersize=7),
        Line2D([0], [0], marker="x", color="#8d877d", label="DNF", markersize=7, linewidth=0),
        Line2D([0], [0], color="#1d2733", lw=2.8, label="Best kept path"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02), fontsize=12)
    fig.suptitle(
        "Karpathy-Style AutoResearcher Sped Up CartPole",
        fontsize=20,
        fontweight="bold",
        y=0.972,
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.16, hspace=0.34)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
