"""Rich + plotext TUI wrapper for the training scripts.

Two reporters with the same interface:

  Reporter      : plain prints (default, what the scripts always did).
  TUIReporter   : Rich Live layout — top row of plotext graphs, bottom panel
                  is a scrolling log. Picks up the same `.log(...)` calls.

Usage:

    with make_reporter(use_tui=args.tui, eval_series=("zebra_exact", "owt_recon_cer")) as r:
        r.log("loading ...")
        r.train_point(step, loss=loss_val, lr=cur_lr)
        r.eval_point(step, zebra_exact=0.45, owt_recon_cer=0.12)

Train points feed the loss / lr graph. Eval points feed one line per series
on the eval graph (series names declared up-front so colors stay stable).

Plain mode just forwards `log` to `print` and ignores graph updates, so a
script can call the same methods unconditionally.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from typing import Iterable

import plotext as plt
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class Reporter:
    """Plain-print reporter. Methods are no-ops where there's no terminal UI."""

    def __init__(self):
        pass

    def log(self, msg: str = "") -> None:
        print(msg)

    def train_point(self, step: int, *, loss: float | None = None, lr: float | None = None) -> None:
        pass

    def eval_point(self, step: int, **metrics: float) -> None:
        pass

    def set_status(self, msg: str) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TUIReporter(Reporter):
    """Rich Live layout with plotext graphs and a scrolling log."""

    def __init__(
        self,
        eval_series: Iterable[str] = (),
        max_train_points: int = 4000,
        max_log_lines: int = 200,
        title: str = "training",
        refresh_every: int = 100,
    ):
        super().__init__()
        self._console = Console()
        self._title = title
        self._status = ""
        self._train_steps: deque[int] = deque(maxlen=max_train_points)
        self._train_loss: deque[float] = deque(maxlen=max_train_points)
        self._train_lr: deque[float] = deque(maxlen=max_train_points)
        self._eval_steps: dict[str, list[int]] = {s: [] for s in eval_series}
        self._eval_vals: dict[str, list[float]] = {s: [] for s in eval_series}
        self._log: deque[str] = deque(maxlen=max_log_lines)
        self._live: Live | None = None
        self._layout = self._build_layout()
        # Per-step calls (train_point, set_status) refresh at most every
        # `refresh_every` increments; log/eval still refresh immediately so
        # rare structured output never disappears behind throttling.
        self._refresh_every = max(1, int(refresh_every))
        self._refresh_tick = 0

    # ---- public API --------------------------------------------------

    def log(self, msg: str = "") -> None:
        for line in str(msg).splitlines() or [""]:
            self._log.append(line)
        self._refresh(force=True)

    def train_point(self, step: int, *, loss: float | None = None, lr: float | None = None) -> None:
        # Subsample to one point per `refresh_every` steps so the deque stays
        # small and the plotted curve is smooth instead of a dense noise band.
        if step % self._refresh_every != 0:
            return
        self._train_steps.append(step)
        self._train_loss.append(float(loss) if loss is not None else float("nan"))
        self._train_lr.append(float(lr) if lr is not None else float("nan"))
        self._refresh()

    def eval_point(self, step: int, **metrics: float) -> None:
        for name, val in metrics.items():
            if name not in self._eval_steps:
                self._eval_steps[name] = []
                self._eval_vals[name] = []
            self._eval_steps[name].append(step)
            self._eval_vals[name].append(float(val))
        self._refresh(force=True)

    def set_status(self, msg: str) -> None:
        self._status = msg
        self._refresh()

    def __enter__(self):
        # Rich Live takes over stdout; auto_refresh off so we control timing
        # and don't fight with our own log appends.
        self._live = Live(
            self._layout,
            console=self._console,
            refresh_per_second=0.5,
            screen=False,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)
            self._live = None
        # After Live tears down, dump the final log so the terminal still
        # shows what happened (Live's last frame is cleared otherwise on
        # some terminals).
        for line in self._log:
            self._console.print(line, highlight=False)
        return False

    # ---- internals ---------------------------------------------------

    def _build_layout(self) -> Layout:
        root = Layout(name="root")
        root.split_column(
            Layout(name="status", size=3),
            Layout(name="graphs", ratio=2),
            Layout(name="log", ratio=3),
        )
        root["graphs"].split_row(
            Layout(name="train"),
            Layout(name="eval"),
        )
        return root

    def _refresh(self, force: bool = False) -> None:
        if self._live is None:
            return
        # Coalesce per-step calls — only do the actual layout/plot work every
        # `refresh_every` ticks. log/eval pass force=True to bypass.
        self._refresh_tick += 1
        if not force and (self._refresh_tick % self._refresh_every) != 0:
            return
        w, h = self._console.size
        # Cap graph dimensions so they don't hog the terminal at large sizes.
        graph_h = min(14, max(6, (h - 5) * 2 // 5))
        graph_w = min(60, max(40, w // 2 - 4))

        self._layout["status"].update(
            Panel(Text(self._status or self._title, style="bold cyan"), title=self._title)
        )
        self._layout["train"].update(
            Panel(self._render_train_graph(graph_w, graph_h), title="train loss / lr")
        )
        self._layout["eval"].update(
            Panel(self._render_eval_graph(graph_w, graph_h), title="eval metrics")
        )
        self._layout["log"].update(self._render_log())

    def _render_train_graph(self, width: int, height: int):
        if not self._train_steps:
            return Text("(no train points yet)", style="dim")
        steps = list(self._train_steps)
        losses = [v for v in self._train_loss]
        lrs = [v for v in self._train_lr]
        plt.clf()
        plt.plot_size(width, height)
        plt.theme("pro")
        # Drop NaNs per series — a train point may carry only one of the two.
        sl = [(s, v) for s, v in zip(steps, losses) if v == v]
        if sl:
            plt.plot([s for s, _ in sl], [v for _, v in sl], label="loss", color="cyan", marker="braille")
        rl = [(s, v) for s, v in zip(steps, lrs) if v == v]
        if rl:
            # LR is on a wildly different scale than loss — put it on a yside
            # so the loss curve stays readable.
            plt.plot([s for s, _ in rl], [v for _, v in rl], label="lr", color="magenta", yside="right", marker="braille")
        plt.xlabel("step")
        try:
            ansi = plt.build()
        except Exception as e:  # pragma: no cover - plotext is occasionally fussy
            return Text(f"(plot error: {e})", style="red")
        return Text.from_ansi(ansi)

    def _render_eval_graph(self, width: int, height: int):
        active = [(n, self._eval_steps[n], self._eval_vals[n])
                  for n in self._eval_steps if self._eval_steps[n]]
        if not active:
            return Text("(no eval points yet)", style="dim")
        plt.clf()
        plt.plot_size(width, height)
        plt.theme("pro")
        colors = ["cyan", "magenta", "green", "yellow", "red", "blue", "white"]
        for i, (name, xs, ys) in enumerate(active):
            color = colors[i % len(colors)]
            if len(xs) == 1:
                plt.scatter(xs, ys, label=name, color=color, marker="braille")
            else:
                plt.plot(xs, ys, label=name, color=color, marker="braille")
        plt.xlabel("step")
        try:
            ansi = plt.build()
        except Exception as e:  # pragma: no cover
            return Text(f"(plot error: {e})", style="red")
        return Text.from_ansi(ansi)

    def _render_log(self) -> Panel:
        # Show only the tail that fits in the log panel — the panel doesn't
        # scroll on its own; rendering more lines than the height just hides
        # the most recent (most important) ones above the fold.
        _, h = self._console.size
        # status (3) + capped graph_h + log panel chrome (2) → leave the rest.
        graph_h = min(14, max(6, (h - 5) * 2 // 5))
        log_h = max(5, h - 3 - graph_h - 2)
        tail = list(self._log)[-log_h:]
        body = Text("\n".join(tail) if tail else "(no log yet)", style="white")
        return Panel(body, title=f"log ({len(self._log)} lines)")


@contextmanager
def make_reporter(
    use_tui: bool,
    eval_series: Iterable[str] = (),
    title: str = "training",
    refresh_every: int = 100,
):
    if use_tui:
        with TUIReporter(
            eval_series=eval_series, title=title, refresh_every=refresh_every,
        ) as r:
            yield r
    else:
        with Reporter() as r:
            yield r
