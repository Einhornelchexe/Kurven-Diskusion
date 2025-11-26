import tkinter as tk
from tkinter import messagebox, scrolledtext
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import (E, S, Derivative, Symbol, diff, lambdify, pi, simplify,
                   solveset, sympify)
import sympy as sp
from sympy.calculus.util import continuous_domain
from sympy.core.expr import Expr


# Tkinter based GUI for automated curve sketching
x = Symbol("x")

ALLOWED_NAMES = {
    "x": x,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "E": E,
    "pi": pi,
}


def _format_value(val: Expr, digits: int = 4) -> str:
    try:
        numeric = float(val)
        return f"{numeric:.{digits}g}"
    except Exception:
        return str(val)


def _real_solutions(expr: Expr) -> List[Expr]:
    solutions = []
    try:
        for sol in solveset(expr, x, domain=S.Reals):
            solutions.append(sol)
    except Exception:
        pass
    return solutions


def _classify_extrema(f: Expr, f1: Expr, f2: Expr, points: List[Expr]) -> Tuple[List[Tuple[Expr, Expr]], List[Tuple[Expr, Expr]], List[Tuple[Expr, Expr]]]:
    maxima, minima, saddle = [], [], []
    for p in points:
        f2_val = f2.subs(x, p)
        y_val = f.subs(x, p)
        if f2_val.is_real:
            if f2_val.is_positive:
                minima.append((p, y_val))
            elif f2_val.is_negative:
                maxima.append((p, y_val))
            else:
                saddle.append((p, y_val))
        else:
            saddle.append((p, y_val))
    return maxima, minima, saddle


def _find_inflection_points(f: Expr, f2: Expr) -> List[Tuple[Expr, Expr]]:
    points = []
    try:
        for p in _real_solutions(f2):
            third = diff(f, x, 3).subs(x, p)
            if third != 0:
                points.append((p, f.subs(x, p)))
    except Exception:
        pass
    return points


class CurveApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Kurvendiskussion")
        self.geometry("1000x750")
        self._build_ui()

    def _build_ui(self) -> None:
        input_frame = tk.Frame(self, padx=10, pady=10)
        input_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(input_frame, text="Funktion f(x) =").grid(row=0, column=0, sticky="w")
        self.function_entry = tk.Entry(input_frame, width=50)
        self.function_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        self.function_entry.insert(0, "x**3 - 3*x")

        tk.Label(input_frame, text="x-Min").grid(row=1, column=0, sticky="w")
        self.xmin_entry = tk.Entry(input_frame, width=10)
        self.xmin_entry.insert(0, "-5")
        self.xmin_entry.grid(row=1, column=1, sticky="w", padx=5)

        tk.Label(input_frame, text="x-Max").grid(row=2, column=0, sticky="w")
        self.xmax_entry = tk.Entry(input_frame, width=10)
        self.xmax_entry.insert(0, "5")
        self.xmax_entry.grid(row=2, column=1, sticky="w", padx=5)

        button_frame = tk.Frame(input_frame)
        button_frame.grid(row=0, column=2, rowspan=3, padx=10, sticky="ns")

        tk.Button(button_frame, text="Analyse starten", command=self.run_analysis).pack(fill=tk.X, pady=2)
        tk.Button(button_frame, text="Eingabe-Hilfe", command=self.show_help).pack(fill=tk.X, pady=2)

        input_frame.columnconfigure(1, weight=1)

        content_frame = tk.Frame(self)
        content_frame.pack(fill=tk.BOTH, expand=True)

        plot_frame = tk.Frame(content_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.figure, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        result_frame = tk.Frame(content_frame, padx=10, pady=10)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(result_frame, text="Ergebnisse", font=("Arial", 12, "bold")).pack(anchor="w")
        self.result_box = scrolledtext.ScrolledText(result_frame, width=40, height=30, state=tk.DISABLED)
        self.result_box.pack(fill=tk.BOTH, expand=True, pady=5)

    def show_help(self) -> None:
        help_text = (
            "So gibst du Funktionen ein:\n"
            "- Potenzen: x**2 für x², x**5 für x^5\n"
            "- E-Funktion: exp(x) oder E**x\n"
            "- Pi: pi\n"
            "- Wurzeln: sqrt(x)\n"
            "- Trigonometrie: sin(x), cos(x), tan(x)\n"
            "- Beträge: abs(x)\n"
            "Verwende x als Variable. Beispiele: x**3-3*x, sin(x)+0.5\n"
        )
        messagebox.showinfo("Eingabe-Hilfe", help_text)

    def run_analysis(self) -> None:
        func_text = self.function_entry.get().strip()
        try:
            xmin = float(self.xmin_entry.get())
            xmax = float(self.xmax_entry.get())
            if xmin >= xmax:
                raise ValueError("x-Min muss kleiner als x-Max sein.")
        except Exception as exc:
            messagebox.showerror("Fehler", f"Ungültiger Bereich: {exc}")
            return

        try:
            f_expr = sympify(func_text, locals=ALLOWED_NAMES)
        except Exception as exc:
            messagebox.showerror("Fehler", f"Funktion konnte nicht verstanden werden: {exc}")
            return

        domain = continuous_domain(f_expr, x, S.Reals)
        if not domain:
            messagebox.showerror("Fehler", "Die Funktion ist nicht auf den reellen Zahlen definiert.")
            return

        f1 = simplify(Derivative(f_expr, x).doit())
        f2 = simplify(Derivative(f1, x).doit())

        zeros = _real_solutions(f_expr)
        crit_points = _real_solutions(f1)
        inflections = _find_inflection_points(f_expr, f2)
        maxima, minima, saddle = _classify_extrema(f_expr, f1, f2, crit_points)

        self._update_results(f_expr, f1, f2, zeros, maxima, minima, saddle, inflections)
        self._update_plot(f_expr, zeros, maxima, minima, saddle, inflections, xmin, xmax)

    def _update_results(
        self,
        f: Expr,
        f1: Expr,
        f2: Expr,
        zeros: List[Expr],
        maxima: List[Tuple[Expr, Expr]],
        minima: List[Tuple[Expr, Expr]],
        saddle: List[Tuple[Expr, Expr]],
        inflections: List[Tuple[Expr, Expr]],
    ) -> None:
        self.result_box.configure(state=tk.NORMAL)
        self.result_box.delete("1.0", tk.END)

        def add_line(text: str) -> None:
            self.result_box.insert(tk.END, text + "\n")

        add_line(f"f(x) = {f}")
        add_line(f"f'(x) = {f1}")
        add_line(f"f''(x) = {f2}\n")

        add_line("Nullstellen:")
        if zeros:
            for z in zeros:
                add_line(f"  x = {_format_value(z)}")
        else:
            add_line("  keine reellen Nullstellen gefunden")

        add_line("\nExtrempunkte:")
        if maxima:
            add_line("  Maxima:")
            for x_val, y_val in maxima:
                add_line(f"    ({_format_value(x_val)}, {_format_value(y_val)})")
        if minima:
            add_line("  Minima:")
            for x_val, y_val in minima:
                add_line(f"    ({_format_value(x_val)}, {_format_value(y_val)})")
        if saddle:
            add_line("  Sattelpunkte (f' = f'' = 0):")
            for x_val, y_val in saddle:
                add_line(f"    ({_format_value(x_val)}, {_format_value(y_val)})")
        if not (maxima or minima or saddle):
            add_line("  keine Extrempunkte gefunden")

        add_line("\nWendepunkte (f''=0, f'''≠0):")
        if inflections:
            for x_val, y_val in inflections:
                add_line(f"  ({_format_value(x_val)}, {_format_value(y_val)})")
        else:
            add_line("  keine Wendepunkte gefunden")

        self.result_box.configure(state=tk.DISABLED)

    def _update_plot(
        self,
        f: Expr,
        zeros: List[Expr],
        maxima: List[Tuple[Expr, Expr]],
        minima: List[Tuple[Expr, Expr]],
        saddle: List[Tuple[Expr, Expr]],
        inflections: List[Tuple[Expr, Expr]],
        xmin: float,
        xmax: float,
    ) -> None:
        self.ax.clear()

        f_lambdified = lambdify(x, f, modules=["numpy"])
        xs = np.linspace(xmin, xmax, 400)
        try:
            ys = f_lambdified(xs)
        except Exception:
            messagebox.showerror("Fehler", "Funktion konnte nicht geplottet werden.")
            return

        self.ax.plot(xs, ys, label="f(x)")

        def scatter_points(points: List[Tuple[Expr, Expr]], color: str, label: str) -> None:
            if not points:
                return
            px = [float(p[0]) for p in points]
            py = [float(p[1]) for p in points]
            self.ax.scatter(px, py, color=color, label=label)

        if zeros:
            zx = [float(z) for z in zeros]
            self.ax.scatter(zx, [0] * len(zx), color="black", marker="x", label="Nullstellen")

        scatter_points(maxima, "red", "Maxima")
        scatter_points(minima, "blue", "Minima")
        scatter_points(saddle, "purple", "Sattelpunkte")
        scatter_points(inflections, "orange", "Wendepunkte")

        self.ax.axhline(0, color="gray", linewidth=0.8)
        self.ax.axvline(0, color="gray", linewidth=0.8)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.legend()
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.canvas.draw()


def main() -> None:
    app = CurveApp()
    app.mainloop()


if __name__ == "__main__":
    main()
