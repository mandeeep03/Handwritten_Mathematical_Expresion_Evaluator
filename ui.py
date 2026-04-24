import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw
import numpy as np
import os
import sys

from predict import SymbolPredictor
from solver import evaluate_expression


# ─────────────────────────────────────────────────────────
#  Reusable mini-canvas for handwriting input
# ─────────────────────────────────────────────────────────
class DrawPad(tk.Frame):
    """A self-contained drawing surface backed by a PIL image."""

    def __init__(self, master, width=380, height=160, brush_size=12, **kw):
        super().__init__(master, bg='#e94560', padx=2, pady=2, **kw)

        self.pad_width = width
        self.pad_height = height
        self.brush_size = brush_size
        self.drawing = False
        self.last_x = None
        self.last_y = None

        self.pil_image = Image.new('L', (width, height), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self.canvas = tk.Canvas(
            self, width=width, height=height,
            bg='#0f3460', cursor='crosshair', highlightthickness=0,
        )
        self.canvas.pack()

        # centre guideline
        self.canvas.create_line(
            0, height // 2, width, height // 2,
            fill='#1a2a50', width=1, dash=(4, 4), tags='guideline',
        )

        self.canvas.bind('<Button-1>', self._start)
        self.canvas.bind('<B1-Motion>', self._draw)
        self.canvas.bind('<ButtonRelease-1>', self._stop)

    # ── drawing callbacks ───────────────────────────────
    def _start(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def _draw(self, event):
        if not self.drawing:
            return
        x, y = event.x, event.y
        r = self.brush_size // 2
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill='white', outline='white', tags='ink',
        )
        if self.last_x is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                fill='white', width=self.brush_size,
                capstyle=tk.ROUND, smooth=True, tags='ink',
            )
            self.pil_draw.line(
                [(self.last_x, self.last_y), (x, y)],
                fill=255, width=self.brush_size,
            )
        self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self.last_x, self.last_y = x, y

    def _stop(self, _event):
        self.drawing = False
        self.last_x = self.last_y = None

    # ── public helpers ──────────────────────────────────
    def clear(self):
        self.canvas.delete('ink')
        self.canvas.delete('answer')
        self.pil_image = Image.new('L', (self.pad_width, self.pad_height), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

    def is_blank(self):
        return np.array(self.pil_image).max() < 10

    def get_image(self):
        return self.pil_image


# ─────────────────────────────────────────────────────────
#  Labelled panel   (title + draw-pad + recognised text)
# ─────────────────────────────────────────────────────────
class LabelledPad(tk.Frame):
    """Title ➜ DrawPad ➜ status/recognised label, with Clear button."""

    def __init__(self, master, title, predictor, pad_w=380, pad_h=160, **kw):
        super().__init__(master, bg='#1a1a2e', **kw)
        self.predictor = predictor

        try:
            self._title_font = tkfont.Font(family='Segoe UI', size=11, weight='bold')
            self._status_font = tkfont.Font(family='Consolas', size=11)
            self._btn_font = tkfont.Font(family='Segoe UI', size=9, weight='bold')
        except Exception:
            self._title_font = tkfont.Font(size=11, weight='bold')
            self._status_font = tkfont.Font(size=11)
            self._btn_font = tkfont.Font(size=9, weight='bold')

        # title row with clear button
        top = tk.Frame(self, bg='#1a1a2e')
        top.pack(fill=tk.X, padx=4, pady=(6, 2))
        tk.Label(top, text=title, font=self._title_font,
                 fg='#e94560', bg='#1a1a2e').pack(side=tk.LEFT)
        tk.Button(
            top, text='✕ Clear', font=self._btn_font,
            fg='white', bg='#533483', activebackground='#6a4c93',
            activeforeground='white', relief=tk.FLAT, padx=8, pady=2,
            command=self._clear,
        ).pack(side=tk.RIGHT)

        # draw pad
        self.pad = DrawPad(self, width=pad_w, height=pad_h, brush_size=12)
        self.pad.pack(padx=4, pady=4)

        # recognised expression label
        self.status_var = tk.StringVar(value='...')
        tk.Label(
            self, textvariable=self.status_var, font=self._status_font,
            fg='#53d769', bg='#16213e', anchor='w', padx=8, pady=4,
        ).pack(fill=tk.X, padx=4, pady=(0, 6))

    def _clear(self):
        self.pad.clear()
        self.status_var.set('...')

    def recognise(self):
        """Run prediction on the pad contents.  Returns (expr_str, confidences, bboxes) or None."""
        if self.pad.is_blank():
            self.status_var.set('(empty)')
            return None
        try:
            expr, confs, bboxes = self.predictor.segment_and_predict(self.pad.get_image())
        except Exception as e:
            self.status_var.set(f'Error: {e}')
            return None
        if not expr:
            self.status_var.set('Could not recognise')
            return None
        spaced = ' '.join(list(expr))
        self.status_var.set(spaced)
        return expr, confs, bboxes


# ─────────────────────────────────────────────────────────
#  Main application window
# ─────────────────────────────────────────────────────────
class MathSolverApp:
    VAR_NAMES = ['a', 'b', 'c']  # three variable slots

    def __init__(self, root, predictor):
        self.root = root
        self.predictor = predictor
        self.root.title('Handwritten Math Solver')
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(True, True)

        try:
            self.title_font = tkfont.Font(family='Segoe UI', size=20, weight='bold')
            self.subtitle_font = tkfont.Font(family='Segoe UI', size=11)
            self.btn_font = tkfont.Font(family='Segoe UI', size=13, weight='bold')
            self.result_title_font = tkfont.Font(family='Segoe UI', size=12, weight='bold')
            self.result_font = tkfont.Font(family='Consolas', size=16, weight='bold')
            self.detail_font = tkfont.Font(family='Consolas', size=11)
        except Exception:
            self.title_font = tkfont.Font(size=20, weight='bold')
            self.subtitle_font = tkfont.Font(size=11)
            self.btn_font = tkfont.Font(size=13, weight='bold')
            self.result_title_font = tkfont.Font(size=12, weight='bold')
            self.result_font = tkfont.Font(size=16, weight='bold')
            self.detail_font = tkfont.Font(size=11)

        self._build_ui()

    # ── layout ──────────────────────────────────────────
    def _build_ui(self):
        # ---------- header ----------
        hdr = tk.Frame(self.root, bg='#1a1a2e', pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text='✦ Handwritten Math Solver',
                 font=self.title_font, fg='#e94560', bg='#1a1a2e').pack()
        tk.Label(hdr, text='Assign variables  ▸  Write an equation  ▸  Evaluate',
                 font=self.subtitle_font, fg='#a0a0b0', bg='#1a1a2e').pack(pady=(2, 0))

        # ---------- separator ----------
        tk.Frame(self.root, bg='#e94560', height=2).pack(fill=tk.X, padx=30, pady=(0, 10))

        # ---------- body ----------
        body = tk.Frame(self.root, bg='#1a1a2e')
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 6))

        # ── left column: 3 variable pads ────────────────
        left = tk.Frame(body, bg='#1a1a2e')
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 8))

        section_lbl = tk.Label(left, text='📝  Variable Assignments',
                               font=self.subtitle_font, fg='#a0a0b0', bg='#1a1a2e')
        section_lbl.pack(anchor='w', pady=(0, 4))

        self.var_pads = []
        for i, vname in enumerate(self.VAR_NAMES):
            lp = LabelledPad(left, title=f'Variable {vname.upper()}   (e.g.  {vname} = 5)',
                             predictor=self.predictor, pad_w=340, pad_h=120)
            lp.pack(fill=tk.X, pady=3)
            self.var_pads.append(lp)

        # ── right column: equation + output ─────────────
        right = tk.Frame(body, bg='#1a1a2e')
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        eq_lbl = tk.Label(right, text='📐  Equation',
                          font=self.subtitle_font, fg='#a0a0b0', bg='#1a1a2e')
        eq_lbl.pack(anchor='w', pady=(0, 4))

        self.eq_pad = LabelledPad(right, title='Equation   (e.g.  a + b - c)',
                                  predictor=self.predictor, pad_w=420, pad_h=180)
        self.eq_pad.pack(fill=tk.X, pady=3)

        # ── buttons ─────────────────────────────────────
        btn_row = tk.Frame(right, bg='#1a1a2e')
        btn_row.pack(fill=tk.X, pady=8)

        tk.Button(
            btn_row, text='🗑  Clear All', font=self.btn_font,
            fg='white', bg='#533483', activebackground='#6a4c93',
            activeforeground='white', relief=tk.FLAT, padx=20, pady=8,
            command=self._clear_all,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_row, text='⚡  Evaluate', font=self.btn_font,
            fg='white', bg='#e94560', activebackground='#ff6b81',
            activeforeground='white', relief=tk.FLAT, padx=30, pady=8,
            command=self._evaluate,
        ).pack(side=tk.RIGHT, padx=4)

        # ── output panel ────────────────────────────────
        out_lbl = tk.Label(right, text='📊  Output',
                           font=self.subtitle_font, fg='#a0a0b0', bg='#1a1a2e')
        out_lbl.pack(anchor='w', pady=(8, 4))

        out_frame = tk.Frame(right, bg='#16213e', padx=16, pady=14)
        out_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        # variables summary
        tk.Label(out_frame, text='Variables:', font=self.result_title_font,
                 fg='#a0a0b0', bg='#16213e', anchor='w').pack(fill=tk.X)
        self.vars_display = tk.Label(
            out_frame, text='...', font=self.detail_font,
            fg='#53d769', bg='#16213e', anchor='w', wraplength=400, justify='left',
        )
        self.vars_display.pack(fill=tk.X, pady=(2, 10))

        # equation recognised
        tk.Label(out_frame, text='Recognised Equation:', font=self.result_title_font,
                 fg='#a0a0b0', bg='#16213e', anchor='w').pack(fill=tk.X)
        self.eq_display = tk.Label(
            out_frame, text='...', font=self.detail_font,
            fg='#7ec8e3', bg='#16213e', anchor='w',
        )
        self.eq_display.pack(fill=tk.X, pady=(2, 10))

        # substituted form
        tk.Label(out_frame, text='Substituted:', font=self.result_title_font,
                 fg='#a0a0b0', bg='#16213e', anchor='w').pack(fill=tk.X)
        self.sub_display = tk.Label(
            out_frame, text='...', font=self.detail_font,
            fg='#c89bff', bg='#16213e', anchor='w',
        )
        self.sub_display.pack(fill=tk.X, pady=(2, 10))

        # final result
        tk.Label(out_frame, text='Result:', font=self.result_title_font,
                 fg='#a0a0b0', bg='#16213e', anchor='w').pack(fill=tk.X)
        self.result_display = tk.Label(
            out_frame, text='...', font=self.result_font,
            fg='#ffd700', bg='#16213e', anchor='w',
        )
        self.result_display.pack(fill=tk.X, pady=(2, 0))

        # ── confidence footer ───────────────────────────
        self.confidence_display = tk.Label(
            right, text='', font=self.detail_font,
            fg='#666680', bg='#1a1a2e', anchor='w', wraplength=420,
        )
        self.confidence_display.pack(fill=tk.X, padx=4)

    # ── actions ─────────────────────────────────────────
    def _clear_all(self):
        for vp in self.var_pads:
            vp._clear()
        self.eq_pad._clear()
        self.vars_display.config(text='...')
        self.eq_display.config(text='...')
        self.sub_display.config(text='...')
        self.result_display.config(text='...')
        self.confidence_display.config(text='')

    def _evaluate(self):
        # 1. Recognise variables ─────────────────────────
        var_values = {}       # e.g. {'a': '5', 'b': '3', 'c': '2'}
        all_confs = []
        var_lines = []

        for vpad, vname in zip(self.var_pads, self.VAR_NAMES):
            result = vpad.recognise()
            if result is None:
                continue
            expr, confs, _bboxes = result
            all_confs.extend(confs)

            # Expect something like "a=5" or just a number "5"
            # Try to extract the numeric value
            cleaned = expr.replace(' ', '')
            if '=' in cleaned:
                parts = cleaned.split('=', 1)
                value_part = parts[1]  # after the =
            else:
                value_part = cleaned   # treat entire drawing as value

            var_values[vname] = value_part
            var_lines.append(f'{vname} = {value_part}')

        if var_lines:
            self.vars_display.config(text='    '.join(var_lines))
        else:
            self.vars_display.config(text='(no variables assigned)')

        # 2. Recognise equation ──────────────────────────
        eq_result = self.eq_pad.recognise()
        if eq_result is None:
            self.eq_display.config(text='(empty)')
            self.sub_display.config(text='...')
            self.result_display.config(text='Draw an equation first')
            return

        eq_expr, eq_confs, _bboxes = eq_result
        all_confs.extend(eq_confs)

        eq_display_str = ' '.join(list(eq_expr))
        self.eq_display.config(text=eq_display_str)

        # 3. Substitute variables ────────────────────────
        substituted = eq_expr
        for vname, vval in var_values.items():
            substituted = substituted.replace(vname, f'({vval})')

        sub_display_str = ' '.join(list(substituted))
        self.sub_display.config(text=sub_display_str)

        # 4. Evaluate ────────────────────────────────────
        result = evaluate_expression(substituted)
        self.result_display.config(text=result)

        # 5. Confidence details ──────────────────────────
        if all_confs:
            conf_text = ' | '.join(f'{s}: {c:.0%}' for s, c in all_confs)
            self.confidence_display.config(text=conf_text)
        else:
            self.confidence_display.config(text='')


def create_app(model_path='symbol_cnn.pth'):
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        print("Please run 'python train.py' first to train the model.")
        sys.exit(1)

    predictor = SymbolPredictor(model_path)
    root = tk.Tk()
    app = MathSolverApp(root, predictor)
    return root
