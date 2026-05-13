import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw
import numpy as np
import os
import sys

from predict import SymbolPredictor
from solver import evaluate_expression, evaluate_for_canvas, _find_variables


class DrawPad(tk.Frame):
    def __init__(self, master, width=800, height=220, brush_size=14, **kw):
        super().__init__(master, bg='#e94560', padx=2, pady=2, **kw)
        self.pad_width = width
        self.pad_height = height
        self.brush_size = brush_size
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.on_stroke_end = None
        self.pil_image = Image.new('L', (width, height), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.canvas = tk.Canvas(
            self, width=width, height=height,
            bg='#0f3460', cursor='crosshair', highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.create_line(
            0, height // 2, width, height // 2,
            fill='#1a2a50', width=1, dash=(4, 4), tags='guideline',
        )
        self.canvas.bind('<Button-1>', self._start)
        self.canvas.bind('<B1-Motion>', self._draw)
        self.canvas.bind('<ButtonRelease-1>', self._stop)

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
        if self.on_stroke_end:
            self.on_stroke_end()

    def clear(self):
        self.canvas.delete('ink')
        self.canvas.delete('answer')
        self.pil_image = Image.new('L', (self.pad_width, self.pad_height), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

    def is_blank(self):
        return np.array(self.pil_image).max() < 10

    def get_image(self):
        return self.pil_image

    def draw_answer(self, text, x_pos, y_pos):
        self.canvas.delete('answer')
        try:
            answer_font = tkfont.Font(family='Consolas', size=36, weight='bold')
        except Exception:
            answer_font = tkfont.Font(size=36, weight='bold')
        self.canvas.create_text(
            x_pos, y_pos, text=text,
            fill='#53d769', font=answer_font, anchor='w', tags='answer',
        )


class MathSolverApp:
    def __init__(self, root, predictor):
        self.root = root
        self.predictor = predictor
        self.root.title('Handwritten Math Expression Evaluator')
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(True, True)
        self._check_timer = None
        self._answer_shown = False
        try:
            self.title_font = tkfont.Font(family='Segoe UI', size=22, weight='bold')
            self.subtitle_font = tkfont.Font(family='Segoe UI', size=11)
            self.btn_font = tkfont.Font(family='Segoe UI', size=13, weight='bold')
            self.result_font = tkfont.Font(family='Consolas', size=20, weight='bold')
            self.detail_font = tkfont.Font(family='Consolas', size=11)
        except Exception:
            self.title_font = tkfont.Font(size=22, weight='bold')
            self.subtitle_font = tkfont.Font(size=11)
            self.btn_font = tkfont.Font(size=13, weight='bold')
            self.result_font = tkfont.Font(size=20, weight='bold')
            self.detail_font = tkfont.Font(size=11)
        self._build_ui()

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg='#1a1a2e', pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(
            hdr, text='Handwritten Math Expression Evaluator',
            font=self.title_font, fg='#e94560', bg='#1a1a2e',
        ).pack()
        tk.Label(
            hdr, text='Draw expressions like  2 + 4 =  or equations like  2x + 9 = 9',
            font=self.subtitle_font, fg='#a0a0b0', bg='#1a1a2e',
        ).pack(pady=(2, 0))

        tk.Frame(self.root, bg='#e94560', height=2).pack(fill=tk.X, padx=30, pady=(0, 10))

        canvas_frame = tk.Frame(self.root, bg='#1a1a2e')
        canvas_frame.pack(padx=20, pady=(0, 6))

        self.draw_pad = DrawPad(canvas_frame, width=800, height=220, brush_size=14)
        self.draw_pad.pack(pady=4)
        self.draw_pad.on_stroke_end = self._schedule_check

        btn_row = tk.Frame(self.root, bg='#1a1a2e')
        btn_row.pack(fill=tk.X, padx=20, pady=8)

        tk.Button(
            btn_row, text='Clear', font=self.btn_font,
            fg='white', bg='#533483', activebackground='#6a4c93',
            activeforeground='white', relief=tk.FLAT, padx=20, pady=8,
            command=self._clear_all,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_row, text='Evaluate', font=self.btn_font,
            fg='white', bg='#e94560', activebackground='#ff6b81',
            activeforeground='white', relief=tk.FLAT, padx=30, pady=8,
            command=self._evaluate,
        ).pack(side=tk.RIGHT, padx=4)

        status_frame = tk.Frame(self.root, bg='#16213e', padx=16, pady=12)
        status_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        tk.Label(
            status_frame, text='Recognised:',
            font=self.subtitle_font, fg='#a0a0b0', bg='#16213e', anchor='w',
        ).pack(fill=tk.X)
        self.expr_display = tk.Label(
            status_frame, text='...',
            font=self.result_font, fg='#7ec8e3', bg='#16213e', anchor='w',
        )
        self.expr_display.pack(fill=tk.X, pady=(2, 8))

        tk.Label(
            status_frame, text='Result:',
            font=self.subtitle_font, fg='#a0a0b0', bg='#16213e', anchor='w',
        ).pack(fill=tk.X)
        self.result_display = tk.Label(
            status_frame, text='...',
            font=self.result_font, fg='#ffd700', bg='#16213e', anchor='w',
        )
        self.result_display.pack(fill=tk.X, pady=(2, 8))

        self.confidence_display = tk.Label(
            status_frame, text='',
            font=self.detail_font, fg='#666680', bg='#16213e',
            anchor='w', wraplength=760,
        )
        self.confidence_display.pack(fill=tk.X)

    def _schedule_check(self):
        if self._check_timer:
            self.root.after_cancel(self._check_timer)
        self._check_timer = self.root.after(800, self._auto_evaluate)

    def _auto_evaluate(self):
        self._check_timer = None
        if self.draw_pad.is_blank():
            return
        try:
            expression, confidences, bboxes = self.predictor.segment_and_predict(
                self.draw_pad.get_image()
            )
        except Exception:
            return
        if not expression:
            return
        spaced = ' '.join(list(expression))
        self.expr_display.config(text=spaced)
        if confidences:
            conf_text = '  '.join(f'{s}: {c:.0%}' for s, c in confidences)
            self.confidence_display.config(text=conf_text)

        variables = _find_variables(expression)

        if expression.endswith('='):
            expr_before_eq = expression[:-1]
            if variables:
                # Equation with variable — solve it
                result = evaluate_for_canvas(expr_before_eq)
            else:
                # Pure arithmetic
                result = evaluate_for_canvas(expr_before_eq)
            self.result_display.config(text=result)
            if bboxes:
                last_bbox = bboxes[-1]
                x_after_eq = last_bbox[2] + 20
                y_center = (last_bbox[1] + last_bbox[3]) // 2
                self.draw_pad.draw_answer(str(result), x_after_eq, y_center)
                self._answer_shown = True
        elif '=' in expression and variables:
            # Equation like "2x+9=9" (= is in the middle, not trailing)
            result = evaluate_expression(expression)
            self.result_display.config(text=result)
            if bboxes:
                last_bbox = bboxes[-1]
                x_after = last_bbox[2] + 20
                y_center = (last_bbox[1] + last_bbox[3]) // 2
                self.draw_pad.draw_answer(str(result), x_after, y_center)
                self._answer_shown = True
        else:
            if variables:
                self.result_display.config(text='draw = to solve')
            else:
                self.result_display.config(text='draw = to get answer')
            if self._answer_shown:
                self.draw_pad.canvas.delete('answer')
                self._answer_shown = False

    def _evaluate(self):
        self._auto_evaluate()

    def _clear_all(self):
        self.draw_pad.clear()
        self.expr_display.config(text='...')
        self.result_display.config(text='...')
        self.confidence_display.config(text='')
        self._answer_shown = False


def create_app(model_path='symbol_cnn.pth'):
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        print("Please run 'python train.py' first to train the model.")
        sys.exit(1)
    predictor = SymbolPredictor(model_path)
    root = tk.Tk()
    root.app = MathSolverApp(root, predictor)
    return root
