import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import sys

from predict import SymbolPredictor
from solver import evaluate_expression


class DrawingCanvas:
    def __init__(self, root, predictor):
        self.root = root
        self.predictor = predictor
        self.root.title("Handwritten Math Solver")
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(True, True)

        self.canvas_width = 900
        self.canvas_height = 350
        self.brush_size = 14
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.answer_items = []

        self.pil_image = Image.new('L', (self.canvas_width, self.canvas_height), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self._setup_ui()

    def _setup_ui(self):
        try:
            title_font = tkfont.Font(family='Segoe UI', size=18, weight='bold')
            label_font = tkfont.Font(family='Segoe UI', size=11)
            button_font = tkfont.Font(family='Segoe UI', size=11, weight='bold')
            result_font = tkfont.Font(family='Consolas', size=14, weight='bold')
            self.canvas_answer_font = tkfont.Font(family='Segoe UI', size=36, weight='bold')
        except Exception:
            title_font = tkfont.Font(size=18, weight='bold')
            label_font = tkfont.Font(size=11)
            button_font = tkfont.Font(size=11, weight='bold')
            result_font = tkfont.Font(size=14, weight='bold')
            self.canvas_answer_font = tkfont.Font(size=36, weight='bold')

        header_frame = tk.Frame(self.root, bg='#1a1a2e', pady=12)
        header_frame.pack(fill=tk.X)

        title_label = tk.Label(
            header_frame,
            text="Handwritten Equation Solver",
            font=title_font,
            fg='#e94560',
            bg='#1a1a2e'
        )
        title_label.pack()

        subtitle_label = tk.Label(
            header_frame,
            text="Draw an equation and press Evaluate",
            font=label_font,
            fg='#a0a0b0',
            bg='#1a1a2e'
        )
        subtitle_label.pack(pady=(2, 0))

        canvas_frame = tk.Frame(self.root, bg='#e94560', padx=2, pady=2)
        canvas_frame.pack(padx=20, pady=10)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg='#0f3460',
            cursor='crosshair',
            highlightthickness=0
        )
        self.canvas.pack()

        self.canvas.create_line(
            0, self.canvas_height // 2, self.canvas_width, self.canvas_height // 2,
            fill='#1a2a50', width=1, dash=(4, 4), tags='guideline'
        )

        self.canvas.bind('<Button-1>', self._start_draw)
        self.canvas.bind('<B1-Motion>', self._draw)
        self.canvas.bind('<ButtonRelease-1>', self._stop_draw)

        controls_frame = tk.Frame(self.root, bg='#1a1a2e', pady=8)
        controls_frame.pack(fill=tk.X, padx=20)

        brush_frame = tk.Frame(controls_frame, bg='#1a1a2e')
        brush_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(
            brush_frame,
            text="Brush Size:",
            font=label_font,
            fg='#a0a0b0',
            bg='#1a1a2e'
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.brush_slider = tk.Scale(
            brush_frame,
            from_=6,
            to=28,
            orient=tk.HORIZONTAL,
            length=120,
            bg='#16213e',
            fg='#e94560',
            highlightthickness=0,
            troughcolor='#0f3460',
            command=self._update_brush_size
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT)

        btn_frame = tk.Frame(controls_frame, bg='#1a1a2e')
        btn_frame.pack(side=tk.RIGHT, padx=10)

        self.clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            font=button_font,
            fg='white',
            bg='#533483',
            activebackground='#6a4c93',
            activeforeground='white',
            relief=tk.FLAT,
            padx=25,
            pady=8,
            command=self._clear_canvas
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.eval_btn = tk.Button(
            btn_frame,
            text="Evaluate",
            font=button_font,
            fg='white',
            bg='#e94560',
            activebackground='#ff6b81',
            activeforeground='white',
            relief=tk.FLAT,
            padx=25,
            pady=8,
            command=self._evaluate
        )
        self.eval_btn.pack(side=tk.LEFT, padx=5)

        result_frame = tk.Frame(self.root, bg='#16213e', pady=12, padx=20)
        result_frame.pack(fill=tk.X, padx=20, pady=8)

        tk.Label(
            result_frame,
            text="Recognized:",
            font=label_font,
            fg='#a0a0b0',
            bg='#16213e',
            anchor='w'
        ).pack(fill=tk.X)

        self.recognized_label = tk.Label(
            result_frame,
            text="...",
            font=result_font,
            fg='#53d769',
            bg='#16213e',
            anchor='w'
        )
        self.recognized_label.pack(fill=tk.X, pady=(2, 8))

        tk.Label(
            result_frame,
            text="Result:",
            font=label_font,
            fg='#a0a0b0',
            bg='#16213e',
            anchor='w'
        ).pack(fill=tk.X)

        self.result_label = tk.Label(
            result_frame,
            text="...",
            font=result_font,
            fg='#ffd700',
            bg='#16213e',
            anchor='w'
        )
        self.result_label.pack(fill=tk.X, pady=(2, 0))

        confidence_frame = tk.Frame(self.root, bg='#1a1a2e', pady=5)
        confidence_frame.pack(fill=tk.X, padx=20)

        self.confidence_label = tk.Label(
            confidence_frame,
            text="",
            font=label_font,
            fg='#a0a0b0',
            bg='#1a1a2e',
            anchor='w',
            wraplength=860
        )
        self.confidence_label.pack(fill=tk.X)

    def _update_brush_size(self, val):
        self.brush_size = int(val)

    def _start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        self._clear_answer()

    def _draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            r = self.brush_size // 2
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill='white', outline='white', tags='drawing'
            )
            if self.last_x is not None:
                self.canvas.create_line(
                    self.last_x, self.last_y, x, y,
                    fill='white', width=self.brush_size,
                    capstyle=tk.ROUND, smooth=True, tags='drawing'
                )
                self.pil_draw.line(
                    [(self.last_x, self.last_y), (x, y)],
                    fill=255, width=self.brush_size
                )
            self.pil_draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=255
            )
            self.last_x = x
            self.last_y = y

    def _stop_draw(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def _clear_answer(self):
        for item_id in self.answer_items:
            self.canvas.delete(item_id)
        self.answer_items = []

    def _draw_answer_on_canvas(self, answer_text, bboxes):
        self._clear_answer()

        if not bboxes:
            draw_x = self.canvas_width // 2
            draw_y = self.canvas_height // 2
        else:
            last_bbox = bboxes[-1]
            draw_x = last_bbox[2] + 25
            all_tops = [b[1] for b in bboxes]
            all_bottoms = [b[3] for b in bboxes]
            draw_y = (min(all_tops) + max(all_bottoms)) // 2

        if draw_x > self.canvas_width - 30:
            draw_x = self.canvas_width - 150

        answer_display = answer_text

        bg_item = self.canvas.create_rectangle(
            draw_x - 5, draw_y - 30,
            draw_x + len(answer_display) * 22 + 10, draw_y + 30,
            fill='#0a2a4a', outline='#e94560', width=2,
            tags='answer'
        )
        self.answer_items.append(bg_item)

        text_item = self.canvas.create_text(
            draw_x + 5, draw_y,
            text=answer_display,
            fill='#53d769',
            font=self.canvas_answer_font,
            anchor='w',
            tags='answer'
        )
        self.answer_items.append(text_item)

        text_bbox = self.canvas.bbox(text_item)
        if text_bbox:
            pad = 8
            self.canvas.coords(
                bg_item,
                text_bbox[0] - pad, text_bbox[1] - pad,
                text_bbox[2] + pad, text_bbox[3] + pad
            )

    def _clear_canvas(self):
        self.canvas.delete('all')
        self.answer_items = []
        self.pil_image = Image.new('L', (self.canvas_width, self.canvas_height), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.recognized_label.config(text="...")
        self.result_label.config(text="...")
        self.confidence_label.config(text="")
        self.canvas.create_line(
            0, self.canvas_height // 2, self.canvas_width, self.canvas_height // 2,
            fill='#1a2a50', width=1, dash=(4, 4), tags='guideline'
        )

    def _evaluate(self):
        img_array = np.array(self.pil_image)

        if img_array.max() < 10:
            self.recognized_label.config(text="Nothing drawn")
            self.result_label.config(text="Please draw an equation")
            return

        try:
            expression, confidences, bboxes = self.predictor.segment_and_predict(self.pil_image)
        except Exception as e:
            self.recognized_label.config(text="Segmentation error")
            self.result_label.config(text=str(e))
            return

        if not expression:
            self.recognized_label.config(text="Could not recognize symbols")
            self.result_label.config(text="Try drawing more clearly with spacing")
            return

        spaced = " ".join(list(expression))
        self.recognized_label.config(text=spaced)

        conf_text = " | ".join([f"{sym}: {conf:.0%}" for sym, conf in confidences])
        self.confidence_label.config(text=conf_text)

        result = evaluate_expression(expression)
        self.result_label.config(text=result)

        answer_value = self._extract_answer_value(result)
        if answer_value is not None:
            self._draw_answer_on_canvas(str(answer_value), bboxes)

    def _extract_answer_value(self, result_string):
        if result_string.startswith("= "):
            return result_string[2:]
        if result_string.startswith("x = "):
            return result_string
        if "True" in result_string or "False" in result_string:
            return result_string.split("(")[0].strip()
        return None


def create_app(model_path='symbol_cnn.pth'):
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        print("Please run 'python train.py' first to train the model.")
        sys.exit(1)

    predictor = SymbolPredictor(model_path)
    root = tk.Tk()
    app = DrawingCanvas(root, predictor)
    return root
