# Handwritten Mathematical Expression Evaluator

A Python application that recognizes handwritten mathematical expressions and algebraic equations from a drawing canvas and computes the result using a CNN model built with PyTorch.

## What It Does

- Provides a **single drawing canvas** where you write a full math expression or equation
- Recognises **digits (0–9)**, **operators (+, -, \*, /, =)**, **parentheses**, and **letters (a–z)** using a trained CNN
- **Solves linear equations** with one variable (e.g. `2x + 9 = 9` → `x = 0`)
- When you draw `=` at the end of arithmetic, the answer **appears automatically on the canvas**
- Segments drawn symbols via connected-component analysis
- Evaluates arithmetic and algebraic expressions using SymPy

## How It Works

1. **Draw** a math expression or equation on the canvas (e.g. `2 + 4 =` or `2x + 9 = 9`)
2. After you finish drawing, the app **automatically recognises** all symbols
3. The **answer appears in green** directly on the canvas
4. The status panel below shows the recognised expression, result, and per-symbol confidence

## UI Layout

```
+---------------------------------------------------+
|     Handwritten Math Expression Evaluator          |
+---------------------------------------------------+
|                                                    |
|   +-----------------------------------------+     |
|   |                                         |     |
|   |    [ you draw:  2x + 9 = 9 ]  x = 0    |     |
|   |                                ^green   |     |
|   +-----------------------------------------+     |
|                                                    |
|   [ Clear ]                      [ Evaluate ]     |
|                                                    |
|   Recognised:  2 x + 9 = 9                        |
|   Result:  x = 0                                   |
|   Confidence:  2: 98%  x: 95%  +: 97%  ...        |
+---------------------------------------------------+
```

## Supported Expressions

| Expression     | What You Draw   | Answer on Canvas  |
|----------------|-----------------|-------------------|
| Addition       | `2 + 4 =`       | `6`               |
| Subtraction    | `9 - 3 =`       | `6`               |
| Multi-digit    | `12 + 35 =`     | `47`              |
| Chained        | `5 + 3 - 2 =`   | `6`               |
| Linear eq.     | `2x + 9 = 9`    | `x = 0`           |
| Linear eq.     | `3x - 6 = 0`    | `x = 2`           |
| Linear eq.     | `x + 5 = 10`    | `x = 5`           |
| With parens    | `2(x+3) = 10`   | `x = 2`           |

## Recognised Symbols (43 classes)

| Category   | Symbols                       |
|------------|-------------------------------|
| Digits     | `0 1 2 3 4 5 6 7 8 9`        |
| Operators  | `+ - * / ( ) =`              |
| Letters    | `a b c d e ... x y z`         |

## Project Structure

```
model.py      CNN architecture (ResNet-style with SE attention, 43 classes)
train.py      Training script (EMNIST digits + letters + synthetic operators)
predict.py    Symbol segmentation and prediction pipeline with TTA
solver.py     Arithmetic + linear equation solver (SymPy)
ui.py         Tkinter single-canvas interface with auto-evaluation
main.py       Application entry point
```

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

1. **Train the model:**

```bash
python train.py
```

This downloads EMNIST (digits + letters), generates synthetic operator data from system fonts, and trains the CNN for 50 epochs with mixup augmentation and class-balanced sampling. The best model is saved as `symbol_cnn.pth`.

2. **Launch the application:**

```bash
python main.py
```

3. **Use the interface:**
   - Draw a math expression on the canvas (e.g. write `4 + 5 =`)
   - Or draw an equation like `2x + 9 = 9` to solve for x
   - The answer appears automatically in green after the `=` sign
   - Press **Clear** to reset the canvas
   - Press **Evaluate** to manually trigger recognition

## Model Details

- **Architecture**: ResNet-style CNN with Squeeze-and-Excitation attention blocks
- **Training data**: EMNIST digits (60k), EMNIST letters (104k), synthetic operators (35k)
- **Augmentation**: Mixup, random affine, perspective, Gaussian blur, random erasing
- **Optimiser**: AdamW with cosine annealing, label smoothing, gradient clipping
- **Test-time**: TTA (4 augmented views averaged) for maximum accuracy

## Requirements

- Python 3.8+
- PyTorch
- Pillow
- NumPy
- SymPy
- Tkinter (included with Python)
