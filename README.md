# Handwritten Mathematical Expression Evaluator

A Python application that recognizes handwritten mathematical expressions from a drawing canvas and computes the result using a CNN model built with PyTorch.

## What It Does

- Provides a **single drawing canvas** where you write a full math expression
- Recognises digits (0–9) and operators (+, -, =) using a trained CNN
- When you draw `=` at the end, the answer **appears automatically on the canvas** right after the equals sign
- Segments drawn symbols via connected-component analysis
- Evaluates arithmetic expressions using SymPy

## How It Works

1. **Draw** a math expression on the canvas (e.g. `2 + 4 =`)
2. After you finish drawing the `=` sign, the app **automatically recognises** all symbols
3. The **answer appears in green** directly on the canvas after the `=`
4. The status panel below shows the recognised expression, result, and per-symbol confidence

## UI Layout

```
+---------------------------------------------------+
|     Handwritten Math Expression Evaluator          |
+---------------------------------------------------+
|                                                    |
|   +-----------------------------------------+     |
|   |                                         |     |
|   |    [ you draw:  2 + 4 = ]   9           |     |
|   |                             ^green      |     |
|   +-----------------------------------------+     |
|                                                    |
|   [ Clear ]                      [ Evaluate ]     |
|                                                    |
|   Recognised:  2 + 4 =                            |
|   Result:  = 6                                    |
|   Confidence:  2: 98%  +: 95%  4: 97%  =: 92%    |
+---------------------------------------------------+
```

## Supported Expressions

| Expression    | What You Draw | Answer on Canvas |
|---------------|---------------|------------------|
| Addition      | `2 + 4 =`     | `6`              |
| Subtraction   | `9 - 3 =`     | `6`              |
| Multi-digit   | `12 + 35 =`   | `47`             |
| Chained       | `5 + 3 - 2 =` | `6`              |

## Project Structure

```
model.py      CNN architecture (ResNet-style with residual blocks, 13 classes)
train.py      Training script (MNIST digits + synthetic operators)
predict.py    Symbol segmentation and prediction pipeline
solver.py     Arithmetic expression evaluation (SymPy)
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

This downloads MNIST, generates synthetic operator data from system fonts, and trains the CNN for 30 epochs. The best model is saved as `symbol_cnn.pth`.

2. **Launch the application:**

```bash
python main.py
```

3. **Use the interface:**
   - Draw a math expression on the canvas (e.g. write `4 + 5 =`)
   - The answer appears automatically in green after the `=` sign
   - Press **Clear** to reset the canvas
   - Press **Evaluate** to manually trigger recognition

## Requirements

- Python 3.8+
- PyTorch
- Pillow
- NumPy
- SymPy
- Tkinter (included with Python)
