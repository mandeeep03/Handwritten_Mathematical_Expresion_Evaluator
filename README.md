# Handwritten Mathematical Expression Evaluator

A Python application that recognizes handwritten mathematical expressions from a drawing canvas and solves linear equations using a CNN model built with PyTorch.

## What It Does

- Provides **five input/output panels** in a single window:
  - **3 Variable Assignment canvases** — draw assignments like `a = 5`, `b = 3`, `c = 2`
  - **1 Equation canvas** — draw an equation using those variables, e.g. `a + b - c`
  - **1 Output panel** — displays recognised variables, the substituted equation, and the final result
- Uses a trained CNN (ResNet-style) to recognize individual symbols (digits 0–9, operators +, −, =, and variable x)
- Segments the drawn image into characters via connected-component analysis
- Evaluates arithmetic and solves linear equations for x using SymPy

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ✦ Handwritten Math Solver                                  │
│  Assign variables  ▸  Write an equation  ▸  Evaluate        │
├──────────────────────┬──────────────────────────────────────┤
│  Variable A (a = ?)  │  Equation (a + b - c)               │
│  [ draw pad ]        │  [ draw pad ]                       │
│  recognised: a = 5   │  recognised: a + b - c              │
├──────────────────────┤                                      │
│  Variable B (b = ?)  │  [ Clear All ]        [ ⚡ Evaluate ]│
│  [ draw pad ]        ├──────────────────────────────────────┤
│  recognised: b = 3   │  📊 Output                           │
├──────────────────────┤  Variables:  a = 5   b = 3   c = 2  │
│  Variable C (c = ?)  │  Recognised: a + b - c              │
│  [ draw pad ]        │  Substituted: (5) + (3) - (2)       │
│  recognised: c = 2   │  Result: = 6                        │
└──────────────────────┴──────────────────────────────────────┘
```

## Supported Expressions

| Type                | Example          | Output      |
|---------------------|------------------|-------------|
| Arithmetic          | `3 + 5 =`        | `= 8`       |
| Linear equation     | `x + 7 = 9`      | `x = 2`     |
| Variable equation   | `a + b - c`      | `= 6` (with a=5, b=3, c=2) |
| Simple assignment   | `x = 8`          | `x = 8`     |

## Project Structure

```
model.py      CNN architecture (ResNet-style with residual blocks)
train.py      Training script with synthetic data generation
predict.py    Symbol segmentation & prediction pipeline
solver.py     Expression parsing and equation solving (SymPy)
ui.py         Tkinter-based multi-panel drawing interface
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

This generates synthetic training data from system fonts and trains the CNN. The trained model is saved as `symbol_cnn.pth`.

2. **Launch the application:**

```bash
python main.py
```

3. **Use the interface:**
   - Draw a value in each **Variable** pad (e.g. write `5` in Variable A)
   - Draw an equation in the **Equation** pad (e.g. write `a + b - c`)
   - Press **⚡ Evaluate** — the output panel shows recognised symbols, substituted expression, and the final result

## Requirements

- Python 3.8+
- PyTorch
- Pillow
- NumPy
- SymPy
- Tkinter (included with Python)
