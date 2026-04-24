# Handwritten Mathematical Expression Evaluator

A Python application that recognizes handwritten mathematical expressions from a drawing canvas and solves linear equations using a CNN model built with PyTorch.

## What It Does

- Provides a drawing canvas where you write mathematical expressions by hand
- Uses a trained CNN to recognize individual symbols (digits 0-9, operators +, -, =, and variable x)
- Segments the drawn image into individual characters using connected component analysis
- Evaluates arithmetic expressions and solves linear equations for x using SymPy

## Supported Expressions

- Arithmetic: `3 + 5 =` outputs `= 8`
- Equations: `x + 7 = 9` outputs `x = 2`
- Simple assignment: `x = 8` outputs `x = 8`

## Project Structure

```
model.py      CNN architecture definition
train.py      Training script with synthetic data generation
predict.py    Symbol segmentation and prediction
solver.py     Expression parsing and equation solving
ui.py         Tkinter-based drawing interface
main.py       Application entry point
```

## Setup

```
pip install -r requirements.txt
```

## How to Run

1. Train the model:

```
python train.py
```

This generates synthetic training data from system fonts and trains the CNN. The trained model is saved as `symbol_cnn.pth`.

2. Launch the application:

```
python main.py
```

3. Draw an equation on the canvas and press Evaluate.

## Requirements

- Python 3.8+
- PyTorch
- Pillow
- NumPy
- SymPy
- Tkinter (included with Python)
