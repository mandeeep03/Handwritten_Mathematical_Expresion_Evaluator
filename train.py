import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms, datasets
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

from model import SymbolCNN, SYMBOL_CLASSES, SYMBOL_TO_IDX, NUM_CLASSES


# ────────────────────────────────────────────────────────────────────
#  Operator/Symbol classes  (indices 10-16 in SYMBOL_CLASSES)
# ────────────────────────────────────────────────────────────────────
OPERATOR_CLASSES = {
    '+': SYMBOL_TO_IDX['+'],
    '-': SYMBOL_TO_IDX['-'],
    '*': SYMBOL_TO_IDX['*'],
    '/': SYMBOL_TO_IDX['/'],
    '(': SYMBOL_TO_IDX['('],
    ')': SYMBOL_TO_IDX[')'],
    '=': SYMBOL_TO_IDX['='],
}

# EMNIST letters split: labels 1-26 -> a-z
# We remap to our indices 17-42
EMNIST_LETTER_REMAP = {}
for i in range(26):
    letter = chr(ord('a') + i)
    emnist_label = i + 1   # EMNIST letters: 1=a, 2=b, ... 26=z
    our_idx = SYMBOL_TO_IDX[letter]
    EMNIST_LETTER_REMAP[emnist_label] = our_idx


# ────────────────────────────────────────────────────────────────────
#  Datasets
# ────────────────────────────────────────────────────────────────────

class EMNISTDigitDataset(Dataset):
    """Load digits 0-9 from EMNIST digits split."""

    def __init__(self, train=True, transform=None, max_per_digit=6000):
        self.transform = transform
        self.images = []
        self.labels = []
        emnist = datasets.EMNIST(root='./data', split='digits', train=train, download=True)
        digit_counts = {i: 0 for i in range(10)}
        for img, label in emnist:
            if digit_counts[label] < max_per_digit:
                # EMNIST images are transposed — fix them
                img = img.transpose(Image.TRANSPOSE)
                self.images.append(img)
                self.labels.append(label)  # 0-9 maps directly
                digit_counts[label] += 1
            if all(c >= max_per_digit for c in digit_counts.values()):
                break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label


class EMNISTLetterDataset(Dataset):
    """Load lowercase letters a-z from EMNIST letters split."""

    def __init__(self, train=True, transform=None, max_per_letter=4000):
        self.transform = transform
        self.images = []
        self.labels = []
        emnist = datasets.EMNIST(root='./data', split='letters', train=train, download=True)
        letter_counts = {i: 0 for i in range(1, 27)}
        for img, label in emnist:
            if label in EMNIST_LETTER_REMAP and letter_counts.get(label, 0) < max_per_letter:
                img = img.transpose(Image.TRANSPOSE)
                self.images.append(img)
                self.labels.append(EMNIST_LETTER_REMAP[label])
                letter_counts[label] = letter_counts.get(label, 0) + 1
            if all(letter_counts.get(i, 0) >= max_per_letter for i in range(1, 27)):
                break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label


class OperatorDataset(Dataset):
    """Synthetically generated operator/symbol images."""

    def __init__(self, num_samples_per_class=5000, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.fonts = self._get_fonts()
        for symbol, label_idx in OPERATOR_CLASSES.items():
            for _ in range(num_samples_per_class):
                img = self._generate_symbol(symbol)
                self.images.append(img)
                self.labels.append(label_idx)

    def _get_fonts(self):
        font_paths = []
        if os.name == 'nt':
            font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        else:
            font_dir = '/usr/share/fonts'
        if os.path.isdir(font_dir):
            for root, dirs, files in os.walk(font_dir):
                for f in files:
                    if f.lower().endswith(('.ttf', '.ttc', '.otf')):
                        font_paths.append(os.path.join(root, f))
        if not font_paths:
            font_paths = [None]
        return font_paths[:40]

    def _generate_symbol(self, symbol):
        img = Image.new('L', (28, 28), 0)
        draw = ImageDraw.Draw(img)
        font_size = random.randint(14, 24)
        font_path = random.choice(self.fonts)
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except (OSError, IOError):
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), symbol, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        jitter_x = random.randint(-3, 3)
        jitter_y = random.randint(-3, 3)
        x_pos = max(0, (28 - text_w) // 2 + jitter_x)
        y_pos = max(0, (28 - text_h) // 2 + jitter_y)
        draw.text((x_pos - bbox[0], y_pos - bbox[1]), symbol, fill=255, font=font)
        img = self._augment(img)
        return img

    def _augment(self, img):
        img_array = np.array(img, dtype=np.float32)
        if random.random() < 0.4:
            noise = np.random.normal(0, random.uniform(5, 25), img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))
        if random.random() < 0.5:
            angle = random.uniform(-20, 20)
            img = img.rotate(angle, fillcolor=0)
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        if random.random() < 0.4:
            scale = random.uniform(0.75, 1.25)
            new_size = max(10, int(28 * scale))
            img = img.resize((new_size, new_size), Image.BILINEAR)
            result = Image.new('L', (28, 28), 0)
            px = (28 - new_size) // 2
            py = (28 - new_size) // 2
            if new_size <= 28:
                result.paste(img, (px, py))
            else:
                crop_x = (new_size - 28) // 2
                crop_y = (new_size - 28) // 2
                result.paste(img.crop((crop_x, crop_y, crop_x + 28, crop_y + 28)), (0, 0))
            img = result
        if random.random() < 0.4:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            img = img.transform((28, 28), Image.AFFINE, (1, 0, -dx, 0, 1, -dy), fillcolor=0)
        if random.random() < 0.3:
            shear = random.uniform(-0.2, 0.2)
            img = img.transform((28, 28), Image.AFFINE, (1, shear, 0, 0, 1, 0), fillcolor=0)
        if random.random() < 0.3:
            thickness = random.choice([1, 2])
            img = img.filter(ImageFilter.MaxFilter(2 * thickness + 1))
        return img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label


# ────────────────────────────────────────────────────────────────────
#  Augmentation transforms
# ────────────────────────────────────────────────────────────────────

class StrongAugTransform:
    """Strong augmentation for EMNIST data (digits + letters)."""
    def __init__(self):
        self.base = transforms.Compose([
            transforms.RandomAffine(
                degrees=15, translate=(0.12, 0.12),
                scale=(0.82, 1.18), shear=12,
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
            ], p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __call__(self, img):
        return self.base(img)


# ────────────────────────────────────────────────────────────────────
#  Mixup
# ────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ────────────────────────────────────────────────────────────────────
#  Training
# ────────────────────────────────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Classes: {SYMBOL_CLASSES}")

    strong_transform = StrongAugTransform()
    operator_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # ── Load EMNIST Digits ──
    print("\nLoading EMNIST digits...")
    train_digits = EMNISTDigitDataset(train=True, transform=strong_transform, max_per_digit=6000)
    print(f"  Loaded {len(train_digits)} digit samples")

    # ── Load EMNIST Letters ──
    print("Loading EMNIST letters...")
    train_letters = EMNISTLetterDataset(train=True, transform=strong_transform, max_per_letter=4000)
    print(f"  Loaded {len(train_letters)} letter samples")

    # ── Generate Operators ──
    print("Generating operator symbols (+, -, *, /, (, ), =)...")
    train_operators = OperatorDataset(num_samples_per_class=5000, transform=operator_transform)
    print(f"  Generated {len(train_operators)} operator samples")

    # ── Combine ──
    train_dataset = ConcatDataset([train_digits, train_letters, train_operators])
    print(f"\n  Total training samples: {len(train_dataset)}")

    # ── Class-balanced sampling ──
    print("Building class-balanced sampler...")
    all_labels = []
    for ds in [train_digits, train_letters, train_operators]:
        all_labels.extend(ds.labels)
    class_counts = {}
    for lb in all_labels:
        class_counts[lb] = class_counts.get(lb, 0) + 1
    weights = [1.0 / class_counts[lb] for lb in all_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(all_labels), replacement=True)

    # ── Validation ──
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    val_digits = EMNISTDigitDataset(train=False, transform=val_transform, max_per_digit=800)
    val_letters = EMNISTLetterDataset(train=False, transform=val_transform, max_per_letter=600)
    val_operators = OperatorDataset(num_samples_per_class=600, transform=val_transform)
    val_dataset = ConcatDataset([val_digits, val_letters, val_operators])
    print(f"  Total validation samples: {len(val_dataset)}")

    # ── Loaders ──
    train_loader = DataLoader(
        train_dataset, batch_size=128, sampler=sampler,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ── Model / Optimiser ──
    model = SymbolCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

    num_epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Mixup
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2)

            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (lam * predicted.eq(y_a).sum().item()
                        + (1 - lam) * predicted.eq(y_b).sum().item())

        train_acc = 100. * correct / total
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    class_total[label] = class_total.get(label, 0) + 1
                    if label == pred:
                        class_correct[label] = class_correct.get(label, 0) + 1

        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}% "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'symbol_cnn.pth')
            print(f"  -> Saved best model (Val Acc: {val_acc:.2f}%)")

        if (epoch + 1) % 10 == 0:
            print("  Per-class accuracy:")
            for idx, sym in enumerate(SYMBOL_CLASSES):
                c = class_correct.get(idx, 0)
                t = class_total.get(idx, 0)
                acc = 100.0 * c / t if t > 0 else 0
                print(f"    {sym}: {acc:.1f}% ({c}/{t})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to symbol_cnn.pth ({NUM_CLASSES} classes)")


if __name__ == '__main__':
    train()
