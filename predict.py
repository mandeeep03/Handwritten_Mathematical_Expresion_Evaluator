import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

from model import SymbolCNN, SYMBOL_CLASSES, NUM_CLASSES


class SymbolPredictor:
    def __init__(self, model_path='symbol_cnn.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SymbolCNN(num_classes=NUM_CLASSES).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def preprocess_symbol(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert('L')

        img_array = np.array(image)

        coords = np.argwhere(img_array > 30)
        if len(coords) == 0:
            return self.transform(Image.new('L', (28, 28), 0)).unsqueeze(0)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        cropped = img_array[y_min:y_max+1, x_min:x_max+1]
        cropped_img = Image.fromarray(cropped)

        target_size = 20
        h, w = cropped.shape
        scale = target_size / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cropped_img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new('L', (28, 28), 0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))

        tensor = self.transform(canvas).unsqueeze(0)
        return tensor

    def predict_single(self, image):
        tensor = self.preprocess_symbol(image).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = probabilities.max(1)
        return SYMBOL_CLASSES[predicted.item()], confidence.item()

    def predict_symbols(self, symbol_images):
        expression = ""
        confidences = []
        for img in symbol_images:
            symbol, conf = self.predict_single(img)
            expression += symbol
            confidences.append((symbol, conf))
        return expression, confidences

    def segment_and_predict(self, full_image):
        if isinstance(full_image, np.ndarray):
            full_image = Image.fromarray(full_image)

        gray = full_image.convert('L')
        img_array = np.array(gray)

        _, binary = threshold_image(img_array, 25)

        raw_components = find_connected_components(binary)
        if not raw_components:
            return "", [], []

        raw_components.sort(key=lambda s: s[0])

        segments = merge_equals_signs(raw_components)

        symbol_images = []
        bboxes = []
        for x, y, w, h in segments:
            pad = 6
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_array.shape[1], x + w + pad)
            y2 = min(img_array.shape[0], y + h + pad)
            symbol_crop = img_array[y1:y2, x1:x2]
            symbol_img = Image.fromarray(symbol_crop)
            symbol_images.append(symbol_img)
            bboxes.append((x1, y1, x2, y2))

        expression, confidences = self.predict_symbols(symbol_images)
        return expression, confidences, bboxes


def threshold_image(img_array, threshold):
    binary = np.zeros_like(img_array)
    binary[img_array > threshold] = 255
    return threshold, binary


def find_connected_components(binary):
    if binary.max() == 0:
        return []

    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for y in range(h):
        for x in range(w):
            if binary[y, x] > 0 and not visited[y, x]:
                min_x, min_y, max_x, max_y = x, y, x, y
                stack = [(y, x)]
                visited[y, x] = True
                pixel_count = 0
                while stack:
                    cy, cx = stack.pop()
                    pixel_count += 1
                    min_x = min(min_x, cx)
                    min_y = min(min_y, cy)
                    max_x = max(max_x, cx)
                    max_y = max(max_y, cy)
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and binary[ny, nx] > 0:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                comp_w = max_x - min_x + 1
                comp_h = max_y - min_y + 1
                if pixel_count > 10 and comp_w > 2 and comp_h > 2:
                    components.append((min_x, min_y, comp_w, comp_h))

    return components


def merge_equals_signs(components):
    if len(components) < 2:
        return components

    components = sorted(components, key=lambda c: c[0])
    merged = []
    used = set()

    for i in range(len(components)):
        if i in used:
            continue

        x1, y1, w1, h1 = components[i]
        cx1 = x1 + w1 / 2
        is_bar_1 = w1 > h1 * 1.2

        best_match = -1
        best_dist = float('inf')

        if is_bar_1:
            for j in range(len(components)):
                if j == i or j in used:
                    continue
                x2, y2, w2, h2 = components[j]
                cx2 = x2 + w2 / 2
                is_bar_2 = w2 > h2 * 1.2

                if not is_bar_2:
                    continue

                horizontal_overlap = abs(cx1 - cx2)
                max_w = max(w1, w2)
                if horizontal_overlap > max_w * 0.7:
                    continue

                vertical_gap = abs((y1 + h1 / 2) - (y2 + h2 / 2))
                max_h = max(h1, h2)

                if vertical_gap < max_h * 6 and vertical_gap > 0:
                    if vertical_gap < best_dist:
                        best_dist = vertical_gap
                        best_match = j

        if best_match >= 0:
            x2, y2, w2, h2 = components[best_match]
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_right = max(x1 + w1, x2 + w2)
            new_bottom = max(y1 + h1, y2 + h2)
            merged.append((new_x, new_y, new_right - new_x, new_bottom - new_y))
            used.add(i)
            used.add(best_match)
        else:
            merged.append((x1, y1, w1, h1))
            used.add(i)

    merged.sort(key=lambda c: c[0])
    return merged
