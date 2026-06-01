# ============================================================
#  map_parser.py  —  Этап 2
#  Читает map.png и строит бинарную сетку препятствий
# ============================================================

import cv2
import numpy as np
import os


def load_map(png_path: str, metrics_path: str = None) -> np.ndarray:
    """
    Загружает map.png и возвращает бинарную сетку.
    0 = свободная клетка, 1 = стена/препятствие
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Ошибка: не могу открыть файл '{png_path}'")
        return None

    # Читаем реальный размер сетки из metrics.json
    if metrics_path and os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            m = json.load(f)
        grid_w, grid_h = m["width"], m["height"]
    else:
        # Запасной вариант: угадываем по размеру картинки
        # (предполагаем что каждая клетка = 10 пикселей)
        grid_h = img.shape[0] // 10
        grid_w = img.shape[1] // 10

    # Масштабируем изображение до размера сетки
    img_resized = cv2.resize(img, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)

    # Бинаризация: тёмные пиксели → стена (1), светлые → свободно (0)
    _, binary = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY_INV)
    grid = (binary > 0).astype(np.uint8)

    return grid


def get_grid_size(grid: np.ndarray) -> tuple:
    """Возвращает (height, width) сетки."""
    return grid.shape


def grid_to_display(grid: np.ndarray, cell_size: int = 8) -> np.ndarray:
    """
    Превращает сетку в RGB-изображение для отображения.
    Стена = чёрный, свободно = белый.
    Используется для быстрой проверки что карта загрузилась верно.
    """
    h, w = grid.shape
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

    for row in range(h):
        for col in range(w):
            color = (30, 30, 30) if grid[row, col] == 1 else (240, 240, 240)
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            img[y1:y2, x1:x2] = color

    return img


# ---- Тест при запуске напрямую ------------------------------
if __name__ == "__main__":
    # Перебираем все 4 кейса
    cases = ["warehouse", "maze", "bottleneck", "thin_walls"]

    for case in cases:
        path = os.path.join("data", case, "map.png")

        if not os.path.exists(path):
            print(f"[{case}] файл не найден, пропускаем")
            continue

        grid = load_map(
    os.path.join("data", case, "map.png"),
    metrics_path=os.path.join("data", case, "metrics.json")
)

        h, w = get_grid_size(grid)
        walls = int(grid.sum())
        free  = h * w - walls
        density = walls / (h * w) * 100

        print(f"[{case}]")
        print(f"  Размер:    {w} x {h} клеток")
        print(f"  Стен:      {walls} ({density:.1f}%)")
        print(f"  Свободно:  {free}")
        print()

        # Показываем карту в окне (нажми любую клавишу чтобы перейти к следующей)
        display = grid_to_display(grid, cell_size=8)
        cv2.imshow(f"Map: {case}", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()