# ============================================================
#  cv_module.py — Computer Vision module
#  Classifies map cells into three categories and renders
#  the simulation frame with agent sprites and task markers.
# ============================================================

import numpy as np
import cv2
from simulator import DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT

# Cell classification classes
BACKGROUND = 0  # free traversable cell
STATIC     = 1  # wall / obstacle
DYNAMIC    = 2  # current agent position
HIDDEN     = 3  # outside agent field of view

CLASS_COLORS = {
    BACKGROUND: (240, 240, 240),
    STATIC:     (30,  30,  30),
}
FOV_COLOR = (70, 40, 15)  # dark brown for hidden cells

# One colour per agent — cycles if more than 12 agents
AGENT_COLORS = [
    (60,  200,  80),
    (60,  140, 240),
    (40,  200, 200),
    (180,  80, 240),
    (80,  200, 240),
    (40,  180, 140),
    (100, 240, 160),
    (200, 160,  60),
    (80,  100, 240),
    (240, 120,  80),
    (160, 200,  60),
    (240,  80, 160),
]


class CVModule:
    """
    Computer Vision module.
    Takes the static obstacle grid and agent positions each tick,
    classifies every cell, and renders the result as an RGB image.
    """

    def __init__(self, static_grid: np.ndarray,
                 cell_size: int = 12, fov_radius: int = 0):
        self.static_grid        = static_grid.copy()
        self.height, self.width = static_grid.shape
        self.cell_size          = cell_size
        self.fov_radius         = fov_radius
        # Pre-build the base class map from the static grid (done once)
        self.background         = self._build_background()
        self._last_agents       = []
        self._last_markers      = []

    def _build_background(self) -> np.ndarray:
        """
        Creates the base classification map from the static obstacle grid.
        Called once at startup — walls and free cells don't change.
        """
        bg = np.zeros((self.height, self.width), dtype=np.uint8)
        bg[self.static_grid == 1] = STATIC
        bg[self.static_grid == 0] = BACKGROUND
        return bg

    def classify(self, agent_positions, task_markers=None):
        """
        Updates the class map for the current tick.
        Marks agent positions as DYNAMIC on top of the static base map.
        If FoV is enabled, hides all cells outside each agent's radius.
        """
        agents = []
        for i, pos in enumerate(agent_positions):
            if len(pos) == 5:
                agents.append(pos)
            else:
                agents.append((pos[0], pos[1], pos[2], DIR_DOWN, False))
        self._last_agents  = agents
        self._last_markers = task_markers or []

        # Start from the static base and mark agent positions
        class_grid = self.background.copy()
        for (_, row, col, _, _) in agents:
            if 0 <= row < self.height and 0 <= col < self.width:
                class_grid[row, col] = DYNAMIC

        # Apply field of view mask if enabled
        if self.fov_radius > 0:
            class_grid = self._apply_fov(
                class_grid, [(r, c) for (_, r, c, _, _) in agents])
        return class_grid

    def _apply_fov(self, class_grid, positions):
        """
        Hides all cells outside each agent's field of view radius.
        Each agent sees a square of (2*radius+1) x (2*radius+1) cells.
        Hidden cells are marked as HIDDEN class.
        """
        visible = np.zeros((self.height, self.width), dtype=bool)
        for (ar, ac) in positions:
            r0 = max(0, ar - self.fov_radius)
            r1 = min(self.height, ar + self.fov_radius + 1)
            c0 = max(0, ac - self.fov_radius)
            c1 = min(self.width,  ac + self.fov_radius + 1)
            visible[r0:r1, c0:c1] = True
        class_grid[~visible] = HIDDEN
        return class_grid

    def render(self, class_grid: np.ndarray) -> np.ndarray:
        """
        Converts the class map into an RGB image.
        Draws background/walls/hidden areas, then task markers,
        then agent sprites on top.
        """
        cs   = self.cell_size
        h_px = self.height * cs
        w_px = self.width  * cs

        image = np.full((h_px, w_px, 3), 45, dtype=np.uint8)

        # Fill each class with its colour using fast numpy broadcasting
        for cls, color in [(BACKGROUND, CLASS_COLORS[BACKGROUND]),
                           (STATIC,     CLASS_COLORS[STATIC]),
                           (HIDDEN,     FOV_COLOR)]:
            mask    = (class_grid == cls)
            mask_px = np.kron(mask, np.ones((cs, cs), dtype=bool))
            image[mask_px] = color

        # Draw task markers: green diamond = pickup, blue circle = dropoff
        for m in self._last_markers:
            row, col = m["pos"]
            cx = col * cs + cs // 2
            cy = row * cs + cs // 2
            r  = max(4, cs // 2 - 1)

            if m["type"] == "pickup":
                pts = np.array([
                    [cx,     cy - r],
                    [cx + r, cy    ],
                    [cx,     cy + r],
                    [cx - r, cy    ],
                ], dtype=np.int32)
                cv2.fillPoly(image, [pts], (50, 180, 50))
                cv2.polylines(image, [pts], True, (20, 100, 20), 1)
                if cs >= 10:
                    cv2.putText(image, "P", (cx - 3, cy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                max(0.2, cs / 65),
                                (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(image, (cx, cy), r, (50, 50, 210), -1)
                cv2.circle(image, (cx, cy), r, (20, 20, 140), 1)
                if cs >= 10:
                    cv2.putText(image, "D", (cx - 4, cy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                max(0.2, cs / 65),
                                (255, 255, 255), 1, cv2.LINE_AA)

        # Draw agent sprites on top of everything
        ROBOT_PX = self.cell_size
        for (aidx, row, col, direction, has_cargo) in self._last_agents:
            cx    = col * cs + cs // 2
            cy    = row * cs + cs // 2
            color = AGENT_COLORS[aidx % len(AGENT_COLORS)]
            dark  = tuple(max(0, c - 70) for c in color)
            light = tuple(min(255, c + 70) for c in color)
            self._draw_robot(image, cx, cy, ROBOT_PX,
                             color, dark, light,
                             aidx + 1, direction, has_cargo)
        return image

    def _draw_robot(self, img, cx, cy, px,
                    color, dark, light, number, direction, has_cargo):
        """
        Draws a robot sprite into a temporary buffer, rotates it
        to face the movement direction, then pastes it onto the image.
        Drawing into a buffer first allows clean rotation without
        affecting the rest of the map.
        """
        buf = np.zeros((px, px, 3), dtype=np.uint8)
        bx  = px // 2
        by  = px // 2
        s   = px // 3

        # Shadow
        cv2.ellipse(buf, (bx, by + s),
                    (s, max(2, s // 3)), 0, 0, 360, (20, 20, 20), -1)

        # Wheels
        ww = max(3, s // 2)
        wh = max(2, s // 3)
        cv2.rectangle(buf, (bx-s, by+s//2-wh), (bx-s+ww, by+s//2+wh), dark, -1)
        cv2.rectangle(buf, (bx+s-ww, by+s//2-wh), (bx+s, by+s//2+wh), dark, -1)

        # Body
        bx1 = bx - s + ww // 2
        bx2 = bx + s - ww // 2
        by1 = by - s // 3
        by2 = by + s // 2
        cv2.rectangle(buf, (bx1, by1), (bx2, by2), color, -1)
        cv2.rectangle(buf, (bx1, by1), (bx2, by2), dark, 1)

        # Highlight stripe on body
        cv2.rectangle(buf, (bx1+2, by1+2), (bx2-2, by1+4), light, -1)

        # Cargo indicator (shown when agent is carrying an item)
        if has_cargo:
            cv2.rectangle(buf,
                (bx1+2, by1+2),
                (bx2-2, by1 + max(3, (by2-by1)//2)),
                (200, 220, 60), -1)

        # Head
        hw  = max(s, 6)
        hh  = max(s // 2 + 2, 4)
        hx1 = bx - hw // 2
        hx2 = bx + hw // 2
        hy1 = by - s - hh
        hy2 = by - s // 3
        cv2.rectangle(buf, (hx1, hy1), (hx2, hy2), color, -1)
        cv2.rectangle(buf, (hx1, hy1), (hx2, hy2), dark, 1)

        # Eyes
        ey     = (hy1 + hy2) // 2
        er     = max(2, hw // 5)
        offset = max(2, hw // 4)
        cv2.circle(buf, (bx - offset, ey), er, (220, 240, 255), -1)
        cv2.circle(buf, (bx + offset, ey), er, (220, 240, 255), -1)

        # Antenna
        ay = max(2, hy1 - s // 2)
        cv2.line(buf, (bx, hy1), (bx, ay), dark, 1)
        cv2.circle(buf, (bx, ay), max(2, s // 4), light, -1)

        # Agent number label
        lbl = str(number)
        fnt = cv2.FONT_HERSHEY_SIMPLEX
        sc  = 0.35
        (tw, th), _ = cv2.getTextSize(lbl, fnt, sc, 1)
        cv2.putText(buf, lbl,
                    (bx - tw//2, (by1+by2)//2 + th//2),
                    fnt, sc, (255,255,255), 1, cv2.LINE_AA)

        # Rotate sprite to face movement direction
        angle_map = {DIR_UP: 0, DIR_DOWN: 180,
                     DIR_LEFT: 270, DIR_RIGHT: 90}
        angle = angle_map.get(direction, 180)
        if angle != 0:
            M   = cv2.getRotationMatrix2D((bx, by), -angle, 1.0)
            buf = cv2.warpAffine(buf, M, (px, px))

        # Paste sprite onto the main image (only non-black pixels)
        y1 = cy - px // 2;  y2 = y1 + px
        x1 = cx - px // 2;  x2 = x1 + px

        iy1 = max(0, y1);  iy2 = min(img.shape[0], y2)
        ix1 = max(0, x1);  ix2 = min(img.shape[1], x2)
        by1c = iy1 - y1;   by2c = by1c + (iy2 - iy1)
        bx1c = ix1 - x1;   bx2c = bx1c + (ix2 - ix1)

        if iy2 > iy1 and ix2 > ix1:
            region = buf[by1c:by2c, bx1c:bx2c]
            mask   = region.any(axis=2)
            img[iy1:iy2, ix1:ix2][mask] = region[mask]

    def show(self, class_grid, window_name="CV Vision"):
        """Renders and displays the frame. Returns False if user pressed Q or Esc."""
        img = self.render(class_grid)
        cv2.imshow(window_name, img)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)