"""
make_placement_grid.py
Generates a 5x5 grid figure showing each of the 5 trained canaries
inserted at each of the 5 placement positions on a sample image.
Rows = canaries (zebra, elephant, gaussian, checkerboard, gradient)
Cols = positions (cc, uc, bc, cl, cr)
Saves to figures/placement_grid.png
"""
import cv2
import numpy as np
import os

# -----------------------------------------------
# Paths
# -----------------------------------------------
CANARIES = {
    'Zebra':        'FJNTraining/canary_zebra/exp_VOC07_120_22_80_50/canary_050.png',
    'Elephant':     'FJNTraining/canary_elephant/exp_VOC07_120_22_80_50/canary_050.png',
    'Gaussian':     'FJNTraining/canary_gaussian/exp_VOC07_120_22_80_50/canary_050.png',
    'Checkerboard': 'FJNTraining/canary_checkerboard_correct/exp_VOC07_120_22_80_50/canary_050.png',
    'Gradient':     'FJNTraining/canary_gradient_correct/exp_VOC07_120_22_80_50/canary_050.png',
}

# Use one clean person image as background
# Pick a benign AdvPatch image that clearly shows a person
SAMPLE_IMG = 'Data/testeval/VOC07_YOLOv8/test/AdvPatch/benign/000032.jpg'

POSITIONS = {
    'cc': (0,   0),    # centre
    'uc': (-30, 0),    # up 30px
    'bc': (+30, 0),    # down 30px
    'cl': (0,  -30),   # left 30px
    'cr': (0,  +30),   # right 30px
}
POS_LABELS = {
    'cc': 'Centre',
    'uc': 'Up 30px',
    'bc': 'Down 30px',
    'cl': 'Left 30px',
    'cr': 'Right 30px',
}

CELL_W = 160
CELL_H = 160
LABEL_H = 22
HEADER_H = 28
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.42
FONT_COLOR = (30, 30, 30)
HEADER_BG = (220, 220, 220)
CANARY_SIZE = 80

def place_canary_on_img(img, canary, dy, dx):
    """Place canary at image centre + (dy, dx) offset."""
    h, w = img.shape[:2]
    cy = h // 2 + dy - CANARY_SIZE // 2
    cx = w // 2 + dx - CANARY_SIZE // 2
    # Clamp to image bounds
    cy = max(0, min(cy, h - CANARY_SIZE))
    cx = max(0, min(cx, w - CANARY_SIZE))
    out = img.copy()
    out[cy:cy+CANARY_SIZE, cx:cx+CANARY_SIZE] = canary
    # Draw red rectangle around canary
    cv2.rectangle(out, (cx, cy), (cx+CANARY_SIZE, cy+CANARY_SIZE), (0, 0, 220), 2)
    return out

def make_label_bar(text, w, h, bg):
    bar = np.ones((h, w, 3), dtype=np.uint8)
    bar[:] = bg
    text_size = cv2.getTextSize(text, FONT, FONT_SCALE, 1)[0]
    tx = max(2, (w - text_size[0]) // 2)
    ty = h - 6
    cv2.putText(bar, text, (tx, ty), FONT, FONT_SCALE, FONT_COLOR, 1, cv2.LINE_AA)
    return bar

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)

    # Load background image
    bg = cv2.imread(SAMPLE_IMG, 1)
    if bg is None:
        # fallback to any available image
        for atk in ['AdvPatch', 'UPC', 'TCEGA']:
            d = f'Data/testeval/VOC07_YOLOv8/test/{atk}/benign'
            files = sorted(os.listdir(d))
            if files:
                bg = cv2.imread(os.path.join(d, files[0]), 1)
                if bg is not None:
                    break
    bg = cv2.resize(bg, (CELL_W, CELL_H))

    canary_names = list(CANARIES.keys())
    pos_names    = list(POSITIONS.keys())

    # Load and resize all canaries
    canary_imgs = {}
    for name, path in CANARIES.items():
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (CANARY_SIZE, CANARY_SIZE))
        canary_imgs[name] = img

    # Build grid
    # Layout: top-left corner cell blank, then column headers (positions)
    # Row headers = canary names

    total_w = (len(pos_names) + 1) * CELL_W   # +1 for row label column
    total_h = HEADER_H + len(canary_names) * (CELL_H + LABEL_H)
    grid = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    # Top-left blank header
    corner = make_label_bar('', CELL_W, HEADER_H, HEADER_BG)
    grid[0:HEADER_H, 0:CELL_W] = corner

    # Column headers (position labels)
    for ci, pos in enumerate(pos_names):
        x0 = (ci + 1) * CELL_W
        hdr = make_label_bar(POS_LABELS[pos], CELL_W, HEADER_H, HEADER_BG)
        grid[0:HEADER_H, x0:x0+CELL_W] = hdr

    # Rows
    for ri, canary_name in enumerate(canary_names):
        y0 = HEADER_H + ri * (CELL_H + LABEL_H)

        # Row label
        row_lbl = make_label_bar(canary_name, CELL_W, CELL_H + LABEL_H, (235, 235, 235))
        grid[y0:y0+CELL_H+LABEL_H, 0:CELL_W] = row_lbl

        # Cells
        for ci, pos in enumerate(pos_names):
            dy, dx = POSITIONS[pos]
            cell_img = place_canary_on_img(bg, canary_imgs[canary_name], dy, dx)

            # Add position label below cell
            cell_with_label = np.ones((CELL_H + LABEL_H, CELL_W, 3), dtype=np.uint8) * 255
            cell_with_label[:CELL_H] = cell_img
            cv2.putText(cell_with_label, pos, (CELL_W//2 - 10, CELL_H + 15),
                        FONT, FONT_SCALE, (100, 100, 100), 1, cv2.LINE_AA)

            x0 = (ci + 1) * CELL_W
            grid[y0:y0+CELL_H+LABEL_H, x0:x0+CELL_W] = cell_with_label

    # Add thin grid lines
    for i in range(len(canary_names) + 1):
        y = HEADER_H + i * (CELL_H + LABEL_H)
        grid[y:y+1, :] = 180
    for j in range(len(pos_names) + 2):
        x = j * CELL_W
        grid[:, x:x+1] = 180

    out_path = 'figures/placement_grid.png'
    cv2.imwrite(out_path, grid)
    print(f'Saved: {out_path}  ({grid.shape[1]}x{grid.shape[0]} px)')
    print('Upload figures/placement_grid.png to Overleaf.')
