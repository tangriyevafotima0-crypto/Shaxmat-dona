#!/usr/bin/env python3
"""Generate a professional printable A4 layout from 4 chess piece template photos."""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from scipy import ndimage
import os

# Configuration
DPI = 300
A4_WIDTH_PX = 2480   # A4 width at 300 DPI
A4_HEIGHT_PX = 3508  # A4 height at 300 DPI

PIECE_WIDTH_CM = 5.0
PIECE_HEIGHT_CM = 6.0
PIECE_WIDTH_PX = round(PIECE_WIDTH_CM / 2.54 * DPI)   # 591
PIECE_HEIGHT_PX = round(PIECE_HEIGHT_CM / 2.54 * DPI)  # 709

TOP_MARGIN_CM = 2.5
LEFT_MARGIN_CM = 3.0
SPACING_CM = 2.0

TOP_MARGIN_PX = int(TOP_MARGIN_CM / 2.54 * DPI)     # ~295
LEFT_MARGIN_PX = int(LEFT_MARGIN_CM / 2.54 * DPI)   # ~354
SPACING_PX = int(SPACING_CM / 2.54 * DPI)           # ~236

IMAGE_FILES = [
    "20260608_083059.jpg",
    "20260608_083106.jpg",
    "20260608_083124.jpg",
    "20260608_083147.jpg",
]

FONT_BOLD_PATH = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
FONT_REGULAR_PATH = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"


def extract_piece(image_path):
    """Extract the chess piece from the photo, returning an RGBA image."""
    img = Image.open(image_path)
    w, h = img.size

    # Crop 10% margins from sides, 8% from top/bottom
    left = int(w * 0.10)
    right = int(w * 0.90)
    top = int(h * 0.08)
    bottom = int(h * 0.92)
    img_cropped = img.crop((left, top, right, bottom))

    # Convert to grayscale and threshold at 160
    gray = img_cropped.convert('L')
    gray_arr = np.array(gray)
    binary = (gray_arr < 160).astype(np.uint8)

    # Morphological cleanup
    binary = ndimage.binary_closing(binary, iterations=3).astype(np.uint8)
    binary = ndimage.binary_opening(binary, iterations=2).astype(np.uint8)

    # Find largest connected component
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        # Fallback: use the whole cropped image
        piece_img = img_cropped.convert('RGBA')
        return resize_piece(piece_img)

    # Find the largest component
    component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    largest_label = np.argmax(component_sizes) + 1
    mask = (labeled == largest_label).astype(np.uint8)

    # Get bounding box of the piece with padding
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding (2% of dimensions)
    pad_y = int((rmax - rmin) * 0.02)
    pad_x = int((cmax - cmin) * 0.02)
    rmin = max(0, rmin - pad_y)
    rmax = min(mask.shape[0] - 1, rmax + pad_y)
    cmin = max(0, cmin - pad_x)
    cmax = min(mask.shape[1] - 1, cmax + pad_x)

    # Crop the piece region
    piece_crop = img_cropped.crop((cmin, rmin, cmax + 1, rmax + 1))
    mask_crop = mask[rmin:rmax + 1, cmin:cmax + 1]

    # Dilate mask slightly for smoother edges
    mask_dilated = ndimage.binary_dilation(mask_crop, iterations=2).astype(np.uint8)

    # Create RGBA version with transparency outside piece
    piece_rgba = piece_crop.convert('RGBA')
    piece_arr = np.array(piece_rgba)
    # Set alpha channel based on mask
    piece_arr[:, :, 3] = mask_dilated * 255
    piece_rgba = Image.fromarray(piece_arr)

    return resize_piece(piece_rgba)


def resize_piece(piece_img):
    """Resize piece to fit within the target dimensions with inner margin, centered on canvas."""
    # Leave some inner margin (5% on each side)
    inner_margin = 0.05
    target_w = int(PIECE_WIDTH_PX * (1 - 2 * inner_margin))
    target_h = int(PIECE_HEIGHT_PX * (1 - 2 * inner_margin))

    # Calculate scale to fit within target
    w, h = piece_img.size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    piece_resized = piece_img.resize((new_w, new_h), Image.LANCZOS)

    # Center on canvas of exact piece dimensions
    canvas = Image.new('RGBA', (PIECE_WIDTH_PX, PIECE_HEIGHT_PX), (255, 255, 255, 0))
    offset_x = (PIECE_WIDTH_PX - new_w) // 2
    offset_y = (PIECE_HEIGHT_PX - new_h) // 2
    canvas.paste(piece_resized, (offset_x, offset_y), piece_resized)

    return canvas


def create_a4_layout(pieces):
    """Create an A4 layout with the 4 pieces arranged in a 2x2 grid."""
    # White background A4 canvas
    a4 = Image.new('RGB', (A4_WIDTH_PX, A4_HEIGHT_PX), (255, 255, 255))
    draw = ImageDraw.Draw(a4)

    # Load fonts
    try:
        font_bold = ImageFont.truetype(FONT_BOLD_PATH, 40)
    except (IOError, OSError):
        font_bold = ImageFont.load_default()
    try:
        font_regular = ImageFont.truetype(FONT_REGULAR_PATH, 24)
    except (IOError, OSError):
        font_regular = ImageFont.load_default()
    try:
        font_label = ImageFont.truetype(FONT_REGULAR_PATH, 20)
    except (IOError, OSError):
        font_label = ImageFont.load_default()

    # Title
    title = "Shaxmat Donalari Maketi"
    title_bbox = draw.textbbox((0, 0), title, font=font_bold)
    title_w = title_bbox[2] - title_bbox[0]
    title_x = (A4_WIDTH_PX - title_w) // 2
    title_y = 80
    draw.text((title_x, title_y), title, fill=(0, 0, 0), font=font_bold)

    # Subtitle
    subtitle = "Har bir dona: 5 sm \u00d7 6 sm | A4 format"
    sub_bbox = draw.textbbox((0, 0), subtitle, font=font_regular)
    sub_w = sub_bbox[2] - sub_bbox[0]
    sub_x = (A4_WIDTH_PX - sub_w) // 2
    sub_y = title_y + 60
    draw.text((sub_x, sub_y), subtitle, fill=(128, 128, 128), font=font_regular)

    # Calculate grid positions for title area offset
    # Top margin starts after the title area
    grid_top = TOP_MARGIN_PX + 200  # Extra space for title/subtitle

    # Place pieces in 2x2 grid
    positions = []
    for row in range(2):
        for col in range(2):
            x = LEFT_MARGIN_PX + col * (PIECE_WIDTH_PX + SPACING_PX)
            y = grid_top + row * (PIECE_HEIGHT_PX + SPACING_PX)
            positions.append((x, y))

    for i, (piece, (x, y)) in enumerate(zip(pieces, positions)):
        # Draw light gray cut-line border
        border_color = (200, 200, 200)
        draw.rectangle(
            [x - 2, y - 2, x + PIECE_WIDTH_PX + 2, y + PIECE_HEIGHT_PX + 2],
            outline=border_color, width=1
        )

        # Paste piece (convert RGBA to RGB with white background for compositing)
        piece_bg = Image.new('RGB', (PIECE_WIDTH_PX, PIECE_HEIGHT_PX), (255, 255, 255))
        piece_bg.paste(piece, (0, 0), piece)
        a4.paste(piece_bg, (x, y))

        # Add size label below each piece
        label = "5\u00d76 sm"
        label_bbox = draw.textbbox((0, 0), label, font=font_label)
        label_w = label_bbox[2] - label_bbox[0]
        label_x = x + (PIECE_WIDTH_PX - label_w) // 2
        label_y = y + PIECE_HEIGHT_PX + 10
        draw.text((label_x, label_y), label, fill=(128, 128, 128), font=font_label)

    return a4


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Extracting chess pieces from photos...")
    pieces = []
    for i, fname in enumerate(IMAGE_FILES):
        print(f"  Processing {fname}...")
        piece = extract_piece(fname)
        pieces.append(piece)
        # Save individual piece
        piece_path = f"final_piece_{i}.png"
        # Save with white background for individual pieces
        piece_bg = Image.new('RGBA', (PIECE_WIDTH_PX, PIECE_HEIGHT_PX), (255, 255, 255, 255))
        piece_bg.paste(piece, (0, 0), piece)
        piece_bg.save(piece_path, dpi=(DPI, DPI))
        print(f"    Saved {piece_path}")

    print("Creating A4 layout...")
    a4_layout = create_a4_layout(pieces)

    # Save as PNG
    png_path = "shaxmat_donalari_a4.png"
    a4_layout.save(png_path, dpi=(DPI, DPI))
    print(f"Saved {png_path}")

    # Save as PDF
    pdf_path = "shaxmat_donalari_a4.pdf"
    a4_layout.save(pdf_path, dpi=(DPI, DPI), resolution=DPI)
    print(f"Saved {pdf_path}")

    print("Done!")


if __name__ == "__main__":
    main()
