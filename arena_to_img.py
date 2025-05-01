from PIL import Image
import numpy as np

TILE_ENCODING = ['=', '.', '#', '@']
WEAPON_ENCODING = ['K', 'S', 'A', 'B', 'M', 'C']


def to_image(file: str):
    with open(file, "r") as f:
        content = f.readlines()

    tile_pixels = {k: i for i, k in enumerate(TILE_ENCODING)}
    weapon_pixels = {k: i for i, k in enumerate(WEAPON_ENCODING, start=10)}

    player = 50

    potion_pixel = 100
    menhir_pixel = 200

    fire = 250
    mist = 255

    pixel_mapping = tile_pixels | weapon_pixels

    lines = [[pixel_mapping[c] for c in list(line.strip())] for line in content]
    img = Image.fromarray(np.array(lines, dtype=np.uint8))
    img.save("map.png")


if __name__ == '__main__':
    to_image("resources/arenas/dungeon.gupb")
