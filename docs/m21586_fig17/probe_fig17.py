"""Machine-read the gauge positions H1..H4 of the Kleefsman (2005) dam-break setup (Fig. 17).

Fig. 17 is a raster JPEG (1182x1007) on page 21 of the paper; the probe coordinates exist only
in that picture.  The picture is dimensioned (1.000 / 1.228 / 1.248 / 0.744 / 0.550 / 0.161),
so the pixel->metre scale is calibrated from the drawing itself and the gauge x follows.
"""
import sys
import numpy as np
import cv2

def load(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (im < 150).astype(np.uint8)

def ascii_art(g, y0, y1, x0, x1, sy=4, sx=4, thr=0.15):
    out = []
    for y in range(y0, y1, sy):
        row = ''
        for x in range(x0, x1, sx):
            blk = g[y:y+sy, x:x+sx]
            f = blk.mean() if blk.size else 0
            row += ' ' if f < thr else ('.' if f < 0.35 else ('+' if f < 0.6 else '#'))
        out.append(f'{y:4d} {row}')
    return '\n'.join(out)

if __name__ == '__main__':
    g = load('docs/m21586_fig17/fig17_raw.jpeg')
    print('shape', g.shape, 'ink', g.sum())
    if len(sys.argv) > 1 and sys.argv[1] == 'art':
        y0,y1,x0,x1,sy,sx = (int(v) for v in sys.argv[2:8])
        print(ascii_art(g, y0,y1,x0,x1,sy,sx))
    else:
        print('row profile (ink per 10 rows):')
        ink = g.sum(axis=1)
        for y in range(0, g.shape[0], 10):
            print(f'{y:4d} {ink[y:y+10].sum():6d}')
