# Render ảnh clean TỪ CHÍNH file PDF -> ảnh và đáp án (gt) luôn khớp tuyệt đối.
import sys
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
W, H = 1240, 1754                      # kích thước chuẩn của cả nhóm
OUT = ROOT / "data" / "clean_new"      # ghi ra thư mục MỚI, chưa đè ảnh cũ
OUT.mkdir(parents=True, exist_ok=True)

ids = sys.argv[1:] or [f"doc{i:02d}" for i in range(1, 11)]
for d in ids:
    page = pymupdf.open(ROOT / "data" / "pdf" / f"{d}.pdf")[0]
    m = pymupdf.Matrix(W / page.rect.width, H / page.rect.height)   # cùng hệ số với gen_gt
    pix = page.get_pixmap(matrix=m, colorspace=pymupdf.csRGB)       # ảnh màu, nền trắng
    n = int(d[3:])                                                   # doc03 -> clean3.png
    pix.save(str(OUT / f"clean{n}.png"))
    print(d, "->", f"clean{n}.png", pix.width, "x", pix.height)