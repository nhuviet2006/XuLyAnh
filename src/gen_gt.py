import json
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from docx import Document

ROOT = Path(__file__).resolve().parent.parent

W, H = 1240, 1754

MARKER_RE = re.compile(r"^(\uf0b7|\u2022|[-\u2013\u25cf]|\d+[.)])$")

#PHAN 1: HAM TIEN ICH
def imread_vn(path, flags=cv2.IMREAD_COLOR):
    # cv2.imread hay lỗi khi đường dẫn có dấu tiếng Việt (Windows) -> đọc bằng numpy
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def imwrite_vn(path, img):
    # tương tự: ghi ảnh an toàn với đường dẫn có dấu
    ok, buf = cv2.imencode(Path(path).suffix, img)
    buf.tofile(str(path))


def norm(s):
    # Chuẩn hóa chữ để SO KHỚP: đưa Unicode về dạng NFC và bỏ hết khoảng trắng.
    # Cần vì PDF và docx có thể cắt khoảng trắng / dấu tiếng Việt khác nhau.
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def find_clean(doc_id):
    # doc03 -> tìm clean3.png, nếu không có thì tìm clean03.png
    n = int(doc_id[3:])
    for name in (f"clean{n}.png", f"clean{n:02d}.png"):
        p = ROOT / "data" / "clean" / name
        if p.exists():
            return p
    return None

#PHAN 2: DOC PDF -> DANH SACH DONG CHU + HOP HINH

def read_pdf_lines(pdf_path):
    """Trả về (danh sách dòng theo thứ tự đọc, danh sách hộp hình), đơn vị pixel."""
    page = pymupdf.open(pdf_path)[0]          # tài liệu chỉ 1 trang
    # Tọa độ PDF tính bằng "point". Nhân hệ số s để đổi sang pixel của ảnh sạch.
    s = W / page.rect.width
    lines, figs = [], []
    pending_marker = None                      # nhớ ký hiệu đầu dòng đang chờ gộp
    for b in page.get_text("dict")["blocks"]:
        if b["type"] == 1:                     # type 1 = HÌNH -> lưu hộp bao của hình
            figs.append([round(v * s) for v in b["bbox"]])
            continue
        for ln in b["lines"]:                  # type 0 = chữ; duyệt từng dòng
            text = "".join(sp["text"] for sp in ln["spans"]).strip()
            if not text:
                continue
            bbox = [v * s for v in ln["bbox"]]     # [x0, y0, x1, y1] theo pixel
            if MARKER_RE.match(text):          # dòng chỉ chứa "•" hay "1." ...
                pending_marker = bbox          # chưa lưu, gộp vào dòng chữ kế tiếp
                continue
            sp0 = ln["spans"][0]               # span đầu tiên quyết định kiểu chữ
            item = {
                "bbox": bbox,
                "text": unicodedata.normalize("NFC", text),
                "size": round(sp0["size"], 1),         # cỡ chữ (pt)
                "bold": bool(sp0["flags"] & 16),       # cờ bit 16 = in đậm
                "italic": bool(sp0["flags"] & 2),      # cờ bit 2 = in nghiêng
                "marker": pending_marker is not None,  # dòng này là mục danh sách?
            }
            if pending_marker is not None:
                # mở rộng hộp dòng để bao luôn cả ký hiệu "•"
                item["bbox"] = [min(bbox[0], pending_marker[0]), min(bbox[1], pending_marker[1]),
                                max(bbox[2], pending_marker[2]), max(bbox[3], pending_marker[3])]
                pending_marker = None
            item["bbox"] = [round(v) for v in item["bbox"]]
            lines.append(item)
    return lines, figs

# PHAN 3: GOM DONG THANH DOAN, DUA VAO VAN BAN TRONG DOCX

def group_paragraphs(lines, docx_path, doc_id):
    doc = Document(docx_path)
    paras = [p for p in doc.paragraphs if p.text.strip()]   # bỏ đoạn rỗng
    result, i, warns = [], 0, []
    for p in paras:
        target = norm(p.text)          # văn bản của đoạn trong docx (đã chuẩn hóa)
        acc, got = "", []
        # Lấy lần lượt từng dòng PDF cho đến khi chữ gom được dài bằng đoạn docx
        while i < len(lines) and len(acc) < len(target):
            acc += norm(lines[i]["text"])
            got.append(lines[i])
            i += 1
        # Danh sách đánh số: PDF có "1." dính trong dòng, docx thì không -> bỏ "1." đi rồi so
        numbered = False
        if acc != target and re.sub(r"^\d+[.)]", "", acc) == target:
            acc, numbered = target, True
        # Nếu vẫn không khớp -> in cảnh báo để mình biết đáp án có thể sai
        if acc != target:
            warns.append(f"[{doc_id}] đoạn không khớp: docx='{p.text[:40]}' pdf='{acc[:40]}'")
        if not got:
            continue
        has_numpr = p._p.pPr is not None and p._p.pPr.numPr is not None  # docx đánh dấu danh sách
        is_list = has_numpr or got[0]["marker"] or numbered
        result.append({"lines": got, "is_list": is_list})
    if i < len(lines):
        warns.append(f"[{doc_id}] còn {len(lines) - i} dòng PDF chưa gán vào đoạn nào")
    return result, warns


def union_box(boxes):
    # Hộp nhỏ nhất bao trùm nhiều hộp (hộp của đoạn = bao các dòng của nó)
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]

def assign_roles(paras):
#gan vai tro cho tung doan: title, list, paraparaph...
    sizes = [ln["size"] for p in paras for ln in p["lines"]]
    body = float(np.median(sizes))        # cỡ chữ thân bài = số trung vị
    for k, p in enumerate(paras):
        ls = p["lines"]
        big = max(ln["size"] for ln in ls)
        all_bold = all(ln["bold"] for ln in ls)
        if p["is_list"]:
            p["role"] = "list_item"
        elif k == 0 and (big > body or all_bold):
            p["role"] = "title"
        elif k == 1 and len(ls) == 1 and ls[0]["italic"]:
            p["role"] = "subtitle"
        elif all_bold and len(ls) == 1:
            p["role"] = "heading"
        else:
            p["role"] = "paragraph"
        p["bbox"] = union_box([ln["bbox"] for ln in ls])

# PHAN 4: TAO MASK DIEM ANH (MUC PIXEL) CHO CHU VA HINH

def make_masks(pdf_path, figs):
    page = pymupdf.open(pdf_path)[0]
    # Xóa hết hình khỏi PDF (phủ trắng) để khi render chỉ còn CHỮ
    for b in page.get_text("dict")["blocks"]:
        if b["type"] == 1:
            page.add_redact_annot(pymupdf.Rect(b["bbox"]), fill=(1, 1, 1))
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_REMOVE)
    # Render ra đúng 1240 x 1754, ảnh xám
    m = pymupdf.Matrix(W / page.rect.width, H / page.rect.height)
    pm = page.get_pixmap(matrix=m, colorspace=pymupdf.csGRAY)
    g = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w)
    text_mask = ((g < 160) * 255).astype(np.uint8)    # điểm tối = chữ -> 255
    # Mask hình: tô trắng hình chữ nhật của từng ảnh minh họa
    fig_mask = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in figs:
        fig_mask[y0:y1, x0:x1] = 255
    return text_mask, fig_mask

# PHAN 5: VE OVERLAY DE TU KIEM TRA

def draw_overlay(clean_path, paras, figs, out_path):
    img = imread_vn(clean_path)
    for p in paras:
        for ln in p["lines"]:
            x0, y0, x1, y1 = ln["bbox"]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 160, 0), 1)        # DÒNG: xanh lá
        x0, y0, x1, y1 = p["bbox"]
        cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (255, 0, 0), 2)  # ĐOẠN: xanh dương
        cv2.putText(img, p["role"], (x0, max(12, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)       # ghi vai trò
    for x0, y0, x1, y1 in figs:
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3)            # HÌNH: đỏ
    imwrite_vn(out_path, img)

# PHAN 6: HAM CHINH

def process(doc_id):
    pdf = ROOT / "data" / "pdf" / f"{doc_id}.pdf"
    docx_p = ROOT / "data" / "docx" / f"{doc_id}.docx"
    clean = find_clean(doc_id)
    for p in (pdf, docx_p):
        if not p.exists():
            raise FileNotFoundError(p)     # thiếu file đầu vào thì báo ngay

    # Bước 1: đọc dòng + hình từ PDF
    lines, figs = read_pdf_lines(pdf)
    # Bước 2: gom dòng thành đoạn nhờ docx, rồi gán vai trò
    paras, warns = group_paragraphs(lines, docx_p, doc_id)
    assign_roles(paras)

    # Bước 3: đóng gói thành layout.json (đáp án về VỊ TRÍ)
    layout = {
        "id": doc_id, "size": [W, H],
        "figures": [{"bbox": f} for f in figs],
        "paragraphs": [{"role": p["role"], "bbox": p["bbox"],
                        "lines": [{k: ln[k] for k in ("bbox", "text", "size", "bold", "italic")}
                                  for ln in p["lines"]]} for p in paras],
        "n_lines": sum(len(p["lines"]) for p in paras),    # đáp án "đếm dòng"
    }

    # Bước 4: đếm cấu trúc (đáp án về SỐ LƯỢNG, Phát dùng ở bước dựng Word)
    roles = [p["role"] for p in paras]
    n_lists, prev = 0, False
    for r in roles:                        # một danh sách = các list_item liên tiếp nhau
        if r == "list_item" and not prev:
            n_lists += 1
        prev = (r == "list_item")
    structure = {
        "id": doc_id,
        "n_paragraphs": roles.count("paragraph"),
        "n_titles": roles.count("title"),
        "n_subtitles": roles.count("subtitle"),
        "n_headings": roles.count("heading"),
        "n_lists": n_lists,
        "n_list_items": roles.count("list_item"),
        "n_figures": len(figs),
        "n_lines": layout["n_lines"],
    }

    # Bước 5: ghi tất cả ra thư mục gt/
    for sub in ("layout", "structure", "mask_text", "mask_fig", "overlay"):
        (ROOT / "gt" / sub).mkdir(parents=True, exist_ok=True)
    (ROOT / "gt/layout" / f"{doc_id}.json").write_text(
        json.dumps(layout, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "gt/structure" / f"{doc_id}.json").write_text(
        json.dumps(structure, ensure_ascii=False, indent=1), encoding="utf-8")
    tm, fm = make_masks(pdf, figs)
    imwrite_vn(ROOT / "gt/mask_text" / f"{doc_id}.png", tm)
    imwrite_vn(ROOT / "gt/mask_fig" / f"{doc_id}.png", fm)
    if clean is not None:
        draw_overlay(clean, paras, figs, ROOT / "gt/overlay" / f"{doc_id}.png")
    else:
        warns.append(f"[{doc_id}] không thấy ảnh sạch, bỏ qua overlay")
    return structure, warns

# ĐIỂM VÀO: chạy file này sẽ xử lý các tài liệu và in cảnh báo (nếu có)

if __name__ == "__main__":
    ids = sys.argv[1:] or [f"doc{i:02d}" for i in range(1, 11)]
    all_warns = []
    for d in ids:
        st, w = process(d)
        all_warns += w
        print(d, st)
    print("\nCẢNH BÁO:" if all_warns else "\nKhông có cảnh báo.")
    for w in all_warns:
        print(" ", w)