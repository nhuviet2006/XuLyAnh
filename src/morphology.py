import numpy as np

def _shift_stack_reduce(img, kernel, pad_value, reducer):
    kh, kw = kernel.shape
    cy, cx = kh // 2, kw // 2                  # tâm kernel (kernel nên có kích thước lẻ)
    h, w = img.shape
    # Thêm viền quanh ảnh để dịch không bị tràn ra ngoài.
    # Viền của CO phải là 255 (không làm mất điểm ở mép), viền của GIÃN là 0.
    padded = np.pad(img, ((cy, kh - 1 - cy), (cx, kw - 1 - cx)),
                    mode="constant", constant_values=pad_value)
    out = None
    for dy in range(kh):
        for dx in range(kw):
            if not kernel[dy, dx]:             # ô 0 của kernel thì bỏ qua
                continue
            window = padded[dy:dy + h, dx:dx + w]    # ảnh dịch đi (dy, dx)
            out = window.copy() if out is None else reducer(out, window)
    return out


def erode(img, kernel):
    # CO: lấy min. Viền 255 để mép ảnh không bị bào oan.
    return _shift_stack_reduce(img, kernel, 255, np.minimum)


def dilate(img, kernel):
    # GIÃN: lấy max. Viền 0.
    return _shift_stack_reduce(img, kernel, 0, np.maximum)


def opening(img, kernel):
    # MỞ = co trước, giãn sau: xóa đốm nhỏ hơn kernel
    return dilate(erode(img, kernel), kernel)


def closing(img, kernel):
    # ĐÓNG = giãn trước, co sau: lấp khe nhỏ hơn kernel
    return erode(dilate(img, kernel), kernel)


def rect_kernel(h, w):
    # Kernel hình chữ nhật h x w toàn số 1.
    # Kernel ngang (1 x n) dùng để nối chữ thành dòng, kernel đứng (n x 1) để nối dòng thành đoạn.
    return np.ones((h, w), dtype=np.uint8)