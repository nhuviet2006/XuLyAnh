import numpy as np


def rlsa_horizontal(binary, threshold):
    """RLSA theo từng hàng (ngang). Khoảng trống dài <= threshold giữa hai điểm trắng sẽ được lấp."""
    out = (binary > 0).astype(np.uint8) * 255
    for y in range(out.shape[0]):
        row = out[y] > 0
        # d = +1 ở đầu mỗi đoạn trắng, -1 ngay sau cuối đoạn trắng
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts = np.flatnonzero(d == 1)          # đầu các đoạn trắng
        ends = np.flatnonzero(d == -1)           # sau cuối các đoạn trắng (loại trừ)
        if len(starts) < 2:
            continue                             # có ít hơn 2 đoạn thì không có khoảng trống ở giữa
        gaps = starts[1:] - ends[:-1]            # độ dài khoảng trống giữa hai đoạn liên tiếp
        for i in np.flatnonzero(gaps <= threshold):
            out[y, ends[i]:starts[i + 1]] = 255  # lấp khoảng trống bằng điểm trắng
    return out


def rlsa_vertical(binary, threshold):
    """RLSA theo từng cột (dọc) = làm RLSA ngang trên ảnh đã chuyển vị rồi chuyển lại."""
    return rlsa_horizontal(binary.T, threshold).T


def rlsa(binary, t_h, t_v, t_smooth=None):
    """RLSA cổ điển: (ngang với t_h) AND (dọc với t_v), rồi làm mịn ngang thêm một lần (nếu có t_smooth)."""
    h = rlsa_horizontal(binary, t_h)
    v = rlsa_vertical(binary, t_v)
    combined = np.minimum(h, v)                  # AND của hai ảnh 0/255
    if t_smooth:
        combined = rlsa_horizontal(combined, t_smooth)
    return combined