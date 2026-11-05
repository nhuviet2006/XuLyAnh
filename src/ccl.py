import numpy as np


def connected_components(binary, connectivity=8):
    """Gán nhãn thành phần liên thông.
    binary      : ảnh 2 chiều, điểm > 0 là điểm trắng (đối tượng)
    connectivity: 8 (tính đường chéo, giống mặc định của cv2) hoặc 4
    Trả về (labels, num): labels cùng cỡ ảnh, nền = 0, các vùng = 1..num
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity phải là 4 hoặc 8")
    fg = binary > 0
    h, w = fg.shape

    # Mảng nhãn có thêm 1 hàng phía trên và 1 cột mỗi bên (toàn số 0).
    # Nhờ viền này, hàng xóm nằm ngoài ảnh tự động là 0, khỏi phải kiểm tra biên.
    lab = np.zeros((h + 1, w + 2), dtype=np.int32)

    parent = [0]                      # sổ union-find: parent[x] = "cha" của nhãn x (nhãn 0 là nền)

    def find(x):
        # Lần theo cha cho đến khi gặp gốc (gốc có parent[x] == x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # nén đường đi: nối thẳng mọi nhãn trên đường về gốc
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        # Gộp hai nhóm. Luôn cho gốc LỚN hơn trỏ về gốc NHỎ hơn.
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # ---------------- LƯỢT 1 ----------------
    # np.nonzero trả về tọa độ điểm trắng theo đúng thứ tự quét (hàng trước, cột sau)
    ys, xs = np.nonzero(fg)
    nxt = 1                           # nhãn mới sẽ cấp
    for y, x in zip(ys.tolist(), xs.tolist()):
        y1, x1 = y + 1, x + 1         # vị trí tương ứng trong mảng có viền
        nb = [lab[y1, x1 - 1],        # W : bên trái
              lab[y1 - 1, x1]]        # N : bên trên
        if connectivity == 8:
            nb.append(lab[y1 - 1, x1 - 1])   # NW: trên-trái
            nb.append(lab[y1 - 1, x1 + 1])   # NE: trên-phải
        nb = [v for v in nb if v > 0]        # chỉ giữ hàng xóm đã có nhãn

        if not nb:                    # không hàng xóm nào có nhãn -> đảo mới
            lab[y1, x1] = nxt
            parent.append(nxt)        # nhãn mới tự là gốc của chính nó
            nxt += 1
        else:
            m = min(nb)               # lấy nhãn nhỏ nhất
            lab[y1, x1] = m
            for v in nb:              # các nhãn khác m thật ra cùng một đảo -> gộp
                if v != m:
                    union(v, m)

    # ---------------- LƯỢT 2 ----------------
    # Tìm gốc của từng nhãn tạm, rồi đánh số lại 1..k theo thứ tự gốc tăng dần
    roots = [find(i) for i in range(nxt)]
    uniq = sorted(set(roots) - {0})
    new_id = {r: i + 1 for i, r in enumerate(uniq)}
    lut = np.zeros(nxt, dtype=np.int32)          # bảng tra: nhãn tạm -> nhãn chính thức
    for i in range(1, nxt):
        lut[i] = new_id[roots[i]]
    labels = lut[lab[1:, 1:-1]]                  # bỏ viền, đổi nhãn bằng bảng tra (rất nhanh)
    return labels, len(uniq)


def bounding_boxes(labels, num):
    """Hộp bao của từng vùng. Trả về mảng (num, 4): [x0, y0, x1, y1] (gồm cả điểm cuối).
    Hàng i ứng với vùng có nhãn i + 1. Dùng ở bước sau để lấy hộp của dòng, đoạn, hình."""
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs] - 1
    x0 = np.full(num, labels.shape[1], dtype=np.int64)
    y0 = np.full(num, labels.shape[0], dtype=np.int64)
    x1 = np.full(num, -1, dtype=np.int64)
    y1 = np.full(num, -1, dtype=np.int64)
    np.minimum.at(x0, ids, xs)
    np.minimum.at(y0, ids, ys)
    np.maximum.at(x1, ids, xs)
    np.maximum.at(y1, ids, ys)
    return np.stack([x0, y0, x1, y1], axis=1)


def areas(labels, num):
    """Diện tích (số điểm ảnh) của từng vùng; phần tử i ứng với vùng i + 1."""
    return np.bincount(labels.ravel(), minlength=num + 1)[1:]