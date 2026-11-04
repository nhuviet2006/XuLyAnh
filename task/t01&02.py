import cv2
import numpy as np
from scipy.signal import convolve2d

#TÌM VIỀN TÀI LIỆU
def custom_gaussian_blur(image_gray, kernel_size=5, sigma=1.0):
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel = kernel / np.sum(kernel)
    blurred = convolve2d(image_gray, kernel, mode='same', boundary='symm')
    return blurred

def custom_sobel_edge(blurred_img, threshold=50):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Gx = convolve2d(blurred_img, Kx, mode='same', boundary='symm')
    Gy = convolve2d(blurred_img, Ky, mode='same', boundary='symm')
    
    G = np.sqrt(Gx**2 + Gy**2)
    binary_edge = np.zeros_like(G, dtype=np.uint8)
    binary_edge[G > threshold] = 255
    return binary_edge

def get_ordered_corners(binary_edge_img):
    contours, _ = cv2.findContours(binary_edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    pts = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            break
            
    # Fallback: Tờ giấy tràn viền hoặc nền quá lộn xộn
    if pts is None:
        h, w = binary_edge_img.shape
        pts = np.array([[0,0], [w,0], [w,h], [0,h]])

    # Sắp xếp tọa độ: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

#NẮN PHẲNG VÀ NỘI SUY
def compute_homography(src_pts, dst_pts):
    A = []
    for i in range(4):
        x, y = src_pts[i]  # Unpacking gọn gàng
        u, v = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    
    A = np.array(A)
    # SVD: Chỉ lấy ma trận V (index 2), bỏ qua U và S
    V = np.linalg.svd(A)[2] 
    H = V[-1].reshape(3, 3)
    H = H / H[2, 2] # Chuẩn hóa hệ số
    return H

def warp_perspective_bilinear(image, H, output_size=(1240, 1754)):
    w, h = output_size
    H_inv = np.linalg.inv(H)
    
    # Kỹ thuật Vectorization: Trải lưới đích thay vì dùng vòng lặp for
    x_dst, y_dst = np.meshgrid(np.arange(w), np.arange(h))
    x_dst_flat = x_dst.flatten()
    y_dst_flat = y_dst.flatten()
    
    # Gom tọa độ 3D đồng nhất
    dst_coords = np.vstack((x_dst_flat, y_dst_flat, np.ones_like(x_dst_flat)))

    # Ánh xạ về ảnh gốc
    src_coords = H_inv @ dst_coords
    src_x = (src_coords[0, :] / src_coords[2, :]).reshape(h, w)
    src_y = (src_coords[1, :] / src_coords[2, :]).reshape(h, w)

    # Chặn tọa độ tràn ảnh
    max_x, max_y = image.shape[1] - 2, image.shape[0] - 2
    src_x = np.clip(src_x, 0, max_x)
    src_y = np.clip(src_y, 0, max_y)

    # Lấy 4 điểm lân cận để nội suy
    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Tính trọng số phân bổ
    wa = (x1 - src_x) * (y1 - src_y)
    wb = (src_x - x0) * (y1 - src_y)
    wc = (x1 - src_x) * (src_y - y0)
    wd = (src_x - x0) * (src_y - y0)

    # Trải trục để nhân được với ảnh màu 3 kênh
    if len(image.shape) == 3:
        wa = wa[..., np.newaxis]
        wb = wb[..., np.newaxis]
        wc = wc[..., np.newaxis]
        wd = wd[..., np.newaxis]

    warped = (wa * image[y0, x0] + wb * image[y0, x1] + wc * image[y1, x0] + wd * image[y1, x1])
    return warped.astype(np.uint8)


if __name__ == "__main__":
    image = cv2.imread("images/anh (17).jpg")
    
    if image is None:
        print("Lỗi: Không đọc được ảnh!")
    else:
        orig_image = image.copy()
        
        # 1. Thu nhỏ ảnh (Downscale) để tăng tốc và chống nhiễu
        ratio = orig_image.shape[0] / 800.0 
        new_width = int(orig_image.shape[1] / ratio)
        image_resized = cv2.resize(orig_image, (new_width, 800))
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        
        # 2. Xử lý Task 1 trên ảnh nhỏ
        blurred_img = custom_gaussian_blur(image_gray, kernel_size=5, sigma=1.0)
        binary_edge = custom_sobel_edge(blurred_img, threshold=50)
        pts_small = get_ordered_corners(binary_edge)
        
        # Vẽ mô phỏng viền lên ảnh nhỏ
        preview_corners = image_resized.copy()
        cv2.drawContours(preview_corners, [pts_small.astype(int)], -1, (0, 255, 0), 2)
        
        # 3. Nhân trả tỷ lệ và thực thi Task 2 trên ảnh siêu nét
        src_pts = pts_small * ratio 
        dst_pts = np.array([[0, 0], [1240, 0], [1240, 1754], [0, 1754]], dtype="float32")
        
        H = compute_homography(src_pts, dst_pts)
        warped = warp_perspective_bilinear(orig_image, H, output_size=(1240, 1754))
        
        # 4. Hiển thị demo
        def show_step(title, img):
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 500, 700)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            cv2.imshow(title, img)

        show_step("Buoc 1: Bien Canny tu cai", binary_edge)
        show_step("Buoc 2: Chot 4 goc tai lieu", preview_corners)
        show_step("Buoc 3: Ket qua A4", warped)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()