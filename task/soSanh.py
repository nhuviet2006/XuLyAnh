import cv2
import numpy as np
import os
import time
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d 
import json 
with open("ground_truth.json", "r") as f:
    gt_data = json.load(f)
# TÌM VIỀN TÀI LIỆU VÀ CHỐT 4 GÓC
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

#Hàm tìm và sắp xếp 4 góc bị thiếu
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
            
    if pts is None:
        h, w = binary_edge_img.shape
        pts = np.array([[0,0], [w,0], [w,h], [0,h]])

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# NẮN PHẲNG VÀ NỘI SUY
def compute_homography(src_pts, dst_pts):
    A = []
    for i in range(4):
        x, y = src_pts[i]  
        u, v = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    
    A = np.array(A)
    V = np.linalg.svd(A)[2] 
    H = V[-1].reshape(3, 3)
    H = H / H[2, 2] 
    return H

def warp_perspective_bilinear(image, H, output_size=(1240, 1754)):
    w, h = output_size
    H_inv = np.linalg.inv(H)
    
    x_dst, y_dst = np.meshgrid(np.arange(w), np.arange(h))
    x_dst_flat = x_dst.flatten()
    y_dst_flat = y_dst.flatten()
    
    dst_coords = np.vstack((x_dst_flat, y_dst_flat, np.ones_like(x_dst_flat)))

    src_coords = H_inv @ dst_coords
    src_x = (src_coords[0, :] / src_coords[2, :]).reshape(h, w)
    src_y = (src_coords[1, :] / src_coords[2, :]).reshape(h, w)

    max_x, max_y = image.shape[1] - 2, image.shape[0] - 2
    src_x = np.clip(src_x, 0, max_x)
    src_y = np.clip(src_y, 0, max_y)

    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = (x1 - src_x) * (y1 - src_y)
    wb = (src_x - x0) * (y1 - src_y)
    wc = (x1 - src_x) * (src_y - y0)
    wd = (src_x - x0) * (src_y - y0)

    if len(image.shape) == 3:
        wa = wa[..., np.newaxis]
        wb = wb[..., np.newaxis]
        wc = wc[..., np.newaxis]
        wd = wd[..., np.newaxis]

    warped = (wa * image[y0, x0] + wb * image[y0, x1] + wc * image[y1, x0] + wd * image[y1, x1])
    return warped.astype(np.uint8)

# ĐO LƯỜNG VÀ XUẤT HÀNG LOẠT
def evaluate_and_batch_process(input_folder="images", output_folder="output"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    
    # ĐỌC FILE GROUND TRUTH TỪ TRƯỚC VÒNG LẶP
    gt_data = {}
    if os.path.exists("ground_truth.json"):
        with open("ground_truth.json", "r") as f:
            gt_data = json.load(f)
    else:
        print("Cảnh báo: Không tìm thấy file ground_truth.json. Sẽ dùng thuật toán tự động.")
    
    print(f"{'Tên file':<15} | {'T/g Tự Code (s)':<15} | {'T/g OpenCV (s)':<15} | {'SSIM Score':<10}")
    print("-" * 65)

    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        orig_image = cv2.imread(img_path)
        if orig_image is None: continue

        dst_pts = np.array([[0, 0], [1240, 0], [1240, 1754], [0, 1754]], dtype="float32")

        #Lấy 4 góc từ file json đã gán nhãn thủ công trước đó
        if filename in gt_data:
            # Lấy thẳng 4 góc chuẩn xác từ file đã click thủ công
            src_pts = np.array(gt_data[filename], dtype="float32")
        else:
            # Nếu ảnh nào chưa gán nhãn thì thuật toán tự tìm
            ratio = orig_image.shape[0] / 800.0 
            new_width = int(orig_image.shape[1] / ratio)
            img_resized = cv2.resize(orig_image, (new_width, 800))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            blurred = custom_gaussian_blur(img_gray, kernel_size=5, sigma=1.0)
            binary = custom_sobel_edge(blurred, threshold=50)
            pts_small = get_ordered_corners(binary)
            src_pts = pts_small * ratio 

        
        # Đo bằng thuật toán tự code
        start_time = time.time()
        H_custom = compute_homography(src_pts, dst_pts)
        warped_custom = warp_perspective_bilinear(orig_image, H_custom, output_size=(1240, 1754))
        time_custom = time.time() - start_time
        
        # Đo bằng thuật toán của OPENCV 
        start_time = time.time()
        H_cv2 = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_cv2 = cv2.warpPerspective(orig_image, H_cv2, (1240, 1754))
        time_cv2 = time.time() - start_time
        
        #Tính chỉ số SSIM và lưu ảnh 
        score, _ = ssim(warped_custom, warped_cv2, channel_axis=2, full=True)
        
        out_path = os.path.join(output_folder, f"aligned_{filename}")
        cv2.imwrite(out_path, warped_custom)
        
        print(f"{filename:<15} | {time_custom:<15.4f} | {time_cv2:<15.4f} | {score:.4f}")

if __name__ == "__main__":
    evaluate_and_batch_process()