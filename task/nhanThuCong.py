import cv2
import os
import json

# Biến lưu tọa độ click
current_clicks = []
scale_ratio = 1.0 

def mouse_callback(event, x, y, flags, param):
    global current_clicks, scale_ratio
    img_display = param['img_copy']
    
    # Khi nhấn chuột trái
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_clicks) < 4:
            # Lưu tọa độ thực trên ảnh gốc (nhân ngược lại với tỷ lệ thu nhỏ)
            real_x, real_y = int(x * scale_ratio), int(y * scale_ratio)
            current_clicks.append([real_x, real_y])
            
            # Vẽ chấm đỏ để xác nhận đã click
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Labeling", img_display)
            
            if len(current_clicks) == 4:
                print("Đã chọn đủ 4 góc. Nhấn phím 's' để lưu, hoặc 'r' để chọn lại.")

def run_labeling(image_folder="images", output_file="ground_truth.json"):
    global current_clicks, scale_ratio
    
    # Tải dữ liệu cũ nếu đã làm dở
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    
    for filename in image_files:
        if filename in data:
            continue 
        
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is None: continue

        # Thu nhỏ ảnh để vừa màn hình
        display_height = 800
        scale_ratio = img.shape[0] / display_height
        display_width = int(img.shape[1] / scale_ratio)
        img_resized = cv2.resize(img, (display_width, display_height))
        
        img_copy = img_resized.copy()
        current_clicks = []
        
        cv2.namedWindow("Labeling")
        # Truyền img_copy vào callback để vẽ
        cv2.setMouseCallback("Labeling", mouse_callback, param={'img_copy': img_copy})
        
        print(f"Đang mở: {filename}. Click 4 góc (Trái trên, Phải trên, Phải dưới, Trái dưới).")
        
        while True:
            cv2.imshow("Labeling", img_copy)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and len(current_clicks) == 4:
                data[filename] = current_clicks
                # Lưu file ngay lập tức
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"Đã lưu {filename}!")
                break
            elif key == ord('r'): #Reset nếu click sai
                print("Chọn lại 4 góc.")
                current_clicks = []
                img_copy = img_resized.copy()
                cv2.setMouseCallback("Labeling", mouse_callback, param={'img_copy': img_copy})
            elif key == ord('q'): #Nhấn q để thoát tool
                print("Thoát chương trình.")
                return

    cv2.destroyAllWindows()
    print("Hoàn thành gán nhãn toàn bộ thư mục!")

if __name__ == "__main__":
    run_labeling()