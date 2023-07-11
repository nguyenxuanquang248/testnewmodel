import os
import cv2
import numpy as np

# Đường dẫn đến thư mục chứa ảnh hoa tulip gốc
image_folder = "/Users/thephaothutre/Desktop/pythonProject/testnewmodel/inpainting/test"

# Đường dẫn đến thư mục lưu trữ ảnh hoa tulip với masked
output_folder = "/Users/thephaothutre/Desktop/pythonProject/testnewmodel/inpainting/models"

# Tạo thư mục đầu ra nếu nó chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lấy danh sách các tệp tin ảnh trong thư mục tulip gốc
image_files = os.listdir(image_folder)

# Kích thước của masked vuông
mask_size = (128, 128)

# Tạo masked vuông ở giữa cho mỗi ảnh hoa tulip
for image_file in image_files:
    # Đường dẫn đến ảnh gốc
    image_path = os.path.join(image_folder, image_file)

    # Đọc ảnh gốc
    image = cv2.imread(image_path)

    # Kích thước của ảnh gốc
    image_height, image_width, _ = image.shape

    # Tính toán vị trí hình vuông masked
    mask_x = int((image_width - mask_size[0]) / 2)
    mask_y = int((image_height - mask_size[1]) / 2)

    # Tạo masked vuông ở giữa
    mask = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    mask[mask_y:mask_y + mask_size[1], mask_x:mask_x + mask_size[0]] = 255

    # Áp dụng masked lên ảnh gốc
    masked_image = np.where(mask > 0, mask, image)

    # Lưu ảnh hoa tulip với masked
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, masked_image)
