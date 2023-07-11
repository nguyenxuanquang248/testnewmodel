
import  cv2
import numpy as np
def create_mask_image(image_path):
    # Đọc ảnh gốc
    mask_size = (128, 128)
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
    output_path = '/Users/thephaothutre/Desktop/pythonProject/testnewmodel/flask_app/static/masked.jpg'
        # Lưu ảnh hoa tulip với masked
    cv2.imwrite(output_path, masked_image)

