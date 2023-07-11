import cv2
from flask import Flask, render_template, request, redirect
import torch
from PIL import Image
import torchvision.transforms as transforms
from testnewmodel.dataset import DaisyDataset
from testnewmodel.model import InpaintingGAN
from testnewmodel.create_mask import create_mask_image

app = Flask(__name__)

# Tải mô hình
model = InpaintingGAN()
model_path = "/Users/thephaothutre/Desktop/pythonProject/testnewmodel/inpainting/models/model_epoch_10.pth"
model.load_state_dict(torch.load(model_path))
model.eval()
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/', methods=['POST'])
def upload_image():
    # Nhận bức ảnh từ người dùng
    image1 = request.files['image']
    image1.save("static/original.jpg")
    image2 = create_mask_image('static/original.jpg')

    image = '/Users/thephaothutre/Desktop/pythonProject/testnewmodel/flask_app/static/masked.jpg'
    #
    # image2 = create_mask_image(image1)
    # image = cv2.imread("/Users/thephaothutre/Desktop/pythonProject/testnewmodel/flask_app/static/masked.jpg")

    # Tiền xử lý ảnh
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(image).convert("RGB")

    input_image = image_transform(img).unsqueeze(0)



    # Áp dụng inpainting
    with torch.no_grad():
        output_image = model(input_image)

    # Xử lý kết quả
    output_image = output_image.squeeze(0)
    output_image = transforms.ToPILImage()(output_image)

    # Lưu ảnh vào thư mục tạm
    output_image.save("static/output.jpg")

    return redirect('/result')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(port=5001,debug=True)
