import random
import numpy as np
from PIL.ImageTk import PhotoImage
from setuptools.windows_support import windows_only
from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import uuid
import os

# Đường dẫn đến model YOLO
model_path = "model.pt"
plate_path = "plate_img"
model = YOLO(model_path)

# Biến global để lưu trữ biển số hiện tại
current_plate_img = None
current_plate_array = None


def select_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if video_path:
        file_label.config(text=f"File đã chọn: {os.path.basename(video_path)}")
        process_video(video_path)


def resize_frame(frame, width, height):
    h, w = frame.shape[:2]
    aspect_ratio = w / h
    if width / height > aspect_ratio:
        new_width = int(height * aspect_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width / aspect_ratio)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame


def are_images_similar(img1, img2, threshold=0.95):
    """So sánh hai ảnh để xác định xem chúng có giống nhau không"""
    if img1 is None or img2 is None:
        return False

    # Đảm bảo hai ảnh có cùng kích thước
    height, width = img1.shape[:2]
    img2 = cv2.resize(img2, (width, height))

    # Tính toán sự khác biệt
    diff = cv2.absdiff(img1, img2)
    diff_sum = np.sum(diff)
    max_diff = img1.shape[0] * img1.shape[1] * 255 * 3  # Max possible difference

    similarity = 1 - (diff_sum / max_diff)
    return similarity > threshold


def update_plate_display(plate_img):
    """Cập nhật hiển thị biển số với kích thước cố định"""
    global plateImg

    # Tạo một ảnh nền trắng với kích thước cố định
    display_width = 300
    display_height = 150
    background = Image.new('RGB', (display_width, display_height), 'white')

    if plate_img is not None:
        # Chuyển đổi plate_img thành ảnh PIL
        if isinstance(plate_img, np.ndarray):
            plate_img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))

        # Tính toán tỷ lệ để giữ nguyên tỷ lệ khung hình
        aspect_ratio = plate_img.size[0] / plate_img.size[1]
        if display_width / display_height > aspect_ratio:
            new_height = display_height
            new_width = int(display_height * aspect_ratio)
        else:
            new_width = display_width
            new_height = int(display_width / aspect_ratio)

        # Resize plate_img
        plate_img = plate_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Tính toán vị trí để căn giữa
        x = (display_width - new_width) // 2
        y = (display_height - new_height) // 2

        # Paste plate_img vào background
        background.paste(plate_img, (x, y))

    # Chuyển đổi thành PhotoImage và cập nhật hiển thị
    plateImg = ImageTk.PhotoImage(background)
    plate_preview.config(image=plateImg)
    plate_preview.image = plateImg


def process_video(video_path):
    global plateImg, current_plate_array
    plateImg = None
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        status_label.config(text="Lỗi: Không thể mở video", foreground="red")
        return

    # Tạo khung hiển thị trống ban đầu
    update_plate_display(None)
    plate_preview.config(text="Đang chờ phát hiện biển số...")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Lấy kích thước của video_frame
        video_width = video_frame.winfo_width()
        video_height = video_frame.winfo_height()

        original_frame_resized = resize_frame(frame, video_width, video_height)

        # Dự đoán kết quả với YOLO
        results = model.predict(frame)

        # Xử lý kết quả của YOLO
        found_new_plate = False
        for r in results:
            original_frame = r.orig_img
            im_arr = r.plot()

            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id == 27:  # Nếu là biển số xe
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    new_plate = original_frame[y1:y2, x1:x2]

                    # Kiểm tra xem biển số mới có khác biệt đáng kể so với biển số hiện tại không
                    if not are_images_similar(current_plate_array, new_plate):
                        current_plate_array = new_plate.copy()
                        update_plate_display(new_plate)
                        found_new_plate = True
                        status_label.config(text="Trạng thái: Phát hiện biển số mới!", foreground="green")

        # Cập nhật video frame
        processed_frame_resized = resize_frame(im_arr, video_width, video_height)
        processed_img = cv2.cvtColor(processed_frame_resized, cv2.COLOR_BGR2RGB)
        processed_img = Image.fromarray(processed_img)
        processed_img = ImageTk.PhotoImage(processed_img)

        video_label.config(image=processed_img)
        video_label.image = processed_img

        if not found_new_plate and current_plate_array is None:
            plate_preview.config(text="Đang chờ phát hiện biển số...")
            status_label.config(text="Trạng thái: Đang tìm kiếm biển số xe...", foreground="orange")

        window.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    status_label.config(text="Trạng thái: Đã dừng xử lý video", foreground="blue")


def save_plate():
    if current_plate_array is not None:
        # Chuyển đổi từ numpy array sang PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(current_plate_array, cv2.COLOR_BGR2RGB))

        while True:
            random_id = random.randint(10000000, 99999999)
            save_path = f"{plate_path}/plate_{random_id}.png"
            if not os.path.exists(save_path):
                break

        pil_image.save(save_path)
        status_label.config(text=f"Đã lưu biển số! ID: {random_id}", foreground="green")
    else:
        status_label.config(text="Lỗi: Không có biển số để lưu", foreground="red")


def show_plate():
    plate_id = entry_id.get().strip()
    if not plate_id.isdigit() or len(plate_id) != 8:
        search_result_label.config(text="ID không hợp lệ\nVui lòng nhập 8 chữ số", foreground="red")
        return

    file_path = f"{plate_path}/plate_{plate_id}.png"

    if os.path.exists(file_path):
        img = Image.open(file_path)
        # Sử dụng hàm update_plate_display để hiển thị kết quả tìm kiếm
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        update_plate_display(img_array)
        search_result_label.config(image=plateImg, text="")
        search_result_label.image = plateImg
        status_label.config(text=f"Đã tìm thấy biển số với ID: {plate_id}", foreground="green")
    else:
        search_result_label.config(image="", text="Không tìm thấy\nbiển số xe", foreground="red")
        status_label.config(text=f"Không tìm thấy biển số với ID: {plate_id}", foreground="red")


# Khởi tạo giao diện
window = tk.Tk()
window.title("Hệ Thống Nhận Diện Biển Số Xe")
window.geometry("1280x720")

# Style configuration
style = ttk.Style()
style.configure("TButton", padding=10, font=('Helvetica', 10))
style.configure("TFrame", background="#f0f0f0")
style.configure("Header.TLabel", font=('Helvetica', 12, 'bold'))

# Main container
main_container = ttk.Frame(window, padding="10")
main_container.grid(row=0, column=0, sticky="nsew")

# Header frame
header_frame = ttk.Frame(main_container)
header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))

select_button = ttk.Button(header_frame, text="Chọn Video", command=select_video)
select_button.grid(row=0, column=0, padx=5)

file_label = ttk.Label(header_frame, text="Chưa chọn file nào", font=('Helvetica', 10))
file_label.grid(row=0, column=1, padx=5)

status_label = ttk.Label(header_frame, text="Trạng thái: Chờ xử lý", font=('Helvetica', 10))
status_label.grid(row=0, column=2, padx=5)

# Video frame
video_frame = ttk.Frame(main_container, borderwidth=2, relief="solid")
video_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
video_label = ttk.Label(video_frame)
video_label.grid(row=0, column=0, sticky="nsew")

# Right sidebar
sidebar = ttk.Frame(main_container)
sidebar.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

# License plate preview section
preview_frame = ttk.LabelFrame(sidebar, text="Biển Số Phát Hiện", padding="10")
preview_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

# Tạo frame có kích thước cố định cho plate preview
plate_preview_container = ttk.Frame(preview_frame, width=300, height=150)
plate_preview_container.grid(row=0, column=0, pady=5)
plate_preview_container.grid_propagate(False)  # Ngăn không cho frame tự động resize

plate_preview = ttk.Label(plate_preview_container, text="Chưa phát hiện biển số")
plate_preview.place(relx=0.5, rely=0.5, anchor="center")  # Căn giữa label trong container

save_button = ttk.Button(preview_frame, text="Lưu Biển Số", command=save_plate)
save_button.grid(row=1, column=0, pady=5)

# Search section
search_frame = ttk.LabelFrame(sidebar, text="Tìm Kiếm Biển Số", padding="10")
search_frame.grid(row=1, column=0, sticky="ew")

entry_id = ttk.Entry(search_frame)
entry_id.grid(row=0, column=0, pady=5)

search_button = ttk.Button(search_frame, text="Tìm Kiếm", command=show_plate)
search_button.grid(row=1, column=0, pady=5)

# Tạo frame có kích thước cố định cho search result
search_result_container = ttk.Frame(search_frame, width=300, height=150)
search_result_container.grid(row=2, column=0, pady=5)
search_result_container.grid_propagate(False)

search_result_label = ttk.Label(search_result_container, text="Kết quả tìm kiếm")
search_result_label.place(relx=0.5, rely=0.5, anchor="center")

# Configure grid weights
window.grid_columnconfigure(0, weight=1)
window.grid_rowconfigure(0, weight=1)
main_container.grid_columnconfigure(0, weight=3)
main_container.grid_columnconfigure(1, weight=3)
main_container.grid_columnconfigure(2, weight=1)
main_container.grid_rowconfigure(1, weight=1)
video_frame.grid_columnconfigure(0, weight=1)
video_frame.grid_rowconfigure(0, weight=1)

# Create plate_img directory if it doesn't exist
if not os.path.exists(plate_path):
    os.makedirs(plate_path)

window.mainloop()