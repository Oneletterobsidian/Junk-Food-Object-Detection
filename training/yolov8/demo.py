import gradio as gr
from ultralytics import YOLO
from PIL import Image

model = YOLO(r"C:\Users\Administrator\Desktop\junk food\code\yolov8\best.pt")

NAMES = {
    0: 'Biryani', 1: 'Burger', 2: 'Unknown', 3: 'Hotdog', 4: 'Chips',
    15: 'Pizza', 20: 'Kabab', 21: 'Mac and Cheese', 22: 'Meatloaf',
    23: 'Muffin', 24: 'Nachos', 25: 'Cookies', 26: 'French Fry',
    27: 'Ice Cream', 28: 'Pizza', 29: 'Processed Cheese',
    30: 'Cheesecake', 31: 'Crispy Chicken', 32: 'Sandwich',
    33: 'Noodles', 34: 'Waffles'
}

def detect(image, confidence):
    results = model.predict(source=image, conf=confidence, verbose=False)
    result_img = results[0].plot()
    result_img = Image.fromarray(result_img[..., ::-1])

    boxes = results[0].boxes
    if len(boxes) == 0:
        text = "No objects detected"
    else:
        lines = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = NAMES.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            lines.append(f"{cls_name}: {conf:.2%}")
        text = "\n".join(lines)

    return result_img, text

demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="Upload Image", sources=["upload"]),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Confidence Threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="Detection Result"),
        gr.Textbox(label="Detected Objects", lines=10),
    ],
    title="Junk Food Object Detection (YOLOv8)",
    description="Upload an image and the model will detect junk food categories with bounding boxes.",
)

demo.launch()