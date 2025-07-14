import gradio as gr
from yolov7.detect_faces import recognize_faces_from_image

iface = gr.Interface(
    fn=recognize_faces_from_image,
    inputs=gr.Image(type="D:\facerecognition\photos"),
    outputs="image"
)

iface.launch()
