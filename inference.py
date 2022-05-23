import gradio as gr
import numpy as np
import torch
from torchvision import transforms

from src.litmodules import MNISTLitModule

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = MNISTLitModule.load_from_checkpoint("checkpoints/last.ckpt")
model.to(device)
model.eval()


def inference(input_image: np.array) -> dict:
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return {i: float(probabilities[i]) for i in range(len(probabilities))}


gr.Interface(
    inference,
    inputs=gr.Image(
        image_mode="L", source="canvas", shape=(28, 28), invert_colors=True
    ),
    outputs=gr.outputs.Label(type="confidences", num_top_classes=10),
    live=False,
    title="Handwritten Digit Recognition",
).launch()
