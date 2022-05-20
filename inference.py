import os

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

torch.hub.download_url_to_file(
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg"
)

model = torch.hub.load("pytorch/vision:v0.9.0", "alexnet", pretrained=True)
model.eval()

# Download ImageNet labels
os.system(
    "wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)


def inference(input_image: Image) -> dict:

    preprocess = transforms.Compose(
        [
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Grayscale(1),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    with open("imagenet_classes.txt") as f:
        categories = [s.strip() for s in f.readlines()]
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = {}
    for i in range(top5_prob.size(0)):
        result[categories[top5_catid[i]]] = top5_prob[i].item()
    return result


inputs = gr.inputs.Image(type="pil")
outputs = gr.outputs.Label(type="confidences", num_top_classes=5)

title = "ALEXNET"
description = "Gradio demo for Alexnet, the 2012 ImageNet winner."
article = "One weird trick for parallelizing convolutional neural networks"

examples = [["dog.jpg"]]
gr.Interface(
    inference,
    inputs,
    outputs,
    title=title,
    description=description,
    article=article,
    examples=examples,
    analytics_enabled=False,
).launch()
