from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification
from PIL import Image
import requests

REPO_MODEL="shehan97/mobilevitv2-1.0-imagenet1k-256"

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = MobileViTImageProcessor.from_pretrained(REPO_MODEL)
model = MobileViTV2ForImageClassification.from_pretrained(REPO_MODEL)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
