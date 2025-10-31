from flask import Flask, render_template, request
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 🧠 Load Hugging Face model and processor
model_name = "nateraw/plant-disease-model"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file!"
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # 🖼️ Open image
    image = Image.open(file_path)

    # 🔍 Process image
    inputs = processor(images=image, return_tensors="pt")

    # 🔮 Prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    label = model.config.id2label[predicted_class_idx]

    return render_template('result.html', result=label, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
