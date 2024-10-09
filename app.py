from flask import Flask, request, jsonify
import os
from PIL import Image
import torch
from torchvision import transforms, models
import json
import torch.nn as nn

# Import the model (make sure to adjust the import based on your structure)
app = Flask(__name__)

alexnet = models.alexnet(pretrained=True)

for param in alexnet.parameters():
    param.requires_grad = False

alexnet.classifier[6] = nn.Linear(4096, 4) #Changing final layer
alexnet.classifier.add_module("7", nn.LogSoftmax(dim = 1)) #adding a layer to classify

# Load the trained model
model = alexnet
model.load_state_dict(torch.load('model.pt', weights_only=True))
model.eval()

# Define the image transformations
def img_transform(img):
    img_transformations = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return img_transformations(img)

# Prediction function
def predict_plastic_type(image_path):
    test_image = Image.open(image_path).convert('RGB')
    test_image_tensor = img_transform(test_image).unsqueeze(0)  # Add batch dimension

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.cuda()

    with torch.no_grad():
        out = model(test_image_tensor)
        ps = torch.exp(out)
        
        topk = ps.topk(4, dim=1)  # Get top 4 predictions
        results = topk.values.cpu().numpy()[0]
        predictions = topk.indices.cpu().numpy()[0]

    return torch.argmax(out).item()

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    # Predict plastic type
    prediction = predict_plastic_type(img_path)
    if prediction == 0:
        prediction = "PET - 1"
    elif prediction == 1:
        prediction = "HDPE - 2"
    elif prediction == 2:
        prediction = "PP - 5"
    else:
        prediction = "PS - 6"

    # Remove the uploaded file after processing
    os.remove(img_path)

    # Prepare the response
    response = {
        'predictions': prediction,
    }

    return jsonify(response)

if __name__ == '__main__':
    # Ensure the upload directory exists
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
