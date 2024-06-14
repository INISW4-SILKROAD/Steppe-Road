from flask import Flask, request, jsonify
import torch
import torch.nn as nn

from models.kostantiniyye import Kostantiniyye

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Kostantiniyye().to(device)
model.load_state_dict(torch.load('.checkpoints/model.pth', map_location=device))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_tensor = torch.tensor(data['input']).float().to(device)
    with torch.no_grad():
        output = model(input_tensor)
    return jsonify({'prediction': output.item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)