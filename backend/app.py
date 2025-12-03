from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import mbti_service

app = Flask(__name__)
CORS(app)  # Enabled CORS for all routes

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            users = mbti_service.parse_chat_users(filepath)
            return jsonify({'filename': file.filename, 'users': users})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    filename = data.get('filename')
    selected_user = data.get('selected_user')
    
    if not filename or not selected_user:
        return jsonify({'error': 'Missing filename or selected_user'}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
        
    try:
        timeline = mbti_service.generate_timeline(filepath, selected_user)
        return jsonify({'timeline': timeline})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    mbti_service.load_model()
    app.run(debug=True, port=5000)
