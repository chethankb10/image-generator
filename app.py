import os
from flask import Flask, render_template, request, send_file, jsonify
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from io import BytesIO
import uuid

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt', 'Astronaut riding a horse')
        
        if not prompt or len(prompt.strip()) == 0:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        # Generate image
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0",
        )
        
        # Save to bytes buffer
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=False,
            download_name='generated_image.png'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
