import base64
import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from worker import speech_to_text, text_to_speech, openai_process_message


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
def main():
    app.run(debug=True)
    
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    """Route qui reçoit un fichier audio et le convertit en texte."""
    
    if 'audio' not in request.files:
        return jsonify({"error": "Aucun fichier audio reçu"}), 400

    audio_file = request.files['audio']
    audio_binary = audio_file.read()  # Lire le fichier en binaire

    transcript = speech_to_text(audio_binary)

    if transcript:
        return jsonify({"transcript": transcript})
    else:
        return jsonify({"error": "Échec de la conversion audio → texte"}), 500


@app.route('/process-message', methods=['POST'])
def process_prompt_route():
    """Route qui traite un message texte avec OpenAI et génère une réponse vocale."""
    
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Message manquant"}), 400

    user_message = data["message"]
    openai_response_text = openai_process_message(user_message)

    if not openai_response_text:
        return jsonify({"error": "Erreur OpenAI"}), 500

    # Générer l'audio avec IBM Watson TTS
    audio_binary = text_to_speech(openai_response_text)

    if not audio_binary:
        return jsonify({"error": "Erreur lors de la génération de l'audio"}), 500

    # Encoder l'audio en base64 pour l'envoyer en JSON
    audio_base64 = base64.b64encode(audio_binary).decode("utf-8")

    return jsonify({
        "openaiResponseText": openai_response_text,
        "openaiResponseSpeech": audio_base64
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


