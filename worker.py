import os
import requests
import openai
import base64
from dotenv import load_dotenv

load_dotenv()

# Configurer la clé API OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def speech_to_text(audio_binary):
    """Convertit un fichier audio en texte avec IBM Watson STT."""

    api_key = os.getenv("STT_API_KEY")
    base_url = os.getenv("STT_URL")

    if not api_key or not base_url:
        print("Erreur: Clé API ou URL STT manquante.")
        return None

    api_url = f"{base_url}/v1/recognize"

    headers = {
        "Authorization": f"Basic {api_key}",
        "Content-Type": "audio/wav"
    }

    params = {"model": "en-US_Multimedia"}

    response = requests.post(api_url, params=params, data=audio_binary, headers=headers)

    if response.status_code != 200:
        print(f"Erreur STT: {response.status_code} - {response.text}")
        return None

    response_json = response.json()

    if response_json.get("results"):
        text = response_json["results"][0]["alternatives"][0]["transcript"]
        print("Texte reconnu:", text)
        return text
    else:
        print("Aucun texte reconnu.")
        return None

def openai_process_message(user_message):
    """Envoie un message à OpenAI et récupère la réponse."""

    if not openai.api_key:
        print("Erreur: Clé API OpenAI manquante.")
        return None

    prompt = "Act as a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations."

    try:
        # Utilisation du modèle GPT-4 ou GPT-3.5 Turbo avec la nouvelle API
        openai_response = openai.ChatCompletion.create(
            model="gpt-4",  # Ou utiliser "gpt-3.5-turbo" selon tes préférences
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=4000  # Ajuste le nombre de tokens en fonction de tes besoins
        )

        # Le texte de la réponse se trouve maintenant dans 'choices[0]['message']['content']'
        response_text = openai_response['choices'][0]['message']['content']
        return response_text

    except Exception as e:
        print("Erreur OpenAI:", e)
        return None

def text_to_speech(text, voice="en-US_AllisonV3Voice"):
    """Convertit un texte en audio avec IBM Watson TTS."""

    # Charger la clé API et l'URL depuis les variables d'environnement
    api_key = os.getenv("TTS_API_KEY")
    base_url = os.getenv("TTS_URL")

    # Vérifier si les informations d'API sont présentes
    if not api_key or not base_url:
        print("Erreur: Clé API ou URL TTS manquante.")
        return None

    # URL de l'API Watson TTS
    api_url = f"{base_url}/v1/synthesize"

    # Encodage de la clé API pour l'authentification Basic
    encoded_api_key = base64.b64encode(f"apikey:{api_key}".encode("utf-8")).decode("utf-8")

    # Ajouter l'authentification de type Basic (IBM Watson utilise un utilisateur:mot de passe pour l'authentification)
    headers = {
        "Authorization": f"Basic {encoded_api_key}",
        "Accept": "audio/wav",
        "Content-Type": "application/json"
    }

    # Créer les données JSON avec le texte et la voix
    json_data = {
        "text": text,
        "voice": voice,
        "accept": "audio/wav"
    }

    try:
        # Effectuer la requête POST pour la synthèse vocale
        response = requests.post(api_url, headers=headers, json=json_data)

        # Vérifier si la réponse est correcte
        if response.status_code != 200:
            print(f"Erreur TTS: {response.status_code} - {response.text}")
            return None

        print("Conversion TTS réussie.")

        # Sauvegarder l'audio dans un fichier .wav
        with open("output_audio.wav", "wb") as audio_file:
            audio_file.write(response.content)

        print("Fichier audio sauvegardé sous 'output_audio.wav'.")
        return response.content  # Retourne directement le fichier audio en binaire

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête TTS: {e}")
        return None

