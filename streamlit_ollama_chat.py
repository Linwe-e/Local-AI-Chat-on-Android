"""
Application de chat local avec Ollama et Streamlit.
Interface web pour discuter avec le modèle Gemma 3 270M en local.
"""
import json
import os
from datetime import datetime

import requests
import streamlit as st

# Configuration sécurisée
OLLAMA = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = "gemma3:270m"
HISTORY_FILE = "chat_history.json"

def load_chat_history():
    """Charge l'historique depuis le fichier JSON"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("messages", 
                               [{"role": "system", "content": "Tu es utile et concis."}])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            st.error(f"Erreur lors du chargement de l'historique : {e}")
    return [{"role": "system", "content": "Tu es utile et concis."}]

def save_chat_history(messages):
    """Sauvegarde l'historique dans le fichier JSON"""
    try:
        data = {
            "last_updated": datetime.now().isoformat(),
            "total_messages": len(messages),
            "messages": messages
        }
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except (OSError, IOError) as e:
        st.error(f"Erreur lors de la sauvegarde : {e}")

# Configuration Streamlit
st.set_page_config(
    page_title="Gemma 3 270M — Chat local", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)
st.title("Chat local — Gemma 3 270M (Ollama)")

# Initialisation de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

def chat(messages):
    """Fonction qui appelle l'API Ollama en streaming"""
    with requests.post(f"{OLLAMA}/api/chat",
                       json={
                           "model": MODEL,
                           "messages": messages,
                           "stream": True,
                           "options": {
                               "num_ctx": 512,
                               "num_predict": 64,
                           }
                       },
                       stream=True,
                       timeout=30) as r:
        r.raise_for_status()
        response = ""
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "message" in data and "content" in data["message"]:
                chunk = data["message"]["content"]
                response += chunk
                yield chunk
            if data.get("done"):
                break

# Ajouter cette variable d'état
if "generating" not in st.session_state:
    st.session_state.generating = False

# Interface avec 2 boutons
col1, col2 = st.columns([3, 1])
with col1:
    user = st.text_input("Message", "", disabled=st.session_state.generating)
with col2:
    if st.session_state.generating:
        # Pendant la génération : bouton Stop bien visible
        if st.button("⏹️ Stop", type="secondary", use_container_width=True):
            st.session_state.generating = False
            st.rerun()
    else:
        # Quand pas de génération : bouton Envoyer
        send_pressed = st.button("📤 Envoyer", type="primary", use_container_width=True)

# Logique d'envoi
if not st.session_state.generating and 'send_pressed' in locals() and send_pressed and user.strip():
    st.session_state.generating = True
    st.rerun()  # Force le rafraîchissement pour montrer le bouton Stop

# Traitement de la génération
if st.session_state.generating and user.strip():
    # Ajouter le message utilisateur (seulement si pas déjà ajouté)
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != user.strip():
        st.session_state.messages.append({"role": "user", "content": user.strip()})

    # Zone d'affichage avec indicateur
    box = st.empty()
    box.markdown("**IA :** 🤔 *Réflexion en cours...*")

    accumulated_response = ""
    try:
        # Streaming de la réponse
        for delta in chat(st.session_state.messages):
            if not st.session_state.generating:  # Check si stop demandé
                accumulated_response += " *[Interrompu]*"
                break
            accumulated_response += delta
            box.markdown(f"**IA :** {accumulated_response}")
    except (requests.RequestException, json.JSONDecodeError) as e:
        box.error(f"Erreur : {str(e)}")
    finally:
        if accumulated_response:
            # Sauvegarder la réponse complète
            st.session_state.messages.append({"role": "assistant", "content": accumulated_response})
            # Sauvegarder l'historique après chaque échange
            save_chat_history(st.session_state.messages)
        
        st.session_state.generating = False


# Affichage de l'historique (12 derniers messages)
for message in st.session_state.messages[-12:]:
    if message["role"] != "system":
        speaker = "Vous" if message["role"] == "user" else "IA"
        st.markdown(f"**{speaker}:** {message['content']}")

# Bouton pour effacer l'historique (en bas)
if st.button("🗑️ Effacer l'historique", type="secondary"):
    st.session_state.messages = [{"role": "system", "content": "Tu es utile et concis."}]
    save_chat_history(st.session_state.messages)
    st.rerun()