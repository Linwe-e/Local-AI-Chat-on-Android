import json
import requests
import streamlit as st

# Configuration
OLLAMA = "http://127.0.0.1:11434"
MODEL = "gemma3:270m"

# Configuration Streamlit
st.set_page_config(page_title="Gemma 3 270M — Chat local", layout="wide", initial_sidebar_state="collapsed")
st.title("Chat local — Gemma 3 270M (Ollama)")

# Initialisation de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "Tu es utile et concis."}]

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
                       stream=True) as r:
        r.raise_for_status()
        acc = ""
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "message" in data and "content" in data["message"]:
                delta = data["message"]["content"]
                acc += delta
                yield delta
            if data.get("done"):
                break

# Ajouter cette variable d'état
if "generating" not in st.session_state:
    st.session_state.generating = False

# Interface avec 2 boutons
col1, col2 = st.columns([3, 1])
with col1:
    user = st.text_input("Message", "")
with col2:
    if st.session_state.generating:
        if st.button("⏹️ Stop", type="secondary"):
            st.session_state.generating = False
            st.rerun()
    else:
        send_pressed = st.button("📤 Envoyer", type="primary")

# Logique d'envoi
if send_pressed and user.strip() and not st.session_state.generating:
    st.session_state.generating = True

    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": user.strip()})

    # Zone d'affichage avec indicateur
    box = st.empty()
    box.markdown("**IA :** 🤔 *Réflexion en cours...*")

    acc = ""
    try:
        # Streaming de la réponse
        for d in chat(st.session_state.messages):
            if not st.session_state.generating:  # Check si stop demandé
                acc += " *[Interrompu]*"
                break
            acc += d
            box.markdown(f"**IA :** {acc}")
    except Exception as e:
        box.error(f"Erreur : {str(e)}")
        st.session_state.generating = False
    else:
        # Sauvegarder la réponse complète
        st.session_state.messages.append({"role": "assistant", "content": acc})

    st.session_state.generating = False


# Affichage de l'historique (12 derniers messages)
for m in st.session_state.messages[-12:]:
    if m["role"] != "system":
        who = "Vous" if m["role"] == "user" else "IA"
        st.markdown(f"**{who}:** {m['content']}")
