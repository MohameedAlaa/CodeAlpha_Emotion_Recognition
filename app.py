import streamlit as st
import tempfile
import os

from src.inference import predict

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a WAV file to predict the emotion. The model was trained on the RAVDESS dataset.")

uploaded_file = st.file_uploader("Choose a WAV audio file...", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Predict Emotion"):
        with st.spinner('Analyzing audio...'):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                
            try:
                emotion, probs = predict(tmp_path)
                
                st.success(f"### Predicted Emotion: **{emotion.upper()}**")
                
                st.write("#### Confidence Scores")
                # Sort probabilities descending
                sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                for emo, prob in sorted_probs:
                    st.progress(float(prob), text=f"{emo}: {prob:.2%}")
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")
            finally:
                os.remove(tmp_path)
