import streamlit as st
from src.predict import predict

st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

st.title("Sentiment Analyzer")
st.write("Type a review or comment and the model will predict if it is positive, neutral or negative.")

text = st.text_area("Enter your text here", height=150)

if st.button("Analyze"):
    if text.strip():
        label, confidence, all_scores = predict(text)

        if label == "Positive":
            st.success(f"Positive 😊  ({confidence:.1%} confident)")
        elif label == "Negative":
            st.error(f"Negative 😞  ({confidence:.1%} confident)")
        else:
            st.info(f"Neutral 😐  ({confidence:.1%} confident)")

        st.subheader("Score breakdown")
        for name in ["Positive", "Neutral", "Negative"]:
            st.write(f"{name}: {all_scores.get(name, 0):.1%}")
            st.progress(all_scores.get(name, 0))
    else:
        st.warning("Please enter some text first.")