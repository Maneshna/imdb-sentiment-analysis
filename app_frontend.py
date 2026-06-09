import streamlit as st
import requests

st.set_page_config(
    page_title="Sentiment Analysis", 
    page_icon="🎬",
    layout="centered"
)
st.title("🎬 IMDb Sentiment Analysis")

st.write("Enter a movie rewiew and see whether it is positive or negative :)")

review_text = st.text_area(
    "Movie review here",
    height=200,
    placeholder="Type your movie review here..."
)

if st.button("Predict sentiment"):
    if not review_text.strip():
        st.warning("Please enter the review here")
    else:
        payload = {"text": review_text}
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                sentiment = result["sentiment"]
                confidence = result["confidence"]

                st.subheader("Prediction Result")

                if sentiment.lower() == "positive":
                    st.success(f"Sentiment: {sentiment}")
                else:
                    st.error(f"Sentiment: {sentiment}")

                st.metric(
                    label="Confidence Score",
                    value=f"{confidence:.2%}"
                )
                st.progress(confidence)
            else:
                st.error(f"API Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to FastAPI backend.\n\n"
                "Make sure app_backend.py is running."
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")


