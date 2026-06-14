import streamlit as st
import requests

st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stButton button {
        width: 100%;
        height: 3rem;
        border-radius: 10px;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 IMDb Sentiment Analyzer")

st.markdown(
    """
    Analyze movie reviews using Machine Learning and Deep Learning models.
    """
)

model_type = st.selectbox(
    "Choose Model",
    [
        "logistic_regression",
        "pytorch"
    ]
)

review_text = st.text_area(
    "Movie Review",
    height=200,
    placeholder="Type your movie review here..."
)

if st.button("Predict Sentiment"):

    if not review_text.strip():

        st.warning(
            "Please enter a movie review."
        )

    else:

        payload = {
            "text": review_text,
            "model_type": model_type
        }

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

                st.divider()

                st.subheader(
                    "Prediction Result"
                )

                if sentiment.lower() == "positive":

                    st.success(
                        "positive Review"
                    )

                else:

                    st.error(
                        "Negative Review"
                    )

                st.metric(
                    label="Confidence Score",
                    value=f"{confidence:.2%}"
                )

                st.progress(confidence)

                st.info(
                    f"Model Used: {model_type}"
                )

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI backend.\n\n"
                "Make sure app_backend.py is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "Request timed out."
            )

        except Exception as e:

            st.error(
                f"Unexpected Error: {str(e)}"
            )

st.divider()

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Logistic Regression",
        "~89%"
    )

with col2:

    st.metric(
        "PyTorch",
        "~88%"
    )

st.caption(
    "Built with FastAPI, Streamlit, Scikit-Learn and PyTorch."
)