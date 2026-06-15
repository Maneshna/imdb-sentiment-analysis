import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

.stButton > button {
    width: 100%;
    height: 3.2rem;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 600;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <h1 style='text-align:center;'>
        🎬 IMDb Sentiment Analyzer
    </h1>

    <p style='text-align:center; font-size:18px;'>
        Analyze movie reviews using Machine Learning and Deep Learning models
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

left_col, right_col = st.columns([2, 1])

with left_col:

    review_text = st.text_area(
        "Movie Review",
        height=250,
        placeholder="Type your movie review here..."
    )

with right_col:

    model_type = st.selectbox(
        "Choose Model",
        [
            "logistic_regression",
            "pytorch"
        ]
    )

    st.info(
        f"Current Model:\n\n{model_type}"
    )

predict_button = st.button(
    "🚀 Analyze Sentiment"
)

if predict_button:

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

                result_col, chart_col = st.columns([1, 1])

                with result_col:

                    st.subheader(
                        "Prediction Result"
                    )

                    if sentiment.lower() == "positive":

                        st.success(
                            "😊 Positive Review"
                        )

                    else:

                        st.error(
                            "😞 Negative Review"
                        )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2%}"
                    )

                    st.progress(confidence)

                    st.info(
                        f"Model Used: {model_type}"
                    )

                with chart_col:

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=confidence * 100,
                            title={
                                "text": "Confidence Score"
                            },
                            gauge={
                                "axis": {
                                    "range": [0, 100]
                                }
                            }
                        )
                    )

                    fig.update_layout(
                        height=350
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI backend."
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

st.subheader(
    "Model Performance"
)

col1, col2 = st.columns(2)

with col1:

    st.image(
        "artifacts/training_accuracy.png",
        caption="Training Accuracy"
    )

with col2:

    st.image(
        "artifacts/confusion_matrix.png",
        caption="Confusion Matrix"
    )

st.divider()

metric1, metric2 = st.columns(2)

with metric1:

    st.metric(
        "Logistic Regression",
        "~89%"
    )

with metric2:

    st.metric(
        "PyTorch",
        "~88%"
    )

st.caption(
    "Built using Streamlit, FastAPI, Scikit-Learn and PyTorch."
)