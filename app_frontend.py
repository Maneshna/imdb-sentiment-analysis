import streamlit as st
import requests

st.set_page_config(
    page_title="AI Movie Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #050505 0%,
        #09090b 50%,
        #0f172a 100%
    );
}

.main {
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.big-title {
    text-align: center;
    font-size: 3.4rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.4rem;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.result-positive {
    background: rgba(59,130,246,0.06);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 20px;
    padding: 24px;
}

.result-negative {
    background: rgba(148,163,184,0.06);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 20px;
    padding: 24px;
}

.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 600;
}

@keyframes growBar {
    from {
        width: 0%;
    }
}

.sentiment-bar {
    width: 100%;
    height: 14px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 8px;
}

.sentiment-fill-positive {
    height: 100%;
    background: linear-gradient(
        90deg,
        #4f46e5,
        #6366f1
    );
    animation: growBar 1.2s ease-out;
}

.sentiment-fill-negative {
    height: 100%;
    background: linear-gradient(
        90deg,
        #334155,
        #475569
    );
    animation: growBar 1.2s ease-out;
}

hr {
    border-color: rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='big-title'>
🎬 AI Movie Review Analyzer
</div>

<div class='subtitle'>
Analyze movie reviews using Machine Learning and Deep Learning
</div>
""", unsafe_allow_html=True)

left, right = st.columns([3, 1])

with left:

    review_text = st.text_area(
        "Movie Review",
        height=260,
        placeholder="Write your movie review here..."
    )

with right:

    st.markdown("### 🤖 Model")

    model_choice = st.radio(
        "",
        [
            "Logistic Regression",
            "PyTorch"
        ]
    )

    if model_choice == "Logistic Regression":
        model_type = "logistic_regression"
    else:
        model_type = "pytorch"

predict = st.button("🚀 Analyze Sentiment")

if predict:

    if not review_text.strip():

        st.warning("Please enter a movie review.")

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
                confidence = result.get(
                    "confidence",
                    0
                )

                positive_prob = result.get(
                    "positive_probability",
                    0
                )

                negative_prob = result.get(
                    "negative_probability",
                    0
                )

                st.markdown("---")

                st.subheader("Prediction Result")

                if sentiment.lower() == "positive":

                    st.markdown(
                        f"""
                        <div class='result-positive'>
                            <h2>😊 Positive Review</h2>
                            <h1>{confidence:.2%}</h1>
                            <p>Confidence Score</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class='result-negative'>
                            <h2>😞 Negative Review</h2>
                            <h1>{confidence:.2%}</h1>
                            <p>Confidence Score</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("### Sentiment Breakdown")

                st.caption(
                    f"Model Used: {model_choice}"
                )

                st.markdown(
                    f"""
                    <p>Positive</p>

                    <div class="sentiment-bar">
                        <div
                            class="sentiment-fill-positive"
                            style="width:{positive_prob*100:.1f}%;">
                        </div>
                    </div>

                    <p>{positive_prob:.2%}</p>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <p>Negative</p>

                    <div class="sentiment-bar">
                        <div
                            class="sentiment-fill-negative"
                            style="width:{negative_prob*100:.1f}%;">
                        </div>
                    </div>

                    <p>{negative_prob:.2%}</p>
                    """,
                    unsafe_allow_html=True
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

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;color:#64748B'>
        Built with FastAPI • PyTorch • Scikit-Learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)