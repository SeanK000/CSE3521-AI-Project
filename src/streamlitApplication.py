"""Interactive sentiment prediction using the model saved by pipeline.py."""

import joblib
import streamlit as st

from pipeline import ARTIFACTS, clean


@st.cache_resource
def load_model(artifact_path, modified_ns):
    # Include the modification time so retraining refreshes the cached model.
    artifacts = joblib.load(artifact_path)
    return artifacts["tfidf"], artifacts["models"]["Linear SVM"]


def main():
    st.title("Food review sentiment")
    st.write("Predict whether a food review is negative, neutral, or positive.")
    st.caption("Group Phi · CSE 3521 · TF-IDF + Linear SVM")

    with st.form("review_form"):
        summary = st.text_input("Summary (optional)")
        review = st.text_area("Review")
        submitted = st.form_submit_button("Analyze")

    if not submitted:
        return
    if not clean(review):
        st.warning("Enter a review to analyze.")
        return
    if not ARTIFACTS.is_file():
        st.error("The trained model is missing. Run this from the project folder first:")
        st.code("python src/pipeline.py", language="bash")
        return

    vectorizer, model = load_model(ARTIFACTS, ARTIFACTS.stat().st_mtime_ns)
    document = clean(f"{summary} . {review}")
    features = vectorizer.transform([document])
    if features.nnz == 0:
        st.warning("No familiar words found. Try a more detailed food review.")
        return

    sentiment = model.predict(features)[0]
    st.success(f"Predicted sentiment: {sentiment.capitalize()}")


if __name__ == "__main__":
    main()
