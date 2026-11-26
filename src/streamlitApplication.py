
"""Group Phi - Food Review Sentiment Analysis
CSE 3521 - Milestone 3 Streamlit Application
"""

import joblib
import pandas as pd
import streamlit as st

from pipeline import ARTIFACTS, ROOT, clean

M3_ARTIFACTS = ROOT / "models" / "m3_models.joblib"

# M2 baselines retain their original fitted vectorizers.
MODEL_FEATURES = {
    "Majority class": "bow",
    "Multinomial NB": "bow",
    "Complement NB": "tfidf",
    "Logistic Regression": "tfidf",
    "Linear SVM": "tfidf",
}


@st.cache_resource(show_spinner="Loading models...")
def load_models(path, modified_ns):
    """Load saved models.

    The modification timestamp is included in the
    cache key so updated artifacts are reloaded.
    """
    return joblib.load(path)


def get_predictions(artifacts, document, m3_artifacts=None, review_frame=None):
    """Predict with M2 baselines and each M3 model's saved preprocessing.

    M3 transformers receive raw Summary/Text columns, just as in training.
    A broken or unavailable model does not prevent the other models predicting.
    """

    predictions = []

    models = artifacts.get("models", {})

    # Reuse feature matrices for models sharing
    # the same vectorizer.
    feature_cache = {}

    for model_name, model in models.items():

        feature_type = MODEL_FEATURES.get(model_name)

        # Unknown models need a matching feature
        # configuration before they can be used.
        if feature_type is None:
            continue

        vectorizer = artifacts.get(feature_type)

        if vectorizer is None:
            continue

        try:
            if feature_type not in feature_cache:
                feature_cache[feature_type] = (
                    vectorizer.transform([document])
                )

            features = feature_cache[feature_type]

            if features.nnz == 0:
                continue

            sentiment = model.predict(features)[0]

            predictions.append({
                "Model": model_name,
                "Feature Type": feature_type.upper(),
                "Prediction": str(sentiment).capitalize()
            })

        except Exception as error:
            st.warning(
                f"{model_name} could not predict: {error}"
            )

    if m3_artifacts and review_frame is not None:
        for model_name, entry in m3_artifacts.get("models", {}).items():
            try:
                features = entry["transformer"].transform(review_frame)
                if hasattr(features, "nnz") and features.nnz == 0:
                    continue
                sentiment = entry["model"].predict(features)[0]
                predictions.append({
                    "Model": model_name,
                    "Feature Type": entry.get("feature_label", "M3 features"),
                    "Prediction": str(sentiment).capitalize(),
                })
            except Exception as error:
                st.warning(f"{model_name} could not predict: {error}")

    return predictions


def display_results(predictions, primary_model="Linear SVM"):
    """Display model results and comparisons."""

    df = pd.DataFrame(predictions)

    st.divider()
    st.header("Sentiment Analysis Results")

    by_name = {prediction["Model"]: prediction for prediction in predictions}
    primary = by_name.get(primary_model, by_name.get("Linear SVM", predictions[0]))

    sentiment = primary["Prediction"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Primary Model",
            primary["Model"]
        )

    with col2:
        st.metric(
            "Predicted Sentiment",
            sentiment
        )

    if sentiment.lower() == "positive":
        st.success("Positive Review")

    elif sentiment.lower() == "negative":
        st.error("Negative Review")

    else:
        st.warning("Neutral Review")

    st.subheader("Model Comparison")

    st.dataframe(
        df,
        width="stretch",
        hide_index=True
    )

    # Prediction distribution
    st.subheader("Prediction Distribution")

    counts = (
        df["Prediction"]
        .value_counts()
        .reindex(
            ["Negative", "Neutral", "Positive"],
            fill_value=0
        )
    )

    st.bar_chart(counts)

    st.caption(
        "This chart shows how many models predicted "
        "each sentiment. It does not represent "
        "prediction confidence."
    )


def main():

    st.set_page_config(
        page_title="Food Review Sentiment",
        layout="wide"
    )

    st.title("Food Review Sentiment Analysis")

    st.write(
        "Enter a food product review to predict "
        "whether its sentiment is positive, "
        "neutral, or negative."
    )

    st.caption(
        "Group Phi | CSE 3521 | Milestone 3"
    )

    st.divider()

    artifacts = {}
    m3_artifacts = {}

    # Each bundle is optional so the app also works with only M2 or only M3.
    with st.sidebar:

        st.header("Project Information")

        st.write(
            "This application uses machine learning "
            "models trained on Amazon Fine Food Reviews."
        )

        for title, path in (("M3 models", M3_ARTIFACTS), ("M2 baselines", ARTIFACTS)):
            st.subheader(title)
            if not path.is_file():
                st.info(f"{title} have not been trained on this computer.")
                continue
            try:
                bundle = load_models(str(path), path.stat().st_mtime_ns)
                if not isinstance(bundle, dict) or not isinstance(bundle.get("models"), dict):
                    raise ValueError("Model bundle must contain a models dictionary.")
                if path == M3_ARTIFACTS:
                    m3_artifacts = bundle
                else:
                    artifacts = bundle
                for name in bundle["models"]:
                    st.write(f"• {name}")
            except Exception as error:
                st.error(f"Could not load {title}: {error}")

        if not m3_artifacts.get("models") or not artifacts.get("models"):
            with st.expander("Model setup"):
                st.write("Run these commands from the project folder to build missing models.")
                if not m3_artifacts.get("models"):
                    st.code("python src/train_m3.py", language="bash")
                if not artifacts.get("models"):
                    st.code("python src/pipeline.py", language="bash")

        st.divider()

        st.caption(
            "Models and preprocessing are maintained "
            "by the Group Phi project team."
        )

    # User input
    st.header("Enter Your Review")

    with st.form("review_form"):

        summary = st.text_input(
            "Review Summary (Optional)",
            placeholder="Great flavor and quality"
        )

        review = st.text_area(
            "Food Product Review",
            placeholder=(
                "I really enjoyed this product. "
                "The flavor was amazing and "
                "I would definitely buy it again."
            ),
            height=170
        )

        submitted = st.form_submit_button(
            "Analyze Sentiment",
            type="primary"
        )

    if not submitted:
        st.info(
            "Enter a review above to begin analysis."
        )
        return

    # Call the clean function
    if not clean(review):
        st.warning("Please enter a review.")
        return

    if not artifacts.get("models") and not m3_artifacts.get("models"):
        st.error("No trained models are available. See Model setup in the sidebar.")
        return

    try:

        document = clean(
            f"{summary} . {review}"
        )
        review_frame = pd.DataFrame({"Summary": [summary], "Text": [review]})

        predictions = get_predictions(
            artifacts,
            document,
            m3_artifacts=m3_artifacts,
            review_frame=review_frame,
        )

        if not predictions:

            st.warning(
                "No compatible models were able "
                "to analyze this review."
            )

            return

        display_results(
            predictions,
            primary_model=m3_artifacts.get("primary_model", "Linear SVM"),
        )

    except Exception as error:

        st.error(
            "An error occurred while analyzing "
            "the review."
        )

        with st.expander("Technical Details"):
            st.exception(error)


if __name__ == "__main__":
    main()
