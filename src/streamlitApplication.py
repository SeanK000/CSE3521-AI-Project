
"""Group Phi - Food Review Sentiment Analysis
CSE 3521 - Milestone 3 Streamlit Application
"""

import joblib
import pandas as pd
import streamlit as st

from pipeline import ARTIFACTS, clean # calls from pipeline

# New models can be added here if needed
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


def get_predictions(artifacts, document):
    """Generate predictions using models."""

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

    return predictions


def display_results(predictions):
    """Display model results and comparisons."""

    df = pd.DataFrame(predictions)

    st.divider()
    st.header("Sentiment Analysis Results")

    primary = next(
        (
            p for p in predictions
            if p["Model"] == "Linear SVM"
        ),
        predictions[0]
    )

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
        use_container_width=True,
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

    # Sidebar information
    with st.sidebar:

        st.header("Project Information")

        st.write(
            "This application uses machine learning "
            "models trained on Amazon Fine Food Reviews."
        )

        st.subheader("Available Models")

        if ARTIFACTS.is_file():

            try:
                artifacts = load_models(
                    str(ARTIFACTS),
                    ARTIFACTS.stat().st_mtime_ns
                )

                models = artifacts.get("models", {})

                for name in models:
                    st.write(f"• {name}")

            except Exception as error:
                st.error(f"Could not load models: {error}")

        else:
            st.info("No trained models found yet.")

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

    if not ARTIFACTS.is_file():

        st.error("Trained model artifacts not found.")

        st.write(
            "Run the team's training pipeline "
            "before making predictions."
        )

        st.code(
            "python src/pipeline.py",
            language="bash"
        )

        return

    try:

        artifacts = load_models(
            str(ARTIFACTS),
            ARTIFACTS.stat().st_mtime_ns
        )

        document = clean(
            f"{summary} . {review}"
        )

        predictions = get_predictions(
            artifacts,
            document
        )

        if not predictions:

            st.warning(
                "No compatible models were able "
                "to analyze this review."
            )

            return

        display_results(predictions)

    except Exception as error:

        st.error(
            "An error occurred while analyzing "
            "the review."
        )

        with st.expander("Technical Details"):
            st.exception(error)


if __name__ == "__main__":
    main()
