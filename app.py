import streamlit as st
import pandas as pd
import joblib

from utils.preprocess import (
    extract_features,
    extract_summary_stats,
    extract_top_tokens,
    extract_top_bigrams,
    extract_review_highlights
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Product Sales Executive Dashboard",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("📊 Product Sales Prediction using Reviews")
st.caption(
    "AI-powered decision-support system that analyzes customer reviews "
    "to estimate sales performance and provide actionable business insights."
)

st.divider()

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    clf = joblib.load("models/sales_classifier.pkl")
    reg = joblib.load("models/sales_regressor.pkl")

    clf_features = joblib.load("models/feature_order.pkl")
    reg_features = joblib.load("models/reg_feature_order.pkl")

    score_min = joblib.load("models/sales_score_p05.pkl")
    score_max = joblib.load("models/sales_score_p95.pkl")

    return clf, reg, clf_features, reg_features, score_min, score_max


classifier, regressor, clf_features, reg_features, score_min, score_max = load_models()

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload a CSV file containing reviews of a single product",
    type=["csv"]
)

if "df_time" not in st.session_state:
    st.session_state.df_time = None

# =========================
# MAIN FLOW
# =========================
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df_raw = df.copy()

        st.success("File uploaded successfully.")

        # =========================
        # PREVIEW
        # =========================
        with st.expander("🔍 Preview Uploaded Data"):
            st.dataframe(df.head())

        st.divider()

        # =========================
        # CONTROLS
        # =========================
        st.subheader("⚙️ Analysis Control Panel")

        num_reviews = st.selectbox(
            "Number of representative review highlights to display",
            options=[1, 2, 3],
            index=0
        )

        time_window = st.selectbox(
            "📅 Select time range for review activity",
            [
                "Last 7 days",
                "Last 30 days",
                "Last 12 months",
                "Last 2 years",
                "Last 5 years",
                "Last 10 years"
            ],
            index=1,
            key="time_window"
        )

        predict_button = st.button("🚀 Predict Sales Performance")

        st.divider()

        # =========================
        # PREDICTION
        # =========================
        if predict_button:

            with st.spinner("Analyzing reviews and generating predictions..."):

                features_df = extract_features(df)
                summary = extract_summary_stats(df)

                positive_reviews, negative_reviews = extract_review_highlights(
                    df, top_n=num_reviews
                )
                top_tokens = extract_top_tokens(df, top_n=10)
                top_bigrams = extract_top_bigrams(df, top_n=10)

                X_clf = features_df[clf_features]
                X_reg = features_df[reg_features]

                # TIME DATA
                if "Time" in df_raw.columns:
                    df_time = df_raw.copy()
                    df_time["review_date"] = pd.to_datetime(
                        df_time["Time"], unit="s", errors="coerce"
                    )
                    df_time = df_time.dropna(subset=["review_date"])
                    st.session_state.df_time = df_time
                else:
                    st.session_state.df_time = None

                sales_class = classifier.predict(X_clf)[0]
                sales_score = regressor.predict(X_reg)[0]

                raw_score = float(sales_score)
                normalized_score = (raw_score - score_min) / (score_max - score_min)
                normalized_score = max(0.0, min(1.0, normalized_score))

                total_reviews = summary["total_reviews"]
                positive_pct = summary["positive_pct"]
                negative_pct = summary["negative_pct"]

                if total_reviews >= 200 and (positive_pct >= 60 or negative_pct >= 60):
                    confidence_level = "High"
                elif total_reviews >= 50:
                    confidence_level = "Medium"
                else:
                    confidence_level = "Low"

            # =========================
            # SUMMARY
            # =========================
            st.subheader("📈 Reviews Summary")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Reviews", summary["total_reviews"])
            col2.metric("Average Rating", summary["avg_rating"])
            col3.metric("Positive (%)", summary["positive_pct"])
            col4.metric("Negative (%)", summary["negative_pct"])

            st.divider()

            # =========================
            # SALES PERFORMANCE
            # =========================
            st.subheader("📊 Sales Performance Evaluation")

            colA, colB, colC = st.columns(3)

            if sales_class == 1:
                colA.success("🟢 Predicted Sales Category: HIGH Sales Product")
            else:
                colA.warning("🔴 Predicted Sales Category: LOW Sales Product")

            colB.metric("Sales Strength Index (0–1)", round(normalized_score, 3))
            colC.metric("Prediction Confidence", confidence_level)

            st.progress(normalized_score)

            st.info(
                "Sales Strength Index is a normalized indicator of relative sales performance, "
                "not a probability or future sales forecast.\n\n"
                "Prediction Confidence indicates how reliable the result is based on the amount "
                "and consistency of review data."
            )

            st.divider()

            # =========================
            # REVIEW ACTIVITY
            # =========================
            st.subheader("📊 Review Activity Over Time")

            df_time = st.session_state.df_time

            if df_time is not None and not df_time.empty:

                if time_window == "Last 7 days":
                    cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
                    plot_df = df_time[df_time["review_date"] >= cutoff]
                    counts = plot_df.groupby(plot_df["review_date"].dt.date).size()

                elif time_window == "Last 30 days":
                    cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
                    plot_df = df_time[df_time["review_date"] >= cutoff]
                    counts = plot_df.groupby(plot_df["review_date"].dt.date).size()

                elif time_window == "Last 12 months":
                    cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
                    plot_df = df_time[df_time["review_date"] >= cutoff]
                    counts = plot_df.groupby(
                        plot_df["review_date"].dt.to_period("M")
                    ).size()
                    counts.index = counts.index.to_timestamp()

                elif time_window == "Last 2 years":
                    cutoff = pd.Timestamp.now() - pd.DateOffset(years=2)
                    plot_df = df_time[df_time["review_date"] >= cutoff]
                    counts = plot_df.groupby(
                        plot_df["review_date"].dt.to_period("M")
                    ).size()
                    counts.index = counts.index.to_timestamp()

                elif time_window == "Last 5 years":
                    cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
                    plot_df = df_time[df_time["review_date"] >= cutoff]
                    counts = plot_df.groupby(
                        plot_df["review_date"].dt.to_period("Y")
                    ).size()
                    counts.index = counts.index.to_timestamp()

                else:
                    cutoff = pd.Timestamp.now() - pd.DateOffset(years=10)
                    plot_df = df_time[df_time["review_date"] >= cutoff]
                    counts = plot_df.groupby(
                        plot_df["review_date"].dt.to_period("Y")
                    ).size()
                    counts.index = counts.index.to_timestamp()

                counts = counts.sort_index()

                if not counts.empty:
                    st.line_chart(counts)
                else:
                    st.info("No review activity found in the selected time range.")

            else:
                st.info("Run prediction to view review activity over time.")

            st.divider()

            # =========================
            # BUSINESS INTERPRETATION
            # =========================
            st.subheader("🧠 Business Interpretation")

            interpretation_points = []

            if total_reviews >= 200:
                interpretation_points.append(
                    "The product shows strong customer engagement with a high number of reviews."
                )
            elif total_reviews >= 50:
                interpretation_points.append(
                    "The product has moderate customer engagement based on review volume."
                )
            else:
                interpretation_points.append(
                    "The product has limited customer engagement due to a low number of reviews."
                )

            if summary["avg_rating"] >= 4.0:
                interpretation_points.append(
                    "Customer ratings indicate high perceived product quality."
                )
            elif summary["avg_rating"] >= 3.0:
                interpretation_points.append(
                    "Customer ratings suggest acceptable but improvable product quality."
                )
            else:
                interpretation_points.append(
                    "Low customer ratings indicate potential quality issues."
                )

            if positive_pct >= 60:
                interpretation_points.append(
                    "A majority of customer reviews express positive sentiment."
                )
            elif negative_pct >= 30:
                interpretation_points.append(
                    "A significant portion of reviews express negative sentiment."
                )
            else:
                interpretation_points.append(
                    "Customer sentiment is mixed with no strong polarity."
                )

            if normalized_score >= 0.6:
                interpretation_points.append(
                    "Overall, the product demonstrates strong sales performance relative to other products."
                )
            elif normalized_score >= 0.3:
                interpretation_points.append(
                    "Overall, the product shows moderate sales performance in the market."
                )
            else:
                interpretation_points.append(
                    "Overall, the product shows weak sales performance compared to most products."
                )

            for point in interpretation_points:
                st.write("• " + point)

            st.divider()

            # =========================
            # KEY THEMES
            # =========================
            st.subheader("📌 Key Themes Detected")

            theme_keywords = [
                "delivery", "delay", "shipping", "courier", "arrival",
                "quality", "defect", "issue", "problem", "damage",
                "broken", "faulty", "cracked", "leak",
                "packaging", "package", "box", "seal", "wrap",
                "price", "cost", "value", "worth",
                "refund", "return", "replacement", "support", "service",
                "original", "fake", "authentic", "duplicate"
            ]

            filtered_themes = []

            for phrase, _ in top_bigrams:
                words = phrase.split()
                for kw in theme_keywords:
                    if kw in words:
                        filtered_themes.append(phrase)
                        break

            filtered_themes = list(dict.fromkeys(filtered_themes))[:5]

            if filtered_themes:
                st.info(
                    "The following business-relevant themes frequently appear in customer reviews:\n\n"
                    + ", ".join(filtered_themes)
                )
            else:
                st.info(
                    "No strong delivery, quality, packaging, or service-related themes were detected."
                )

            st.divider()

            # =========================
            # REVIEW HIGHLIGHTS
            # =========================
            st.subheader("🗣️ Review Highlights")

            with st.expander("👍 Top Positive Reviews"):
                for i, review in enumerate(positive_reviews, start=1):
                    st.success(f"{i}. {review}")

            with st.expander("👎 Top Negative Reviews"):
                for i, review in enumerate(negative_reviews, start=1):
                    st.error(f"{i}. {review}")

            st.divider()

            # =========================
            # ACTIONABLE INSIGHTS
            # =========================
            st.subheader("📌 Actionable Insights from Reviews")

            actions = []

            bigram_rules = {
                "late delivery": "Improve delivery timelines and logistics management.",
                "delivery delay": "Optimize order fulfillment and shipping processes.",
                "poor quality": "Investigate and improve product quality issues.",
                "bad quality": "Address quality control problems.",
                "good quality": "Highlight product quality in marketing campaigns.",
                "great taste": "Promote taste as a key product strength.",
                "poor packaging": "Improve packaging to prevent product damage.",
                "damaged product": "Enhance handling and packaging standards.",
                "fast delivery": "Use fast delivery as a competitive advantage.",
                "value money": "Position the product as good value for money."
            }

            unigram_rules = {
                "delay": "Review delivery and logistics operations.",
                "refund": "Improve refund and return experience.",
                "packaging": "Enhance packaging quality.",
                "quality": "Maintain consistent product quality.",
                "taste": "Emphasize taste in promotions."
            }

            for phrase, _ in top_bigrams:
                if phrase in bigram_rules:
                    actions.append(
                        f"• Frequent mention of **'{phrase}'** → {bigram_rules[phrase]}"
                    )

            if not actions:
                for word, _ in top_tokens:
                    if word in unigram_rules:
                        actions.append(
                            f"• Frequent mention of **'{word}'** → {unigram_rules[word]}"
                        )

            if actions:
                for action in actions[:5]:
                    st.write(action)
            else:
                st.write(
                    "No strong recurring themes detected in customer reviews. "
                    "Customer feedback appears balanced."
                )

    except Exception as e:
        st.error("❌ Error processing the file.")
        st.write(str(e))

st.divider()
st.caption("Machine Learning Project | Sales Prediction using Review Analytics")