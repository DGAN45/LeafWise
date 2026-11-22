import streamlit as st
import plotly.express as px
import pandas as pd

from config import ALLOWED_IMAGE_EXTENSIONS
from utils.disease_info import get_disease_info
from utils.image_processing import (
    ImageValidationError,
    load_and_prepare_image,
)
from utils.predictor import PredictionError, load_model, predict_from_array
from utils.history import add_history_record, clear_history, load_history


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --leafwise-forest: #173F2B;
            --leafwise-green: #237A4B;
            --leafwise-emerald: #2F9A62;
            --leafwise-sage: #E7F1E9;
            --leafwise-ink: #203229;
            --leafwise-muted: #63736A;
            --leafwise-border: #E0E8E1;
            --leafwise-canvas: #F5F8F5;
        }
        .stApp {
            background: var(--leafwise-canvas);
            color: var(--leafwise-ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container {
            max-width: 1380px;
            padding: 2rem 2.4rem 3rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #173F2B 0%, #205D3C 100%);
            border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #D3E5D8;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] > label {
            color: #BFD4C5;
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .1em;
            text-transform: uppercase;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 10px;
            padding: 8px 10px;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255,255,255,.10);
        }
        [data-testid="stMain"] {
            color: var(--leafwise-ink);
        }
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4 {
            color: var(--leafwise-forest);
            letter-spacing: -.02em;
        }
        [data-testid="stMain"] p,
        [data-testid="stMain"] label,
        [data-testid="stMain"] [data-testid="stCaptionContainer"] {
            color: var(--leafwise-muted);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF;
            border: 1px solid var(--leafwise-border) !important;
            border-radius: 16px !important;
            box-shadow: 0 5px 18px rgba(23,63,43,.055);
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            border: 0 !important;
            background: transparent !important;
        }
        .lw-hero {
            background: linear-gradient(115deg, #E9F3EA 0%, #F9FBF8 78%);
            border: 1px solid #DCE9DE;
            border-radius: 20px;
            padding: 1.8rem 2rem;
            margin: .2rem 0 1.35rem;
        }
        .lw-eyebrow {
            color: var(--leafwise-green) !important;
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .lw-hero h1 {
            margin: .35rem 0 .45rem;
            font-size: clamp(1.7rem, 3vw, 2.25rem);
            line-height: 1.15;
        }
        .lw-hero p {
            max-width: 740px;
            margin: 0;
            font-size: .98rem;
            color: #53665A !important;
        }
        .lw-section-title {
            margin: 1.3rem 0 .8rem;
            color: var(--leafwise-forest);
            font-size: 1.05rem;
            font-weight: 700;
        }
        .lw-metric-value {
            color: var(--leafwise-forest);
            font-size: 1.7rem;
            font-weight: 750;
            line-height: 1.15;
        }
        .lw-metric-label {
            margin-top: .35rem;
            color: var(--leafwise-muted);
            font-size: .83rem;
        }
        .lw-result-confidence {
            color: var(--leafwise-green);
            font-size: 2.5rem;
            font-weight: 750;
            letter-spacing: -.04em;
            line-height: 1;
        }
        .lw-warning {
            background: #FFF8E8;
            border: 1px solid #F1E4C9;
            border-left: 4px solid #D9A441;
            border-radius: 10px;
            padding: .8rem 1rem;
            color: #65532F;
        }
        [data-testid="stFileUploaderDropzone"] {
            min-height: 150px;
            background: #FBFDFC;
            border: 1.5px dashed #AFC7B5;
            border-radius: 14px;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--leafwise-emerald);
            background: #F4FAF5;
        }
        [data-testid="stMain"] .stButton button[kind="primary"] {
            min-height: 2.8rem;
            background: var(--leafwise-green);
            border: 1px solid var(--leafwise-green);
            border-radius: 10px;
            color: #FFFFFF;
            font-weight: 650;
            transition: background .15s ease;
        }
        [data-testid="stMain"] .stButton button[kind="primary"]:hover {
            background: var(--leafwise-forest);
            border-color: var(--leafwise-forest);
        }
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stMain"] .stButton button[kind="secondary"] {
            background: #FFFFFF;
            border: 1px solid var(--leafwise-border);
            border-radius: 9px;
            color: var(--leafwise-ink);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--leafwise-border);
            border-radius: 12px;
            overflow: hidden;
        }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid var(--leafwise-border);
            border-radius: 14px;
            padding: .85rem 1rem;
        }
        [data-testid="stPlotlyChart"] {
            border-radius: 12px;
        }
        @media (max-width: 760px) {
            .block-container { padding: 1.2rem 1rem 2rem; }
            .lw-hero { padding: 1.35rem; }
            .lw-result-confidence { font-size: 2.1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:.7rem;margin:.5rem 0 1.5rem;">
          <div style="width:38px;height:38px;border-radius:12px;background:#E7F1E9;
                      color:#173F2B;display:flex;align-items:center;justify-content:center;
                      font-weight:800;letter-spacing:-.04em;">LW</div>
          <div>
            <div style="font-size:1.16rem;font-weight:750;line-height:1.1;">LeafWise</div>
            <div style="font-size:.72rem;color:#D3E5D8;margin-top:.25rem;">
              PLANT HEALTH INTELLIGENCE
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.sidebar.radio(
        "Workspace",
        [
            "Dashboard",
            "Disease Detection",
            "Prediction History",
            "Analytics",
            "About LeafWise",
        ],
        label_visibility="collapsed",
    )


@st.cache_resource(show_spinner=False)
def load_cached_model():
    return load_model()


def _healthy_mask(history: pd.DataFrame) -> pd.Series:
    return history["disease_name"].fillna("").str.lower().str.contains(
        "healthy",
        regex=False,
    )


def _recent_history(history: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    return history.sort_values(
        "timestamp",
        ascending=False,
        na_position="last",
    ).head(limit)


def render_home() -> None:
    history = load_history()
    healthy_count = int(_healthy_mask(history).sum()) if not history.empty else 0
    diseased_count = len(history) - healthy_count

    st.markdown(
        """
        <div class="lw-hero">
            <div class="lw-eyebrow">AI-powered crop intelligence</div>
            <h1>Plant health, made clearer.</h1>
            <p>
                LeafWise screens leaf images with your trained MobileNetV2 model,
                then presents likely results and practical next steps.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(3, gap="medium")
    metric_data = [
        ("Total predictions", len(history), "Saved analyses"),
        ("Healthy leaves", healthy_count, "Model classified as healthy"),
        ("Disease predictions", diseased_count, "Model detected a possible issue"),
    ]
    for column, (label, value, help_text) in zip(metric_columns, metric_data):
        with column:
            with st.container(border=True):
                st.metric(label, value, help=help_text)

    chart_column, recent_column = st.columns([1.15, 0.85], gap="large")
    with chart_column:
        with st.container(border=True):
            st.markdown("### Recent activity")
            trend = history.dropna(subset=["timestamp"]).copy()
            if trend.empty:
                st.info("Recent prediction activity will appear here after your first analysis.")
            else:
                trend["date"] = trend["timestamp"].dt.date
                daily_counts = (
                    trend.groupby("date", as_index=False)
                    .size()
                    .rename(columns={"size": "Predictions"})
                )
                figure = px.area(
                    daily_counts,
                    x="date",
                    y="Predictions",
                    markers=True,
                    color_discrete_sequence=["#2F9A62"],
                    labels={"date": "Date"},
                )
                figure.update_layout(
                    height=290,
                    margin=dict(l=8, r=8, t=12, b=8),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis=dict(dtick=1, rangemode="tozero"),
                )
                st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    with recent_column:
        with st.container(border=True):
            st.markdown("### Latest predictions")
            recent = _recent_history(history, limit=5)
            if recent.empty:
                st.info("No predictions saved yet.")
            else:
                for _, row in recent.iterrows():
                    timestamp = row["timestamp"]
                    time_label = (
                        timestamp.strftime("%b %d, %H:%M")
                        if pd.notna(timestamp)
                        else "Time unavailable"
                    )
                    st.markdown(
                        f"**{row['plant_name']}** · {row['disease_name']}  \n"
                        f"<span style='color:#63736A;font-size:.82rem;'>"
                        f"{time_label} · "
                        f"{row['confidence']:.1%} confidence</span>"
                        if pd.notna(row["confidence"])
                        else f"**{row['plant_name']}** · {row['disease_name']}  \n"
                        f"<span style='color:#63736A;font-size:.82rem;'>"
                        f"{time_label} · confidence unavailable</span>",
                        unsafe_allow_html=True,
                    )
                    st.divider()

    st.markdown(
        '<div class="lw-warning"><strong>For screening, not diagnosis.</strong> '
        'Confirm concerning results with a local agricultural expert.</div>',
        unsafe_allow_html=True,
    )


def render_detection() -> None:
    st.markdown("## Disease Detection")
    st.caption("Upload a clear leaf photo to get a model-assisted screening result.")

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=list(ALLOWED_IMAGE_EXTENSIONS),
            accept_multiple_files=False,
            help="JPG, JPEG, or PNG. Maximum accepted size: 10 MB.",
        )
        st.caption("For best results, use a well-lit close-up with the leaf in focus.")

    if uploaded_file is None:
        st.info("Your image preview and prediction results will appear here after upload.")
        return

    try:
        display_image, image_array = load_and_prepare_image(
            uploaded_file,
            uploaded_file.name,
            uploaded_file.size,
        )
    except ImageValidationError as exc:
        st.error(str(exc))
        return

    image_column, action_column = st.columns([1.15, 0.85], gap="large")
    with image_column:
        with st.container(border=True):
            st.markdown("### Image preview")
            st.image(display_image, caption="Uploaded leaf image", width="stretch")

    with action_column:
        with st.container(border=True):
            st.markdown("### Ready for screening")
            st.write("LeafWise will resize the photo to 224 × 224 pixels and return the top three classes.")
            st.caption("The model confidence is not a guaranteed diagnosis.")
            analyze = st.button(
                "Analyze leaf",
                type="primary",
                width="stretch",
            )

    if not analyze:
        return

    try:
        with st.spinner("Analyzing the leaf image..."):
            load_cached_model()
            predictions = predict_from_array(image_array)
    except PredictionError as exc:
        st.error(str(exc))
        return

    best = predictions[0]
    info = get_disease_info(best["class_name"])
    confidence = best["probability"] * 100

    try:
        add_history_record(
            plant_name=info["plant_name"],
            disease_name=info["disease_name"],
            predicted_class=best["class_name"],
            confidence=best["probability"],
        )
    except (OSError, ValueError) as exc:
        st.warning(f"The prediction was completed, but it could not be saved: {exc}")

    st.markdown("---")
    result_column, detail_column = st.columns([0.85, 1.15], gap="large")

    with result_column:
        with st.container(border=True):
            st.markdown("### Primary prediction")
            st.markdown(
                f'<div class="lw-result-confidence">{confidence:.1f}%</div>'
                '<div class="lw-metric-label">model confidence</div>',
                unsafe_allow_html=True,
            )
            st.subheader(info["disease_name"])
            st.write(f"**Plant:** {info['plant_name']}")
            st.markdown(
                '<div class="lw-warning">This is a model prediction, '
                'not a guaranteed diagnosis.</div>',
                unsafe_allow_html=True,
            )

    with detail_column:
        with st.container(border=True):
            st.markdown("### About this result")
            st.write(info["description"])
            st.markdown("**Common visible signs**")
            st.write(info["symptoms"])
            st.markdown("**General management suggestions**")
            for suggestion in info["management"]:
                st.markdown(f"- {suggestion}")
            st.caption(info["disclaimer"])

    with st.container(border=True):
        st.markdown("### Top 3 predictions")
        for index, prediction in enumerate(predictions, start=1):
            percent = prediction["probability"] * 100
            disease_name = get_disease_info(prediction["class_name"])["disease_name"]
            prediction_column, confidence_column = st.columns([3, 1])
            with prediction_column:
                st.markdown(f"**{index}. {disease_name}**")
            with confidence_column:
                st.markdown(f"**{percent:.1f}%**")
            st.progress(min(max(prediction["probability"], 0.0), 1.0))


def render_about() -> None:
    st.markdown(
        """
        <div class="lw-hero">
            <div class="lw-eyebrow">About the project</div>
            <h1>AI-assisted plant health screening</h1>
            <p>
                LeafWise brings image classification and clear, cautious guidance
                together in an approachable agricultural dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    intro_column, model_column = st.columns([1.1, .9], gap="large")
    with intro_column:
        with st.container(border=True):
            st.markdown("### What is LeafWise?")
            st.write(
                "LeafWise is a student project that demonstrates how deep learning "
                "can support early plant-leaf screening. A user uploads a photo, "
                "and the application presents likely classes with general context."
            )
            st.write(
                "The classifier uses the provided MobileNetV2 transfer-learning "
                "model. Its learned image features are passed to a classification "
                "head that predicts among the original 38 labels."
            )

    with model_column:
        with st.container(border=True):
            st.markdown("### Model at a glance")
            st.metric("Supported classes", "38")
            st.caption("Input: 224 × 224 RGB · Output: 38-class softmax")
            st.caption("The model's own preprocessing layer is preserved.")

    with st.container(border=True):
        st.markdown("### Limitations and responsible use")
        st.markdown(
            """
- Predictions depend on image quality, lighting, background, and similarity to
  the model's training data.
- A confidence percentage is a model score, not a probability of guaranteed
  correctness or a professional diagnosis.
- The model only predicts among its 38 known classes; other plants or conditions
  may be misclassified.
- Disease descriptions are general information. Confirm concerns with a local
  agricultural expert before making crop-management decisions.
"""
        )
        st.warning(
            "LeafWise is for educational screening and does not replace field "
            "inspection or expert advice."
        )

def render_history() -> None:
    st.markdown("## Prediction History")
    st.caption(
        "Your latest model results are stored locally. Uploaded images are not retained."
    )

    history = load_history()

    header_left, header_right = st.columns([3, 1])
    with header_left:
        with st.container(border=True):
            st.metric("Saved predictions", len(history))
    with header_right:
        with st.container(border=True):
            st.markdown("**Manage records**")
            confirm = st.checkbox("Confirm clear")
            if st.button(
                "Clear history",
                type="secondary",
                disabled=not confirm,
                width="stretch",
            ):
                clear_history()
                st.success("Prediction history cleared.")
                st.rerun()

    if history.empty:
        st.info("No predictions have been saved yet.")
        return

    display_history = _recent_history(history, limit=len(history)).copy()
    display_history["Confidence"] = display_history["confidence"].map(
        lambda value: f"{value * 100:.2f}%"
        if pd.notna(value)
        else "Unavailable"
    )
    display_history["Timestamp"] = display_history["timestamp"].map(
        lambda value: value.strftime("%Y-%m-%d %H:%M")
        if pd.notna(value)
        else "Unknown"
    )

    st.markdown("### All predictions")
    st.caption("Images are intentionally not stored; the table contains prediction metadata only.")
    st.dataframe(
        display_history[
            ["Timestamp", "plant_name", "disease_name", "Confidence", "predicted_class"]
        ].rename(
            columns={
                "plant_name": "Plant",
                "disease_name": "Prediction",
                "predicted_class": "Model class",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def render_analytics() -> None:
    st.markdown("## Analytics Dashboard")

    history = load_history()
    if history.empty:
        st.info("No prediction data is available yet. Analyze a leaf to begin.")
        return

    valid_confidence = history["confidence"].dropna()
    healthy_count = int(_healthy_mask(history).sum())
    diseased_count = len(history) - healthy_count
    disease_counts = (
        history.loc[~_healthy_mask(history), "disease_name"]
        .fillna("Unknown")
        .value_counts()
    )

    total, avg_confidence, healthy, diseased = st.columns(4)
    total.metric("Total predictions", len(history))
    avg_confidence.metric(
        "Average confidence",
        f"{valid_confidence.mean() * 100:.2f}%"
        if not valid_confidence.empty
        else "N/A",
    )
    healthy.metric("Healthy predictions", int(healthy_count))
    diseased.metric("Disease predictions", int(diseased_count))

    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        with st.container(border=True):
            st.markdown("### Healthy vs. diseased")
            health_distribution = {
                "Healthy": healthy_count,
                "Possible disease": diseased_count,
            }
            health_distribution = {
                label: count
                for label, count in health_distribution.items()
                if count > 0
            }
            health_chart = px.pie(
                names=list(health_distribution.keys()),
                values=list(health_distribution.values()),
                hole=.62,
                color_discrete_sequence=["#2F9A62", "#173F2B"],
            )
            health_chart.update_layout(
                height=310,
                margin=dict(l=8, r=8, t=8, b=8),
                paper_bgcolor="white",
                legend_title_text="",
            )
            st.plotly_chart(health_chart, width="stretch", config={"displayModeBar": False})

    with chart_right:
        with st.container(border=True):
            st.markdown("### Most frequent disease predictions")
            if disease_counts.empty:
                st.info("No disease predictions have been recorded.")
            else:
                disease_chart = px.bar(
                    x=disease_counts.head(8).values,
                    y=disease_counts.head(8).index,
                    orientation="h",
                    labels={"x": "Predictions", "y": ""},
                    color_discrete_sequence=["#2F9A62"],
                )
                disease_chart.update_layout(
                    height=310,
                    margin=dict(l=8, r=8, t=8, b=8),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis={"categoryorder": "total ascending"},
                )
                st.plotly_chart(disease_chart, width="stretch", config={"displayModeBar": False})

    distribution_column, trend_column = st.columns(2, gap="large")
    with distribution_column:
        with st.container(border=True):
            st.markdown("### Prediction distribution")
            class_counts = history["disease_name"].fillna("Unknown").value_counts()
            top_classes = class_counts.head(7)
            remaining = int(class_counts.iloc[7:].sum())
            if remaining:
                top_classes.loc["Other predictions"] = remaining
            distribution_chart = px.pie(
                names=top_classes.index,
                values=top_classes.values,
                hole=.58,
                color_discrete_sequence=px.colors.sequential.Greens,
            )
            distribution_chart.update_layout(
                height=310,
                margin=dict(l=8, r=8, t=8, b=8),
                paper_bgcolor="white",
                legend_title_text="",
            )
            st.plotly_chart(
                distribution_chart,
                width="stretch",
                config={"displayModeBar": False},
            )

    with trend_column:
        with st.container(border=True):
            st.markdown("### Prediction activity over time")
            trend = history.dropna(subset=["timestamp"]).copy()
            if trend.empty:
                st.info("No valid timestamps are available for the trend chart.")
            else:
                trend["date"] = trend["timestamp"].dt.date
                daily_counts = (
                    trend.groupby("date", as_index=False)
                    .size()
                    .rename(columns={"size": "Predictions"})
                )
                trend_chart = px.line(
                    daily_counts,
                    x="date",
                    y="Predictions",
                    markers=True,
                    labels={"date": "Date"},
                    color_discrete_sequence=["#237A4B"],
                )
                trend_chart.update_layout(
                    height=310,
                    margin=dict(l=8, r=8, t=8, b=8),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis=dict(dtick=1, rangemode="tozero"),
                )
                st.plotly_chart(trend_chart, width="stretch", config={"displayModeBar": False})


def main() -> None:
    st.set_page_config(
        page_title="LeafWise",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    page = render_sidebar()

    if page == "Dashboard":
        render_home()
    elif page == "Disease Detection":
        render_detection()
    elif page == "Prediction History":
        render_history()
    elif page == "Analytics":
        render_analytics()
    else:
        render_about()


if __name__ == "__main__":
    main()
