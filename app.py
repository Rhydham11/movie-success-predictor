"""Interactive dashboard for predicting movie success from pre-release metadata."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(page_title="Movie Success Predictor", page_icon="🎬", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "movie_metadata.xlsx"
RANDOM_STATE = 42
CLASS_ORDER = ["Flop", "Average", "Hit"]

# These fields can reasonably be known before release. Review counts, vote counts,
# gross earnings and movie Facebook likes are deliberately excluded to avoid leakage.
NUMERIC_FEATURES = [
    "duration",
    "director_facebook_likes",
    "actor_1_facebook_likes",
    "actor_2_facebook_likes",
    "actor_3_facebook_likes",
    "cast_total_facebook_likes",
    "facenumber_in_poster",
    "budget",
    "title_year",
    "aspect_ratio",
]
CATEGORICAL_FEATURES = [
    "color",
    "genre_primary",
    "language",
    "country",
    "content_rating",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load, de-duplicate, and prepare the supplied movie dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Keep data/movie_metadata.xlsx next to app.py."
        )

    movies = pd.read_excel(DATA_PATH)
    movies = movies.drop_duplicates().copy()
    movies["movie_title"] = movies["movie_title"].astype(str).str.replace("\u00a0", " ").str.strip()
    movies["genre_primary"] = (
        movies["genres"].fillna("Unknown").astype(str).str.split("|").str[0]
    )

    # Assignment rule: [1, 3) Flop, [3, 6) Average, [6, 10] Hit.
    movies["success_class"] = pd.cut(
        movies["imdb_score"],
        bins=[-np.inf, 3, 6, np.inf],
        labels=CLASS_ORDER,
        right=False,
    ).astype(str)
    return movies


def build_preprocessor() -> ColumnTransformer:
    """Return a preprocessing pipeline shared by every candidate model."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=10
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


@st.cache_resource(show_spinner="Training and comparing models...")
def train_models() -> dict:
    """Train three classifiers and retain the held-out test-set results."""
    movies = load_data()
    X = movies[FEATURES].copy()
    y = movies["success_class"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    fitted_models: dict[str, Pipeline] = {}
    comparison_rows: list[dict] = []
    for name, classifier in classifiers.items():
        model = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)]
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        comparison_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Macro F1": f1_score(y_test, predictions, average="macro"),
            }
        )
        fitted_models[name] = model

    random_forest = fitted_models["Random Forest"]
    rf_predictions = random_forest.predict(X_test)
    rf_probabilities = random_forest.predict_proba(X_test)
    comparison = pd.DataFrame(comparison_rows).sort_values(
        "Macro F1", ascending=False
    )

    transformed_names = random_forest.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    importances = pd.DataFrame(
        {
            "Feature": [
                name.replace("numeric__", "").replace("categorical__", "")
                for name in transformed_names
            ],
            "Importance": random_forest.named_steps["classifier"].feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return {
        "models": fitted_models,
        "comparison": comparison,
        "X_test": X_test,
        "y_test": y_test,
        "rf_predictions": rf_predictions,
        "rf_probabilities": rf_probabilities,
        "report": pd.DataFrame(
            classification_report(
                y_test,
                rf_predictions,
                labels=CLASS_ORDER,
                output_dict=True,
                zero_division=0,
            )
        ).T,
        "importances": importances,
    }


def class_distribution_chart(movies: pd.DataFrame) -> go.Figure:
    counts = (
        movies["success_class"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .rename_axis("Success class")
        .reset_index(name="Movies")
    )
    return px.bar(
        counts,
        x="Success class",
        y="Movies",
        color="Success class",
        color_discrete_map={"Flop": "#e76f51", "Average": "#e9c46a", "Hit": "#2a9d8f"},
        text="Movies",
    ).update_layout(showlegend=False, yaxis_title="Number of movies")


def pick_default(options: list[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def render_overview(movies: pd.DataFrame) -> None:
    st.title("Movie Success Predictor")
    st.write(
        "Classify a movie as a **Flop**, **Average**, or **Hit** using pre-release metadata "
        "from the supplied IMDb movie dataset."
    )
    st.caption(
        "Class rule: Flop < 3.0, Average 3.0–5.9, Hit ≥ 6.0 IMDb score."
    )

    counts = movies["success_class"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Movies after de-duplication", f"{len(movies):,}")
    col2.metric("Hit movies", f"{counts['Hit']:,}")
    col3.metric("Average movies", f"{counts['Average']:,}")
    col4.metric("Flop movies", f"{counts['Flop']:,}")

    left, right = st.columns([1.15, 1])
    with left:
        fig = px.histogram(
            movies,
            x="imdb_score",
            nbins=30,
            color="success_class",
            category_orders={"success_class": CLASS_ORDER},
            color_discrete_map={"Flop": "#e76f51", "Average": "#e9c46a", "Hit": "#2a9d8f"},
            labels={"imdb_score": "IMDb score", "success_class": "Class"},
            title="IMDb score distribution",
        )
        fig.add_vline(x=3, line_dash="dash", line_color="#6c757d")
        fig.add_vline(x=6, line_dash="dash", line_color="#6c757d")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.plotly_chart(class_distribution_chart(movies), use_container_width=True)

    st.subheader("Why the model uses only pre-release information")
    st.write(
        "The model excludes gross earnings, review counts, vote counts, and movie-page likes. "
        "Those values are only known after a release and would make an unrealistic prediction "
        "look better than it really is."
    )


def render_explore(movies: pd.DataFrame) -> None:
    st.title("Explore the data")
    st.write(
        "The original spreadsheet contains 5,043 records and 28 variables. The dashboard removes "
        "45 exact duplicate rows before analysis."
    )
    tab1, tab2, tab3 = st.tabs(["Data quality", "Relationships", "Movie records"])

    with tab1:
        missing = (
            movies.isna()
            .sum()
            .sort_values(ascending=False)
            .rename("Missing values")
            .to_frame()
        )
        missing = missing[missing["Missing values"] > 0]
        st.dataframe(missing, use_container_width=True)
        st.info(
            "Missing numeric values are filled with the training-set median. Missing categories are "
            "filled with the most frequent training-set category."
        )

    with tab2:
        numeric_choice = st.selectbox(
            "Choose a numeric feature", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("budget")
        )
        scatter = px.scatter(
            movies,
            x=numeric_choice,
            y="imdb_score",
            color="success_class",
            hover_name="movie_title",
            category_orders={"success_class": CLASS_ORDER},
            color_discrete_map={"Flop": "#e76f51", "Average": "#e9c46a", "Hit": "#2a9d8f"},
            labels={"imdb_score": "IMDb score", numeric_choice: numeric_choice.replace("_", " ").title()},
        )
        st.plotly_chart(scatter, use_container_width=True)

        category_choice = st.selectbox(
            "Compare scores by category", ["genre_primary", "language", "country", "content_rating"]
        )
        category_stats = (
            movies.groupby(category_choice, dropna=False)
            .agg(mean_imdb_score=("imdb_score", "mean"), movies=("imdb_score", "size"))
            .query("movies >= 10")
            .sort_values("mean_imdb_score", ascending=False)
            .head(15)
            .reset_index()
        )
        category_stats[category_choice] = category_stats[category_choice].fillna("Missing")
        category_chart = px.bar(
            category_stats.sort_values("mean_imdb_score"),
            x="mean_imdb_score",
            y=category_choice,
            orientation="h",
            hover_data=["movies"],
            labels={"mean_imdb_score": "Mean IMDb score", category_choice: category_choice.replace("_", " ").title()},
        )
        st.plotly_chart(category_chart, use_container_width=True)

    with tab3:
        search = st.text_input("Find a movie title")
        records = movies[
            ["movie_title", "title_year", "genres", "country", "budget", "gross", "imdb_score", "success_class"]
        ].copy()
        if search:
            records = records[
                records["movie_title"].str.contains(search, case=False, na=False)
            ]
        st.dataframe(records.head(200), use_container_width=True, hide_index=True)


def render_performance(movies: pd.DataFrame) -> None:
    results = train_models()
    st.title("Model performance")
    st.write(
        "Every model uses the same 80% training and 20% held-out test split with stratification. "
        "Macro F1 is emphasized because the Flop class is much smaller than the other classes."
    )

    comparison = results["comparison"].copy()
    st.dataframe(
        comparison.style.format({"Accuracy": "{:.1%}", "Macro F1": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.success("Random Forest is the selected final model because it is the assignment focus and handles non-linear relationships well.")

    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(results["y_test"], results["rf_predictions"], labels=CLASS_ORDER)
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=CLASS_ORDER,
                y=CLASS_ORDER,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Movies: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Random Forest confusion matrix",
            xaxis_title="Predicted class",
            yaxis_title="Actual class",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        report = results["report"].loc[CLASS_ORDER, ["precision", "recall", "f1-score", "support"]]
        st.subheader("Class-level report")
        st.dataframe(
            report.style.format(
                {"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}
            ),
            use_container_width=True,
        )
        st.warning(
            "There are only 41 Flop examples in the full dataset. Treat this class’s metrics "
            "carefully and collect more low-rated examples before a real business rollout."
        )

    st.subheader("Most important Random Forest features")
    importance = results["importances"].head(15).sort_values("Importance")
    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        labels={"Feature": "Feature", "Importance": "Importance"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_prediction(movies: pd.DataFrame) -> None:
    results = train_models()
    model = results["models"]["Random Forest"]
    st.title("Predict a movie")
    st.write(
        "Enter information that would be available before the movie releases, then ask the final "
        "Random Forest model for a predicted class."
    )
    st.caption("Inputs are pre-filled with medians or common categories from the dataset.")

    with st.form("prediction_form"):
        left, right = st.columns(2)
        values: dict[str, object] = {}
        with left:
            for feature in NUMERIC_FEATURES[:5]:
                default = float(movies[feature].median())
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=0.0,
                    value=default,
                    step=1.0,
                )
        with right:
            for feature in NUMERIC_FEATURES[5:]:
                default = float(movies[feature].median())
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=0.0,
                    value=default,
                    step=1.0,
                )

        st.subheader("Movie categories")
        category_columns = st.columns(5)
        preferred_defaults = {
            "color": "Color",
            "genre_primary": "Action",
            "language": "English",
            "country": "USA",
            "content_rating": "PG-13",
        }
        for column, feature in zip(category_columns, CATEGORICAL_FEATURES):
            options = sorted(movies[feature].dropna().astype(str).unique().tolist())
            with column:
                values[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options,
                    index=pick_default(options, preferred_defaults[feature]),
                )

        submitted = st.form_submit_button("Predict movie success", type="primary")

    if submitted:
        prediction_input = pd.DataFrame([values], columns=FEATURES)
        prediction = model.predict(prediction_input)[0]
        probabilities = model.predict_proba(prediction_input)[0]
        probability_frame = pd.DataFrame(
            {"Class": model.named_steps["classifier"].classes_, "Probability": probabilities}
        ).sort_values("Probability", ascending=False)

        st.subheader(f"Predicted class: {prediction}")
        st.plotly_chart(
            px.bar(
                probability_frame,
                x="Class",
                y="Probability",
                text=probability_frame["Probability"].map("{:.1%}".format),
                range_y=[0, 1],
                color="Class",
                color_discrete_map={"Flop": "#e76f51", "Average": "#e9c46a", "Hit": "#2a9d8f"},
            ).update_layout(showlegend=False),
            use_container_width=True,
        )
        st.info(
            "This is a classroom prediction based on historical IMDb data. It is not a guarantee of commercial performance."
        )


def render_project_guide() -> None:
    st.title("Project guide")
    st.subheader("Pipeline you can explain")
    st.markdown(
        """
1. **Load and inspect** the movie metadata.
2. **Remove exact duplicates** and create a primary-genre feature.
3. **Create the target** from IMDb score: Flop, Average, or Hit.
4. **Choose pre-release predictors** and deliberately exclude post-release information.
5. **Preprocess inside the pipeline**: median imputation for numbers, mode imputation and one-hot encoding for categories.
6. **Split the data** into stratified training and test sets.
7. **Compare three classifiers** and select Random Forest as the final model.
8. **Evaluate on held-out data** with accuracy, macro F1, a confusion matrix, and a classification report.
9. **Use the final model** in the prediction page.
        """
    )
    st.subheader("Key limitations")
    st.markdown(
        """
- IMDb score measures audience/critic rating, not box-office profit.
- The dataset ends in 2016 and may not represent current movie audiences.
- The Flop class is very small, so predictions for that class are less reliable.
- Facebook likes are an old popularity signal and should be replaced with current signals in a real project.
        """
    )


def main() -> None:
    try:
        movies = load_data()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Open a page",
        ["Overview", "Explore data", "Model performance", "Predict a movie", "Project guide"],
    )
    st.sidebar.caption("Dataset: 5,043 movie records supplied with this project.")

    if page == "Overview":
        render_overview(movies)
    elif page == "Explore data":
        render_explore(movies)
    elif page == "Model performance":
        render_performance(movies)
    elif page == "Predict a movie":
        render_prediction(movies)
    else:
        render_project_guide()


if __name__ == "__main__":
    main()
