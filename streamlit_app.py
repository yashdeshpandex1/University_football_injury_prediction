import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Football Injury Prediction", page_icon="⚽", layout="wide")
st.title("⚽ University Football Injury Prediction")

st.markdown(
    """
    <style>
    .stApp {
        background: url("https://github.com/yashdeshpandex1/University_football_injury_prediction/blob/main/football_bg.png?raw=true");
        background-size: cover;
        background-attachment: fixed;
    }

    [data-testid="collapsedControl"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Load and prepare dataset
# -----------------------
df = pd.read_csv("data.csv")
target = "Injury_Next_Season"
x = df.drop(target, axis=1)
y = df[target]

num_cols = x.select_dtypes(include=["number"]).columns
cat_cols = x.select_dtypes(exclude=["number"]).columns

# Preprocessor
ct = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", ct),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train model once on all data (so we can use it for predictions)
model.fit(x, y)

# -----------------------
# Sidebar inputs for prediction
# -----------------------
st.sidebar.header("🔮 Predict New Player Injury Risk")

input_data = {}
for col in num_cols:
    min_val, max_val = float(df[col].min()), float(df[col].max())
    default_val = float(df[col].mean())
    input_data[col] = st.sidebar.number_input(
        f"{col}", min_value=min_val, max_value=max_val, value=default_val
    )

for col in cat_cols:
    options = df[col].unique().tolist()
    input_data[col] = st.sidebar.selectbox(f"{col}", options)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction button
if st.sidebar.button("Predict Injury Risk"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Player is likely to be **Injured Next Season** (Probability: {prob[1]:.2f})")
    else:
        st.success(f"✅ Player is likely to be **Safe** (Probability: {prob[0]:.2f})")

# -----------------------
# Show feature importance
# -----------------------
st.write("### 🔎 Top Feature Importances")

feature_names = (
    ct.named_transformers_["num"].get_feature_names_out(num_cols).tolist()
    + ct.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
)
importances = model.named_steps["classifier"].feature_importances_
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

# Create transparent figure
fig, ax = plt.subplots(figsize=(8,6), facecolor="none")

# Draw semi-transparent rectangle behind bars
ax.add_patch(
    plt.Rectangle(
        (0, -0.5),                  # x, y start point
        1,                          # width (dummy, will scale)
        len(feat_imp.head(15)),     # height (number of bars)
        color='white',               # banner color
        alpha=0.7,                   # opacity (0 = transparent, 1 = opaque)
        transform=ax.transAxes,      # scale rectangle to axes coordinates
        zorder=0                     # behind all bars
    )
)

# Plot bars (slightly transparent)
sns.barplot(
    x="Importance",
    y="Feature",
    data=feat_imp.head(15),
    ax=ax,
    alpha=0.9,   # bars slightly transparent
    zorder=1      # on top of rectangle
)

# Make axes background transparent
ax.patch.set_alpha(0)

# Optional: make labels white for readability
ax.set_facecolor("none")
ax.tick_params(colors="white")        # axis ticks
ax.yaxis.label.set_color("white")     # y-axis label
ax.xaxis.label.set_color("white")     # x-axis label
ax.title.set_color("white")           # title if any

st.pyplot(fig, transparent=True)

