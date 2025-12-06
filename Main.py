import streamlit as st
import joblib

model = joblib.load('model.joblib')

st.set_page_config(
    page_title="Iris-Rishabh",
    page_icon="🌸",
    layout="wide",
)

with st.sidebar:
    st.header("Iris Feature Inputs")

    sl = st.slider("Sepal Length", 4.3, 7.9, 5.8)
    sw = st.slider("Sepal Width", 2.0, 4.4, 3.0)
    pl = st.slider("Petal Length", 1.0, 6.9, 4.3)
    pw = st.slider("Petal Width", 0.1, 2.5, 1.3)



st.title("🌸 Iris Flower Prediction")


prediction = model.predict([[sl, sw, pl, pw]])
prediction = prediction[0]  # array -> scalar

# creating dictionary
label_map = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

flower_name = label_map.get(prediction)

# Layout: text on left, image on right
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")
    st.write(f"**Predicted species:** `{flower_name.capitalize()}`")
    st.write("### Input values")
    st.write(
        f"- Sepal Length: **{sl} cm**\n"
        f"- Sepal Width: **{sw} cm**\n"
        f"- Petal Length: **{pl} cm**\n"
        f"- Petal Width: **{pw} cm**"
    )

with col2:
    st.image(f"{flower_name}.jpg", caption=flower_name.capitalize(), width=400)






