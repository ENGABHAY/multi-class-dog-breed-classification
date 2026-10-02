"""
Streamlit interface for the 120-breed Dog Classifier (MobileNetV2 transfer learning model).

Run with:
    streamlit run app.py

Expects the trained model file "30_epoch_model.keras" (as saved in the training
notebook) to be in the same folder as this script, or a custom path set in the
sidebar.
"""

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
IMG_SIZE = (224, 224)
DEFAULT_MODEL_PATH = "30_epoch_model.keras"

# Class names, in the exact label order (0-119) used during training
# (extracted from the notebook's `class_names` list).
CLASS_NAMES = ["n02085620-Chihuahua", "n02085782-Japanese_spaniel", "n02085936-Maltese_dog", "n02086079-Pekinese", 
    "n02086240-Shih-Tzu", "n02086646-Blenheim_spaniel", "n02086910-papillon", "n02087046-toy_terrier", 
    "n02087394-Rhodesian_ridgeback", "n02088094-Afghan_hound", "n02088238-basset", "n02088364-beagle", 
    "n02088466-bloodhound", "n02088632-bluetick", "n02089078-black-and-tan_coonhound", "n02089867-Walker_hound", 
    "n02089973-English_foxhound", "n02090379-redbone", "n02090622-borzoi", "n02090721-Irish_wolfhound", 
    "n02091032-Italian_greyhound", "n02091134-whippet", "n02091244-Ibizan_hound", "n02091467-Norwegian_elkhound", 
    "n02091635-otterhound", "n02091831-Saluki", "n02092002-Scottish_deerhound", "n02092339-Weimaraner", 
    "n02093256-Staffordshire_bullterrier", "n02093428-American_Staffordshire_terrier", "n02093647-Bedlington_terrier", 
    "n02093754-Border_terrier", "n02093859-Kerry_blue_terrier", "n02093991-Irish_terrier", "n02094114-Norfolk_terrier", 
    "n02094258-Norwich_terrier", "n02094433-Yorkshire_terrier", "n02095314-wire-haired_fox_terrier", 
    "n02095570-Lakeland_terrier", "n02095889-Sealyham_terrier", "n02096051-Airedale", "n02096177-cairn", 
    "n02096294-Australian_terrier", "n02096437-Dandie_Dinmont", "n02096585-Boston_bull", "n02097047-miniature_schnauzer", 
    "n02097130-giant_schnauzer", "n02097209-standard_schnauzer", "n02097298-Scotch_terrier", "n02097474-Tibetan_terrier", 
    "n02097658-silky_terrier", "n02098105-soft-coated_wheaten_terrier", "n02098286-West_Highland_white_terrier", 
    "n02098413-Lhasa", "n02099267-flat-coated_retriever", "n02099429-curly-coated_retriever", "n02099601-golden_retriever", 
    "n02099712-Labrador_retriever", "n02099849-Chesapeake_Bay_retriever", "n02100236-German_short-haired_pointer", 
    "n02100583-vizsla", "n02100735-English_setter", "n02100877-Irish_setter", "n02101006-Gordon_setter", 
    "n02101388-Brittany_spaniel", "n02101556-clumber", "n02102040-English_springer", "n02102177-Welsh_springer_spaniel", 
    "n02102318-cocker_spaniel", "n02102480-Sussex_spaniel", "n02102973-Irish_water_spaniel", "n02104029-kuvasz", 
    "n02104365-schipperke", "n02105056-groenendael", "n02105162-malinois", "n02105251-briard", "n02105412-kelpie", 
    "n02105505-komondor", "n02105641-Old_English_sheepdog", "n02105855-Shetland_sheepdog", "n02106030-collie", 
    "n02106166-Border_collie", "n02106382-Bouvier_des_Flandres", "n02106550-Rottweiler", "n02106662-German_shepherd", 
    "n02107142-Doberman", "n02107312-miniature_pinscher", "n02107574-Greater_Swiss_Mountain_dog", "n02107683-Bernese_mountain_dog", 
    "n02107908-Appenzeller", "n02108000-EntleBucher", "n02108089-boxer", "n02108422-bull_mastiff", "n02108551-Tibetan_mastiff", 
    "n02108915-French_bulldog", "n02109047-Great_Dane", "n02109525-Saint_Bernard", "n02109961-Eskimo_dog", 
    "n02110063-malamute", "n02110185-Siberian_husky", "n02110627-affenpinscher", "n02110806-basenji", 
    "n02110958-pug", "n02111129-Leonberg", "n02111277-Newfoundland", "n02111500-Great_Pyrenees", "n02111889-Samoyed", 
    "n02112018-Pomeranian", "n02112137-chow", "n02112350-keeshond", "n02112706-Brabancon_griffon", "n02113023-Pembroke", 
    "n02113186-Cardigan", "n02113624-toy_poodle", "n02113712-miniature_poodle", "n02113799-standard_poodle", 
    "n02113978-Mexican_hairless", "n02115641-dingo", "n02115913-dhole", "n02116738-African_hunting_dog", 
]


def clean_breed_name(raw_name: str) -> str:
    """Turn 'n02085620-Chihuahua' into 'Chihuahua'."""
    name = raw_name.split("-", 1)[-1] if "-" in raw_name else raw_name
    return name.replace("_", " ").title()


# ----------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize + cast to match the training pipeline's `load_image` function.
    (MobileNetV2's preprocess_input is already applied inside the model itself
    via its Lambda layer, so it must NOT be applied again here.)
    """
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image, top_k: int = 5):
    batch = preprocess_image(image)
    probs = model.predict(batch, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(clean_breed_name(CLASS_NAMES[i]), float(probs[i])) for i in top_indices]


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="centered")

st.title("🐶 Dog Breed Classifier")
st.caption("120-breed classifier — MobileNetV2 transfer learning (test accuracy: 80.24%)")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model file path", value=DEFAULT_MODEL_PATH)
    top_k = st.slider("Number of predictions to show", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.markdown(
        "**About**\n\n"
        "Trained with a frozen MobileNetV2 backbone + a custom classification "
        "head (GlobalAveragePooling2D → Dense(256) → Dropout(0.3) → Dense(120, softmax)) "
        "on 120 dog breeds."
    )

try:
    model = load_model(model_path)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model from '{model_path}'. Error: {e}")

uploaded_file = st.file_uploader(
    "Upload a dog photo", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Predicting..."):
        results = predict(model, image, top_k=top_k)

    top_breed, top_conf = results[0]

    with col2:
        st.subheader("Prediction")
        st.markdown(f"### {top_breed}")
        st.metric("Confidence", f"{top_conf * 100:.2f}%")

    st.markdown("---")
    st.subheader(f"Top {top_k} Predictions")
    st.bar_chart(
        {breed: conf for breed, conf in results},
        horizontal=True,
    )
    for breed, conf in results:
        st.write(f"**{breed}** — {conf * 100:.2f}%")

elif uploaded_file is not None and not model_loaded:
    st.warning("Fix the model path in the sidebar before uploading an image.")
else:
    st.info("Upload a dog image above to get a breed prediction.")