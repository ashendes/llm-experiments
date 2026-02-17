import streamlit as st
import torch
import pandas as pd
import spacy
import os
import gdown
from annoy import AnnoyIndex
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Load models
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading spaCy model 'en_core_web_lg'...")
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

@st.cache_resource
def load_llm(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Force loading weights to CPU then move to GPU to avoid meta tensor issues
    # Using float16 for GPU efficiency
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype=torch.float16
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
    return model, device

model, device = load_llm(model_name)

# Download & Load Ingredient Data
GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1-qf8ZIrBlsEixBJULmXyDJk4M4ktRurH"
CSV_FILE = "processed_ingredients_with_id.csv"

@st.cache_data
def load_ingredient_data():
    if not os.path.exists(CSV_FILE):  
        gdown.download(GDRIVE_FILE_URL, CSV_FILE, quiet=False)
    return pd.read_csv(CSV_FILE)["processed"].dropna().unique().tolist()

ingredient_list = load_ingredient_data()

# Compute Embeddings (Filter out zero vectors)
@st.cache_resource
def compute_embeddings():
    filtered_ingredients = []
    vectors = []

    for ing in ingredient_list:
        vec = nlp(ing.lower()).vector
        if np.any(vec):  # Exclude zero vectors
            filtered_ingredients.append(ing)
            vectors.append(vec)

    return np.array(vectors, dtype=np.float32), filtered_ingredients

ingredient_vectors, filtered_ingredient_list = compute_embeddings()

# Build Annoy Index (Fast Approximate Nearest Neighbors)
@st.cache_resource
def build_annoy_index():
    dim = ingredient_vectors.shape[1]
    index = AnnoyIndex(dim, metric="angular")  #  Uses angular distance (1 - cosine similarity)
    
    for i, vec in enumerate(ingredient_vectors):
        index.add_item(i, vec)
    
    index.build(50)  #  More trees = better accuracy
    return index
annoy_index = build_annoy_index()

#  Direct Cosine Similarity Search (Most Accurate)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) if np.any(vec1) and np.any(vec2) else 0

def direct_search_alternatives(ingredient):
    return 

#  Annoy Search (Fixed for Correct Cosine Similarity)
def annoy_search_alternatives(ingredient):
    return 

#  Generate Recipe
def generate_recipe(ingredients, cuisine, temperature=1.0, num_beams=1, top_k=50, top_p=1.0):
    input_text = (
        f"Ingredients: {', '.join(ingredients.split(', '))}\n"
        f"Cuisine: {cuisine}\n"
        f"Let's create a dish inspired by {cuisine} cuisine with these ingredients. Here are the preparation and cooking instructions:"
    )    
    generation_config = GenerationConfig(
        max_length=250, 
        num_return_sequences=1, 
        repetition_penalty=1.2,
        do_sample=True,
        temperature=temperature,
        num_beams=num_beams,
        top_k=top_k,
        top_p=top_p,
    )
    outputs = model.generate(tokenizer(input_text, return_tensors="pt").to(device)["input_ids"], 
                             generation_config=generation_config)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()

#  Streamlit App UI
st.title("🤖🧑🏻‍🍳 ChefBot: AI Recipe Chatbot")
st.sidebar.markdown(f"**Device:** `{device.upper()}`")
if device == "cuda":
    st.sidebar.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")
    
ingredients = st.text_input("🥑🥦🥕 Ingredients (comma-separated):")
cuisine = st.selectbox("Select a cuisine:", ["Any", "Asian", "Indian", "Middle Eastern", "Mexican",  "Western", "Mediterranean", "African"])
if st.button("Generate Recipe", use_container_width=True) and ingredients:
    st.session_state["recipe"] = generate_recipe(ingredients, cuisine)

if "recipe" in st.session_state:
    st.markdown("### 🍽️ Generated Recipe:")
    st.text_area("Recipe:", st.session_state["recipe"], height=200)

    st.download_button(label="📂 Save Recipe", 
                       data=st.session_state["recipe"], 
                       file_name="recipe.txt", 
                       mime="text/plain")

    #  Alternative Ingredient Section
    st.markdown("---")
    st.markdown("## 🔍 Find Alternative Ingredients")

    ingredient_to_replace = st.text_input("Enter an ingredient:")
    search_method = st.radio("Select Search Method:", ["Annoy (Fastest)", "Direct Search (Best Accuracy)"], index=0)

    if st.button("🔄 Find Alternatives", use_container_width=True) and ingredient_to_replace:
        search_methods = {
            "Annoy (Fastest)": annoy_search_alternatives,
            "Direct Search (Best Accuracy)": direct_search_alternatives
        }
        alternatives = search_methods[search_method](ingredient_to_replace)
        st.markdown(f"### 🌿 Alternatives for **{ingredient_to_replace.capitalize()}**:")
        st.markdown(f"➡️ {' ⟶ '.join(alternatives)}")
