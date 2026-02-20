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

# model_name = "Qwen/Qwen3-0.6B-Base"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
token = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=token
)

@st.cache_resource
def load_llm(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Force loading weights to CPU then move to GPU to avoid meta tensor issues
    # Using float16 for GPU efficiency
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=token,
            dtype=torch.float16,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
        )
        
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
def generate_recipe(ingredients, 
                    cuisine, 
                    format_response=False,
                    response_detail_level="Brief",
                    creative_level="Basic",
                    temperature=1.0, 
                    num_beams=1, 
                    do_sample=False, 
                    top_k=50, 
                    top_p=1.0):
    """Generate a recipe using the LLM with configurable generation parameters and prompt styles.
    
    Args:
        ingredients: Comma-separated ingredient string
        cuisine: Selected cuisine type
        format_response: Whether to generate a structured response with title, ingredients, and instructions
        response_detail_level: Level of detail for the recipe (Brief or Detailed)
        creative_level: Level of creativity for the recipe (Basic or Surprise Me!)
        temperature: Controls randomness
        num_beams: Number of beams for beam search
        do_sample: Whether to use sampling
        top_k: Limits vocabulary to top-k tokens at each step
        top_p: Nucleus sampling threshold
    """

    #  Format Response Prompt
    if format_response:
        format_text = (f"Format the response **EXACTLY** as follows: \n"
                       f"Recipe Name: [Recipe Name]\n"
                       f"Ingredients: [Ingredients]\n"
                       f"Instructions: [Instructions]\n"
                       f"Do not include any other text.\n")
    else:
        format_text = ""

    #  Recipe Detail Prompt
    if response_detail_level == "Brief":
        detail_text = "brief 3-4 sentence recipe without extra details"
    elif response_detail_level == "Detailed":
        detail_text = "highly detailed recipe with descriptive instructions"
    else:
        detail_text = "recipe"

    #  Recipe Creative Prompt
    if creative_level == "Basic":
        creative_text = f"Generate a {detail_text} inspired by {cuisine} cuisine with these ingredients.\n"
    else:
        creative_text = (f"You are an adventurous, world-class fusion chef. Create a surprising and unconventional "
                         f"{detail_text} inspired by {cuisine} cuisine with the ingredients below. Combine "
                         f"unexpected flavors, unusual cooking techniques, innovative presentation styles, and give the "
                         f"dish a fun name.\n")
    
    #  Final Input Prompt
    input_text = (f"{creative_text}\n"
                  f"Ingredients: {', '.join(ingredients.split(', '))}\n"
                  f"Do not reference any images or visual content.\n"   # Prevent Llama from referencing images
                  f"{format_text}\n"
                #   f"Here are the recipe and instructions: "
                )
    
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)["input_ids"]
    attention_mask = torch.ones_like(input_ids)  # Fixes attention mask warning

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=500,
        num_return_sequences=1,
        repetition_penalty=1.2,
        do_sample=do_sample,
        temperature=temperature,
        num_beams=num_beams,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,  # Fix eos token warning
    )
    return input_text, tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()



# Task 5 - Recipe Remix

def get_remixed_ingredients(ingredients):
    ing_list = [ing.strip().lower() for ing in ingredients.split(",")]

    ing_to_swap = random.sample(ing_list, min(2, len(ing_list)))

    subs = {}
    for ing in ing_to_swap:
        subs[ing] = annoy_search_alternatives(ing)[-1]

    new_ingredients = [subs.get(ing, ing) for ing in ing_list]

    return ",".join(new_ingredients)


#  Streamlit App UI
st.title("🤖🧑🏻‍🍳 ChefBot: AI Recipe Chatbot")

# Sidebar Environment Info
st.sidebar.markdown(f"**Model:** `{model_name}`")
st.sidebar.markdown(f"**Device:** `{device.upper()}`")
if device == "cuda":
    st.sidebar.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")

ingredients = st.text_input("🥑🥦🥕 Ingredients (comma-separated):")
cuisine = st.selectbox("Select a cuisine:", ["Any", "Asian", "Indian", "Middle Eastern", "Mexican",  "Western", "Mediterranean", "African"])

# Response control parameters
format_response = st.checkbox("Format response (Title, ingredients, and instructions)", value=False)
response_detail_level = st.selectbox("Response detail level:", ["Brief", "Normal", "Detailed"], index=0)
creative_level = st.selectbox("Creativity Level: ",["Basic", "Surprise Me!"], index=0)

# Generation Parameters in sidebar
st.sidebar.header("Generation Parameters")
temperature = st.sidebar.select_slider(
    "Temperature:",
    options=[0.5, 1.0, 2.0],
    value=1.0
)

decoding_strategy = st.sidebar.radio(
    "Decoding Strategy:",
    options=["Greedy", "Beam Search"],
    index=0
)
num_beams = st.sidebar.slider(
    "Number of Beams:",
    min_value=1,
    max_value=10,
    value=1 if decoding_strategy == "Greedy" else 5,
    disabled=decoding_strategy == "Greedy"
)

do_sample = st.sidebar.checkbox("Enable Sampling: ", value=False)
top_k = st.sidebar.select_slider(
    "Top-K:",
    options=[5, 50],
    value=50,
    disabled=not do_sample
)

top_p = st.sidebar.select_slider(
    "Top-P:",
    options=[0.7, 0.95, 1.0],
    value=1.0,
    disabled=not do_sample
)

#  Display current config summary
st.sidebar.markdown("---")
st.sidebar.markdown("**Current Config:**")
st.sidebar.markdown(f"- Temperature: `{temperature}`")
st.sidebar.markdown(f"- Decoding Strategy: `{decoding_strategy}`")
st.sidebar.markdown(f"- Number of Beams: `{num_beams}`")
st.sidebar.markdown(f"- Sampling: `{do_sample}`")
st.sidebar.markdown(f"- Top-K: `{top_k}`, Top-P: `{top_p}`")

if st.button("Generate Recipe", use_container_width=True) and ingredients:
    st.session_state["input_text"], st.session_state["recipe"] = generate_recipe(
        ingredients, 
        cuisine, 
        format_response=format_response,
        response_detail_level=response_detail_level,
        creative_level=creative_level,
        temperature=temperature,
        num_beams=num_beams,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p
    )

disable_remix = "recipe" not in st.session_state or not st.session_state["recipe"] or not ingredients
if st.button("Remix Recipe", use_container_width=True, disabled=disable_remix):
    ingredients = get_remixed_ingredients(ingredients)
    st.session_state["input_text"], st.session_state["recipe"] = generate_recipe(
        ingredients,
        cuisine, 
        format_response=format_response,
        response_detail_level=response_detail_level,
        creative_level=creative_level,
        temperature=temperature,
        num_beams=num_beams,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p
    )

if "recipe" in st.session_state:
    st.markdown("### 🍽️ Generated Recipe:")
    st.text_area("Updated Prompt:", st.session_state["input_text"], height=200)
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
        start_time = time.time()
        alternatives = search_methods[search_method](ingredient_to_replace)
        end_time = time.time()
        execution_time = end_time - start_time        
        st.markdown(f"### 🌿 Alternatives for **{ingredient_to_replace.capitalize()}**:")
        st.markdown(f"➡️ {' ⟶ '.join(alternatives)}")
        st.markdown(f"Time taken: {execution_time:.4f} seconds")
