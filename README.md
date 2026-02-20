# ChefBot: AI Recipe Chatbot 🤖🧑🏻‍🍳

An intelligent, Streamlit-based web application that serves as your personal AI Sous-Chef. ChefBot takes your available ingredients and preferred cuisine to generate creative recipes using a Large Language Model (LLM). It also features advanced capabilities like intelligent ingredient substitution and a "Smart Recipe Remix".

## What This Project Does

- **AI Recipe Generation**: Enter the ingredients you have on hand, select a cuisine (e.g., Asian, Mexican, Mediterranean), and the app will generate a custom recipe. It uses `meta-llama/Llama-3.2-1B-Instruct` (or other HuggingFace models) to create the instructions.
- **Smart Recipe Remix**: Want to try something different? The "Remix Recipe!" feature randomly swaps out ingredients from your list with similar alternatives, creating a fun twist while maintaining the culinary coherence of the dish.
- **Find Alternative Ingredients**: Missing an ingredient? ChefBot can search for the best alternatives using two methods:
  - **Annoy Index (Fastest)**: Uses approximate nearest neighbors for blazing-fast lookups.
  - **Direct Search (Best Accuracy)**: Uses exact cosine similarity on word embeddings.
- **Configurable Generation**: Power users can tweak generation parameters exactly how they want (Temperature, Decoding Strategy, Beam Search, Top-K, Top-P, and more) from the UI sidebar.

## Educational Uses & LLM Experimentation

Beyond generating fun recipes, this project is perfectly suited as an interactive testbed for learning and experimenting with Large Language Models (LLMs):

- **Hyperparameter Tuning**: Observe firsthand how different generation hyperparameters (like temperature, top-k, top-p, and beam search vs. greedy decoding) affect the model's output and creativity.
- **Prompt Configurations**: Test various prompt structures, such as modifying the detail level, creative tone ("Basic" vs "Surprise Me!"), and formatting requests, to see how the LLM adheres to complex instructions.
- **Model Comparisons**: Easily swap out the underlying `model_name` in `Recipe_Bot.py` (e.g., trying out `Qwen/Qwen3-0.6B-Base` or swapping to Mistral) to evaluate and compare how different LLMs behave under the exact same configurations.

## Prerequisites

- **OS**: Windows / Linux / macOS (Windows requires a C++ build environment for the `annoy` library).
- **Python**: 3.9+ 
- **Conda** (Recommended for managing PyTorch and CUDA dependencies)
- **Hugging Face Account**: You need a Hugging Face API token to access the generative LLM.

## Setup and Installation

### 1. Environment Setup

It is highly recommended to use Conda + Pip to avoid dependency conflicts, especially for GPU/CUDA setups.

```bash
# Create a new conda environment called "genai"
conda create -n genai python=3.9 pytorch pytorch-cuda=12.1 spacy pandas numpy -c pytorch -c nvidia -c conda-forge -y

# Activate the environment
conda activate genai

# Install the remaining pip dependencies
pip install -r requirements.txt

# Download the required spaCy language model for ingredient embeddings
python -m spacy download en_core_web_lg
```

### 2. Environment Variables

This project uses `python-dotenv` to load secrets. Create a `.env` file in the root directory of the project and add your Hugging Face API token:

```env
HF_TOKEN=your_hugging_face_token_here
```

*Note: Ensure your Hugging Face account has been granted access to `meta-llama/Llama-3.2-1B-Instruct` on the Hugging Face website.*

## How to Run

### 1. Verify GPU Setup (Optional but Recommended)

For faster model inference, ensure your PyTorch installation is correctly picking up your NVIDIA GPU:

```bash
python check_gpu.py
```
If properly configured, it will print your CUDA device name. Otherwise, the app will fall back to sluggish CPU generation.

### 2. Launch the Application

Start the Streamlit web server:

```bash
streamlit run Recipe_Bot.py
```

The application will automatically download the required datasets (ingredients `.csv` file via Google Drive) on the first run, initialize the models (which might take a moment), and open a browser window at `http://localhost:8501`.
