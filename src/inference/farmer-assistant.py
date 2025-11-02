"""
@author: Naveen N G
@date: 02-11-2025
"""

import modal
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) 
data_directory = os.path.join(src_dir, 'data')  # src/data/

# === Modal setup ===
app = modal.App("farmer-assistant")

def download_translate_model():
    from transformers import MarianMTModel, MarianTokenizer
    import os
    
    model_name = "Helsinki-NLP/opus-mt-mul-en"
    print(f"Downloading Translate model: {model_name}")
    
    # Set cache directory
    cache_dir = "/root/.cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download and cache the model
    MarianTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    MarianMTModel.from_pretrained(model_name, cache_dir=cache_dir)
    
    print("Translate Model downloaded successfully!")

def download_sentence_model():
    from sentence_transformers import SentenceTransformer
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Modal image with dependencies
# Stage 1: Base dependencies
base_image = modal.Image.debian_slim().apt_install("git")

# Stage 2: Python packages (can be cached)
python_deps_image = base_image.pip_install(
    "torch",
    "transformers",
    "sentencepiece",
    "sacremoses",
    "peft",
    "accelerate",
    "bitsandbytes",
    "chromadb",
    "sentence-transformers",
    "gradio",
    "tqdm"
)

# Stage 3: Download translation model (separate, can be cached)
translation_and_sentence_image = python_deps_image.run_function(
    download_translate_model, 
    timeout=3600  # 60 minutes
).run_function(
    download_sentence_model, 
    timeout=1200  # 20 minutes
)


# Stage 4: Add local files
image = (
    translation_and_sentence_image
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file(f"{current_dir}/constants.py", remote_path="/root/constants.py")
    .add_local_dir(data_directory, remote_path="/root/src/data")
)

# Database path
DB_PATH = '/root/src/data/chroma_database'
DATA_PATH = '/root/src/data'

# Secrets
secrets = modal.Secret.from_name("huggingface-secret")

# Constants
GPU = "T4"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
FINETUNED_MODEL = "naveenng10/farmer-assistant"


@app.cls(
    image=image,
    secrets=[secrets],
    gpu=GPU, 
    min_containers=1,
    container_idle_timeout=600,    # Keep GPU container alive for 10 minutes
    allow_concurrent_inputs=15,    # Allow up to 15 concurrent model inferences
    timeout=3600
)
class FarmerAssistantModel:

    @modal.enter()
    def setup(self):
        
        """Initialize all models - imports happen here at runtime"""
        import sys

        # Add paths for imports
        sys.path.insert(0, '/root')
        sys.path.insert(0, '/root/src')
        
        # Import constants
        import constants as DATA_CONSTANTS
        self.DATA_CONSTANTS = DATA_CONSTANTS

        # Add paths for imports
        sys.path.insert(0, '/root')
        sys.path.insert(0, '/root/src')

        try:
            print("Initializing FarmerAssistantModel STARTS.")
            self.load_finetuned_model()
            self.load_croma_vector_db()
            self.load_translation_model()
            print("Initializing FarmerAssistantModel COMPLETED")
        except Exception as e:
            print(f"Setup failed: {e}")
            raise
    
    def load_finetuned_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed, pipeline
        from peft import PeftModel
    
        print("load_finetuned_model...")
        # Quant Config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            quantization_config=quant_config,
            device_map="auto"
        )
        self.fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
        # Optional: merge adapter for faster inference
        self.fine_tuned_model = self.fine_tuned_model.merge_and_unload()
        set_seed(42)
        self.generator = pipeline(
            "text-generation",
            model=self.fine_tuned_model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=80,  
            temperature=0.5,
            do_sample=False,
            top_p=0.9,
            top_k=30,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        device = next(self.fine_tuned_model.parameters()).device
        print(f"Finetuned model loaded on: {device}")
        print("load_finetuned_model completed...")

    def get_saved_data(self):
        import pickle

        with open(f'{DATA_PATH}/train.pkl', 'rb') as file:
            kcc_data = pickle.load(file)
        return kcc_data

    def load_croma_vector_db(self):
        import chromadb
        from chromadb.config import Settings
        from tqdm import tqdm
        from sentence_transformers import SentenceTransformer

        print(f"Loading ChromaDB from: {DB_PATH}") 
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 
        try:
            client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=True))
            self.collection = client.get_or_create_collection('kcc_data_collection', metadata={"hnsw:space": "cosine"})

            existing_count = self.collection.count()
            if existing_count > 0:
                print(f"Found existing Chroma collection with {existing_count} documents. Skipping embedding.")
                return

            kcc_data = self.get_saved_data()
            print('Recived the data.....')
            NUMBER_OF_DOCUMENTS = len(kcc_data)
            for i in tqdm(range(0, NUMBER_OF_DOCUMENTS, 1000)):
                documents = [self.user_query(item) for item in kcc_data[i: i+1000]]
                vectors = self.embedder.encode(documents, normalize_embeddings=True, batch_size=64).tolist()
                metadatas = [{"QueryType": item['QueryType'], "answer": item['Answer']} for item in kcc_data[i: i+1000]]
                ids = [f"doc_{j}" for j in range(i, i+len(documents))]
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=vectors,
                    metadatas=metadatas
                )
            print("ChromaDB Collections data loads completed")
        except Exception as e:
            print("Failed to load ChromaDB")   
            raise e
        
    def user_query(self, item):
        return item['prompt'].split("\n\nAnswer is:")[0]

    def load_translation_model(self):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        print("Loading load_translation_model STARTS:")
        # This model translates from multiple languages to English
        model_name = "Helsinki-NLP/opus-mt-mul-en"
        # Use CPU to save GPU memory for main model
        self.translate_device = "cpu"
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.translate_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translate_model = MarianMTModel.from_pretrained(model_name).to(self.translate_device)
            print("Loading load_translation_model COMPLETED:")
        except Exception as e:
            print(f"Translation model load failed: {e}")
            raise

    def translate_to_english(self, text):
        import torch

        if not text or not text.strip():
            return text
        
        # Tokenize
        inputs = self.translate_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.translate_device)
        
        # Generate translation without gradients (saves memory)
        with torch.no_grad():
            outputs = self.translate_model.generate(**inputs)

        # Decode
        translated = self.translate_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

    def is_query_relevant(self, vector_result, threshold=0.6):
        try:
            # Lower distance = more similar (depends on your metric)
            if 'distances' in vector_result and vector_result['distances']:
                avg_distance = sum(vector_result['distances'][0]) / len(vector_result['distances'][0])
                # Adjust threshold based on your data
                pint_msg= f"distance:{vector_result['distances'][0]} and sum:{sum(vector_result['distances'][0])} and avg:{avg_distance}"
                print(pint_msg)
                print(vector_result['distances'][0])
                return avg_distance < threshold
                    # Fallback: check if we got any results
            # return len(vector_result['documents'][0][:]) > 0
            return True
        except:
            return False

    def find_similars(self, query):
        try:
            results = self.collection.query(
                query_embeddings=[self.embedder.encode(query, normalize_embeddings=True).tolist()],
                n_results=3
            )
            # Check the queruy relevance with chromadb distances/similarities
            isRelevantQuery = self.is_query_relevant(results)
            if(isRelevantQuery):
                questions = results['documents'][0][:]
                answers = [m['answer'] for m in results['metadatas'][0][:]]
                return questions, answers
            return [], []
        except Exception as e:
            print(f"Cromadb Vector search error: {e}")
            return [], []
    
    def create_context(self, questions, answers):
        if questions is not None and answers is not None and len(questions) > 0 and len(answers) > 0:
            message = "To provide some context, here are some of the questions and answers that might be similar to the question you need to answer.\n\n"
            for question, answer in zip(questions, answers):
                message += f"Potentially related question:\n{question}\nAnswer is: {answer}\n\n"
            return message
        return ""
    
    def build_internal_query(self, query,  query_type = ''):
        message = ''
        message += query
        if query_type and len(query_type) > 0:
            message += f"This question is related to {query_type}"
        return message
    

    def get_system_message(self):
        import constants as DATA_CONSTANTS

        allowed_categories = ", ".join(DATA_CONSTANTS.QUERY_TYPE)
        system_message = f"""You are a specialized farming assistant. You can ONLY answer questions about:
            {allowed_categories}.\n\n
            STRICT RULES:
            1. If a question is about sports, politics, entertainment, technology, or ANY non-farming topic, you MUST respond EXACTLY with:
            "I'm sorry, but I can only answer questions related to Agriculture or Horticulture topics."

            2. DO NOT provide any information about:
            - Sports (cricket, football, etc.)
            - Politics or elections
            - Movies, actors, or entertainment
            - Technology (unless directly related to farming equipment)
            - General knowledge unrelated to farming

            3. ONLY answer questions about:
            - Crops, plants, seeds
            - Soil, fertilizers, irrigation
            - Pests, diseases, plant protection
            - Farming techniques and equipment
            - Livestock and animal husbandry

            Examples of REJECTED questions:
            Q: "Who won the cricket match?"
            A: "I'm sorry, but I can only answer questions related to Agriculture or Horticulture topics."

            Q: "Who is the prime minister?"
            A: "I'm sorry, but I can only answer questions related to Agriculture or Horticulture topics."

            Examples of ACCEPTED questions:
            Q: "How do I control aphids on my tomato plants?"
            A: [Provide helpful farming advice]

            Always reply in English. Be clear, accurate, and practical.
        """
        return system_message

    def is_agriculture_related(self, query):
        # Comprehensive list of blocked keywords
        blocked_keywords = [
            # Sports
            'sport', 'cricket', 'football', 'soccer', 'tennis', 'basketball',
            'match', 'player', 'team', 'goal', 'wicket', 'bat', 'ball game',
            # Politics (fixed typo!)
            'politics', 'election', 'minister', 'president', 'government',
            'parliament', 'vote', 'political party',
            # Entertainment
            'movie', 'cinema', 'film', 'actor', 'actress', 'director',
            'music', 'song', 'singer', 'album', 'concert',
            'celebrity', 'bollywood', 'hollywood', 'star',
            # Technology (unless farming-related)
            'smartphone', 'laptop', 'computer game', 'social media',
            'facebook', 'instagram', 'whatsapp', 'coding', 'programming'
        ]
        # Block if contains blocked keywords
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in blocked_keywords):
            return False
        return True

    def default_response(self):
        return {
            "answer": "I'm sorry, but I can only answer questions related to Agriculture or Horticulture topics.",
            "original_response": "",
            "context_used": ""
        }

    @modal.method()
    def predict_answer(self, category, query):
        print('predict_answer called with...:', category, query)
        try:
            
            # Pre-filter: Check if query is agriculture-related
            if self.is_agriculture_related(query) == False:
                print(f'Query is not related to Agriculture: {query.lower()}')
                return self.default_response()
            # Pre-filter: Check similarity check in chroma batabase
            questions, answers   = self.find_similars(query)
            # if len(questions) == 0 :
            #     return self.default_response()

            internal_query = self.build_internal_query(query, category)
            system_message = self.get_system_message()
            context = self.create_context(questions, answers)
            print(f'context:{context}')
            user_prompt = context + f"Now the question for you:\n\n"
            user_prompt += internal_query
            messages =  [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "Answer is:"}
            ]
            prompt = "\n".join([m["content"] for m in messages])
            response = self.generator(prompt, return_full_text=False)[0]["generated_text"]
            print(f"Response generated")
            # Translate response to English
            english_response = self.translate_to_english(response)
            return {"answer": english_response,"original_response": response, "context_used": context}
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                "answer": f"Error processing query: {query}",
                "context_used": "",
                "original_response": ""
            }

@app.function(image=image, 
    secrets=[secrets], 
    min_containers=1, 
    max_containers=1,
    container_idle_timeout=300,  # Keep container alive for 5 minutes
    timeout=600,                 # Max 10 minutes per request
    allow_concurrent_inputs=100  # Allow up to 100 concurrent requests
    )
@modal.asgi_app()
def gradio_app():
    import gradio as gr
    import sys

    # Add paths for constants import
    sys.path.insert(0, '/root')
    sys.path.insert(0, '/root/src')
    import constants as DATA_CONSTANTS

    rag_handle = FarmerAssistantModel()
    
    def query_chat(category, query, progress=gr.Progress()):

        print(f"Inputs: {category}, {query}")
        # Considering category as query_type for now.
        if not category or category not in DATA_CONSTANTS.QUERY_TYPE:
            return "Please select a valid category."
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            return "Please enter your question."
        try:
            result = rag_handle.predict_answer.remote(category, query)
            return f"Answer:\n\n{result.get('answer', 'No answer returned.')} \n\n For more information, Visit regional office near by you."
        except Exception as e:
            return f" Error: {str(e)}"

    with gr.Blocks(title="Farmer Query Assistant", 
                   css="#output-box {border: 2px solid #4CAF50; padding: 1rem; border-radius: 10px; min-height: 200px; background-color: #f9fff9;}") as ui:
        gr.Markdown("# 🌾 Farmer Query Assistant")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                category = gr.Dropdown(DATA_CONSTANTS.QUERY_TYPE, label="category", value="Plant Protection")
                question = gr.Textbox(label="Your Question", lines=3)
                submit = gr.Button("Submit")
            with gr.Column(scale=2, min_width=300):
                output = gr.Markdown(elem_id="output-box")
        submit.click(
            fn=query_chat,
            inputs=[category, question],
            outputs=output,
            api_name="predict",
            queue=False 
        )
        question.submit(
            fn=query_chat,
            inputs=[category, question],
            outputs=output,
            api_name="predict",
            queue=False
        )

         # Configure queue with concurrency settings
        ui.queue(
            default_concurrency_limit=20,  # Max 20 concurrent requests
            max_size=100,                  # Queue up to 100 requests
            api_open=True                  # Enable API access
        )
    
    return ui.app

# Follow the steps to deploy to the Modal cloud:
# 1. Login here https://modal.com, and setup the secrete token of hugging face. 
# 2. Install Modal: pip install modal
# 3. Setup : "modal setup"  to generate a modl platform tokens.
# 4. create a workspace in modal platform with meaningful name: modal environment create <ENV_NAME>
# 5. switch to that workspace if multiple exisits, using: modal profile activate <workspace-name>
# 6. View all the available workspaces using: modal profile list
# 7. Run to deploy: modal deploy src/inference/farmer-assistant.py
# 8. Run hot deploymnet from local: modal serve src/inference/farmer-assistant.py