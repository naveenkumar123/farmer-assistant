
"""
@author: Naveen N G
@date: 31-10-2025
"""

# imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from dotenv import load_dotenv
from hg_login import login_hf
from sentence_transformers import SentenceTransformer
import chromadb
from llama_chat import LlamaChat
import gradio as gr
import constants as DATA_CONSTANTS
from langdetect import detect, LangDetectException
import langcodes
from language_convert import LanguageConverter

load_dotenv()
login_hf()


DB_PATH = "chroma_database"
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(current_dir, '..', 'data' , DB_PATH))


class RagFinetuneModel:
    def __init__(self):
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        # self.translator = LanguageConverter()
        self.get_vector_db()


    async def translate_text(self, text: str, lang: str = 'en'):
        translated = (await self.translator.translate(text, src='hi', dest=lang)).text
        return translated

    def get_vector_db(self):
        client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = client.get_or_create_collection('kcc_data_collection')

    def vector(self, query):
        return self.embedder.encode([query])
    
    def find_similars(self, query):
        results = self.collection.query(query_embeddings=self.vector(query).astype(float).tolist(), n_results=5)
        questions = results['documents'][0][:]
        answers = [m['answer'] for m in results['metadatas'][0][:]]
        return questions, answers
    
    async def create_context(self, questions, answers):
        message = "To provide some context, here are some of the questions and answers that might be similar to the question you need to answer.\n\n"
        for question, answer in zip(questions, answers):
            message += f"Potentially related question:\n{question}\nAnswer is: ${answer}\n\n"
        return message
    
    def build_internal_query(self, query, catgory = '', query_type = ''):
        message = ''
        if catgory and len(catgory) > 0:
             message += f"Under {catgory} category."
        message += self.translator.translate(query)
        if query_type and len(query_type) > 0:
            message += f"This question is related to ${query_type}"
        return message
    
    async def get_messages(self, query, catgory, query_type):
        # try:
        #     lang = detect(query) or ''
        # except LangDetectException:
        #     lang = 'en'
        # language = langcodes.Language.make(lang).display_name() or 'English'
        # self.translator.set_tokenizer(language)
        # language_info = ''
        # if len(language) > 0:
        #     language_info = f"The user’s language is {language}.\n\n"
        # query = self.translator.translate(query)

        questions, answers   = self.find_similars(query)
        # internal_query = self.build_internal_query(query, catgory, query_type)
        system_message = f"You are an assistant. You provide the response to the user questions. Reply only with the answer, no explanation."
        user_prompt = await self.create_context(questions, answers)
        user_prompt += f"Now the question for you:\n\n"
        user_prompt += query
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Answer is:"}
        ]
    
    async def predict_answer(self, catgory, query_type, query):
        messages = await self.get_messages(query, catgory, query_type)
        response = LlamaChat().chat(messages)
        response += "\n\n*For more information, Visit reginal office near by you.*"
        return response
    
def advanced_chat_ui_render(callFunc):
    view =  gr.Interface(
                fn=callFunc,
                inputs=[
                    gr.Dropdown(DATA_CONSTANTS.CATEGORIES, label="Select Category"),
                    gr.Dropdown(DATA_CONSTANTS.QUERY_TYPE, label="Select Query Type"),
                    gr.Textbox(label="Ask your question:")
                ], 
                outputs=[gr.Markdown(label="Response:")],
                flagging_mode="never")
    view.launch(share=False)



if __name__ == "__main__":
   ragFinetuneModel = RagFinetuneModel()
   advanced_chat_ui_render(ragFinetuneModel.predict_answer)


# Use this command to run : python src/model/rag_finetune_model.py 