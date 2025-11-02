

"""
@author: Naveen N G
@date: 28-10-2025
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hg_login import login_hf
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.manifold import TSNE
import plotly.graph_objects as go


from load_data import DataLoader

login_hf()  

MAXIMUM_DATAPOINTS = 3000


DB = "chroma_database"
collection_name = "kcc_data_collection"
current_dir = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(current_dir, DB)

class RagPipline:

    def __init__(self):
        self.client = chromadb.PersistentClient(path=database_path)   
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 
        # self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        


    def load_dataset(self):
        # Loading the dataset form local, which was saved in pickle files.
        train, test = DataLoader().get_saved_data()
        return [train, test]

    def create_chroma_database(self):
        existing_collection_names = [collection.name for collection in  self.client.list_collections()]
        # if collection_name in existing_collection_names:
        #     self.client.delete_collection(collection_name)
        #     print(f"Deleted existing collection: {collection_name}")
        # self.collection = self.client.create_collection(collection_name)
        self.collection  = self.client.get_collection(name=collection_name)

    def user_query(self, item):
        return item['prompt'].split("\n\nAnswer is:")[0]
    
    def save_data_in_chroma_datstore(self):
        train, test = self.load_dataset()
        print('Recived the data.....')
        NUMBER_OF_DOCUMENTS = len(train)
        for i in tqdm(range(0, NUMBER_OF_DOCUMENTS, 1000)):
            documents = [self.user_query(item) for item in train[i: i+1000]]
            vectors = self.embedder.encode(documents).tolist()
            metadatas = [{"QueryType": item['QueryType'], "answer": item['Answer']} for item in train[i: i+1000]]
            ids = [f"doc_{j}" for j in range(i, i+len(documents))]
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=vectors,
                metadatas=metadatas
            )

    def render_vector_db_graph(self):
        print('inside render_vector_db_graph...')
        CATEGORIES = ['Post Harvest Preservation', 'Government Schemes', 'Varieties', 'Market Information','Cultural Practices', 'Plant Protection', 'Weed Management', 'Fertilizer Use and Availability', 'Nutrient Management']
        COLORS = ['red', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'cyan', 'black']     

        result = self.collection.get(include=['embeddings', 'documents', 'metadatas'], limit=MAXIMUM_DATAPOINTS)
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        categories = [metadata['QueryType'] for metadata in result['metadatas']]
        query_types = set(metadata['QueryType'] for metadata in result['metadatas'])
        for c in categories:
            if c not in CATEGORIES:
                CATEGORIES.append(c)
                COLORS.append('gray')  # Assign a default color for unknown categories
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        tsne = TSNE(n_components=3, random_state=42, perplexity=30, method='exact') 
        reduced_vectors = tsne.fit_transform(vectors)

        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode='markers',
            marker=dict(size=3, color=colors, opacity=0.7),
            hovertext=[f"Query Type: {t}<br>Text: {d}..." for t, d in zip(categories, documents)],
            hovertemplate='%{hovertext}<extra></extra>'
        )])

        fig.update_layout(
            title='3D Chroma Vectorstore Visualization',
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
            width=1200,
            height=800,
            margin=dict(r=20, b=10, l=10, t=40)
        )
        # fig.show()
        fig.write_html(f"{current_dir}/chroma_database/vector_database_3d.html", auto_open=True) 


def view_trained_data_vector_store():
    llmAgentRagPipline = RagPipline()
    llmAgentRagPipline.load_dataset()
    llmAgentRagPipline.create_chroma_database()
    llmAgentRagPipline.save_data_in_chroma_datstore()
    llmAgentRagPipline.render_vector_db_graph()


if __name__ == "__main__":
    view_trained_data_vector_store()

    


# Use this command to run : python src/data/data_rag_pipeline.py 