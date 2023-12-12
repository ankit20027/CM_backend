import numpy as np
import spacy, torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# # Loading model from the caller script
# nlp = spacy.load("en_core_web_sm")
# df = pd.read_csv("Training.csv")

class NLP:
    
    df = None
    nlp = None
    
    def __init__(self, df, nlp) -> None:
        self.df = df
        self.nlp = nlp
    
    # Exact matches
    # TODO: return encodings insted of matching columns
    def __direct_matches(self, query):
        # NLP
        query = query.strip()
        tokenized_query = [token.text for token in self.nlp(query)]
        # Matchings
        matching_columns = []
        for column in self.df.columns:
            if any(token in column.lower() for token in tokenized_query):
                matching_columns.append(column)
        return matching_columns

    # TODO: return encodings
    def __preprocess_bert(self, query):
        model_name = "dmis-lab/biobert-v1.1"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        
        query_embedding = None
        query_token = [token.text for token in self.nlp(query)]
        token_ids = tokenizer.convert_tokens_to_ids(query_token)
        input_ids = torch.tensor(token_ids).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids)
            token_embeddings = outputs.last_hidden_state
            query_embedding = torch.mean(token_embeddings, dim=1).numpy()
        
        symptom_embeddings = []
        symptom_tokens = [tokenizer.tokenize(symptom) for symptom in self.df.columns]
        for tokens in symptom_tokens:
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(token_ids).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_ids)
                token_embeddings = outputs.last_hidden_state
                symptom_embedding = torch.mean(token_embeddings, dim=1).numpy()
                symptom_embeddings.append(symptom_embedding)
        
        # now we will calcualte  similarity between tokenized queries and symptoms
        query_embedding_flat = np.concatenate([query_embedding], axis=0)
        symptom_embedding_flat = np.concatenate(symptom_embeddings, axis=0)
        similarities = cosine_similarity(query_embedding_flat, symptom_embedding_flat)
        
        top_matches = np.argsort(similarities, axis=1)[:, -3:]
        symptoms = list(self.df.columns)
        matched_columns = [symptoms[idx] for idx in top_matches[0] if idx < len(symptoms)]
        return matched_columns
    
    # Helper methods convert matches to encodings
    def get_encodings(self, query):
        matched_columns = None
        direct_matched_columns = self.__direct_matches(query)
        biobert_matched_columns = self.__preprocess_bert(query)
        if len(direct_matched_columns) >= len(biobert_matched_columns): matched_columns = direct_matched_columns
        else: matched_columns = biobert_matched_columns

        if len(matched_columns) == 0: return -1
        
        matched_columns_set = set(matched_columns)
        encodings = [1 if (symptom in matched_columns_set) else 0 for symptom in self.df.columns[:-2]]
        return encodings
        

