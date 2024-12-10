
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from SourceFiles.utils import load_config
config = load_config()
class bert_sentiment:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    def sentiment(self,df,is_df = True):
        reviews = df[config["review_column"]].apply(lambda x: x[:512]).to_list() if is_df else df
        predictions = []
        for review in reviews:
            predictions.append(self.sentiment_score(review))
        if is_df:
            df["sentiment_score"] = predictions
            df["sentiment_label"] = df["sentiment_score"].apply(lambda x: "positive" if x >= 3 else "negative") 
            return df
        return predictions
    def sentiment_score(self,text):
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        result = self.model(tokens)
        return int(torch.argmax(result.logits))+1  