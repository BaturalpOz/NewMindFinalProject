
from typing import List
from SourceFiles.model_files.transformers_llm import transformers_llm
from fastapi import FastAPI
from pydantic import BaseModel
from SourceFiles.preprocessing import get_dataset
from SourceFiles.sentiment_analysis import bert_sentiment
from SourceFiles.utils import prepare_prompt, load_config, prepare_prompt_from_list,save_to_json
from SourceFiles.database_handler import file_handler



app = FastAPI()

class ModelResponse (BaseModel):
    response_text:str
    token_usage:int
class NewComments (BaseModel):
    comments: List[str]
class ConfigData(BaseModel):
    dataset_path: str
    review_column:str
    rating_column:str
    def to_dict(self):
        return {
            "dataset_path":self.dataset_path,
            "review_column":self.review_column,
            "rating_column":self.rating_column
        }
_model = transformers_llm("microsoft/Phi-3.5-mini-instruct")
_sentiment = bert_sentiment()
_db = file_handler()
config = load_config()

@app.get("/summary", response_model=ModelResponse)
def get_summary():
    cleaned_reviews = get_dataset()
    cleaned_reviews = cleaned_reviews.head(20)
    classified_reviews = _sentiment.sentiment(cleaned_reviews)

    neg_prompt, avg_neg_score = prepare_prompt(classified_reviews,query = "sentiment_label == 'negative'", extra_text="Following comments are mostly negative. Summarize the reason of negativity")
    pos_prompt, avg_pos_score = prepare_prompt(classified_reviews,query="sentiment_label == 'positive'",extra_text="Following comments are mostly positive. summarize the reason of positivity")   
    neg_summary, token1 = _model.run_model(neg_prompt)
    pos_summary, token2 = _model.run_model(pos_prompt)

    avg_rating = classified_reviews["sentiment_score"].mean()
    prompt = f'''You are a customer relations expert. Your task is to summarize the general opinion about a product,
    the summary of the negative overall negative feedback: {neg_summary}, the summary of the positive overall
    negative feedback: {pos_summary}. Average rating is {avg_rating} out of 5. Give an overview that is understandable to the producers of the product'''
    response, token= _model.run_model(prompt) 
    token =  token + token1 + token2
    return {
    "response_text":response,
    "token_usage":token
    }
@app.get("/summary_recursive", response_model=ModelResponse)
def get_recursive_summary():
    cleaned_reviews = get_dataset()
    cleaned_reviews = cleaned_reviews.head(20)
    classified_reviews = _sentiment.sentiment(cleaned_reviews)
    response, token = _model.recursively_summarize(classified_reviews)
    return {
    "response_text":response,
    "token_usage": token
        }
@app.post("/summarize_new_reviews", response_model = ModelResponse)
def get_new_comment_summaries(new_comments: NewComments):
    batch_summaries_list = _db.load_batch_summaries()
    batch_summary= [text["Summary"] for text in batch_summaries_list]
    comments = batch_summary + new_comments.comments

    comment_ratings = _sentiment.sentiment(comments,is_df=False)
    avg_rating = (sum(comment_ratings) / len(comment_ratings))
    overall_prompt = prepare_prompt_from_list(comments, extra_text =f"The average rating for these comments is {avg_rating}.")
    overall_response, token1 = _model.run_model(overall_prompt)

    new_comments_rating = _sentiment.sentiment(new_comments.comments,is_df=False)
    new_comments_rating_avg = sum(new_comments_rating) / len(new_comments_rating)
    new_comments_prompt = prepare_prompt_from_list(new_comments.comments, extra_text=f"The average rating for these comments is {new_comments_rating_avg}.")
    new_comments_summary, token2 = _model.run_model(new_comments_prompt)

    token = token1 + token2
    response = f"Summary of the new comments: {new_comments_summary}. \n\n Overall summary: {overall_response}"
   
    return {
     "response_text":response,
    "token_usage":token
    }
@app.post("/change_config",response_model = dict)
def change_config(config_data: ConfigData):
    config_dict = config_data.to_dict()
    save_to_json("config.json",config_dict)
    return config_dict