from SourceFiles.utils import load_config
from SourceFiles.model_files.base_model import base_model
import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd

config = load_config()
class transformers_llm(base_model): #This will be invoked for all model names starting with transformers:
    def __init__(self,model_name):
        load_dotenv()
        HF_KEY= os.environ.get("HF_KEY")
        if HF_KEY:
            os.environ["HF_TOKEN"] = HF_KEY       
        torch.random.manual_seed(0)
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
        )       
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    def run_model(self,prompt,max_new_tokens = 250):
        

        messages = [
            {"role": "user", "content": prompt},
        ]

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature": 0.1,
            "do_sample": True,   
        }

        output = pipe(messages, **generation_args)
        return output[0]['generated_text'],generation_args["max_new_tokens"]
    def recursively_summarize(self, df, chunk_size=15, max_length=100):
        """
        Recursively summarizes reviews from a DataFrame using an LLM with hierarchical summarization.

        Parameters:
        - _model: An object with a `run_model(prompt)` method that generates a response.
        - df: A pandas DataFrame containing the reviews.
        - column_name: The column name in the DataFrame containing the review text.
        - chunk_size: Number of reviews per chunk for the LLM to summarize in one pass.
        - max_length: The maximum token length for each summary.

        Returns:
        - A single string containing the final summarized output.
        """
        column_name = config["rating_column"]
        # Extract the reviews from the DataFrame
        reviews = df[column_name].to_list()

        # Base case: If the number of reviews is small enough, summarize directly
        if len(reviews) <= chunk_size:
            prompt = (
                "Summarize the following reviews into a concise and coherent paragraph:\n\n"
                + "\n\n".join(reviews)
            )
            return self.run_model(prompt)
        
        # Divide reviews into chunks of `chunk_size`
        chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            prompt = (
                "Summarize the following reviews into a concise and coherent paragraph:\n\n"
                + "\n\n".join(chunk)
            )
            summary,_ = self.run_model(prompt)
            chunk_summaries.append(summary)
        
        # Convert summaries into a DataFrame for the recursive step
        summaries_df = pd.DataFrame(chunk_summaries, columns=[column_name])
        
        # Recursive step: Summarize the summaries
        return self.recursively_summarize( summaries_df,chunk_size, max_length)
   



