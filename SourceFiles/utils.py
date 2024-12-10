import json

def load_config(config_path = "config.json"):
    return load_json(config_path)

def load_json(path) -> json:
    with open(path) as f:
        file  = json.load(f)
    return file
def save_to_json(path,dict):
    file = load_json(path)
    file[config["dataset_path"]] = dict
    with open(path, "w") as outfile: 
        json.dump(file, outfile)
config = load_config()
    
def prepare_prompt(df,size = None, query =  None, extra_text = ""):
    config = load_config()
    relevant_comments = df.query(query) if query else df
    relevant_comments = relevant_comments.head(size) if size else relevant_comments
    all_comments  = relevant_comments[config["review_column"]].str.cat(sep=' \n')
    avg_rating = relevant_comments["sentiment_score"].mean()
    prompt = f'''Summarize the following comments in a single sentence. Focus on the most prominent parts and come to a conclusion at the end. {extra_text}: \n{all_comments} '''
    return prompt, avg_rating

def prepare_prompt_from_list(list, size = None, extra_text = ""):

    relevant_comments = relevant_comments[:size] if size else list
    all_comments = ""
    for i in list:       
        all_comments += i
        all_comments + "\n"
    prompt = f'''Summarize the following comments in a single sentence. Focus on the most prominent parts and come to a conclusion at the end. {extra_text}: \n{all_comments}'''
    return prompt