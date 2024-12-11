from SourceFiles.utils import load_json,load_config,save_to_json
config = load_config()
class file_handler:
    def __init__(self,path = "Data/batch_summaries.json"):
        self.path = path
    def save_batch_summaries(self,summaries):
        file = load_json(self.path)
        file[config["dataset_path"]] = summaries
        save_to_json(self.path,file)
    def load_batch_summaries(self):
        batch_summaries = load_json(self.path)
        return batch_summaries[config["dataset_path"]]