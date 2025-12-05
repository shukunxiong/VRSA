import os
import json
from toxic_judge import extract_content, judge_template
from VRSA_Context import  VRSA
import time
from utils import multi_model_gpt_api,multi_model_gpt_api_image,Intenvl_3_5_chat
import logging
import datetime
from pathlib import Path

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"generation/log/{current_time}.log"
Path(log_filename).parent.mkdir(parents=True, exist_ok=True)
os.makedirs("res", exist_ok=True)
# log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

"""
    To facilitate the monitoring of the Visual Reasoning Sequence generation process, in addition to log files, 
    we also provide a structured JSON file to record key information during generation. Specifically, its structure is as follows:
    
    "Harmful_question.json":{                         # The filename of the JSON file currently being read that contains the original harmful queries.
        "Harmful_Query_i":{                           # Represents that the currently read item is the $i$-th question
            "original_harmful_query":                 # The complete harmful query
            "initial_scene":                          # The most relevant scene matched from the scene library
            "scene_in_adaptive_refinement":           # All intermediate scenes that have appeared in the scene adaptive refinement stage
            "scene_score_in_adaptive_refinement":     # The corresponding scores of the intermediate scenes
            "scene_after_adaptive_refinement”：       # The final scene obtained after scene adaptive refinement
            "initial_instruction_4":                  # The instruction after the initial splitting
            "instruction_4_after_scene_embedding":    # The instruction after scene embedding
            "instruction_4_after_completion":         # The final instruction after semantic coherent completion
            "capture_content":                        # The instruction for make picture
            "image_path":                             # The path of the generated image
        }
        "Harmful_Query_i+1":{
            .......
        }
    }
"""

class Generation:
    def __init__(self):
        self.save_idx = 0
        
        self.folder_path = 'generation/json'
        self.image_save_path = "visual_reasoning_images/"
        self.harmful_query_json_path = 'data/harmful_query/harmful_query.json'
        self.scene_library_path = "data/env.json"
    def read_harmful_query_json(self):

        folder_path = self.harmful_query_json_path
        output = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                logging.info(f"--- read file: {filename} ---")
                self.filename = filename
                
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        logging.info(f"error: {filename} is not a valid json file")
                        continue

                file_idx = filename.split("-")[0]

                if not file_idx in output:
                    output[file_idx] = []

                for entry_id, entry in data.items():
                    question = entry.get("Question", "No Question")
                    output[file_idx].append((entry_id, question))

        return output
        
    def generation(self):
        
        json_name = f'{current_time}.json'
        json_path = os.path.join(self.folder_path,json_name)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        
        data = {}
        
        input = self.read_harmful_query_json()
        for file_idx, Data in input.items():
            
            data[f'{file_idx}'] = {}
            question_idx = 0
            for idx in range(len(Data)):
                
                data_ = Data[idx]
                harmful_query = data_[1]
                # MCA
                self.save_idx += 1
                
                # Used to save each image
                save_every_picture = 0
                
                # Define the data format of the JSON file
                data[f'{file_idx}'][f'Harmful_Query_{question_idx}'] = {}
                data_one_question = data[f'{file_idx}'][f'Harmful_Query_{question_idx}']
                data_one_question['original_harmful_query'] = harmful_query
                data_one_question['initial_scene'] = []
                data_one_question['scene_in_adaptive_refinement'] = []
                data_one_question['scene_score_in_adaptive_refinement'] = []
                data_one_question['scene_after_adaptive_refinement'] = []
                data_one_question['initial_instruction_4'] = []
                data_one_question['instruction_4_after_scene_embedding'] = []
                data_one_question['instruction_4_after_completion'] = []
                data_one_question['capture_content'] = []
                data_one_question['image_path'] = []
                
                
                save_every_picture += 1
                mca = VRSA(decomposed_prompt=harmful_query, env_json_path = self.scene_library_path, image_save_dir=self.image_save_path,
                            save_idx=self.save_idx,save_every_picture=save_every_picture,
                            data=data_one_question,file_idx=file_idx,file_name = self.filename)
                mca.run()
                



if __name__ == "__main__":
    eval = Generation()
    eval.generation()
                  