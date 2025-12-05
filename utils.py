from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import io
import base64

import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

#-----------------------------------Closed-source model calls-------------------------------------#

def multi_model_gpt_api_image(prompt, mode="judge", choice="gpt-3.5-turbo", image_path=None):
    

    with Image.open(image_path) as img:

        img_io = io.BytesIO()
        img.save(img_io, format="JPEG", quality=50)  
        img_io.seek(0)  
        base64_image = base64.b64encode(img_io.read()).decode("utf-8")
    messages = [{'role': 'user', 
                 'content':[
                            {
                                "type":"image_url",
                                "image_url":{
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text" : prompt
                            }
                            ]
                }
                ]
    
    model_configs = [       
        {
            "api_key": Your_API_Key,
            "base_url": "Official model API calling address",
            "model": "gpt-5-mini"
        },
        {
            "api_key": Your_API_Key,
            "base_url": "Official model API calling address",
            "model": "gpt-5"
        },
        {
            "api_key": Your_API_Key,
            "base_url": "Official model API calling address",
            "model": "gpt-4o"
        },
        {
            "api_key": Your_API_Key,
            "base_url": "Official model API calling address",
            "model": "o3" # gpt-o3
        },
    ]

    
    for config in model_configs[1:]:
        if mode == "judge":

            try:
                print(f"attempt to load model: {config['model']}")

                # create client
                client = OpenAI(
                    api_key=config["api_key"],
                    base_url=config["base_url"]
                )

                completion = client.chat.completions.create(
                    model=config["model"],
                    messages=messages
                )

                msg = completion.choices[0].message.content
                print(f"model {config['model']} load successfully ")
                print(msg)
                return msg

            except (AuthenticationError, RateLimitError, APIError) as e:
                print(f"model {config['model']} failed to load:  {str(e)}")
                continue  # next model
            except Exception as e:
                print(f"accident error: {str(e)}")
                continue

        elif mode=="test":
            if config["model"] == choice:

                try:
                    print(f"attempt to load model: {config['model']}")
                    client = OpenAI(
                        api_key=config["api_key"],
                        base_url=config["base_url"]
                    )

                    completion = client.chat.completions.create(
                        model = config["model"],
                        messages=messages
                    )
                    msg = completion.choices[0].message.content
                    print(msg)
                    print(f"model {config['model']} load successfully ")
                    return msg

                except (AuthenticationError, RateLimitError, APIError) as e:
                    print(f"model {config['model']} failed to load:  {str(e)}")
                    continue  # next model
                except Exception as e:
                    print(f"accident error: {str(e)}")
                    continue
                

    


    # if all model failed
    print("No model can be used")
    return None

def multi_model_gpt_api(prompt, mode="judge", choice="gpt-5-mini"):
    messages = [{'role': 'user', 'content': prompt}]
    model_configs = [       
        {
            "api_key": Your_API_Key,
            "base_url": "Official model API calling address",
            "model": "gpt-5-mini"
        },
        {
            "api_key": Your_API_Key,
            "base_url": "Official model API calling address",
            "model": "gpt-5"
        },
        {
            "api_key": Your_API_Key,
            "base_url": "Official model API calling address",
            "model": "gpt-4o"
        },
        {
            "api_key": Your_API_Key,
            "base_url": "Official model API calling address",
            "model": "o3" # gpt-o3
        },
    ]

    
    for config in model_configs[1:]:
        if mode == "judge":
            if config["model"] == choice: 
                try:
                    print(f"attempt to load model: {config['model']}")

                    # create client
                    client = OpenAI(
                        api_key=config["api_key"],
                        base_url=config["base_url"]
                    )

                    completion = client.chat.completions.create(
                        model=config["model"],
                        messages=messages
                    )

                    msg = completion.choices[0].message.content
                    print(f"model {config['model']} load successfully ")
                    print(msg)
                    return msg

                except (AuthenticationError, RateLimitError, APIError) as e:
                    print(f"model {config['model']} failed to load:  {str(e)}")
                    continue  # next model
                except Exception as e:
                    print(f"accident error: {str(e)}")
                    continue

        elif mode=="test":
            if config["model"] == choice:

                try:
                    print(f"attempt to load model: {config['model']}")

                    # create client
                    client = OpenAI(
                        api_key=config["api_key"],
                        base_url=config["base_url"]
                    )

                    completion = client.chat.completions.create(
                        model=config["model"],
                        messages=messages
                    )

                    msg = completion.choices[0].message.content
                    print(f"model {config['model']} load successfully ")
                    print(msg)
                    return msg

                except (AuthenticationError, RateLimitError, APIError) as e:
                    print(f"model {config['model']} failed to load:  {str(e)}")
                    continue  # next model
                except Exception as e:
                    print(f"accident error: {str(e)}")
                    continue
    # if all model failed
    print("No model can be used")
    return None
#----------------------------------End of closed-source model calls--------------------------------#


#-----------------------------------Open-source model calls----------------------------------------#
import base64

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

# Match the optimal aspect ratio for image cropping.
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

# Dynamically match the optimal aspect ratio and then perform cropping.
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def Intenvl_3_5_chat(question_text, image_path,model_path='OpenGVLab/InternVL3_5-8B',use_flash_attn=True):
    model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # Load image
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=2048, do_sample=True)
    
    # Single-turn image-text conversation
    question = '<image>\n'+question_text
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response

#---------------------------------End of open-source model calls---------------------------------------#





if __name__ == "__main__":
    import random
    s = "Analyze the sentiment of the input, and respond only positive or negative."
    words = s.split()
    random.shuffle(words)
    s_shuffle = " ".join(words)    
    print(s_shuffle)
    #qwen_chat(prompt="introduce")
    
    