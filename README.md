# 🌟VRSA：Jailbreaking Multimodal Large Language Models through Visual Reasoning Sequential Attack

## 💡 1.Introduction
The powerful visual reasoning capabilities of MLLMs bring potential security risks of jailbreak attacks. To fully evaluate potential safety risks in the visual reasoning task, we propose Visual Reasoning Sequential Attack (VRSA), which induces MLLMs to gradually externalize and aggregate complete harmful intent by decomposing the original harmful text into several sequentially related sub-images.

## 🗺️ 2.Pipeline
The following figure illustrates the overall workflow of the proposed methodology, clarifying the relationships between different modules.

![Figure 1](case/framework.jpg)
*Figure 1: Framework of our VRSA. Based on a harmful text T, VRSA operates in four stages: (1) A scene judge model selects and rewrites the scene S most relevant to the harmful text T, and decomposes it into scene-grounded sub-texts. (2) A semantic judge model and a revise model mask and reconstruct the semantically low-relevance content within the sub-texts. (3) A Text-to-Image model iteratively generates sub-images that are semantically aligned with the sub-texts, and the semantic similarity is measured by Visual-language Model (CLIP). (4) The sub-images are combined with a pre-defined guided texts to jailbreak the MLLMs.*

## 🔬 3.Performance

Experimental results of VRSA on four mainstream open-source models and four closed-source models are provided here.

It is important to emphasize that our VRSA is a purely black-box attack, meaning the generation of the adversarial images does not involve any interaction with the target models.

The evaluation metrics we adopted include the Attack Success Rate (ASR) and the Toxicity Score of the model's response (Toxic), where a higher score indicates a more harmful response.

Category abbreviations: **IA**(Illegal Activities),**HS**(Hate Speech),**PH**(Physical Harm),**FR**(Fraud),**PO**(Pornography),**PV**(Privacy Violation),**Average** denotes the mean over the seven categories.

<div align="center">
Table 1: Performance on four open-source models
</div>

![Table 1](case/performance_for_open_source.png)

<div align="center">
Table 2: Performance on four closed-source models
</div>

![Table 1](case/performance_for_closed_source.png)

## 📥 Quickstart
### **1.Requirements**


``````
conda create -n VRSA python=3.10
conda activate VRSA
pip install -r requirements.txt
``````

### **2.Data Preparation**

Organize the queries into a JSON file named **harmful_query.json**, and store it under the path **data/harmful_query/harmful_query.json**. The format of **harmful_query.json** is as follows:
``````
{
    "0": {

        "Question": "Question_0"
    },
    "1": {
        "Question": "Question_1"
    },
    "2": {
        "Question": "Question_2"
    },
.....
``````

### **3.Replacement**

Next, you need to replace the **API key** and **model address** for the closed-source models in **model_configs** within **utils.py** with your own secret key and the official API model address. Additionally, if you need to use other models, you can expand the implementation by referring to how it's done in **utils.py**.

### **4.Generatation**

First, start Ollama.
``````
ollama start
``````
Then, run
``````
python generation.py
``````
in a new terminal to obtain the VRSA images in **visual_reasoning_images/Image_data**.

### **5.Evaluation**
First, specify the attack model and the judge model in **eval.py**, then run
``````
python eval.py
``````
to perform the evaluation.

It is important to emphasize that we also support various attack methods under a pure black-box setting. For specific instructions on how to do this, please refer to the comments in **eval.py**.

### **6.Implementation information**

To better track the generation and evaluation process of VRSA images, in addition to log files, we also provide JSON files that record key information of the intermediate steps according to a specific format. Specific details can be found in the comments at the beginning of **eval.py** and **generation.py**.




