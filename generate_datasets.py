import os
import json
import pandas as pd
import re
import xml.etree.ElementTree as ET
from pathlib import Path


###############################
### Sentihood #################
###############################

sentihood_sentiments = ["none", "positive", "negative"]
sentihood_locations = ["location - 1", "location - 2"]
sentihood_aspects = ["general", "price", "safety", "transit location"]
sentihood_label2id = {"none": 0, "positive": 1, "negative": 2}


def generate_sentihood_QA_M(data, output_path):
    output = []
    for entry in data:
        id = entry["id"]
        original_sentence = entry["text"]
        for location in sentihood_locations:
            if location in original_sentence:
                for aspect in sentihood_aspects:
                    auxiliary_sentence = f"what do you think of the {aspect} of {location} ?"
                    label = "none"
                    for opinion in entry["opinions"]:
                        if opinion["target_entity"] == location and opinion["aspect"] == aspect:
                            if opinion["sentiment"] == "Positive":
                                label = "positive"
                            elif opinion["sentiment"] == "Negative":
                                label = "negative"
                    output.append([id, original_sentence, auxiliary_sentence, sentihood_label2id[label], label])
    loc1_rows = sorted([row for row in output if "location - 1" in row[2]], key=lambda el: el[0])
    loc2_rows = sorted([row for row in output if "location - 2" in row[2]], key=lambda el: el[0])
    output = [["id", "original_sentence", "auxiliary_sentence", "label_id", "label"]]
    output += loc1_rows + loc2_rows
    df = pd.DataFrame(output)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def generate_sentihood_NLI_M(data, output_path):
    output = []
    for entry in data:
        id = entry["id"]
        original_sentence = entry["text"]
        for location in sentihood_locations:
            if location in original_sentence:
                for aspect in sentihood_aspects:
                    auxiliary_sentence = f"{location} - {aspect}"
                    label = "none"
                    for opinion in entry["opinions"]:
                        if opinion["target_entity"] == location and opinion["aspect"] == aspect:
                            if opinion["sentiment"] == "Positive":
                                label = "positive"
                            elif opinion["sentiment"] == "Negative":
                                label = "negative"
                    output.append([id, original_sentence, auxiliary_sentence, sentihood_label2id[label], label])
    loc1_rows = sorted([row for row in output if "location - 1" in row[2]], key=lambda el: el[0])
    loc2_rows = sorted([row for row in output if "location - 2" in row[2]], key=lambda el: el[0])
    output = [["id", "original_sentence", "auxiliary_sentence", "label_id", "label"]]
    output += loc1_rows + loc2_rows
    df = pd.DataFrame(output)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def generate_sentihood_QA_B(data, output_path):
    output = []
    for entry in data:
        id = entry["id"]
        original_sentence = entry["text"]
        for location in sentihood_locations:
            if location in original_sentence:
                for aspect in sentihood_aspects:
                    current_output = []
                    for sentiment in sentihood_sentiments:
                        auxiliary_sentence = f"the polarity of the aspect {aspect} of {location} is {sentiment}"
                        current_output.append([id,
                                               original_sentence,
                                               auxiliary_sentence,
                                               1 if sentiment == "none" else 0])  # Default label is 1 for None
                    for opinion in entry["opinions"]:
                        if opinion["target_entity"] == location and opinion["aspect"] == aspect:
                            current_output[0][-1] = 0  # Overwrite label of None if there is an opinion
                            if opinion["sentiment"] == "Positive":
                                current_output[1][-1] = 1
                            elif opinion["sentiment"] == "Negative":
                                current_output[2][-1] = 1
                    output.extend(current_output)
    loc1_rows = sorted([row for row in output if "location - 1" in row[2]], key=lambda el: el[0])
    loc2_rows = sorted([row for row in output if "location - 2" in row[2]], key=lambda el: el[0])
    output = [["id", "original_sentence", "auxiliary_sentence", "label"]]
    output += loc1_rows + loc2_rows
    df = pd.DataFrame(output)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def generate_sentihood_NLI_B(data, output_path):
    output = []
    for entry in data:
        id = entry["id"]
        original_sentence = entry["text"]
        for location in sentihood_locations:
            if location in original_sentence:
                for aspect in sentihood_aspects:
                    current_output = []
                    for sentiment in sentihood_sentiments:
                        auxiliary_sentence = f"{location} - {aspect} - {sentiment}"
                        current_output.append([id,
                                               original_sentence,
                                               auxiliary_sentence,
                                               1 if sentiment == "none" else 0])  # Default label is 1 for None
                    for opinion in entry["opinions"]:
                        if opinion["target_entity"] == location and opinion["aspect"] == aspect:
                            current_output[0][-1] = 0  # Overwrite label of None if there is an opinion
                            if opinion["sentiment"] == "Positive":
                                current_output[1][-1] = 1
                            elif opinion["sentiment"] == "Negative":
                                current_output[2][-1] = 1
                    output.extend(current_output)
    loc1_rows = sorted([row for row in output if "location - 1" in row[2]], key=lambda el: el[0])
    loc2_rows = sorted([row for row in output if "location - 2" in row[2]], key=lambda el: el[0])
    output = [["id", "original_sentence", "auxiliary_sentence", "label"]]
    output += loc1_rows + loc2_rows
    df = pd.DataFrame(output)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def convert_sentihood_input(data):
    for entry in data:
        entry["text"] = entry["text"].replace("LOCATION2", "location - 2").replace("LOCATION1", "location - 1")
        for opinion in entry["opinions"]:
            opinion["target_entity"] = opinion["target_entity"]\
                    .replace("LOCATION1", "location - 1").replace("LOCATION2", "location - 2")
            opinion["aspect"] = opinion["aspect"].replace("transit-location", "transit location")
    return data


###############################
### Semeval ###################
###############################

semeval_sentiments = ["positive", "neutral", "negative", "conflict", "none"]
semeval_aspects = ["price", "anecdotes", "food", "ambience", "service"]
semeval_label2id = {"positive": 0, "neutral": 1, "negative": 2, "conflict": 3, "none": 4}


def generate_semeval_QA_M(root, output_path):
    output = [["id", "original_sentence", "auxiliary_sentence", "label_id", "label"]]
    for sentence in root:
        id = sentence.attrib["id"]
        original_sentence = sentence.find("text").text
        for aspect in semeval_aspects:
            auxiliary_sentence = f"what do you think of the {aspect} of it ?"
            label = "none"
            for opinion in sentence.find("aspectCategories"):
                if opinion.attrib["category"] == aspect:
                    label = opinion.attrib["polarity"]
            output.append([id, original_sentence, auxiliary_sentence, semeval_label2id[label], label])
    df = pd.DataFrame(output)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def generate_semeval_NLI_M(root, output_path):
    output = [["id", "original_sentence", "auxiliary_sentence", "label_id", "label"]]
    for sentence in root:
        id = sentence.attrib["id"]
        original_sentence = sentence.find("text").text
        for aspect in semeval_aspects:
            auxiliary_sentence = f"{aspect}"
            label = "none"
            for opinion in sentence.find("aspectCategories"):
                if opinion.attrib["category"] == aspect:
                    label = opinion.attrib["polarity"]
            output.append([id, original_sentence, auxiliary_sentence, semeval_label2id[label], label])
    df = pd.DataFrame(output)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def generate_semeval_QA_B(root, output_path):
    output = [["id", "original_sentence", "auxiliary_sentence", "label"]]
    for sentence in root:
        id = sentence.attrib["id"]
        original_sentence = sentence.find("text").text
        for aspect in semeval_aspects:
            current_output = []
            found_opinion_flag = False
            for sentiment in semeval_sentiments:
                auxiliary_sentence = f"the polarity of the aspect {aspect} is {sentiment}."
                label = 0
                for opinion in sentence.find("aspectCategories"):
                    if opinion.attrib["category"] == aspect:
                        if opinion.attrib["polarity"] == sentiment:
                            label = 1
                            found_opinion_flag = True
                if not found_opinion_flag and sentiment == "none":
                    label = 1
                current_output.append([id, original_sentence, auxiliary_sentence, label])
            output.extend(current_output)
    df = pd.DataFrame(output)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def generate_semeval_NLI_B(root, output_path):
    output = [["id", "original_sentence", "auxiliary_sentence", "label"]]
    for sentence in root:
        id = sentence.attrib["id"]
        original_sentence = sentence.find("text").text
        for aspect in semeval_aspects:
            current_output = []
            found_opinion_flag = False
            for sentiment in semeval_sentiments:
                auxiliary_sentence = f"{aspect} - {sentiment}"
                label = 0
                for opinion in sentence.find("aspectCategories"):
                    if opinion.attrib["category"] == aspect:
                        if opinion.attrib["polarity"] == sentiment:
                            label = 1
                            found_opinion_flag = True
                if not found_opinion_flag and sentiment == "none":
                    label = 1
                current_output.append([id, original_sentence, auxiliary_sentence, label])
            output.extend(current_output)
    df = pd.DataFrame(output)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def convert_semeval_input(root):
    for sentence in root:
        for aspect in sentence.find("aspectCategories"):
            if aspect.attrib["category"] == "anecdotes/miscellaneous":
                aspect.attrib["category"] = "anecdotes"
    return root


if __name__ == '__main__':
    SENTIHOOD_DIR = "data/sentihood/"
    SEMEVAL_DIR = "data/semeval2014/"

    Path(f"{SENTIHOOD_DIR}BERT-pair/").mkdir(parents=True, exist_ok=True)
    for file in os.scandir(SENTIHOOD_DIR):
        if file.name.endswith(".json"):
            with open(file.path, "r") as f:
                data = json.loads(f.read())
                data = convert_sentihood_input(data)
                set = re.findall("(?<=-).*(?=\.)", file.name)[0]
                output_path = f"{SENTIHOOD_DIR}BERT-pair/{set}_QA_M.csv"
                generate_sentihood_QA_M(data, output_path)
                output_path = f"{SENTIHOOD_DIR}BERT-pair/{set}_NLI_M.csv"
                generate_sentihood_NLI_M(data, output_path)
                output_path = f"{SENTIHOOD_DIR}BERT-pair/{set}_QA_B.csv"
                generate_sentihood_QA_B(data, output_path)
                output_path = f"{SENTIHOOD_DIR}BERT-pair/{set}_NLI_B.csv"
                generate_sentihood_NLI_B(data, output_path)

    Path(f"{SEMEVAL_DIR}BERT-pair/").mkdir(parents=True, exist_ok=True)
    for file in os.scandir(SEMEVAL_DIR):
        if file.name.endswith(".xml"):
            with open(file.path, "r") as f:
                tree = ET.parse(f)
                root = convert_semeval_input(tree.getroot())
                set = re.findall("(?<=_).*?(?=[_|.])", file.name)[0].lower()
                output_path = f"{SEMEVAL_DIR}BERT-pair/{set}_QA_M.csv"
                generate_semeval_QA_M(root, output_path)
                output_path = f"{SEMEVAL_DIR}BERT-pair/{set}_NLI_M.csv"
                generate_semeval_NLI_M(root, output_path)
                output_path = f"{SEMEVAL_DIR}BERT-pair/{set}_QA_B.csv"
                generate_semeval_QA_B(root, output_path)
                output_path = f"{SEMEVAL_DIR}BERT-pair/{set}_NLI_B.csv"
                generate_semeval_NLI_B(root, output_path)
