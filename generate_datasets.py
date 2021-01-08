import os
import json

import nltk
import pandas as pd
import re

DATA_DIR = "data"
SENTIHOOD_DIR = "data/sentihood/"
SEMEVAL2014_DIR = "data/semeval2014/"

sentiments = ["None", "Positive", "Negative"]
locations = ["location - 1", "location - 2"]
aspects = ["general", "price", "safety", "transit location"]


def generate_sentihood_QA_M(data, output_path):
    output = []
    for entry in data:
        id = entry["id"]
        original_sentence = entry["text"]
        for location in locations:
            if location in original_sentence:
                for aspect in aspects:
                    auxiliary_sentence = f"what do you think of the {aspect} of {location} ?"
                    label = "None"
                    for opinion in entry["opinions"]:
                        if opinion["target_entity"] == location and opinion["aspect"] == aspect:
                            if opinion["sentiment"] == "Positive":
                                label = "Positive"
                            elif opinion["sentiment"] == "Negative":
                                label = "Negative"
                    output.append([id, original_sentence, auxiliary_sentence, label])
    loc1_rows = sorted([row for row in output if "location - 1" in row[2]], key=lambda el: el[0])
    loc2_rows = sorted([row for row in output if "location - 2" in row[2]], key=lambda el: el[0])
    output = [["id", "original_sentence", "auxiliary_sentence", "label"]]
    output += loc1_rows + loc2_rows
    df = pd.DataFrame(output)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def generate_sentihood_NLI_M(data, output_path):
    output = []
    for entry in data:
        id = entry["id"]
        original_sentence = entry["text"]
        for location in locations:
            if location in original_sentence:
                for aspect in aspects:
                    auxiliary_sentence = f"{location} - {aspect}"
                    label = "None"
                    for opinion in entry["opinions"]:
                        if opinion["target_entity"] == location and opinion["aspect"] == aspect:
                            if opinion["sentiment"] == "Positive":
                                label = "Positive"
                            elif opinion["sentiment"] == "Negative":
                                label = "Negative"
                    output.append([id, original_sentence, auxiliary_sentence, label])
    loc1_rows = sorted([row for row in output if "location - 1" in row[2]], key=lambda el: el[0])
    loc2_rows = sorted([row for row in output if "location - 2" in row[2]], key=lambda el: el[0])
    output = [["id", "original_sentence", "auxiliary_sentence", "label"]]
    output += loc1_rows + loc2_rows
    df = pd.DataFrame(output)
    df.to_csv(output_path, sep='\t', index=False, header=False)


def generate_sentihood_QA_B(data, output_path):
    output = []
    for entry in data:
        id = entry["id"]
        original_sentence = entry["text"]
        for location in locations:
            if location in original_sentence:
                for aspect in aspects:
                    current_output = []
                    for sentiment in sentiments:
                        auxiliary_sentence = f"the polarity of the aspect {aspect} of {location} - is {sentiment} ."
                        current_output.append([id,
                                               original_sentence,
                                               auxiliary_sentence,
                                               1 if sentiment == "None" else 0])  # Default label is 1 for None
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
        for location in locations:
            if location in original_sentence:
                for aspect in aspects:
                    current_output = []
                    for sentiment in sentiments:
                        auxiliary_sentence = f"{sentiment} - {location} - {aspect}"
                        current_output.append([id,
                                               original_sentence,
                                               auxiliary_sentence,
                                               1 if sentiment == "None" else 0])  # Default label is 1 for None
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


def convert_sentihood_text(data):
    for entry in data:
        entry["text"] = " ".join(nltk.word_tokenize(entry["text"]))\
                        .replace("LOCATION1", "location - 1").replace("LOCATION2", "location - 2")\
                        .replace("'", "' ")\
                        .lower()
        entry["text"] = entry["text"]
        for opinion in entry["opinions"]:
            opinion["target_entity"] = opinion["target_entity"]\
                    .replace("LOCATION1", "location - 1").replace("LOCATION2", "location - 2")
            opinion["aspect"] = opinion["aspect"].replace("transit-location", "transit location")
    return data


if __name__ == '__main__':

    for file in os.scandir(SENTIHOOD_DIR):
        if file.name.endswith(".json"):
            with open(file.path, "r") as f:
                data = json.loads(f.read())
                data = convert_sentihood_text(data)
                set = re.findall("(?<=-).*(?=\.)", file.name)[0]
                output_path = f"{SENTIHOOD_DIR}{set}_QA_M.csv"
                generate_sentihood_QA_M(data, output_path)
                output_path = f"{SENTIHOOD_DIR}{set}_NLI_M.csv"
                generate_sentihood_NLI_M(data, output_path)
                output_path = f"{SENTIHOOD_DIR}{set}_QA_B.csv"
                generate_sentihood_QA_B(data, output_path)
                output_path = f"{SENTIHOOD_DIR}{set}_NLI_B.csv"
                generate_sentihood_NLI_B(data, output_path)
