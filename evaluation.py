import pandas as pd
from sklearn import metrics
import numpy as np
import argparse


def get_dataset(data_dir="data", dataset_type="sentihood"):
    original_sentences = []
    auxiliary_sentences = []
    labels = []
    data = pd.read_csv(f"{data_dir}/{dataset_type}/BERT-pair/test_NLI_M.csv", header=0, sep="\t").values.tolist()
    for row in data:
        original_sentences.append(row[1])
        auxiliary_sentences.append(row[2])
        labels.append(row[3])
    return original_sentences, auxiliary_sentences, labels


def get_predictions(predictions_dir, task, dataset_type):
    predicted_labels = []
    scores = []
    if task.endswith("M"):
        data = pd.read_csv(f"{predictions_dir}/{dataset_type}/BERT-pair/{task}.csv", header=0).values.tolist()
        for row in data:
            predicted_labels.append(int(row[0]))
            scores.append([float(el) for el in row[1:]])
    else:
        if dataset_type == "sentihood":
            if task.endswith("B"):
                data = pd.read_csv(f"{predictions_dir}/{dataset_type}/BERT-pair/{task}.csv", header=0).values.tolist()
                count_aspect_rows = 0
                current_aspect_scores = []
                for row in data:
                    current_aspect_scores.append(row[2])
                    count_aspect_rows += 1
                    if count_aspect_rows % 3 == 0:
                        sum_current_aspect_scores = np.sum(current_aspect_scores)
                        current_aspect_scores = [score / sum_current_aspect_scores for score in current_aspect_scores]
                        scores.append(current_aspect_scores)
                        predicted_labels.append(np.argmax(current_aspect_scores))
                        current_aspect_scores = []
            elif task == "single":
                sentihood_locations = ["location_1", "location_2"]
                sentihood_aspects = ["general", "price", "safety", "transit location"]
                data = {}
                for location in sentihood_locations:
                    data[location] = {}
                    for aspect in sentihood_aspects:
                        data[location][aspect] = pd.read_csv(f"{predictions_dir}/{dataset_type}/BERT-single/{location}_{aspect}.csv", header=0).values.tolist()
                for location in sentihood_locations:
                    for i in range(len(data[location][sentihood_aspects[0]])):
                        for aspect in sentihood_aspects:
                            scores.append(data[location][aspect][i][1:])
                            predicted_labels.append(int(data[location][aspect][i][0]))
        elif dataset_type == "semeval2014":
            if task.endswith("B"):
                data = pd.read_csv(f"{predictions_dir}/{dataset_type}/BERT-pair/{task}.csv", header=0).values.tolist()
                count_aspect_rows = 0
                current_aspect_scores = []
                for row in data:
                    current_aspect_scores.append(row[2])
                    count_aspect_rows += 1
                    if count_aspect_rows % 5 == 0:
                        sum_current_aspect_scores = np.sum(current_aspect_scores)
                        current_aspect_scores = [score / sum_current_aspect_scores for score in current_aspect_scores]
                        scores.append(current_aspect_scores)
                        predicted_labels.append(np.argmax(current_aspect_scores))
                        current_aspect_scores = []
            elif task == "single":
                semeval_aspects = ["price", "anecdotes", "food", "ambience", "service"]
                data = {}
                for aspect in semeval_aspects:
                    data[aspect] = pd.read_csv(f"{predictions_dir}/{dataset_type}/BERT-single/{aspect}.csv", header=0).values.tolist()
                for i in range(len(data[semeval_aspects[0]])):
                    for aspect in semeval_aspects:
                        scores.append(data[aspect][i][1:])
                        predicted_labels.append(int(data[aspect][i][0]))
    return predicted_labels, scores


def compute_sentihood_aspect_strict_accuracy(test_labels, predicted_labels):
    correct_count = 0
    num_examples = len(test_labels) // 4
    for i in range(num_examples):
        if test_labels[i * 4] == predicted_labels[i * 4]\
                and test_labels[i * 4 + 1] == predicted_labels[i * 4 + 1]\
                and test_labels[i * 4 + 2] == predicted_labels[i * 4 + 2]\
                and test_labels[i * 4 + 3] == predicted_labels[i * 4 + 3]:
            correct_count += 1
    return correct_count / num_examples


def compute_sentihood_aspect_macro_F1(test_labels, predicted_labels):
    total_precision = 0
    total_recall = 0
    num_examples = len(test_labels) // 4
    count_examples_with_sentiments = 0
    for i in range(num_examples):
        test_aspects = set()
        predicted_aspects = set()
        for j in range(4):
            if test_labels[i * 4 + j] != 0:
                test_aspects.add(j)
            if predicted_labels[i * 4 + j] != 0:
                predicted_aspects.add(j)
        if len(test_aspects) == 0:
            continue
        intersection = test_aspects.intersection(predicted_aspects)
        if len(intersection) > 0:
            precision = len(intersection) / len(predicted_aspects)
            recall = len(intersection) / len(test_aspects)
        else:
            precision = 0
            recall = 0
        total_precision += precision
        total_recall += recall
        count_examples_with_sentiments += 1
    ma_P = total_precision / count_examples_with_sentiments
    ma_R = total_recall / count_examples_with_sentiments
    return (2 * ma_P * ma_R) / (ma_P + ma_R)


def compute_sentihood_aspect_macro_AUC(test_labels, scores):
    aspects_test_labels = [[] for _ in range(4)]
    aspects_none_scores = [[] for _ in range(4)]
    for i in range(len(test_labels)):
        if test_labels[i] != 0:
            new_label = 0
        else:
            new_label = 1   # For metrics.roc_auc_score you need to use the score of the maximum label, so "None" : 1
        aspects_test_labels[i % 4].append(new_label)
        aspects_none_scores[i % 4].append(scores[i][0])
    aspect_AUC = []
    for i in range(4):
        aspect_AUC.append(metrics.roc_auc_score(aspects_test_labels[i], aspects_none_scores[i]))
    aspect_macro_AUC = np.mean(aspect_AUC)
    return aspect_macro_AUC


def compute_sentihood_sentiment_classification_metrics(test_labels, scores):
    """Compute macro AUC and accuracy for sentiment classification ignoring "None" scores"""
    # Macro AUC
    sentiment_test_labels = [[] for _ in range(4)]  # One list for each aspect
    sentiment_negative_scores = [[] for _ in range(4)]
    sentiment_predicted_label = []
    sentiment_test_label = []   # One global list
    for i in range(len(test_labels)):
        if test_labels[i] != 0:
            new_test_label = test_labels[i] - 1  # "Positive": 0, "Negative": 1
            sentiment_test_label.append(new_test_label)
            new_negative_score = scores[i][2] / (scores[i][1] + scores[i][2])   # Prob. of "Negative" ignoring "None"
            if new_negative_score > 0.5:
                sentiment_predicted_label.append(1)
            else:
                sentiment_predicted_label.append(0)
            sentiment_test_labels[i % 4].append(new_test_label)
            sentiment_negative_scores[i % 4].append(new_negative_score)
    sentiment_AUC = []
    for i in range(4):
        sentiment_AUC.append(metrics.roc_auc_score(sentiment_test_labels[i], sentiment_negative_scores[i]))
    sentiment_macro_AUC = np.mean(sentiment_AUC)

    # Accuracy
    sentiment_accuracy = metrics.accuracy_score(sentiment_test_label, sentiment_predicted_label)

    return sentiment_macro_AUC, sentiment_accuracy


def compute_semeval_PRF(test_labels, predicted_labels):
    num_total_intersection = 0
    num_total_test_aspects = 0
    num_total_predicted_aspects = 0
    num_examples = len(test_labels) // 5
    for i in range(num_examples):
        test_aspects = set()
        predicted_aspects = set()
        for j in range(5):
            if test_labels[i * 5 + j] != 4:
                test_aspects.add(j)
            if predicted_labels[i * 5 + j] != 4:
                predicted_aspects.add(j)
        if len(test_aspects) == 0:
            continue
        intersection = test_aspects.intersection(predicted_aspects)
        num_total_test_aspects += len(test_aspects)
        num_total_predicted_aspects += len(predicted_aspects)
        num_total_intersection += len(intersection)
    mi_P = num_total_intersection / num_total_predicted_aspects
    mi_R = num_total_intersection / num_total_test_aspects
    mi_F = (2 * mi_P * mi_R) / (mi_P + mi_R)
    return mi_P, mi_R, mi_F


def compute_semeval_accuracy(test_labels, predicted_labels, scores, num_classes=4):
    count_considered_examples = 0
    count_correct_examples = 0
    if num_classes == 4:
        for i in range(len(test_labels)):
            if test_labels[i] == 4:
                continue
            new_predicted_label = predicted_labels[i]
            if new_predicted_label == 4:
                new_scores = scores[i].copy()
                new_scores[4] = 0
                new_predicted_label = np.argmax(new_scores)
            if test_labels[i] == new_predicted_label:
                count_correct_examples += 1
            count_considered_examples += 1
        semeval_accuracy = count_correct_examples / count_considered_examples

    elif num_classes == 3:
        for i in range(len(test_labels)):
            if test_labels[i] >= 3:
                continue
            new_predicted_label = predicted_labels[i]
            if new_predicted_label >= 3:
                new_scores = scores[i].copy()
                new_scores[3] = 0
                new_scores[4] = 0
                new_predicted_label = np.argmax(new_scores)
            if test_labels[i] == new_predicted_label:
                count_correct_examples += 1
            count_considered_examples += 1
        semeval_accuracy = count_correct_examples / count_considered_examples
    elif num_classes == 2:
        for i in range(len(test_labels)):
            if test_labels[i] == 1 or test_labels[i] >= 3:
                continue
            new_predicted_label = predicted_labels[i]
            if new_predicted_label == 1 or new_predicted_label >= 3:
                new_scores = scores[i].copy()
                new_scores[1] = 0
                new_scores[3] = 0
                new_scores[4] = 0
                new_predicted_label = np.argmax(new_scores)
            if test_labels[i] == new_predicted_label:
                count_correct_examples += 1
            count_considered_examples += 1
        semeval_accuracy = count_correct_examples / count_considered_examples
    else:
        raise ValueError("num_classes must be equal to 2, 3, or 4")
    return semeval_accuracy


def main(task="NLI_M", dataset_type="sentihood", data_dir="", predictions_path=""):
    predicted_labels, scores = get_predictions(predictions_path, task, dataset_type)
    test_original_sentences, test_auxiliary_sentences, test_labels = get_dataset(data_dir, dataset_type)

    if dataset_type == "sentihood":
        sentihood_aspect_strict_acc = compute_sentihood_aspect_strict_accuracy(test_labels, predicted_labels)
        print(f"{task} Sentihood aspect strict accuracy: {sentihood_aspect_strict_acc}")
        sentihood_aspect_macro_F1 = compute_sentihood_aspect_macro_F1(test_labels, predicted_labels)
        print(f"{task} Sentihood aspect macro F1: {sentihood_aspect_macro_F1}")
        sentihood_aspect_macro_AUC = compute_sentihood_aspect_macro_AUC(test_labels, scores)
        print(f"{task} Sentihood aspect macro AUC: {sentihood_aspect_macro_AUC}")

        sentihood_sentiment_macro_AUC, sentihood_sentiment_accuracy = compute_sentihood_sentiment_classification_metrics(
            test_labels, scores)
        print(f"{task} Sentihood sentiment accuracy: {sentihood_sentiment_accuracy}")
        print(f"{task} Sentihood sentiment macro AUC: {sentihood_sentiment_macro_AUC}")

    elif dataset_type == "semeval2014":
        semeval_aspect_precision, semeval_aspect_recall, semeval_aspect_micro_F1 = compute_semeval_PRF(test_labels,
                                                                                                       predicted_labels)
        print(f"{task} Semeval aspect precision: {semeval_aspect_precision}")
        print(f"{task} Semeval aspect recall: {semeval_aspect_recall}")
        print(f"{task} Semeval aspect micro F1: {semeval_aspect_micro_F1}")

        semeval_4_classes_accuracy = compute_semeval_accuracy(test_labels, predicted_labels, scores, 4)
        semeval_3_classes_accuracy = compute_semeval_accuracy(test_labels, predicted_labels, scores, 3)
        semeval_2_classes_accuracy = compute_semeval_accuracy(test_labels, predicted_labels, scores, 2)
        print(f"{task} Semeval 4-classes accuracy: {semeval_4_classes_accuracy}")
        print(f"{task} Semeval 3-classes accuracy: {semeval_3_classes_accuracy}")
        print(f"{task} Semeval 2-classes accuracy: {semeval_2_classes_accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        default="NLI_M",
                        type=str,
                        required=True,
                        choices=["QA_M", "NLI_M", "QA_B", "NLI_B"],
                        help="Name of the task to evaluate.")
    parser.add_argument("--dataset_type",
                        default="sentihood",
                        type=str,
                        required=True,
                        choices=["sentihood", "semeval2014"],
                        help="Dataset for the task")
    parser.add_argument("--data_dir",
                        default="data/",
                        type=str,
                        required=True,
                        help="Test dataset_type (csv format)")
    parser.add_argument("--predictions_path",
                        default="results/sentihood_NLI_M.csv",
                        type=str,
                        required=True,
                        help="Predictions and scores predictions_dir")
    args = parser.parse_args()

    task = args.task
    dataset_type = args.dataset_type
    data_dir = args.data_dir
    predictions_path = args.predictions_path

    main(task, dataset_type, data_dir, predictions_path)

