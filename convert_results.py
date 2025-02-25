import json
import numpy as np
import pandas as pd

def convert_to_inline_char_spans(predictions, original, save_path):

    pred_offset = 2  # Account for [POS] and [NEG]

    observation_ids = []
    texts = []
    polarities = []
    spans = []

    for pred, gold in zip(predictions, original):
        assert pred["sentence"] == gold["sentence"]

        # remove special tokens from prediction results
        pred["sentence"] = pred["sentence"][pred_offset:]
        gold["sentence"] = gold["sentence"][pred_offset:]

        token_char_map = np.array(gold["token_char_map"])

        # Format original sentence
        sentence_formatted = ""
        prev_end = 0
        for token, char_idx in zip(gold["sentence"], token_char_map):
            if sentence_formatted == "":
                sentence_formatted = token
                prev_end = char_idx[1]
            else:
                if char_idx[0] - prev_end > 1:
                    num_spaces = char_idx[0] - prev_end - 1
                    sentence_formatted += " " * num_spaces + token
                elif char_idx[0] - prev_end == 1:
                    sentence_formatted += token
                else:
                    print(char_idx[0] - prev_end)
                    raise Exception("Invalid character indexes.")

            prev_end = char_idx[1]

        if len(pred["entity"]) == 0:  # If no entity, append default
            observation_ids.append(gold["id"])
            texts.append(sentence_formatted)
            polarities.append("NA")
            spans.append("NA")

        for ent in pred["entity"]:
            # Adjust indexes to remove special tokens
            token_indexes = [x - pred_offset for x in ent["index"]]
            char_indexes = [token_char_map[x] for x in token_indexes]

            # #Format entity type
            ent_type = "NA" if ent["type"] == "normf" else "X"

            # Format span
            prev_end = 0
            span_formatted = ""
            for start_idx, end_idx in char_indexes:
                if span_formatted == "":
                    span_formatted += str(start_idx)

                if prev_end == 0:
                    prev_end = end_idx
                    continue

                if start_idx - prev_end > 2:
                    span_formatted += "-" + str(prev_end + 1) + "," + str(start_idx)

                prev_end = end_idx

            span_formatted += "-" + str(end_idx + 1)

            # Record output
            observation_ids.append(gold["id"])
            texts.append(sentence_formatted)
            polarities.append(ent_type)
            spans.append(span_formatted)

    formatted_data = pd.DataFrame({"ObservationID": observation_ids,
                                   "Text": texts,
                                   "HPO Term": [""] * len(texts),
                                   "Polarity": polarities,
                                   "Spans": spans})

    formatted_data.to_csv(save_path, index=False, sep="\t")

def remove_special_tokens(preds):
    for pred in preds:
        assert "[POS]" == pred["sentence"][0]
        pred["sentence"] = pred["sentence"][2:]

        for ent in pred["entity"]:
            ent["index"] = [x - 2 for x in ent["index"]]

    return preds


