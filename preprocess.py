import os
import logging
import argparse
import numpy as np
import re
import json

def convert_from_inline(dataset):
    file_paths = {"train": "%s/train.txt" % (args.input_folder),
                  "dev": "%s/dev.txt" % (args.input_folder),
                  "test": "%s/test.txt" % (args.input_folder)}

    for split, file_path in file_paths.items():
        logging.info("-" * 100)
        logging.info("Loading Data - %s" % split)

        #Load data
        with open(file_path, "r") as file:
            contents = file.readlines()

        sentences = [x.strip() for x in contents[::3]]
        inst_entities = [x.strip() for x in contents[1::3]]
        assert len(sentences) == len(inst_entities)

        #Process sentence
        #NOTE: For cadec, share13, and share14, Dai et al's preprocessed data is used and tokenized by whitespace
        #      For other datasets, separate punctuations from works using preprocess_sentence method
        tokenized_sentences = [x.split(" ") for x in sentences]

        #Process entities
        proc_entities = []
        for inst_ent in inst_entities:
            entities = inst_ent.strip().split("|")

            proc_inst_entities = []
            for ent in entities:
                if ent == "":
                    continue
                ent_type = ent.split(" ")[-1]
                ent_spans = ent.replace(ent_type, "").strip().split(",")
                ent_starts = ent_spans[::2]
                ent_ends = ent_spans[1::2]
                assert len(ent_starts) == len(ent_ends)

                ent_index = []
                for start, end in zip(ent_starts, ent_ends):
                    ent_index.extend([idx for idx in range(int(start), int(end) + 1)])

                proc_inst_entities.append({"index": ent_index,
                                           "type": ent_type})

            proc_entities.append(proc_inst_entities)
        assert len(sentences) == len(proc_entities)

        #Token char map - for recreating original sentence

        token_mapping = []
        if args.generate_token_map:
            for sentence, tokens in zip(sentences, tokenized_sentences):
                token_char_map = []
                end = 0
                for i, token in enumerate(tokens):
                    start = 0 if i == 0 else sentence.find(token, end+1)
                    end = start + len(token) - 1

                    token_char_map.append((start, end))
                token_mapping.append(token_char_map)
            assert len(token_mapping) == len(sentences)

        #Combine data
        final_data = []
        for i in range(len(sentences)):
            final = {"sentence": tokenized_sentences[i],
                     "ner": proc_entities[i]
                     }

            if args.generate_token_map:
                final["token_char_map"] = token_mapping[i]

            final_data.append(final)

        with open("%s/%s/%s.json" % (args.output_folder, dataset, split), "w") as file:
            logging.info("Saving %s json data..." % split)
            json.dump(final_data, file)

    logging.info("End preprocessing %s" % dataset)

def preprocess_sentence(sentence):
    #Tokenize by whitespace and separate punctuations
    return [x for x in re.split("(\W)", sentence.strip()) if x not in ["", " "]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str, default="data")
    parser.add_argument('--generate_token_map', default=False, action="store_true")

    args = parser.parse_args()

    os.makedirs("%s/%s" % (args.output_folder, args.dataset), exist_ok = True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')

    if args.dataset in ["cadec", "share13", "share14"]:
        convert_from_inline(args.dataset)
    else:
        raise Exception("Unhandled dataset. Please implement custom preprocessing code.")



