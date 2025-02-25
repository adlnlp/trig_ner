import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader

import config
import data_loader as data_loader
import utils
from model import W2NER, TriGNER, TriGNER_Conv
from main import Trainer

import random

import time
from tqdm import tqdm
import triplet_mining

from convert_results import convert_to_inline_char_spans, remove_special_tokens

from prettytable import PrettyTable

# class Temp_Config:
#     def __init__(self, d=None):
#         if d is not None:
#             for key, value in d.items():
#                 setattr(self, key, value)
#
#     def __repr__(self):
#         return "{}".format(self.__dict__.items())

class Predictor(object):
    def __init__(self, model):
        self.model = model

    def filter_by_label(self, type_id, instance_list):
        # filtered = [ent for inst in instance_list for ent in inst if ent[-1] == type_id]
        filtered = []
        for inst in instance_list:
            filtered_inst = []
            for ent in inst:
                if ent[-1] == type_id:
                    filtered_inst.append(ent)
            filtered.append(filtered_inst)
        return filtered

    def predict(self, data_loader, data):
        self.model.eval()

        pred_result = []
        label_result = []

        result = []
        pred_indexes = []
        gold_indexes = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i:i + config.batch_size]
                entity_text = data_batch[-1]
                # data_batch = [data.cuda() for data in data_batch[:-1]]
                data_batch = [data.cuda() for data in data_batch[
                                                      :-4]]  # Remove entity text, triplets, positives, and negatives that doesn't need to be in GPU
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs, _ = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text,
                                                                    length.cpu().numpy())

                # entity_index = [[[int(x) for x in ent.split("-")[:-2]] for ent in inst] for inst in entity_text]
                # pred_index = [[x[0] for x in inst] for inst in decode_entities]
                entity_index = [[[int(x) for x in ent.split("-") if x.isdigit()] for ent in inst] for inst in entity_text]
                pred_index = [[x[0] + [x[1]] for x in inst] for inst in decode_entities]

                pred_indexes.extend(pred_index)
                gold_indexes.extend(entity_index)
                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                   "index": ent[0],
                                                   # Rina: Added for discontinuous entity results
                                                   "type": config.vocab.id_to_label(ent[1])})
                    result.append(instance)

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)
        all_results = utils.extract_all_disc_metrics(gold_indexes, pred_indexes)
        assert all_results["all"]["f1"] == e_f1
        assert all_results["all"]["p"] == e_p
        assert all_results["all"]["r"] == e_r

        #Extract results per entity type
        types = config.vocab.get_labels()
        type_results = {}
        for t in types:
            t_id = config.vocab.label_to_id(t)
            filtered_gold = self.filter_by_label(t_id, gold_indexes)
            filtered_pred = self.filter_by_label(t_id, pred_indexes)

            type_results[t] = utils.extract_all_disc_metrics(filtered_gold, filtered_pred)

        overall_results = {"all_entities": all_results,
                           "per_entity": type_results}

        return overall_results, result

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path), strict=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--session_name", type=str, default="Inference_(date)")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--save_name", type=str)

    parser.add_argument("--save_as_inline", default=False, action="store_true")

    parser.add_argument("--predict_train", default=False, action="store_true")
    parser.add_argument("--predict_dev", default=False, action="store_true")
    parser.add_argument("--predict_test", default=False, action="store_true")

    args = parser.parse_args()

    if ~np.any([args.predict_train, args.predict_dev, args.predict_test]):
        print("Please indicate which subset to produce predictions.")
        exit()

    # Load config settings
    config = config.Config(args)

    if args.save_name is None:
        args.save_name = config.session_name

    #Overwrite checkpoint if needed (renamed; transferred folders)
    config.checkpoint = args.checkpoint if args.checkpoint is not None else "./output/" + config.save_path

    logger = utils.get_logger(config)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    datasets, ori_data = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=4,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    logger.info("Building Model")
    if config.model_name == "W2NER":
        model = W2NER(config)
    elif config.model_name == "TriGNER":
        model = TriGNER(config)
    elif config.model_name == "TriGNER_Conv":
        model = TriGNER_Conv(config)
    else:
        raise Exception("Invalid model name")

    model = model.cuda()

    logger.info("Loading Model")
    predictor = Predictor(model)
    predictor.load(config.checkpoint)

    loaders = np.array([train_loader, dev_loader, test_loader])
    loader_types = np.array(["train", "dev", "test"])

    loaders = loaders[[args.predict_train, args.predict_dev, args.predict_test]]
    loader_types = loader_types[[args.predict_train, args.predict_dev, args.predict_test]]
    original_data = np.array(ori_data)[[args.predict_train, args.predict_dev, args.predict_test]]

    for loader, loader_type, o_data in zip(loaders, loader_types, original_data):
        results, predictions = predictor.predict(loader, o_data)

        logger.info(loader_type)
        logger.info(results)

        #Save predictions
        #JSON
        with open("output/%s_%s_preds.json" % (args.save_name, loader_type), "w", encoding="utf-8") as f:
            adjusted_preds = remove_special_tokens(predictions)
            json.dump(adjusted_preds, f, ensure_ascii=False)

        #INLINE CHAR SPAN
        if args.save_as_inline:
            if "token_char_map" not in o_data:
                logger.info("Token-character map required for generating inline format.")
            else:
                convert_to_inline_char_spans(predictions, o_data, "output/%s_%s_preds.tsv" % (args.save_name, loader_type))

        #Save metrics results
        with open("output/%s_%s_results.json" % (args.save_name, loader_type), "w") as file:
            json.dump(results, file, indent=4)

# HPOv2_GridW2NERModel_v3_PubMedBert_scheme_grid_centroid_win_15_maxneg_0_60ep_finetunedenc_perentitynew_unique_notanh_es_lr_5e-4.pt