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

import random

import time
from tqdm import tqdm
import triplet_mining

from convert_results import remove_special_tokens


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        print(config)
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

        # NOTE: RINA addition
        self.tr_criterion = nn.TripletMarginLoss(margin=config.tr_margin)

    @staticmethod
    def filter_by_label(type_id, instance_list):
        # filtered = [ent for inst in instance_list for ent in inst if ent[-1] == type_id]
        filtered = []
        for inst in instance_list:
            filtered_inst = []
            for ent in inst:
                if ent[-1] == type_id:
                    filtered_inst.append(ent)
            filtered.append(filtered_inst)
        return filtered

    @staticmethod
    def extract_encoded_triplets(hidden_states, triplet_batch, positives_batch, negatives_batch, scheme):
        encoded_triplets = []

        # Offline mode
        if config.mining_mode == "offline":
            assert len(triplet_batch) == hidden_states.shape[0]  # Check if batch size matches

            if hidden_states.dim() == 3:  # word representations batch_size x seq_len x hidden_dim
                for j, triplet_instance in enumerate(triplet_batch):

                    # Get encoder representation for each triplet
                    for tripletIds in triplet_instance:  # tripletIds --> [(aId, pId, nId)]
                        encoded_triplets.append(hidden_states[j, tripletIds])

            elif hidden_states.dim() == 4:  # grid with shape batch_size x seq_len x seq_len x hidden_dim
                for j, triplet_instance in enumerate(triplet_batch):

                    aIdx_list, aIdy_list = [], []
                    pIdx_list, pIdy_list = [], []
                    nIdx_list, nIdy_list = [], []
                    for (aIds, pIds, nIds) in triplet_instance:
                        aIdx_list.append(aIds[0])
                        aIdy_list.append(aIds[1])
                        pIdx_list.append(pIds[0])
                        pIdy_list.append(pIds[1])
                        nIdx_list.append(nIds[0])
                        nIdy_list.append(nIds[1])

                    a_enc = hidden_states[j][aIdx_list, aIdy_list]
                    p_enc = hidden_states[j][pIdx_list, pIdy_list]
                    n_enc = hidden_states[j][nIdx_list, nIdy_list]

                    # if len(a_enc) > 0 and len(a_enc) == len(p_enc) and len(a_enc) == len(n_enc):
                    encoded_triplets.extend(torch.stack((a_enc, p_enc, n_enc), dim=1))

        # Online mode
        elif config.mining_mode == "online":
            if scheme in ["grid_centroid", "grid_negcentroid", "grid_hardneg", "grid_semineg", "grid_negcentroid2"]:
                encoded_triplets = triplet_mining.generate_grid_online_triplets(hidden_states, positives_batch,
                                                                                negatives_batch, scheme,
                                                                                dist_metric=config.dist_metric,
                                                                                margin=config.tr_margin)
            else:
                encoded_triplets = triplet_mining.generate_online_triplets(hidden_states, positives_batch,
                                                                           negatives_batch,
                                                                           mining_scheme=scheme,
                                                                           dist_metric=config.dist_metric)

        else:
            raise Exception("Invalid mining mode.")

        # Remove invalid triplets
        if len(encoded_triplets) != 0:
            encoded_triplets = torch.stack(encoded_triplets)
            p_dist = torch.norm(encoded_triplets[:, 0] - encoded_triplets[:, 1], p=2, dim=1)
            n_dist = torch.norm(encoded_triplets[:, 0] - encoded_triplets[:, 2], p=2, dim=1)
            tr = config.tr_margin + p_dist - n_dist
            valid_idx = torch.argwhere(tr > 0).flatten()

            encoded_triplets = encoded_triplets[valid_idx]

        return encoded_triplets

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        ce_loss_list = []
        tr_loss_list = []
        total_triplets = 0

        for i, data_batch in enumerate(tqdm(data_loader)):

            triplet_batch = data_batch[-4].copy()
            positives_batch = data_batch[-3].copy()
            negatives_batch = data_batch[-2].copy()

            # data_batch = [data.cuda() for data in data_batch[:-1]]
            data_batch = [data.cuda() for data in data_batch[:-4]]  # Remove entity text, triplets, positives, and negatives that doesn't need to be in GPU

            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs, hidden_states = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)

            grid_mask2d = grid_mask2d.clone()
            ce_loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])

            #Rina Added
            tr_loss = 0

            # If triplet is enabled, calculate triplet loss based on encoder outputs
            # If no triplets in batch (can occur due to shuffling), calculate loss without triplet loss (ie. triplet loss = 0)
            max_triplets = max([len(x) for x in triplet_batch]) if triplet_batch != [] else 0
            max_anchors = max([0] + [len(x) for positives in positives_batch for x in positives]) if positives_batch != [] else 0
            if (config.use_triplet & (max_triplets > 0 or max_anchors > 0)):
                schemes = [config.mining_scheme]
                scheme_multiplier = 1
                if config.mining_scheme2 is not None:
                    schemes.append(config.mining_scheme2)
                    scheme_multiplier = 0.5

                for scheme in schemes:
                    _tr_loss = 0

                    if config.triplet_source in ["encoder"]:  # , "grid"
                        encoded_triplets = self.extract_encoded_triplets(hidden_states, triplet_batch, positives_batch,
                                                                         negatives_batch, scheme)

                        if len(encoded_triplets) != 0:
                            assert encoded_triplets.dim() == 3 and encoded_triplets.shape[1] == 3

                            total_triplets += encoded_triplets.shape[0]
                            _tr_loss = self.tr_criterion(encoded_triplets[:, 0],  # anchors
                                                         encoded_triplets[:, 1],  # positives
                                                         encoded_triplets[:, 2])  # negatives

                            if _tr_loss == 0:
                                raise Exception("Invalid triplets detected.")


                    elif config.triplet_source == "output":
                        triplets = self.extract_encoded_triplets(outputs, triplet_batch, positives_batch,
                                                                     negatives_batch, scheme)

                        if len(triplets) != 0:
                            total_triplets += triplets.shape[0]
                            _tr_loss = self.tr_criterion(triplets[:, 0],  # anchors
                                                            triplets[:, 1],  # positives
                                                            triplets[:, 2])  # negatives

                    else:
                        raise Exception("Unknown triplet source.")

                    # if config.mining_scheme2 is not None:
                    #     logger.info("TR Loss ({}):{:3.4f}".format(scheme, 0 if _tr_loss == 0 else _tr_loss.cpu().item()))
                    tr_loss += (scheme_multiplier * _tr_loss)

                loss = ce_loss + tr_loss
                ce_loss_list.append(ce_loss.cpu().item())
                tr_loss_list.append(0 if tr_loss == 0 else tr_loss.cpu().item())
            else:
                loss = ce_loss
                ce_loss_list.append(ce_loss.cpu().item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

            self.scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")

        # table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        # table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
        #               ["{:3.4f}".format(x) for x in [f1, p, r]])
        # logger.info("\n{}".format(table))

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall", "Triplets", "CE Loss", "TR Loss"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]] +
                      [total_triplets] +
                      ["{:.4f}".format(np.mean(ce_loss_list)), "{:.4f}".format(np.mean(tr_loss_list))])
        logger.info("\n{}".format(table))

        return f1

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                # data_batch = [data.cuda() for data in data_batch[:-1]]
                data_batch = [data.cuda() for data in data_batch[:-4]]  # Remove entity text, triplets, positives, and negatives that doesn't need to be in GPU
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs, _ = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, _ = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "EVAL" if not is_test else "TEST"
        logger.info('{} Label F1 {}'.format(title, f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))
        return e_f1, e_p, e_r

    def predict(self, epoch, data_loader, data):
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
                sentence_batch = data[i:i+config.batch_size]
                entity_text = data_batch[-1]
                # data_batch = [data.cuda() for data in data_batch[:-1]]
                data_batch = [data.cuda() for data in data_batch[:-4]]  # Remove entity text, triplets, positives, and negatives that doesn't need to be in GPU
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs, _ = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                # entity_index = [[[int(x) for x in ent.split("-")[:-2]] for ent in inst] for inst in entity_text]
                # pred_index = [[x[0] for x in inst] for inst in decode_entities]
                entity_index = [[[int(x) for x in ent.split("-") if x.isdigit()] for ent in inst] for inst in
                                entity_text]
                pred_index = [[x[0] + [x[1]] for x in inst] for inst in decode_entities]

                pred_indexes.extend(pred_index)
                gold_indexes.extend(entity_index)
                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                   "index": ent[0],         #Rina: Added for discontinuous entity results
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

        title = "TEST"
        logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])
        table.add_row(["Disc_Sent"] + ["{:3.4f}".format(x) for x in [all_results["sent_disc"]["f1"], all_results["sent_disc"]["p"], all_results["sent_disc"]["r"]]])
        table.add_row(["Disc_Ent"] + ["{:3.4f}".format(x) for x in [all_results["ent_disc"]["f1"], all_results["ent_disc"]["p"], all_results["ent_disc"]["r"]]])

        logger.info("\n{}".format(table))

        logger.info(all_results)
        overall_results = {"all_entities": all_results}

        # Extract results per entity type
        types = config.vocab.get_labels()
        type_results = {}
        for t in types:
            logger.info(t)
            t_id = config.vocab.label_to_id(t)
            filtered_gold = self.filter_by_label(t_id, gold_indexes)
            filtered_pred = self.filter_by_label(t_id, pred_indexes)

            type_results[t] = utils.extract_all_disc_metrics(filtered_gold, filtered_pred)

            logger.info(type_results[t])

        overall_results["per_entity"] = type_results

        with open("output/" + config.predict_path, "w", encoding="utf-8") as f:
            adjusted_preds = remove_special_tokens(result)
            json.dump(adjusted_preds, f, ensure_ascii=False)

        # return e_f1
        return overall_results

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/cadec.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    # Triplet parameters
    parser.add_argument("--use_triplet", default=None, action="store_true")
    parser.add_argument('--window_size', type=int, default=None)
    # parser.add_argument("--num_triplets", type=int, default=None)  # Only for randomized selection
    parser.add_argument("--mining_scheme", type=str, default=None,
                        choices=["grid_centroid", "grid_negcentroid", "grid_hardneg",
                                 "grid_semineg"])
    parser.add_argument("--mining_scheme2", type=str, default=None,
                        choices=["grid_centroid", "grid_negcentroid", "grid_hardneg",
                                 "grid_semineg"])
    parser.add_argument("--dist_metric", type=str, default=None, choices=["cosine", "euclidean"])
    parser.add_argument("--per_entity", default=None, action="store_true")
    parser.add_argument("--triplet_source", type=str, default=None, choices=["encoder", "output"])
    parser.add_argument("--unique_grid_pairs", default=None, action="store_true")
    parser.add_argument("--tr_margin", type=float, default=1)

    parser.add_argument("--model_name", type=str, default=None,
                        choices=["W2NER", "TriGNER", "TriGNER_Conv"])
    parser.add_argument("--session_name", type=str, default="Run_(date)")

    parser.add_argument("--tune_parameters", default=None, action="store_true")
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--use_finetuned", default=None, action="store_true")
    parser.add_argument("--save_preds", default=None, action="store_true")
    parser.add_argument("--early_stop", type=int, default=None)

    args = parser.parse_args()

    config = config.Config(args)

    # Triplet defaults
    if config.use_triplet:
        config.window_size = 3 if config.window_size is None else config.window_size
        config.num_triplets = 0 if config.num_triplets is None else config.num_triplets
        config.mining_scheme = "grid_centroid" if config.mining_scheme is None else config.mining_scheme
        # config.dist_metric = "cosine" if config.mining_scheme in ["distance_hardest", "distance_hardneg", "centroid"] and config.dist_metric is None else config.dist_metric
        config.triplet_source = "output" if config.triplet_source is None else config.triplet_source
    config.mining_mode = "online" if config.mining_scheme in ["grid_centroid", "grid_negcentroid", "grid_hardneg",
                                                              "grid_semineg"] else "offline"

    # Set grid defaults
    if config.mining_scheme in ["grid", "grid_centroid", "grid_negcentroid", "grid_hardneg", "grid_semineg"]:
        config.unique_grid_pairs = True if config.unique_grid_pairs is None else config.unique_grid_pairs
        assert config.model_name in ["W2NER", "TriGNER", "TriGNER_Conv"] #or (config.model_name in ["W2NER"] and config.triplet_source == "output")
        assert (config.mining_scheme2 is None) or (config.mining_scheme2.split("_")[0] == "grid")

    if config.bert_name == "facebook/bart-large":
        config.bart_hid_size = 1024

    #NOTE: Add supported bert based PLMs here
    if config.use_finetuned and config.bert_name in ["google-bert/bert-base-uncased",                         #BERT
                                                     "Lianglab/PharmBERT-uncased",                            #PharmBERT
                                                     "emilyalsentzer/Bio_ClinicalBERT",                       #BioClinicalBERT
                                                     "dmis-lab/biobert-base-cased-v1.2",                      #BioBERT
                                                     "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"]: #PubMedBERT
        #NOTE: Ensure finetuned version of model is saved on models_del folder
        config.bert_name = "models_del/%s-finetuned-%s" % (config.bert_name.split("/")[-1], config.dataset)

    ####
    # print(config)
    # exit()
    logger = utils.get_logger(config)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    logger.info("Loading Data")
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

    updates_total = len(datasets[0]) // config.batch_size * config.epochs

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

    trainer = Trainer(model)

    best_f1 = 0
    best_test_f1 = 0

    best_epoch = 0
    best_dev_results = {}
    train_start = time.time()

    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        f1, p, r = trainer.eval(i, dev_loader)
        if (f1 > best_f1) or (best_f1 == 0):
            best_f1 = f1
            best_epoch = i
            best_dev_results = {"f1": f1,
                                "p": p,
                                "r": r}
            trainer.save("output/" + config.save_path)

        if (config.early_stop is not None) and (i - best_epoch > int(config.early_stop)):
            break

    train_time = time.time() - train_start

    logger.info("Training time (s): %.4f" % train_time)
    logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    trainer.load("output/" + config.save_path)

    all_results = trainer.predict("Final", test_loader, ori_data[-1])

    # Save results with model
    with open("output/%s.json" % config.session_name, "w") as file:
        save_results = {"config": {k: v for k, v in config.__dict__.items() if k not in ["logger", "vocab"]},
                        "train_time": train_time,
                        "best_epoch": best_epoch,
                        "best_dev_results": best_dev_results
                        }

        for k, v in all_results.items():
            if k == "all":
                k = "test_results"
            save_results[k] = v

        json.dump(save_results, file, indent=4)