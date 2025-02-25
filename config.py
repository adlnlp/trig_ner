import json
import time


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

            if "config" in config:
                config = config["config"]

        self.dataset = config["dataset"]

        self.dist_emb_size = config["dist_emb_size"]
        self.type_emb_size = config["type_emb_size"]
        self.lstm_hid_size = config["lstm_hid_size"]
        self.conv_hid_size = config["conv_hid_size"]
        self.bert_hid_size = config["bert_hid_size"]
        self.biaffine_size = config["biaffine_size"]
        self.ffnn_hid_size = config["ffnn_hid_size"]

        self.dilation = config["dilation"]

        self.emb_dropout = config["emb_dropout"]
        self.conv_dropout = config["conv_dropout"]
        self.out_dropout = config["out_dropout"]

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.bert_name = config["bert_name"]
        self.bert_learning_rate = config["bert_learning_rate"]
        self.warm_factor = config["warm_factor"]

        self.use_bert_last_4_layers = config["use_bert_last_4_layers"]

        self.seed = config["seed"]

        self.use_triplet = config["use_triplet"]
        self.model_name = config["model_name"]
        self.use_triplet = config["use_triplet"]
        self.window_size = config["window_size"]
        self.num_triplets = config["num_triplets"]
        self.mining_scheme = config["mining_scheme"]
        self.mining_scheme2 = config["mining_scheme2"]
        self.dist_metric = config["dist_metric"]
        self.per_entity = config["per_entity"]
        self.triplet_source = config["triplet_source"]
        self.unique_grid_pairs = config["unique_grid_pairs"]
        self.tr_margin = config["tr_margin"]
        self.tune_parameters = config["tune_parameters"]
        self.num_trials = config["num_trials"]
        self.use_finetuned = config["use_finetuned"]
        self.save_preds = config["save_preds"]
        self.early_stop = config["early_stop"]

        args.session_name = args.session_name.replace("(date)", time.strftime("%Y-%m-%d_%H-%M-%S"))
        self.session_name = args.session_name
        self.save_path = self.session_name + ".pt"
        self.predict_path = self.session_name + "_test_preds.json"

        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())
