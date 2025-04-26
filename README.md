# <div align="center">TriG-NER: Triplet-Grid Framework for</br>Discontinuous Entity Recognition</div>

### <div align="center">[Rina Carines Cabral](https://scholar.google.com/citations?user=Cfdv_5IAAAAJ), [Soyeon Caren Han](https://scholar.google.com/citations?user=VH2jVOgAAAAJ), [Josiah Poon](https://scholar.google.com/citations?user=Q7U0O0gAAAAJ)</div>
#### <div align="center">Accepted at The Web Conference 2025 (WWW'25)<br>[paper](https://doi.org/10.1145/3696410.3714639)</div>

**Abstract:** Discontinuous Named Entity Recognition (DNER) presents a challenging problem where entities may be scattered across multiple non-adjacent tokens, making traditional sequence labelling approaches inadequate. Existing methods predominantly rely on custom tagging schemes to handle these discontinuous entities, resulting in models tightly coupled to specific tagging strategies and lacking generalisability across diverse datasets. To address these challenges, we propose TriG-NER, a novel Triplet-Grid Framework that introduces a generalisable approach to learning robust token-level representations for discontinuous entity extraction. Our framework applies triplet loss at the token level, where similarity is defined by word pairs existing within the same entity, effectively pulling together similar and pushing apart dissimilar ones. This approach enhances entity boundary detection and reduces the dependency on specific tagging schemes by focusing on word-pair relationships within a flexible grid structure. We evaluate TriG-NER on three benchmark DNER datasets and demonstrate significant improvements over existing grid-based architectures. These results underscore our framework’s effectiveness in capturing complex entity structures and its adaptability to various tagging schemes, setting a new benchmark for discontinuous entity extraction.

<p align="center">
  <img alt="Overall Architecture" src="https://github.com/adlnlp/trig_ner/blob/main/figures/architecture5.jpg" height="250" />
  <img alt="Triplet Candidates and Grid Class example" src="https://github.com/adlnlp/trig_ner/blob/main/figures/candidate_sample_1_v2.jpg" height="250" /> 
</p>

----
## Datasets
Please download the original datasets from their respective repositories and follow [Dai et al.'s preprocessing](https://github.com/dainlp/acl2020-transition-discontinuous-ner/tree/masterv) to create an inline version of the datasets and put them on the _inline\_data_ folder. 
- [CADEC](https://doi.org/10.4225/08/570FB102BDAD2) (Karimi et al.)
- [ShARe13](https://doi.org/10.13026/rxa7-q798) (Pradhan et al.)
- [ShARe14](https://doi.org/10.13026/0zgk-9j94) (Mowrey et al.)

Run the following code to convert inline format to json format.
```
python preprocess.py --dataset cadec             #dataset name
                     --input_folder inline_data  #folder containing inline data
                     --output_Folder data        #folder to write data on
                     --generate_token_map        #optional; generates character mapping of each token
```

For custom datasets, please follow the json format below and save each train/dev/test split in separate files. ``token_char_map`` is an optional entity that maps each token to it's character span indexes. This is used for converting the final predictions back to the inline format but is not necessary for training.
```
[
  {
    "sentence": ["This", "is", "a", "sample", "sentence", "."],
    "ner": [
      {"index": [3, 4], "type": "sample_type"},
      ...
    ],
    "token_char_map": [[0,4], [5,7], [8,9], [10,16], [18,26], [26,27]]
  },
  ...
]
```

## Config
The dataset and other parameters will be taken from a config file in the _config_ folder. A sample is provided as a starting point for other datasets. We provide the best found configuration for the three datasets.

## Finetuning Pre-trained Language Models
We finetune popular pre-trained language models using a masked language modeling for each dataset. The following code finetunes BioBert using the CADEC dataset. A bash script _finetuning.sh_ is provided to finetune all supported PLMs for all the datasets. Finetuned PLMs will be stored in the _models_ folder.
```
python finetune_lm.py --dataset cadec --epochs 20 --batch_size 16 --pt_name dmis/biobert-base-cased-v1.2
```

Alternatively, [download](https://unisyd-my.sharepoint.com/:u:/g/personal/rina_cabral_sydney_edu_au/EburYqBUqUFAsSA14bwGKHgBnc7GF3AoOa0pSZixeSWGlg?e=MtK2xP) (4.51GB) our finetuned models and add put them int the _models_ folder.

## Training
The following code will run the base setup for the CADEC dataset.
```
python main.py --config ./config/cadec.json
```

### Parameters
The following are parameters that may be used for tuning the models for each dataset.
```
--session_name        --> all saved files will start with this; defaults to Run_[date]
--use_triplet         --> indicates the use of the TriG-NER triplet framework; if set to false, the code will run without using the triplet loss
--window_size         --> size of the window centering on the anchor point
--mining_scheme       --> mining scheme to use (see below for options
--unique_grid_pairs   --> indicates the use of the top-half of the grid; if false, uses the whole grid
--use_finetuned       --> will use finetuned PLMs instead of base models
--bert_name           --> PLM used to initialize encoder weights (see below for options)
--learning_rate       --> learning rate
--batch_size          --> batch size
--epochs              --> number of epochs for training
--early_stop          --> early stop
```

#### Mining Schemes
![Mining Schemes](https://github.com/adlnlp/trig_ner/blob/main/figures/triplet_selection.jpg)

Use the code for ``--mining_scheme`` parameter.

| Mining Scheme | Code |
|---|---|
| Centroid | ``grid_centroid`` | 
| Negative Centroid | ``grid_negcentroid`` | 
| Hard Negative | ``grid_hardneg`` |
| Semihard Negative | ``grid_semihard``|

#### Pretrained Language Models
Tested language models used to initialize encoder weights are below. Use the model name for ``--bert_name``. Any BERT-based model in huggingface may be used.

| Name | Model Name |
|---|---|
| [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) | ``dmis-lab/biobert-base-cased-v1.2`` | 
| [BioClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) | ``emilyalsentzer/Bio_ClinicalBERT`` |
| [PharmBERT](https://huggingface.co/Lianglab/PharmBERT-uncased) | ``Lianglab/PharmBERT-uncased`` |
| [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) | ``microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`` |

### Outputs
Running the training code will produce three files in the _output_ folder.
1. _*\_test\_preds.json_ - all predictions from the test set
2. _*\_results.json_ - json file containing the configuration used for training and the metric scores (F1, P, R) for all entities and for each entities.
3. _*.pt_ - saved state of the best model

## Inference
If generating precitions using an already trained model, run the following code for inference.
```
python inference.py --config ./config/cadec.json           #Config to load model settings
                    --checkpoint ./output/saved_model.pt   #Load state dict from saved model from training
                    --session_name "Session_Name"          #Set to name this run
                    --predict_train                        #Save predictions and results for training set
                    --predict_dev                          #Save predictions and results for dev set
                    --predict_test                         #Save predictions and results for test set
```
To reproduce the results on the paper, [download](https://unisyd-my.sharepoint.com/:u:/g/personal/rina_cabral_sydney_edu_au/ERKfWGry67FGqj7zycaV7GgBZxc2fzpaHYOO_IHH6GOxNg?e=Bm7bd6) (5.10GB) our best setup models and place them on the _output_ folder. Run the bash file ``reproduce_paper_results.sh`` to produce predictions and metric scores.

## Overall Results

<img src="https://github.com/adlnlp/trig_ner/blob/main/figures/overall_results.jpg" width="50%">

## Citation (preprint)
WWW'25 citation will be shared after the conference.
```
@misc{2025-cabral-trig,
      title={TriG-NER: Triplet-Grid Framework for Discontinuous Named Entity Recognition}, 
      author={Rina Carines Cabral and Soyeon Caren Han and Areej Alhassan and Riza Batista-Navarro and Goran Nenadic and Josiah Poon},
      year={2025},
      eprint={2411.01839},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.01839}, 
}
```
