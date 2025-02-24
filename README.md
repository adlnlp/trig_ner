# <div align="center">TriG-NER: Triplet-Grid Framework for</br>Discontinuous Entity Recognition</div>

### <div align="center">Rina Carines Cabral, Soyeon Caren Han, Josiah Poon</div>
#### <div align="center">Accepted at The Web Conference 2025 (WWW'25)<br>[preprint](https://arxiv.org/abs/2411.01839)</div>

**Abstract:** Discontinuous Named Entity Recognition (DNER) presents a challenging problem where entities may be scattered across multiple non-adjacent tokens, making traditional sequence labelling approaches inadequate. Existing methods predominantly rely on custom tagging schemes to handle these discontinuous entities, resulting in models tightly coupled to specific tagging strategies and lacking generalisability across diverse datasets. To address these challenges, we propose TriG-NER, a novel Triplet-Grid Framework that introduces a generalisable approach to learning robust token-level representations for discontinuous entity extraction. Our framework applies triplet loss at the token level, where similarity is defined by word pairs existing within the same entity, effectively pulling together similar and pushing apart dissimilar ones. This approach enhances entity boundary detection and reduces the dependency on specific tagging schemes by focusing on word-pair relationships within a flexible grid structure. We evaluate TriG-NER on three benchmark DNER datasets and demonstrate significant improvements over existing grid-based architectures. These results underscore our framework’s effectiveness in capturing complex entity structures and its adaptability to various tagging schemes, setting a new benchmark for discontinuous entity extraction.

<p align="center">
  <img alt="Overall Architecture" src="https://github.com/adlnlp/trig_ner/blob/main/figures/architecture5.jpg" height="250" />
  <img alt="Triplet Candidates and Grid Class example" src="https://github.com/adlnlp/trig_ner/blob/main/figures/candidate_sample_1_v2.jpg" height="250" /> 
</p>

----
## Datasets
Please download the original datasets from their respective repositories.
- [CADEC](https://doi.org/10.4225/08/570FB102BDAD2) (Karimi et al.)
- [ShARe13](https://doi.org/10.13026/rxa7-q798) (Pradhan et al.)
- [ShARe14](https://doi.org/10.13026/0zgk-9j94) (Mowrey et al.)

Run the following code to preprocess and convert the data into a token-level json format. We follow the preprocessing steps of Dai et al. [] which involves the separation of punctuations from words and sentence-level instances. We also follow Li et al. [] json formatting.
```
CODE HERE
```

For custom datasets, please follow the json format below and save each train/dev/test split in separate files. ``token_char_map`` is an optional entity that maps each token to it's character span indexes. This is used for converting the final predictions back to the inline format used by Dai et al. [] but is not necessary for training.
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

## Training
```
```

### Parameters
The following are parameters that may be used for tuning the models.
```
```

### Outputs
Running the training code will produce three files in the _output_ folder.
1. _*\_testpreds.json_ - all predictions from the test set
2. _*\_results.json_ - json file containing the configuration used for training and the metric scores (F1, P, R) for all entities and for each entities.
3. _*.pt_ - saved state of the best model
