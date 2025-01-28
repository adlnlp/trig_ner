# <div align="center">TriG-NER: Triplet-Grid Framework for</br>Discontinuous Entity Recognition</div>

### <div align="center">Rina Carines Cabral, Soyeon Caren Han, Josiah Poon</div>
#### <div align="center">Accepted at The Web Conference 2025 (WWW'25)<br>[preprint](https://arxiv.org/abs/2411.01839)</div>

**Abstract:** Discontinuous Named Entity Recognition (DNER) presents a challenging problem where entities may be scattered across multiple non-adjacent tokens, making traditional sequence labelling approaches inadequate. Existing methods predominantly rely on custom tagging schemes to handle these discontinuous entities, resulting in models tightly coupled to specific tagging strategies and lacking generalisability across diverse datasets. To address these challenges, we propose TriG-NER, a novel Triplet-Grid Framework that introduces a generalisable approach to learning robust token-level representations for discontinuous entity extraction. Our framework applies triplet loss at the token level, where similarity is defined by word pairs existing within the same entity, effectively pulling together similar and pushing apart dissimilar ones. This approach enhances entity boundary detection and reduces the dependency on specific tagging schemes by focusing on word-pair relationships within a flexible grid structure. We evaluate TriG-NER on three benchmark DNER datasets and demonstrate significant improvements over existing grid-based architectures. These results underscore our framework’s effectiveness in capturing complex entity structures and its adaptability to various tagging schemes, setting a new benchmark for discontinuous entity extraction.

<p align="center">
  <img alt="Overall Architecture" src="https://github.com/adlnlp/trig_ner/blob/main/figures/architecture5.jpg" height="250" />
  <img alt="Triplet Candidates and Grid Class example" src="https://github.com/adlnlp/trig_ner/blob/main/figures/candidate_sample_1_v2.jpg" height="250" /> 
</p>

⚠️Full code to be uploaded soon.
