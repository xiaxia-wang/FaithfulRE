# Faithful Rule Extraction

Data, source codes and experimental results for paper "*Faithful Logical Rule Extraction for Differentiable Rule Learning*". 

> The explainability of knowledge graph reasoning has attracted much research attention. Recent efforts on differentiable rule learning aim to extract logical rules, especially chain-like Datalog rules, from a trained machine learning model as explanation. However, it remains unknown for these methods that, applying the extracted rules to the same dataset can produce faithful (i.e., both sound and complete) predictions w.r.t. the model. By analyzing a representative method, DRUM, we surprisingly notice that the rules extracted by DRUM can be unsound or incomplete. To address this, we provide our modification in two directions. First, we propose two new models MMDRUM and SMDRUM with different expressive power by modifying DRUM. We prove the Datalog rule extraction for both of them is always faithful. Furthermore, we introduce Datalog with inequality (Datalog-neq) as an extension of standard Datalog rule language, and propose an algorithm for faithful Datalog-neq rule extraction from original DRUM models. We empirically evaluate the effectiveness of the proposed models and algorithms.

## Datasets

- All datasets used in our paper are provided in [datasets](https://github.com/xiaxia-wang/FaithfulRE/tree/main/datasets). 
- For inductive knowledge graph completion, we reused the datasets and splits from the [Grail benchmark](https://github.com/kkteru/grail/tree/master/data), i.e., FB15K237 (V1--V4), NELL995 (V1--V4), and WN18RR (V1--V4). We further split each original training set into 3:1 training facts and training labels on a random basis. Analogously, each testing set was also split into 3:1 testing facts and testing labels. 
- For the rule extraction experiments, we reused the *family* dataset and its splits from [the original DRUM project](https://github.com/alisadeghian/DRUM).

## Source Codes and Dependencies

### Dependencies

- Python 3.8
- PyTorch 2.0.1

### Codes

- All source codes for model training, evaluation and rule extraction are provided in [code/src](https://github.com/xiaxia-wang/FaithfulRE/tree/main/code/src). 
- Here we briefly explain the structure of source codes.
  - [data.py]()
  - [model.py]()
  - 

## Experiments

Data and experimental results are provided in xxx.

> If you have any question about the codes or experimental results, please email to xxx.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/xiaxia-wang/FaithfulRE/blob/main/LICENSE) for details.

## Citation

If you use these data or codes, please kindly cite it as follows:

[will be avaliable after double-blind review period]

