# Faithful Rule Extraction

Data, source codes and experimental results for paper "*Faithful Rule Extraction for Differentiable Rule Learning Models*". 

> There is increasing interest in methods for extracting interpretable rules from ML models trained to solve a wide range of tasks over knowledge graphs (KGs), such as KG completion, node classification, question answering and recommendation. Many such approaches, however, lack formal guarantees establishing the precise relationship between the model and the extracted rules, and this lack of assurance becomes especially problematic when the extracted rules are applied in safety-critical contexts or to ensure compliance with legal requirements. Recent research has examined whether the rules derived from the influential Neural-LP model exhibit soundness (or completeness), which means that the results obtained by applying the model to any dataset always contain (or are contained in) the results obtained by applying the rules to the same dataset. In this paper, we extend this analysis to the context of DRUM, an approach that has demonstrated superior practical performance. After observing that the rules currently extracted from a DRUM model can be unsound and/or incomplete, we propose a novel algorithm where the output rules, expressed in an extension of Datalog, ensure both soundness and completeness. This algorithm, however, can be inefficient in practice and hence we propose additional constraints to DRUM models facilitating rule extraction, albeit at the expense of reduced expressive power.

## Datasets

- All datasets used in our paper are provided in [datasets](https://github.com/xiaxia-wang/FaithfulRE/tree/main/datasets). 
- For inductive knowledge graph completion, we reused the datasets and splits from the [Grail benchmark](https://github.com/kkteru/grail/tree/master/data), i.e., FB15K237 (V1--V4), NELL995 (V1--V4), and WN18RR (V1--V4). We further split each original training set into 3:1 training facts and training labels on a random basis. Analogously, each testing set was also split into 3:1 testing facts and (positive) testing labels. We randomly sampled negative examples as the same number of testing labels. The process of random sampling is provided in [utils.py](https://github.com/xiaxia-wang/FaithfulRE/blob/main/code/src/utils.py).
- For the rule extraction experiments, we reused the *family* dataset and its splits from [the original DRUM project](https://github.com/alisadeghian/DRUM).

## Source Codes and Dependencies

### Dependencies

- Python 3.8
- PyTorch 2.0.1

### Codes

- All source codes for model training, evaluation and rule extraction are provided in [code/src](https://github.com/xiaxia-wang/FaithfulRE/tree/main/code/src). 
- Here we briefly explain the structure of source codes.
  - [data.py](https://github.com/xiaxia-wang/FaithfulRE/blob/main/code/src/data.py) provides a data loader, which parses the original dataset and feeds it to the model.
  - [experiment.py](https://github.com/xiaxia-wang/FaithfulRE/blob/main/code/src/experiment.py) contains all the experimental process, including model training, performance evaluation and rule extraction.
  - [model.py](https://github.com/xiaxia-wang/FaithfulRE/blob/main/code/src/model.py) provides the neural network models of MMDRUM, SMDRUM and DRUM.
  - [main.py](https://github.com/xiaxia-wang/FaithfulRE/blob/main/code/src/main.py) is the main entrance for training and evaluating the corresponding model.


## Experiments

### Set up

- Paths of datasets and evaluation results could be customized in [code/config.ini](https://github.com/xiaxia-wang/FaithfulRE/blob/main/code/config.ini). 
- Default hyper-parameters could be modified in [main.py](https://github.com/xiaxia-wang/FaithfulRE/blob/main/code/src/main.py) file.

### Training

- In the main function of [main.py](https://github.com/xiaxia-wang/FaithfulRE/blob/main/code/src/main.py) file, set the `model` parameter to the required model (i.e., one of `"mmdrum"`, `"smdrum"` and `"drum"`). For example, `model="drum"`. Set the `process` parameter to `"train"`. 
- Then execute the main function. The model checkpoint will be automatically saved after each training epoch.
- The MMDRUM, SMDRUM and DRUM model checkpoints (.pt files) could be provided upon request.

### Evaluation

- Analogously set the `model` parameter as the training process, and set the `process` parameter to `"evaluate"`. 
- Then execute the main function. The evaluate process includes getting the predictions of all the test cases, and computing all evaluation metrics on the test sets.

### Rule Extraction

- Analogously set the `model` parameter as above, and set the `process` parameter to `"analyze"`. 
- Execute the main function to extract (Datalog) rules from the MMDRUM or SMDRUM model. For DRUM model, one can input a specific triple and get the multipath rule as its explanation.



> If you have any question about the codes or experimental results, please email [xiaxia.wang@cs.ox.ac.uk](mailto:xiaxia.wang@cs.ox.ac.uk).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/xiaxia-wang/FaithfulRE/blob/main/LICENSE) for details.

## Citation

If you use these data or codes, please kindly cite it as follows:

```
@inproceedings{faithfulre,
  author       = {Xiaxia Wang and David J. Tena Cucala and Bernardo Cuenca Grau and Ian Horrocks},
  title        = {Faithful Rule Extraction for Differentiable Rule Learning Models},
  booktitle    = {The 12th International Conference on Learning Representations},
  year         = {2024}
}
```
