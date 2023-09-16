# Faithful Rule Extraction

Data, source codes and experimental results for paper "*Towards Faithful Rule Extraction for Differentiable Rule Learning Models*". 

> The explainability of knowledge graph reasoning has attracted much research attention. Recent efforts on differentiable rule learning aim to extract logical rules from a trained neural network model to explain the model's prediction. However, it remains unknown for these methods whether applying the extracted rules to the same dataset can derive faithful (i.e., both sound and complete) results w.r.t. the model. By analyzing a representative model, DRUM, we notice that the Datalog rules extracted from a DRUM model by existing algorithm can be unsound or incomplete. To address the problem, we work in two directions. First, to characterize the exact form of rules that can be learned by the DRUM model, we introduce multipath rules based on an extension of standard Datalog rule language. Based on that, we propose an algorithm for faithful multipath rule extraction. Furthermore, to achieve efficient rule extraction, we propose two models MMDRUM and SMDRUM as variants of DRUM with different expressivity. We also present faithful Datalog rule extraction algorithms for them, respectively.

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
  - Additionally, we provide a transformer-based rule learning model that apply transformer layers to learn attention tensors for each predicate. The related experiment process and the main entrance are provided in [experiment_gpu.py](https://github.com/xiaxia-wang/FaithfulRE/blob/main/code/src/experiment_gpu.py) and [main_transformer.py](https://github.com/xiaxia-wang/FaithfulRE/blob/main/code/src/main_transformer.py), respectively. For more details, please refer to the appendix of our paper.

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
- Execute the main function to extract (Datalog) rules from the MMDRUM or SMDRUM model. For DRUM model, one can input a specific triple and get the Datalog-neq rule as its explanation.



> If you have any question about the codes or experimental results, please email to xxx.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/xiaxia-wang/FaithfulRE/blob/main/LICENSE) for details.

## Citation

If you use these data or codes, please kindly cite it as follows:

[will be avaliable after double-blind review period]
