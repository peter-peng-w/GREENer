# Graph-based Extractive Explainer for Recommendations

This repo contains the PyTorch implementation of the model Graph-based Extractive Explainer for Recommendations (GREENer) proposed in our WWW 2022 paper ["Graph-based Extractive Explainer for Recommendations"](https://dl.acm.org/doi/10.1145/3485447.3512168). Please refer to the paper for the details of the algorithm.

## Data

We provide 2 preprocessed datasets that we are using in this work.

* Ratebeer [[link]](https://drive.google.com/file/d/1-aAFvUsGkAJJ1VtLqKKCjGhJ1gTEsea1/view?usp=share_link)
* TripAdvisor [[link]](https://drive.google.com/file/d/1SwDufC_jSYdYIP81u7YtcTWvmF2XV0dR/view?usp=share_link)

Please set up a new issue if you find these links are broken.

Note that these datasets are slightly different from the datasets that we are using in our paper. We sampled a subset of the whole dataset and only preserve positive reviews in these 2 datasets. The whole dataset will be released soon.

## How to Run

### Training

Once you download the data, you can decompress them under the folder ``graph_data``. Since there could be multiple version of data when you run different experiments for each dataset (i.e. with different sample sizes and different preprocess strategies), you can store the data in second-level (or higher-level) directories. For example, for the Ratebeer dataset, after downloading it, you can save it under the directory of ``./graph_data/ratebeer/medium_500_positive/``

The data contains both the dictionaries between user, item, attributes, and sentences, and also the pre-constructed graph for each user-item pair. You can find the pre-constructed graphs under the directory ``geo_graph_batch`` or ``geo_graph_batch_item_attr``, where the in the second one, we only preserve item-side attributes and remove all those attributes that only come from the user-side on the graph.

After the data is correctly being place, you can run the script ``train.sh`` to start training. Make sure that the parameter such as *data_dir*, *graph_dir*, *data_set*, *data_name*, and *model_name* are correctly set (the last 3 will only affects the name/path of the log file and checkpoint). There are many hyper-parameters of the model that you can tune, please refer to our paper for more details on how to set the hyper-parameters. Once you start training, you will be able to see and monitor the saved model parameters under the ``checkpoint`` directory and also the log files under the ``log`` directory.

### Evaluation (without ILP)

After the training complete, please first run the script ``eval_model.sh`` to perform the evaluation and prediction. Same as the ``train.sh``, you should make sure that all the parameters of the data are correctly set and the **model_file** should be set to the path of the model parameter. You should also correctly set the *eval_output_path* which point to the directory where you will store the result. There are several heuristic filter methods to select the top-ranked sentences, including greedy, trigram filtering, trigram-feature-unigram filtering and bleu score filtering. You can choose the method you want by modifying line 32-37 of ``eval.py``. In order to save the predicted results, make sure that line 45 (i.e. ``save_hyps_refs``) is set to be True and line 39 (i.e. ``save_predict``) is set to be False. After the evaluation is complete, you should be able to see the saved hypothesis and references under your *eval_output_path*.

### Evaluation (with ILP)

In order to run the ILP post-processing, please first run the script ``eval_model.sh`` again, with line 39 (i.e. ``save_predict``) being set to True. This will store the predicted logits for all the candidate sentences for each user-item pair. You should be able to see the saved predicted results under the directory ``data_postprocess``. Once this is done, please run the script ``eval_ILP.sh``. You should also make sure that the parameters of dataset, saved model weights and output directory are correctly being set.

We need [Gurobi](https://www.gurobi.com/) as the solver to solve the ILP. In order to correctly run the code, you should install [gurobipy](https://pypi.org/project/gurobipy/) and also the Gurobi optimizer. You can request an academic license to use Gurobi for free. Please refer to there [website](https://www.gurobi.com/features/academic-named-user-license/) for more detailed instructions on how to correctly set up Gurobi on your machine.

The evalaution of ILP is solved via multi-processing. Once it is finished, you should be able to see the hypothesis and references under your *eval_output_path*.

### Evaluation Metrics

We are using Bleu and Rouge to evaluate the performance of explanation sysnthesis. For Bleu score, you can use the perl script under ``data_postprocess``. An example of how to run this perl script is shown in ``compute_bleu.sh`` under the same directory. For Rouge score, you can the [rouge-score](https://pypi.org/project/rouge-score/) package.