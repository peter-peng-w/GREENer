import numpy as np
from numpy.core.numeric import indices
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# %matplotlib inline
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch import nonzero
import torch.nn.functional as F
import torch.nn as nn
import datetime
import statistics
from metric import compute_bleu, get_bleu, get_sentence_bleu, get_example_recall_precision
from metric import get_feature_recall_precision, get_recall_precision_f1, get_recall_precision_f1_random, get_recall_precision_f1_popular, get_recall_precision_f1_sent
from metric import get_recall_precision_f1_gt, get_recall_precision_f1_gt_random
from rouge import Rouge
from nltk.translate import bleu_score
import dgl
import pickle
import random


dataset_name = 'medium_500_pure'
label_format = 'soft_label'

# how to select the top-predicted sentences
use_origin = False
use_trigram = False
use_bleu_filter = False

# select features randomly
random_features = False

# select features based on the popularity
popular_features = False
popular_features_vs_origin = False
popular_features_vs_trigram = False

# select features based on the feature prediction scores
predict_features = True
predict_features_vs_origin = False
predict_features_vs_trigram = False

save_sentence_selected = False
save_feature_selected = True
save_feature_logits = True

use_ground_truth = True

avg_proxy_feature_num = 19
avg_gt_feature_num = 15
total_feature_num = 575
MAX_batch_output = 100


class EVAL_FEATURE(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_batch_size = args.batch_size
        self.m_mean_loss = 0

        self.m_sid2swords = vocab_obj.m_sid2swords
        self.m_feature2fid = vocab_obj.m_feature2fid
        self.m_item2iid = vocab_obj.m_item2iid
        self.m_user2uid = vocab_obj.m_user2uid
        self.m_sent2sid = vocab_obj.m_sent2sid
        self.m_train_sent_num = vocab_obj.m_train_sent_num

        # get item id to item mapping
        self.m_iid2item = {self.m_item2iid[k]: k for k in self.m_item2iid}
        # get user id to user mapping
        self.m_uid2user = {self.m_user2uid[k]: k for k in self.m_user2uid}
        # get fid to feature(id) mapping
        self.m_fid2feature = {self.m_feature2fid[k]: k for k in self.m_feature2fid}
        # get sid to sent_id mapping
        self.m_sid2sentid = {self.m_sent2sid[k]: k for k in self.m_sent2sid}

        self.m_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.m_device = device
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file
        self.m_eval_output_path = args.eval_output_path

        print("Evaluation results are saved under dir: {}".format(self.m_eval_output_path))
        print("Dataset: {0} \t Label: {1}".format(dataset_name, label_format))
        if random_features:
            print("Use the random features.")
        elif popular_features:
            print("Use the popular features.")
        elif popular_features_vs_origin:
            print("Use the popular features vs. origin predict sentences.")
        elif popular_features_vs_trigram:
            print("Use the popular features vs. trigram predict sentences.")
        elif predict_features_vs_origin:
            print("Use the predict features vs. origin predict sentences.")
        elif predict_features_vs_trigram:
            print("Use the predict features vs. trigram predict sentences.")
        elif use_trigram:
            print("Use the features from sentence after trigram blocking.")
        elif use_origin:
            print("Use the features from sentence based on original scores.")
        else:
            print("Use the predicted features based on the feature prediction scores.")

        # need to load some mappings
        id2feature_file = '../../Dataset/ratebeer/{}/train/feature/id2feature.json'.format(dataset_name)
        feature2id_file = '../../Dataset/ratebeer/{}/train/feature/feature2id.json'.format(dataset_name)
        trainset_id2sent_file = '../../Dataset/ratebeer/{}/train/sentence/id2sentence.json'.format(dataset_name)
        testset_id2sent_file = '../../Dataset/ratebeer/{}/test/sentence/id2sentence.json'.format(dataset_name)
        # trainset_useritem_pair_file = '../../Dataset/ratebeer/{}/train/useritem_pairs.json'.format(dataset_name)
        testset_useritem_cdd_withproxy_file = '../../Dataset/ratebeer/{}/test/useritem2sentids_withproxy.json'.format(dataset_name)
        trainset_user2featuretf_file = '../../Dataset/ratebeer/{}/train/user/user2featuretf.json'.format(dataset_name)
        trainset_item2featuretf_file = '../../Dataset/ratebeer/{}/train/item/item2featuretf.json'.format(dataset_name)
        trainset_sentid2featuretfidf_file = '../../Dataset/ratebeer/{}/train/sentence/sentence2feature.json'.format(dataset_name)
        testset_sentid2featuretf_file = '../../Dataset/ratebeer/{}/test/sentence/sentence2featuretf.json'.format(dataset_name)

        with open(id2feature_file, 'r') as f:
            print("Load file: {}".format(id2feature_file))
            self.d_id2feature = json.load(f)
        with open(feature2id_file, 'r') as f:
            print("Load file: {}".format(feature2id_file))
            self.d_feature2id = json.load(f)
        # Load train/test sentence_id to sentence content
        with open(trainset_id2sent_file, 'r') as f:
            print("Load file: {}".format(trainset_id2sent_file))
            self.d_trainset_id2sent = json.load(f)
        with open(testset_id2sent_file, 'r') as f:
            print("Load file: {}".format(testset_id2sent_file))
            self.d_testset_id2sent = json.load(f)
        # # Load trainset user-item pair
        # with open(trainset_useritem_pair_file, 'r') as f:
        #     print("Load file: {}".format(trainset_useritem_pair_file))
        #     self.d_trainset_useritempair = json.load(f)
        # Load testset user-item cdd sents with proxy
        with open(testset_useritem_cdd_withproxy_file, 'r') as f:
            print("Load file: {}".format(testset_useritem_cdd_withproxy_file))
            self.d_testset_useritem_cdd_withproxy = json.load(f)
        # Load trainset user/item to feature tf-value dict
        with open(trainset_user2featuretf_file, 'r') as f:
            print("Load file: {}".format(trainset_user2featuretf_file))
            self.d_trainset_user2featuretf = json.load(f)
        with open(trainset_item2featuretf_file, 'r') as f:
            print("Load file: {}".format(trainset_item2featuretf_file))
            self.d_trainset_item2featuretf = json.load(f)
        # Load testset sentence id to feature tf-value dict
        with open(testset_sentid2featuretf_file, 'r') as f:
            print("Load file: {}".format(testset_sentid2featuretf_file))
            self.d_testset_sentid2featuretf = json.load(f)
        # Load trainset sentence id to feature tf-idf value dict
        with open(trainset_sentid2featuretfidf_file, 'r') as f:
            print("Load file: {}".format(trainset_sentid2featuretfidf_file))
            self.d_trainset_sentid2featuretfidf = json.load(f)

        print("Total number of feature: {}".format(len(self.d_id2feature)))

        # Get the sid2featuretf dict (on Valid/Test Set)
        self.d_testset_sid2featuretf = self.get_sid2featuretf_eval(
            self.d_testset_sentid2featuretf, self.m_sent2sid, self.m_train_sent_num)
        # Get the sid2feature dict (on Train Set)
        self.d_trainset_sid2feature = self.get_sid2feature_train(
            self.d_trainset_sentid2featuretfidf, self.m_sent2sid)

    def f_init_eval(self, network, model_file=None, reload_model=False):
        if reload_model:
            print("reload model")
            if not model_file:
                model_file = "model_best.pt"
            model_name = os.path.join(self.m_model_path, model_file)
            print("model name", model_name)
            check_point = torch.load(model_name)
            network.load_state_dict(check_point['model'])

        self.m_network = network

    def f_eval(self, train_data, eval_data):
        print("Start Eval ...")
        # Feature P/R/F/AUC
        f_recall_list = []
        f_precision_list = []
        f_F1_list = []
        f_auc_list = []

        # average features in proxy/ground-truth
        proxy_feature_num_cnt = []
        gt_feature_num_cnt = []

        topk = 3                # this is used for predict topk sentences
        topk_candidate = 20     # this is used for sanity check for the top/bottom topk sentneces
        # already got feature2fid mapping, need the reverse
        self.m_fid2feature = {value: key for key, value in self.m_feature2fid.items()}

        cnt_useritem_batch = 0
        train_test_overlap_cnt = 0
        train_test_differ_cnt = 0
        save_logging_cnt = 0
        self.m_network.eval()
        with torch.no_grad():
            print("Number of evaluation data: {}".format(len(eval_data)))

            for graph_batch in eval_data:
                if cnt_useritem_batch % 100 == 0:
                    print("... eval ... ", cnt_useritem_batch)

                graph_batch = graph_batch.to(self.m_device)

                # logits: batch_size*max_sen_num
                s_logits, sids, s_masks, target_sids, f_logits, fids, f_masks, target_f_labels = self.m_network.eval_forward(graph_batch)
                batch_size = s_logits.size(0)

                """ Get the topk predicted sentences
                """
                if use_trigram or popular_features_vs_trigram or predict_features_vs_trigram:
                    topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids = self.trigram_blocking_sent_prediction(
                        s_logits, sids, s_masks, batch_size, topk=topk, topk_cdd=topk_candidate
                    )
                else:
                    topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids = self.origin_blocking_sent_prediction(
                        s_logits, sids, s_masks, topk=topk, topk_cdd=topk_candidate
                    )

                # Get the user/item id of this current graph batch.
                # NOTE: They are not the 'real' user/item id in the dataset, still need to be mapped back.
                userid = graph_batch.u_rawid
                itemid = graph_batch.i_rawidå

                # Decide the batch_save_flag. To get shorted results, we only print the first several batches' results
                cnt_useritem_batch += 1
                if cnt_useritem_batch <= MAX_batch_output:
                    batch_save_flag = True
                else:
                    batch_save_flag = False
                # Whether to break or continue(i.e. pass) when the batch_save_flag is false
                if batch_save_flag:
                    save_logging_cnt += 1
                else:
                    # do nothing
                    pass
                    # break

                # Loop through the batch
                for j in range(batch_size):
                    refs_j_list = []
                    hyps_j_list = []
                    hyps_featureid_j_list = []

                    for sid_k in target_sids[j]:
                        refs_j_list.append(self.m_sid2swords[sid_k.item()])

                    for sid_k in pred_sids[j]:
                        hyps_j_list.append(self.m_sid2swords[sid_k.item()])
                        hyps_featureid_j_list.extend(self.d_trainset_sid2feature[sid_k.item()])

                    hyps_j = " ".join(hyps_j_list)
                    refs_j = " ".join(refs_j_list)
                    hyps_featurewords_j_list = [self.d_id2feature[this_fea_id] for this_fea_id in hyps_featureid_j_list]
                    hyps_num_unique_features = len(set(hyps_featureid_j_list))
                    hyps_unique_featurewords_j = [set(hyps_featurewords_j_list)]

                    userid_j = userid[j].item()
                    itemid_j = itemid[j].item()
                    # get the true user/item id
                    true_userid_j = self.m_uid2user[userid_j]
                    true_itemid_j = self.m_iid2item[itemid_j]

                    proxy_j_list = []
                    for sid_k in self.d_testset_useritem_cdd_withproxy[true_userid_j][true_itemid_j][-1]:
                        proxy_j_list.append(self.d_trainset_id2sent[sid_k])
                    proxy_j = " ".join(proxy_j_list)

                    # get feature prediction performance
                    # f_logits, fids, f_masks, target_f_labels
                    f_logits_j = f_logits[j]
                    fid_j = fids[j].cpu()
                    mask_f_j = f_masks[j].cpu()
                    target_f_labels_j = target_f_labels[j].cpu()

                    # get the user-item featuretf
                    user_to_featuretf = self.d_trainset_user2featuretf[true_userid_j]
                    item_to_featuretf = self.d_trainset_item2featuretf[true_itemid_j]
                    useritem_to_featuretf = self.combine_featuretf(user_to_featuretf, item_to_featuretf)

                    """ Get the popular features (and the corresponding tf-value) for this user-item pair
                        The number of popular features has severla options:
                        1. average number of ground-truth sentences' unique features across the test set
                        2. average number of proxy sentences' unique features across the test set
                        3. the top-predicted sentences' unique features of this ui-pair
                        4. the top-predicted (after 3-gram block) sentences' unique features of this ui-pair
                    """
                    if use_ground_truth:
                        if popular_features:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=avg_gt_feature_num)
                        elif popular_features_vs_origin:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=hyps_num_unique_features)
                        elif popular_features_vs_trigram:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=hyps_num_unique_features)
                        else:
                            pass
                    else:
                        if popular_features:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=avg_proxy_feature_num)
                        elif popular_features_vs_origin:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=hyps_num_unique_features)
                        elif popular_features_vs_trigram:
                            useritem_popular_features, ui_popular_features_freq = self.get_popular_features(
                                useritem_to_featuretf, topk=hyps_num_unique_features)
                        else:
                            pass

                    f_num_j = target_f_labels_j.size(0)
                    mask_f_logits_j = f_logits_j[:f_num_j].cpu()
                    mask_fid_j = fid_j[:f_num_j]
                    mask_featureid_j = [self.m_fid2feature[this_f_id.item()] for this_f_id in mask_fid_j]

                    if save_feature_logits and batch_save_flag:
                        self.feature_logits_save_file(true_userid_j, true_itemid_j, mask_f_logits_j)

                    # target is generated from the proxy. These features are unique features (duplications removed)
                    # get the index of the feature labels (feature labels are 1)
                    target_fid_index_j = (target_f_labels_j.squeeze() == 1).nonzero(as_tuple=True)[0]
                    # get the fid of the feature labels
                    target_fid_j = torch.gather(fid_j, dim=0, index=target_fid_index_j)
                    # get the featureid of the feature labels
                    target_featureid_j = [self.m_fid2feature[this_f_id.item()] for this_f_id in target_fid_j]
                    # get the feature word of the feature labels
                    target_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in target_featureid_j]

                    # gt is generated from the ground-truth review. These features are unique features (duplications removed)
                    gt_featureid_j, _ = self.get_gt_review_featuretf(self.d_testset_sid2featuretf, target_sids[j])
                    # get the feature word of the gt feature labels
                    gt_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in gt_featureid_j]

                    # number of feaures in proxy/gt
                    proxy_feature_num_cnt.append(len(target_featureword_j))
                    gt_feature_num_cnt.append(len(gt_featureword_j))

                    # for fea_id in gt_featureid_j:
                    #     if fea_id not in useritem_to_featuretf.keys():
                    #         print("User: {0}\tItem: {1}\tFeature id: {2}".format(
                    #             true_userid_j, true_itemid_j, fea_id
                    #         ))

                    if use_ground_truth:
                        if random_features:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, top_pred_random_featureid_j = get_recall_precision_f1_gt_random(
                                mask_f_logits_j, gt_featureid_j, mask_featureid_j,
                                avg_gt_feature_num, total_feature_num)
                        elif popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j = get_recall_precision_f1_popular(
                                useritem_popular_features, gt_featureid_j, useritem_to_featuretf, total_feature_num)
                        elif use_trigram:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j = get_recall_precision_f1_sent(
                                hyps_featureid_j_list, gt_featureid_j, total_feature_num)
                        elif use_origin:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j = get_recall_precision_f1_sent(
                                hyps_featureid_j_list, gt_featureid_j, total_feature_num)
                        elif predict_features_vs_origin or predict_features_vs_trigram:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, top_pred_featureid_j, topk_preds_featureid_with_logits = get_recall_precision_f1_gt(
                                mask_f_logits_j, gt_featureid_j, mask_featureid_j,
                                hyps_num_unique_features, total_feature_num)
                        else:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, top_pred_featureid_j, topk_preds_featureid_with_logits = get_recall_precision_f1_gt(
                                mask_f_logits_j, gt_featureid_j, mask_featureid_j,
                                avg_gt_feature_num, total_feature_num)
                    else:
                        if random_features:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, topk_pred_f_j = get_recall_precision_f1_random(
                                mask_f_logits_j, target_f_labels_j, avg_proxy_feature_num)
                        elif popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j = get_recall_precision_f1_popular(
                                useritem_popular_features, target_featureid_j, useritem_to_featuretf, total_feature_num)
                        elif use_trigram:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j = get_recall_precision_f1_sent(
                                hyps_featureid_j_list, target_featureid_j, total_feature_num)
                        elif use_origin:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j = get_recall_precision_f1_sent(
                                hyps_featureid_j_list, target_featureid_j, total_feature_num)
                        elif predict_features_vs_origin or predict_features_vs_trigram:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, topk_pred_f_j = get_recall_precision_f1(
                                mask_f_logits_j, target_f_labels_j, hyps_num_unique_features)
                        else:
                            f_prec_j, f_recall_j, f_f1_j, f_auc_j, topk_pred_f_j = get_recall_precision_f1(
                                mask_f_logits_j, target_f_labels_j, avg_proxy_feature_num)

                    f_precision_list.append(f_prec_j)
                    f_recall_list.append(f_recall_j)
                    f_F1_list.append(f_f1_j)
                    f_auc_list.append(f_auc_j)

                    if not use_ground_truth:
                        # Save log results for feature prediction on proxy
                        if use_trigram or use_origin:
                            target_featureid_j_set = set(target_featureid_j)
                            select_featureid_j_set = set(hyps_featureid_j_list)
                            overlap_featureid_j_set = target_featureid_j_set.intersection(select_featureid_j_set)
                            overlap_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in overlap_featureid_j_set]
                            if save_feature_selected and batch_save_flag:
                                predict_features_file = os.path.join(
                                    self.m_eval_output_path,
                                    'eval_features_{0}_{1}.txt'.format(dataset_name, label_format))
                                with open(predict_features_file, 'a') as f:
                                    f.write("User: {0}\tItem: {1}\n".format(true_userid_j, true_itemid_j))
                                    f.write("refs: {}\n".format(refs_j))
                                    f.write("hyps: {}\n".format(hyps_j))
                                    f.write("proxy: {}\n".format(proxy_j))
                                    f.write("target features: {}\n".format(target_featureword_j))
                                    f.write("predict sentences' features: {}\n".format(", ".join(hyps_unique_featurewords_j)))
                                    f.write("overlappings: {}\n".format(overlap_featureword_j))
                                    f.write("Number of features in target: {}\n".format(len(target_featureid_j_set)))
                                    f.write("Number of features in predict sentences: {}\n".format(len(select_featureid_j_set)))
                                    f.write("Number of feature overlap: {}\n".format(len(overlap_featureid_j_set)))
                                    f.write("Precision: {0}\tRecall: {1}\tF1: {2}\tAUC: {3}\n".format(f_prec_j, f_recall_j, f_f1_j, f_auc_j))
                                    f.write("==------==------==------==------==------==------==\n")
                        elif popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                            popular_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in useritem_popular_features]
                            # get the overlapping features
                            target_featureid_j_set = set(target_featureid_j)
                            popular_featureid_j_set = set(useritem_popular_features)
                            overlap_featureid_j_set = target_featureid_j_set.intersection(popular_featureid_j_set)
                            overlap_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in overlap_featureid_j_set]
                            if save_feature_selected and batch_save_flag:
                                predict_features_file = os.path.join(
                                    self.m_eval_output_path,
                                    'eval_features_{0}_{1}.txt'.format(dataset_name, label_format))
                                with open(predict_features_file, 'a') as f:
                                    f.write("User: {0}\tItem: {1}\n".format(true_userid_j, true_itemid_j))
                                    f.write("refs: {}\n".format(refs_j))
                                    f.write("hyps: {}\n".format(hyps_j))
                                    f.write("proxy: {}\n".format(proxy_j))
                                    f.write("target features: {}\n".format(target_featureword_j))
                                    f.write("popular features: {}\n".format(popular_featureword_j))
                                    f.write("popular features frequency: {}\n".format(ui_popular_features_freq))
                                    f.write("overlappings: {}\n".format(overlap_featureword_j))
                                    f.write("Number of features in target: {}\n".format(len(target_featureid_j_set)))
                                    f.write("Number of features in top-pred: {}\n".format(len(popular_featureid_j_set)))
                                    f.write("Number of feature overlap: {}\n".format(len(overlap_featureid_j_set)))
                                    f.write("Precision: {0}\tRecall: {1}\tF1: {2}\tAUC: {3}\n".format(f_prec_j, f_recall_j, f_f1_j, f_auc_j))
                                    f.write("==------==------==------==------==------==------==\n")
                        else:
                            # get the index of the predicted features
                            top_pred_fid_index_j = (topk_pred_f_j == 1).nonzero(as_tuple=True)[0]
                            top_pred_fid_j = torch.gather(fid_j, dim=0, index=top_pred_fid_index_j)
                            top_pred_featureid_j = [self.m_fid2feature[this_f_id.item()] for this_f_id in top_pred_fid_j]
                            top_pred_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in top_pred_featureid_j]
                            # get the overlapping features
                            target_featureid_j_set = set(target_featureid_j)
                            top_pred_featureid_j_set = set(top_pred_featureid_j)
                            overlap_featureid_j_set = target_featureid_j_set.intersection(top_pred_featureid_j_set)
                            overlap_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in overlap_featureid_j_set]
                            if save_feature_selected and batch_save_flag:
                                predict_features_file = os.path.join(
                                    self.m_eval_output_path,
                                    'eval_features_{0}_{1}.txt'.format(dataset_name, label_format))
                                with open(predict_features_file, 'a') as f:
                                    f.write("User: {0}\tItem: {1}\n".format(true_userid_j, true_itemid_j))
                                    f.write("refs: {}\n".format(refs_j))
                                    f.write("hyps: {}\n".format(hyps_j))
                                    f.write("proxy: {}\n".format(proxy_j))
                                    f.write("predict sentences' features: {}\n".format(", ".join(hyps_unique_featurewords_j)))
                                    f.write("target features: {}\n".format(target_featureword_j))
                                    f.write("top predict features: {}\n".format(top_pred_featureword_j))
                                    f.write("overlappings: {}\n".format(overlap_featureword_j))
                                    f.write("Number of features in target: {}\n".format(len(target_featureid_j_set)))
                                    f.write("Number of features in top-pred: {}\n".format(len(top_pred_featureid_j_set)))
                                    f.write("Number of feature overlap: {}\n".format(len(overlap_featureid_j_set)))
                                    f.write("Precision: {0}\tRecall: {1}\tF1: {2}\tAUC: {3}\n".format(f_prec_j, f_recall_j, f_f1_j, f_auc_j))
                                    f.write("==------==------==------==------==------==------==\n")
                    else:
                        # Save log results for feature prediction on proxy
                        if use_trigram or use_origin:
                            gt_featureword_j_set = set(gt_featureword_j)
                            select_featureword_j_set = set(hyps_featurewords_j_list)
                            overlap_featureword_j = list(gt_featureword_j_set.intersection(select_featureword_j_set))
                            if save_feature_selected and batch_save_flag:
                                predict_features_file = os.path.join(
                                    self.m_eval_output_path,
                                    'eval_features_{0}_{1}.txt'.format(dataset_name, label_format))
                                with open(predict_features_file, 'a') as f:
                                    f.write("User: {0}\tItem: {1}\n".format(true_userid_j, true_itemid_j))
                                    f.write("refs: {}\n".format(refs_j))
                                    f.write("hyps: {}\n".format(hyps_j))
                                    f.write("proxy: {}\n".format(proxy_j))
                                    f.write("ground-truth features: {}\n".format(gt_featureword_j_set))
                                    f.write("predict sentences' features: {}\n".format(", ".join(hyps_unique_featurewords_j)))
                                    f.write("overlappings: {}\n".format(overlap_featureword_j))
                                    f.write("Number of features in ground-truth: {}\n".format(len(gt_featureword_j_set)))
                                    f.write("Number of features in predict sentences: {}\n".format(len(select_featureword_j_set)))
                                    f.write("Number of feature overlap: {}\n".format(len(overlap_featureword_j)))
                                    f.write("Precision: {0}\tRecall: {1}\tF1: {2}\tAUC: {3}\n".format(f_prec_j, f_recall_j, f_f1_j, f_auc_j))
                                    f.write("==------==------==------==------==------==------==\n")
                        elif popular_features or popular_features_vs_origin or popular_features_vs_trigram:
                            # popular feature words
                            popular_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in useritem_popular_features]
                            # get the overlapping features
                            popular_featureword_j_set = set(popular_featureword_j)
                            gt_featureword_j_set = set(gt_featureword_j)
                            overlap_featureword_j = list(popular_featureword_j_set.intersection(gt_featureword_j_set))
                            if save_feature_selected and batch_save_flag:
                                predict_features_file = os.path.join(
                                    self.m_eval_output_path,
                                    'eval_features_{0}_{1}.txt'.format(dataset_name, label_format))
                                with open(predict_features_file, 'a') as f:
                                    f.write("User: {0}\tItem: {1}\n".format(true_userid_j, true_itemid_j))
                                    f.write("refs: {}\n".format(refs_j))
                                    f.write("hyps: {}\n".format(hyps_j))
                                    f.write("proxy: {}\n".format(proxy_j))
                                    f.write("predict sentences' features: {}\n".format(", ".join(hyps_unique_featurewords_j)))
                                    f.write("ground-truth features: {}\n".format(gt_featureword_j))
                                    f.write("popular features: {}\n".format(popular_featureword_j))
                                    f.write("overlappings: {}\n".format(overlap_featureword_j))
                                    f.write("Number of features in ground-truth: {}\n".format(len(gt_featureword_j)))
                                    f.write("Number of features in popular: {}\n".format(len(popular_featureword_j)))
                                    f.write("Number of feature overlap: {}\n".format(len(overlap_featureword_j)))
                                    f.write("Precision: {0}\tRecall: {1}\tF1: {2}\tAUC: {3}\n".format(f_prec_j, f_recall_j, f_f1_j, f_auc_j))
                                    f.write("==------==------==------==------==------==------==\n")
                        elif random_features:
                            # predict feature words
                            top_pred_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in top_pred_random_featureid_j]
                            # get the overlapping features
                            top_pred_featureword_j_set = set(top_pred_featureword_j)
                            gt_featureword_j_set = set(gt_featureword_j)
                            overlap_featureword_j = list(top_pred_featureword_j_set.intersection(gt_featureword_j_set))
                            if save_feature_selected and batch_save_flag:
                                predict_features_file = os.path.join(
                                    self.m_eval_output_path,
                                    'eval_features_{0}_{1}.txt'.format(dataset_name, label_format))
                                with open(predict_features_file, 'a') as f:
                                    f.write("User: {0}\tItem: {1}\n".format(true_userid_j, true_itemid_j))
                                    f.write("refs: {}\n".format(refs_j))
                                    f.write("hyps: {}\n".format(hyps_j))
                                    f.write("proxy: {}\n".format(proxy_j))
                                    f.write("ground-truth features: {}\n".format(gt_featureword_j))
                                    f.write("random features: {}\n".format(top_pred_featureword_j))
                                    f.write("overlappings: {}\n".format(overlap_featureword_j))
                                    f.write("Number of features in ground-truth: {}\n".format(len(gt_featureword_j)))
                                    f.write("Number of features in random: {}\n".format(len(top_pred_featureword_j)))
                                    f.write("Number of feature overlap: {}\n".format(len(overlap_featureword_j)))
                                    f.write("Precision: {0}\tRecall: {1}\tF1: {2}\tAUC: {3}\n".format(f_prec_j, f_recall_j, f_f1_j, f_auc_j))
                                    f.write("==------==------==------==------==------==------==\n")
                        else:
                            # predict feature words
                            top_pred_featureword_j = [self.d_id2feature[this_fea_id] for this_fea_id in top_pred_featureid_j]
                            # predict feature words with logits
                            top_pred_featureword_logits_j = [(self.d_id2feature[this_fea_id], this_logits) for this_fea_id, this_logits in topk_preds_featureid_with_logits]
                            # get the overlapping features
                            top_pred_featureword_j_set = set(top_pred_featureword_j)
                            gt_featureword_j_set = set(gt_featureword_j)
                            overlap_featureword_j = list(top_pred_featureword_j_set.intersection(gt_featureword_j_set))
                            if save_feature_selected and batch_save_flag:
                                predict_features_file = os.path.join(
                                    self.m_eval_output_path,
                                    'eval_features_{0}_{1}.txt'.format(dataset_name, label_format))
                                with open(predict_features_file, 'a') as f:
                                    f.write("User: {0}\tItem: {1}\n".format(true_userid_j, true_itemid_j))
                                    f.write("refs: {}\n".format(refs_j))
                                    f.write("hyps: {}\n".format(hyps_j))
                                    f.write("proxy: {}\n".format(proxy_j))
                                    f.write("predict sentences' features: {}\n".format(", ".join(hyps_unique_featurewords_j)))
                                    f.write("ground-truth features: {}\n".format(gt_featureword_j))
                                    f.write("top predict features: {}\n".format(top_pred_featureword_j))
                                    f.write("top predict features with logits: {}\n".format(top_pred_featureword_logits_j))
                                    f.write("overlappings (top-predict vs. gt features): {}\n".format(overlap_featureword_j))
                                    f.write("Number of features in ground-truth: {}\n".format(len(gt_featureword_j)))
                                    f.write("Number of features in top-pred: {}\n".format(len(top_pred_featureword_j)))
                                    f.write("Number of feature overlap: {}\n".format(len(overlap_featureword_j)))
                                    f.write("Precision: {0}\tRecall: {1}\tF1: {2}\tAUC: {3}\n".format(f_prec_j, f_recall_j, f_f1_j, f_auc_j))
                                    f.write("==------==------==------==------==------==------==\n")

        self.m_mean_f_precision = np.mean(f_precision_list)
        self.m_mean_f_recall = np.mean(f_recall_list)
        self.m_mean_f_f1 = np.mean(f_F1_list)
        self.m_mean_f_auc = np.mean(f_auc_list)
        self.m_mean_proxy_feature = np.mean(proxy_feature_num_cnt)
        self.m_mean_gt_feature = np.mean(gt_feature_num_cnt)

        print("Totally {0} batches ({1} data instances).\n Among them, {2} batches are saved into logging files.".format(
            len(eval_data), len(f_precision_list), save_logging_cnt
        ))

        print("Average features num in proxy: {0}. Average features num in ground-truth: {1}".format(
            self.m_mean_proxy_feature, self.m_mean_gt_feature
        ))

        print("feature prediction, precision: %.4f, recall: %.4f, F1: %.4f, AUC: %.4f" % (
            self.m_mean_f_precision, self.m_mean_f_recall, self.m_mean_f_f1, self.m_mean_f_auc
        ))

        metric_log_file = os.path.join(self.m_eval_output_path, 'eval_metrics_{0}_{1}.txt'.format(dataset_name, label_format))
        with open(metric_log_file, 'w') as f:
            print("feature prediction, precision: %.4f, recall: %.4f, F1: %.4f, AUC: %.4f \n" % (
                self.m_mean_f_precision, self.m_mean_f_recall, self.m_mean_f_f1, self.m_mean_f_auc), file=f)
            print("Total number of user-item on testset (not appear in trainset): {}\n".format(train_test_differ_cnt), file=f)
            print("Total number of user-item on testset (appear in trainset): {}\n".format(train_test_overlap_cnt), file=f)

    def combine_featuretf(self, user_featuretf, item_featuretf):
        """ Add 2 dict together to get the feature tf-value on this user and this item
        """

        useritem_featuretf = dict()
        for key, value in user_featuretf.items():
            feature_id = key
            assert isinstance(feature_id, str)
            feature_tf = value
            assert isinstance(feature_tf, int)
            assert feature_id not in useritem_featuretf
            useritem_featuretf[feature_id] = feature_tf
        for key, value in item_featuretf.items():
            feature_id = key
            assert isinstance(feature_id, str)
            feature_tf = value
            assert isinstance(feature_tf, int)
            if feature_id not in useritem_featuretf:
                useritem_featuretf[feature_id] = feature_tf
            else:
                useritem_featuretf[feature_id] += feature_tf

        return useritem_featuretf

    def get_popular_features(self, useritem_featuretf, topk=26):
        """ Get the popular features (id) based on the feature tf value
        """

        sorted_useritem_featuretf = dict(sorted(useritem_featuretf.items(), key=lambda item: item[1], reverse=True))

        topk_popular_features = []
        topk_popular_features_freq = []
        cnt_features = 0
        for key, value in sorted_useritem_featuretf.items():
            topk_popular_features.append(key)   # key is featureid
            topk_popular_features_freq.append(value)  # value is the frequency
            cnt_features += 1
            if cnt_features == topk:
                break
        assert len(topk_popular_features) <= topk

        return topk_popular_features, topk_popular_features_freq

    def get_sid2featuretf_eval(self, testset_sentid2featuretf, sent2sid, train_sent_num):
        """ Get sid to featuretf mapping (on valid/test set).
            During constructing the graph data, we load the valid/test sentences. Since the
            original sentid is seperated from train-set sentence sentid, we first add the
            sentid of valid/test-set with train_sent_num and then mapping the new sent_id
            to sid. Therefore, to simplify the mapping between sid and featureid (and also
            feature tf) we need to construct this mapping here.
        """
        testset_sid2featuretf = dict()
        for key, value in testset_sentid2featuretf.items():
            assert isinstance(key, str)
            sentid = int(key) + train_sent_num
            sentid = str(sentid)
            sid = sent2sid[sentid]
            assert sid not in testset_sid2featuretf
            testset_sid2featuretf[sid] = value
        return testset_sid2featuretf

    def get_sid2feature_train(self, trainset_sentid2featuretfidf, sent2sid):
        trainset_sid2feature = dict()
        for key, value in trainset_sentid2featuretfidf.items():
            assert isinstance(key, str)     # key is the sentid
            sid = sent2sid[key]
            assert sid not in trainset_sid2feature
            trainset_sid2feature[sid] = list(value.keys())
        return trainset_sid2feature

    def get_gt_review_featuretf(self, testset_sid2featuretf, gt_sids):
        """ Get the featureid list and featuretf dict for a list of ground-truth sids
        """
        gt_featureid_set = set()
        gt_featuretf_dict = dict()
        for gt_sid in gt_sids:
            cur_sid_featuretf = testset_sid2featuretf[gt_sid.item()]
            for key, value in cur_sid_featuretf.items():
                gt_featureid_set.add(key)
                if key not in gt_featuretf_dict:
                    gt_featuretf_dict[key] = value
                else:
                    gt_featuretf_dict[key] += value
        return list(gt_featureid_set), gt_featuretf_dict

    def ngram_blocking(self, sents, p_sent, n_win, k):
        """ ngram blocking
        :param sents:   batch of lists of candidate sentence, each candidate sentence is a string. shape: [batch_size, sent_num]
        :param p_sent:  torch tensor. batch of predicted/relevance scores of each candidate sentence. shape: (batch_sizem, sent_num)
        :param n_win:   ngram window size, i.e. which n-gram we are using. n_win can be 2,3,4,...
        :param k:       we are selecting the top-k sentences

        :return:        selected index of sids
        """
        batch_size = p_sent.size(0)
        batch_select_idx = []
        batch_select_proba = []
        batch_select_rank = []
        assert len(sents) == len(p_sent)
        assert len(sents) == batch_size
        assert len(sents[0]) == len(p_sent[0])
        # print(sents)
        # print("batch size (sents): {}".format(len(sents)))
        for i in range(len(sents)):
            # print(len(sents[i]))
            assert len(sents[i]) == len(sents[0])
            assert len(sents[i]) == len(p_sent[i])
        # print(p_sent)
        # print(p_sent.shape)
        for batch_idx in range(batch_size):
            ngram_list = []
            _, sorted_idx = p_sent[batch_idx].sort(descending=True)
            select_idx = []
            select_proba = []
            select_rank = []
            idx_rank = 0
            for idx in sorted_idx:
                idx_rank += 1
                try:
                    cur_sent = sents[batch_idx][idx]
                except:
                    print("i: {0} \t idx: {1}".format(batch_idx, idx))
                cur_tokens = cur_sent.split()
                overlap_flag = False
                cur_sent_ngrams = []
                for i in range(len(cur_tokens)-n_win+1):
                    this_ngram = " ".join(cur_tokens[i:(i+n_win)])
                    if this_ngram in ngram_list:
                        overlap_flag = True
                        break
                    else:
                        cur_sent_ngrams.append(this_ngram)
                if not overlap_flag:
                    select_idx.append(idx)
                    select_proba.append(p_sent[batch_idx][idx])
                    select_rank.append(idx_rank)
                    ngram_list.extend(cur_sent_ngrams)
                    if len(select_idx) >= k:
                        break
            batch_select_idx.append(select_idx)
            batch_select_proba.append(select_proba)
            batch_select_rank.append(select_rank)
        # convert list to torch tensor
        batch_select_idx = torch.LongTensor(batch_select_idx)
        return batch_select_idx, batch_select_proba, batch_select_rank

    def origin_blocking_sent_prediction(self, s_logits, sids, s_masks, topk=3, topk_cdd=20):
        # incase some not well-trained model will predict the logits for all sentences as 0.0, we apply masks on it
        masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        # 1. get the top-k predicted sentences which form the hypothesis
        topk_logits, topk_pred_snids = torch.topk(masked_s_logits, topk, dim=1)
        # topk sentence index
        # pred_sids: shape: (batch_size, topk_sent)
        sids = sids.cpu()
        pred_sids = sids.gather(dim=1, index=topk_pred_snids)
        # 2. get the top-20 predicted sentences' content and proba
        top_cdd_logits, top_cdd_pred_snids = torch.topk(masked_s_logits, topk_cdd, dim=1)
        top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
        # 3. get the bottom-20 predicted sentences' content and proba
        reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
        bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_cdd, dim=1)
        bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

        return topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids

    def trigram_blocking_sent_prediction(self, s_logits, sids, s_masks, batch_size, topk=3, topk_cdd=20):
        # use n-gram blocking
        # get all the sentence content
        batch_sents_content = []
        assert len(sids) == s_logits.size(0)      # this is the batch size
        for i in range(batch_size):
            cur_sents_content = []
            assert len(sids[i]) == len(sids[0])
            for cur_sid in sids[i]:
                cur_sents_content.append(self.m_sid2swords[cur_sid.item()])
            batch_sents_content.append(cur_sents_content)
        assert len(batch_sents_content[0]) == len(batch_sents_content[-1])      # this is the max_sent_len (remember we are using zero-padding for batch data)
        masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        sids = sids.cpu()
        # 1. get the top-k predicted sentences which form the hypothesis
        ngram_block_pred_snids, ngram_block_pred_proba, ngram_block_pred_rank = self.ngram_blocking(
            batch_sents_content, masked_s_logits, n_win=3, k=3
        )
        pred_sids = sids.gather(dim=1, index=ngram_block_pred_snids)
        topk_logits = ngram_block_pred_proba
        # 2. get the top-20 predicted sentences' content and proba
        top_cdd_pred_snids, top_cdd_logits, _ = self.ngram_blocking(
            batch_sents_content, masked_s_logits, n_win=3, k=topk_cdd
        )
        top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
        # 3. get the bottom-20 predicted sentences' content and proba
        reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
        bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_cdd, dim=1)
        bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

        return topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids

    def bleu_filtering_sent_prediction(self):
        pass

    def feature_logits_save_file(self, true_userid_j, true_itemid_j, mask_f_logits_j):
        feature_logits_file = os.path.join(
            self.m_eval_output_path,
            'feature_logits_{0}_{1}.txt'.format(dataset_name, label_format)
        )
        with open(feature_logits_file, 'a') as f_l:
            f_l.write("User: {0}\tItem: {1}\n".format(true_userid_j, true_itemid_j))
            sorted_mask_f_logits_j, _ = torch.topk(mask_f_logits_j, mask_f_logits_j.size(0))
            for logit in sorted_mask_f_logits_j:
                f_l.write("%.4f" % logit.item())
                f_l.write(", ")
            f_l.write("\n")

    def features_result_save_file(
        self, use_ground_truth, proxy_featureids, gt_featureids, hyps_featureids, popular_featureids,
            top_predict_featureids):
        """ Write the results into file.
        :param: use_ground_truth: True if the predicted features is compared with the ground-turth features.
            False if the predicted features is compared with the proxy's features
        :param: proxy_featureids: proxy's featureids, list
        :param: gt_featureids: ground-truth's featureids, list
        :param: hyps_featureids: hypothesis/predict sentences' featureids, list
        :param: popular_featureids: popular features' featureids, list
        :param: top_predict_featureids: top predicted featureids, list
        """

        features_result_file_path = os.path.join(
            self.m_eval_output_path, 'eval_features_{0}_{1}.txt'.format(dataset_name, label_format))

        proxy_featureids_set = set(proxy_featureids)
        hyps_featureids_set = set(hyps_featureids)
        popular_featureids_set = set(popular_featureids)
        top_predict_featureids_set = set(top_predict_featureids)

        assert len(popular_featureids) == len(popular_featureids_set)
        assert len(top_predict_featureids) == len(top_predict_featureids_set)

        proxy_featurewords = [self.d_id2feature[this_fea_id] for this_fea_id in proxy_featureids_set]
        hyps_featurewords = [self.d_id2feature[this_fea_id] for this_fea_id in hyps_featureids_set]
        popular_featurewords = [self.d_id2feature[this_fea_id] for this_fea_id in popular_featureids]
        top_predict_featurewords = [self.d_id2feature[this_fea_id] for this_fea_id in top_predict_featureids]

        if use_ground_truth:
            pass
        else:
            # 1. Use the features from the predicted sentences.
            # Options: use_origin/use_trigram
            if use_origin or use_trigram:
                overlap_featureids_set = proxy_featureids_set.intersection(hyps_featureids_set)
            # 2. Use the popular features
            elif popular_features:
                overlap_featureids_set = proxy_featureids_set.intersection(popular_featureids_set)
            # 3. Use the predicted features by the multi-task model
            else:
                overlap_featureids_set = proxy_featureids_set.intersection(top_predict_featureids_set)
