python eval_ILP.py --data_dir "/p/graph2x/Dataset/ratebeer/medium_500_positive/" --graph_dir "/p/graph2x/Dataset/ratebeer/medium_500_positive/geo_graph_batch_item_attr/" --model_name "graph_sentence_extractor" --data_set "ratebeer/medium_500_positive" --data_name "ratebeer" --model_file "ratebeer_graph_sentence_extractor/model_best_1_28_16_55.pt" --eval_output_path "../result/rb_medium_positive/lr0.0001_1_27_50epoch_item_attr/lr0.0002_bs32/ILP/topk_5/" --select_topk_f 15 --select_topk_s 5 --num_threads 25 --print_freq 20 --alpha 1.0