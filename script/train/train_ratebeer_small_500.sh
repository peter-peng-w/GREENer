CUDA_VISIBLE_DEVICES=1 python main.py --data_dir "/p/reviewde/data/ratebeer/graph/small_500/" --graph_dir "/p/reviewde/data/ratebeer/graph/small_500/geo_graph_batch/" --epoch_num 20 --train --batch_size 64 --learning_rate 0.001 --optimizer "Adam" --user_embed_size 256 --item_embed_size 256 --feature_embed_size 256 --sent_embed_size 768 --hidden_size 256 --head_num 4 --ffn_inner_hidden_size 256 --ffn_dropout_rate 0.01 --attn_dropout_rate 0.01 --print_interval 20