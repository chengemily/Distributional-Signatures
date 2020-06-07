#dataset=amazon
#data_path='data/amazon.json'
#n_train_class=10
#n_val_class=5
#n_test_class=9

#dataset=fewrel
#data_path='data/fewrel.json'
#n_train_class=65
#n_val_class=5
#n_test_class=10

dataset=20newsgroup
data_path='data/20news.json'
embed_path='cached_embeds/20news_shear_vecoffset_embed.json'
n_train_class=8
n_val_class=5
n_test_class=7

#dataset=huffpost
#data_path='data/huffpost.json'
#embed_path='cached_embeds/huffpost_shear_vecoffset_embed.json'
#n_train_class=20
#n_val_class=5
#n_test_class=16

#dataset=rcv1
#data_path='data/rcv1.json'
#n_train_class=37
#n_val_class=10
#n_test_class=24

#dataset=reuters
#data_path='data/reuters.json'
#n_train_class=15
#n_val_class=5
#n_test_class=11


python src/main.py \
    --cuda 0 \
    --way 5 \
    --shot 1 \
    --query 25 \
    --mode 'test' \
    --embedding 'idf'\
    --classifier 'nn' \
    --dataset=$dataset \
    --data_path=$data_path \
    --path_to_embed_cache=$embed_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --zero_shot \
