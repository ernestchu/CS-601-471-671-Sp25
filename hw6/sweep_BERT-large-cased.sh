export CUDA_VISIBLE_DEVICES=7

echo 1e-4 7 >> log_bert
python base_classification.py --device cuda --model BERT-large-cased --batch_size 64 --lr 1e-4 --num_epochs 7 | tail -n2 >> log_bert

echo 5e-4 7 >> log_bert
python base_classification.py --device cuda --model BERT-large-cased --batch_size 64 --lr 5e-4 --num_epochs 7 | tail -n2 >> log_bert

echo 1e-3 7 >> log_bert
python base_classification.py --device cuda --model BERT-large-cased --batch_size 64 --lr 1e-3 --num_epochs 7 | tail -n2 >> log_bert

echo 1e-4 9 >> log_bert
python base_classification.py --device cuda --model BERT-large-cased --batch_size 64 --lr 1e-4 --num_epochs 9 | tail -n2 >> log_bert

echo 5e-4 9 >> log_bert
python base_classification.py --device cuda --model BERT-large-cased --batch_size 64 --lr 5e-4 --num_epochs 9 | tail -n2 >> log_bert

echo 1e-3 9 >> log_bert
python base_classification.py --device cuda --model BERT-large-cased --batch_size 64 --lr 1e-3 --num_epochs 9 | tail -n2 >> log_bert
