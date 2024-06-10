mkdir download
cd download/
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -p $HOME/anaconda3
source $HOME/.bashrc
conda create -n style-venv python=3.6 -y
conda activate style-venv
cd ..
git clone https://github.com/martiansideofthemoon/style-transfer-paraphrase.git
git clone https://github.com/rominf/ordered-set-stubs.git
cd ordered-set-stubs/
pip install .
cd ../style-transfer-paraphrase/
echo -e "torch==1.4.0\ntorchvision==0.5.0\npandas\n" >> requirements.txt
pip install -r requirements.txt
pip install --editable .
cd fairseq/
pip install --editable .
cd ../../download
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -xzvf roberta.large.tar.gz
export ROBERTA_LARGE=$PWD/roberta.large
pip install gdown

cd /content/style-transfer-paraphrase/datasets
gdown --folder https://drive.google.com/drive/folders/1C7QbJizVeJ2xY8r4KeiDp369SKTYeKvL  # paranmt_filtered
cd ..
bash style_paraphrase/examples/run_finetune_paraphrase.sh 


cd /content/style-transfer-paraphrase
mkdir pretrained_models
cd pretrained_models
gdown --folder https://drive.google.com/drive/folders/1bvd1RryArQwx_3VyN_VbHiEmEF4UvRlH
gdown --folder https://drive.google.com/drive/folders/1RmiXX8u1ojj4jorj_QgxOWWkryDIdie-
gdown --folder https://drive.google.com/drive/folders/1R0ch-D3vfGtt5avm7LENEXUfFe5eTcIU
gdown --folder https://drive.google.com/drive/folders/1ApW4J3oT5N48BH_JOPc6z16nW8dy0I7N
cd ..
# or mount and cp:
!cp -r "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/style_transfer_paraphrase/models" "/content/style-transfer-paraphrase/pretrained_models"

sed -i '15s|.*|from style_paraphrase.dataset_config import DATASET_CONFIG|' "/content/style-transfer-paraphrase/style_paraphrase/utils.py"
sed -i '16s|.*|from style_paraphrase.data_utils import update_config, Instance|' "/content/style-transfer-paraphrase/style_paraphrase/utils.py"
python demo_paraphraser.py --model_dir ./pretrained_models/models/paraphraser_gpt2_large --top_p_value 0.6
python paraphrase_many.py --model_dir ./pretrained_models/models/paraphraser_gpt2_large --top_p_value 0.0 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/style-transfer-paraphrase/datasets/sentences/ag_new.txt"

# python demo_paraphraser.py --top_p_value 0.6 --model_dir /content/style-transfer-paraphrase/pretrained_models/models/cds_models/tweets
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/paraphraser_gpt2_large --top_p_value 0.0 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_paraphraser_p_0.0.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/paraphraser_gpt2_large --top_p_value 0.6 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_paraphraser_p_0.6.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/paraphraser_gpt2_large --top_p_value 0.9 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_paraphraser_p_0.9.txt"

python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/formality_models/model_314 --top_p_value 0.0 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_formality_p_0.0.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/formality_models/model_314 --top_p_value 0.6 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_formality_p_0.6.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/formality_models/model_314 --top_p_value 0.9 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_formality_p_0.9.txt"

python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/shakespeare_models/model_300 --top_p_value 0.0 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_shakespeare_p_0.0.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/shakespeare_models/model_300 --top_p_value 0.6 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_shakespeare_p_0.6.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/shakespeare_models/model_300 --top_p_value 0.9 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_shakespeare_p_0.9.txt"

python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/cds_models/aae --top_p_value 0.0 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_aae_p_0.0.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/cds_models/aae --top_p_value 0.6 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_aae_p_0.6.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/cds_models/aae --top_p_value 0.9 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_aae_p_0.9.txt"

python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/cds_models/bible --top_p_value 0.0 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_bible_p_0.0.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/cds_models/bible --top_p_value 0.6 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_bible_p_0.6.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/cds_models/bible --top_p_value 0.9 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_bible_p_0.9.txt"

python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/cds_models/tweets --top_p_value 0.0 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_tweets_p_0.0.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/cds_models/tweets --top_p_value 0.6 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_tweets_p_0.6.txt"
python paraphrase_many.py --model_dir /content/style-transfer-paraphrase/pretrained_models/models/cds_models/tweets --top_p_value 0.9 --input "/content/style-transfer-paraphrase/datasets/sentences/ag.txt" --output "/content/drive/MyDrive/Colab Notebooks/DSC 253 - Adv Data-Driven Text Mining/Project/DSC253/ag_data/tmp/ag_tweets_p_0.9.txt"