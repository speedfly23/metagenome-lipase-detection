# Metagenome Lipase Detection 🧬

## نصب سریع

```bash
git clone https://github.com/your-username/metagenome-lipase-detection.git
cd metagenome-lipase-detection
chmod +x setup.sh
./setup.sh





22222222222




## source ~/miniconda/bin/activate metagenome

##python scripts/prepare_data.py \
    --positive /path/to/all_true_sample.fasta \
    --negative /path/to/all_negative_sample.fasta \
    --output lipase_dataset.csv \
    --subsample 10000


    ##python scripts/train_model.py \
    --dataset lipase_dataset.csv \
    --metagenome /path/to/my_metagenome.faa \
    --threshold 0.7


    # فرض: فایل‌ها در پوشه ~/project/data/ هستند
python scripts/prepare_data.py \
    --positive ~/project/data/all_true_sample.fasta \
    --negative ~/project/data/all_negative_sample.fasta \
    --subsample 10000

python scripts/train_model.py \
    --dataset lipase_dataset.csv \
    --metagenome ~/project/prokka_out/my_metagenome.faa