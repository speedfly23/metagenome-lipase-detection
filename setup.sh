#!/bin/bash
echo "🚀 شروع نصب محیط Metagenome Analysis..."
echo "📦 این اسکریپت فقط پکیج‌های اصلی رو نصب می‌کنه"
echo ""

# بروزرسانی سیستم
echo "🔄 بروزرسانی سیستم..."
sudo apt update -y
sudo apt install -y wget curl git

# چک کردن اینکه Miniconda نصب شده یا نه
if command -v conda &> /dev/null; then
    echo "✅ Conda در دسترس است"
else
    echo "📦 نصب Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    
    # راه‌اندازی conda
    $HOME/miniconda/bin/conda init bash
    source ~/.bashrc
    echo "✅ Miniconda نصب شد"
fi

# تنظیم conda channels
echo "🔧 تنظیم conda channels..."
conda config --add channels defaults
conda config --add channels bioconda  
conda config --add channels conda-forge

# ایجاد محیط conda
echo "🐍 ایجاد محیط metagenome..."
if conda env list | grep -q "metagenome"; then
    echo "⚠️  محیط metagenome قبلاً وجود دارد. حذف و ایجاد مجدد..."
    conda env remove -n metagenome -y
fi

conda create -n metagenome python=3.10 -y

# فعال‌سازی محیط
echo "🔄 فعال‌سازی محیط..."
source $HOME/miniconda/bin/activate metagenome

# نصب پکیج‌های bioinformatics
echo "🧬 نصب ابزارهای bioinformatics..."
conda install -n metagenome -c bioconda -c conda-forge \
    biopython \
    seqkit \
    fastp \
    megahit \
    prokka \
    -y

# نصب پکیج‌های machine learning
echo "🤖 نصب پکیج‌های machine learning..."
conda install -n metagenome -c conda-forge \
    scikit-learn \
    pandas \
    numpy \
    scipy \
    joblib \
    -y

# ایجاد ساختار پوشه‌های پروژه
echo "📁 ایجاد ساختار پروژه..."
mkdir -p ~/project/{data,results,prokka_out,models}

# تنظیم دسترسی‌ها
chmod +x scripts/*.py 2>/dev/null || echo "⚠️  پوشه scripts هنوز وجود ندارد"

echo ""
echo "✅ نصب کامل شد!"
echo ""
echo "📖 برای استفاده:"
echo "1️⃣  فعال‌سازی محیط:"
echo "   source ~/miniconda/bin/activate metagenome"
echo ""
echo "2️⃣  بررسی نصب:"
echo "   python -c 'import biopython, sklearn, pandas; print(\"همه پکیج‌ها آماده!\")'"
echo ""
echo "3️⃣  فایل‌های داده را در مسیر ~/project/data/ قرار دهید"
echo ""
echo "🎯 محیط آماده استفاده است!"