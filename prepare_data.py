#!/usr/bin/env python3
"""
آماده‌سازی داده‌های آموزشی برای تشخیص لیپاز
استفاده: python prepare_data.py --positive فایل_مثبت --negative فایل_منفی
"""
from Bio import SeqIO
import pandas as pd
import random
import argparse
import os

def prepare_dataset(positive_file, negative_file, output_file, subsample_size=None):
    """آماده‌سازی dataset از فایل‌های FASTA"""
    
    # چک کردن وجود فایل‌ها
    if not os.path.exists(positive_file):
        print(f"❌ فایل مثبت پیدا نشد: {positive_file}")
        return False
    
    if not os.path.exists(negative_file):
        print(f"❌ فایل منفی پیدا نشد: {negative_file}")
        return False
    
    print("🔄 در حال لود داده‌های مثبت...")
    positive_sequences = []
    for record in SeqIO.parse(positive_file, "fasta"):
        positive_sequences.append({"sequence": str(record.seq), "label": 1})
    
    print(f"✅ {len(positive_sequences)} نمونه مثبت لود شد")
    
    print("🔄 در حال لود داده‌های منفی...")
    negative_sequences = []
    for record in SeqIO.parse(negative_file, "fasta"):
        negative_sequences.append({"sequence": str(record.seq), "label": 0})
    
    print(f"✅ {len(negative_sequences)} نمونه منفی لود شد")
    
    # Subsample اگه لازم باشه
    if subsample_size:
        positive_sample = random.sample(positive_sequences, 
                                      min(subsample_size, len(positive_sequences)))
        negative_sample = random.sample(negative_sequences, 
                                      min(subsample_size, len(negative_sequences)))
        print(f"📊 Subsample: {len(positive_sample)} مثبت، {len(negative_sample)} منفی")
    else:
        positive_sample = positive_sequences
        negative_sample = negative_sequences
    
    # ترکیب و ذخیره
    all_data = positive_sample + negative_sample
    random.shuffle(all_data)
    
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"💾 Dataset ذخیره شد: {output_file}")
    print(f"📈 تعداد کل رکورد: {len(df)}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="آماده‌سازی داده‌های لیپاز")
    parser.add_argument("--positive", required=True, help="مسیر فایل نمونه‌های مثبت (.fasta)")
    parser.add_argument("--negative", required=True, help="مسیر فایل نمونه‌های منفی (.fasta)") 
    parser.add_argument("--output", default="lipase_dataset.csv", help="نام فایل خروجی")
    parser.add_argument("--subsample", type=int, help="تعداد subsample از هر کلاس (اختیاری)")
    
    args = parser.parse_args()
    
    success = prepare_dataset(args.positive, args.negative, args.output, args.subsample)
    
    if success:
        print("🎉 آماده‌سازی داده‌ها با موفقیت انجام شد!")
    else:
        print("❌ خطا در آماده‌سازی داده‌ها")