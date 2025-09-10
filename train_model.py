# ساخت پوشه scripts اگه نیست
mkdir -p scripts

# انتقال prepare_data به scripts (اگه تو root هست)
mv prepare_data.py scripts/ 2>/dev/null || cp prepare_data.py scripts/

# ساخت فایل آموزش مدل
cat > scripts/train_model.py << 'EOF'
#!/usr/bin/env python3
"""
آموزش مدل تشخیص لیپاز و پیش‌بینی روی متاژنوم
نویسنده: speedfly23
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from Bio import SeqIO
import joblib
import os
import argparse
from datetime import datetime

def train_and_predict(dataset_file, metagenome_file=None, threshold=0.7, n_jobs=12):
    """آموزش مدل و پیش‌بینی"""
    
    print("🤖 Metagenome Lipase Detection - Model Training")
    print("👤 speedfly23 - https://github.com/speedfly23/metagenome-lipase-detection")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # بررسی وجود فایل dataset
    if not os.path.exists(dataset_file):
        print(f"❌ فایل dataset پیدا نشد: {dataset_file}")
        print("🔄 ابتدا scripts/prepare_data.py را اجرا کنید")
        return False
    
    print("🔄 لود dataset...")
    df = pd.read_csv(dataset_file)
    print(f"📊 تعداد رکورد: {len(df)}")
    print(f"📊 توزیع: {sum(df['label'])} مثبت ({sum(df['label'])/len(df)*100:.1f}%), {len(df) - sum(df['label'])} منفی ({(len(df) - sum(df['label']))/len(df)*100:.1f}%)")
    
    # استخراج ویژگی (3-mer)
    print("🧬 استخراج ویژگی‌ها (3-mer)...")
    vectorizer = CountVectorizer(
        analyzer='char', 
        ngram_range=(3,3), 
        max_features=5000  # محدود کردن برای جلوگیری از memory issue
    )
    X = vectorizer.fit_transform(df['sequence'])
    y = df['label']
    
    print(f"📐 ابعاد ماتریس ویژگی: {X.shape}")
    
    # تقسیم داده
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # آموزش مدل
    print("🤖 آموزش مدل Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=n_jobs,
        max_depth=15,  # محدود کردن عمق
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    
    # تست دقت
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ دقت مدل: {accuracy:.4f}")
    
    # گزارش تفصیلی
    print("\n📋 گزارش کلاسیفیکیشن:")
    print(classification_report(y_test, predictions, target_names=['Non-Lipase', 'Lipase']))
    
    # ذخیره مدل و vectorizer
    joblib.dump(model, "trained_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("💾 مدل و vectorizer ذخیره شدند")
    
    # پیش‌بینی روی متاژنوم (اگه وجود داره)
    if metagenome_file and os.path.exists(metagenome_file):
        print(f"🔍 پیش‌بینی روی داده‌های متاژنوم: {metagenome_file}")
        
        metagenome_sequences = []
        for record in SeqIO.parse(metagenome_file, "fasta"):
            if len(str(record.seq)) > 50:  # فقط توالی‌های بلندتر از 50
                metagenome_sequences.append({
                    "id": record.id, 
                    "sequence": str(record.seq),
                    "length": len(str(record.seq))
                })
        
        if len(metagenome_sequences) == 0:
            print("⚠️  هیچ توالی مناسبی در فایل متاژنوم پیدا نشد")
            return True
            
        df_meta = pd.DataFrame(metagenome_sequences)
        print(f"🧬 تعداد ORF‌ها: {len(df_meta)}")
        
        # پیش‌بینی
        X_meta = vectorizer.transform(df_meta['sequence'])
        probabilities = model.predict_proba(X_meta)[:, 1]
        
        # فیلتر کاندیداها با threshold های مختلف
        thresholds = [0.5, 0.7, 0.8, 0.9]
        
        for thresh in thresholds:
            candidates = df_meta[probabilities > thresh].copy()
            candidates['probability'] = probabilities[probabilities > thresh]
            candidates = candidates.sort_values('probability', ascending=False)
            
            print(f"\n🎯 کاندیداها با threshold > {thresh}: {len(candidates)}")
            
            if len(candidates) > 0:
                # ذخیره نتایج
                output_file = f"predicted_lipases_t{thresh}.csv"
                candidates.to_csv(output_file, index=False)
                
                print(f"💾 ذخیره شد: {output_file}")
                print("🏆 بهترین کاندیداها:")
                for i, (_, row) in enumerate(candidates.head(5).iterrows()):
                    print(f"  {i+1}. {row['id']}: {row['probability']:.4f} (طول: {row['length']} aa)")
        
        # خلاصه نهایی
        main_candidates = df_meta[probabilities > threshold].copy()
        if len(main_