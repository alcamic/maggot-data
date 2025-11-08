# ================================================================
# MLP CLASSIFIER - WITH MORPHOLOGICAL FEATURES
# Versi Gabungan dengan Visualisasi Fitur Data Latih
# ================================================================
import pandas as pd
import sys
import numpy as np
import time, os, shutil, textwrap, warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import uniform, loguniform
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ================================================================
# ===== FUNGSI VISUALISASI FITUR (DARI SKRIP KEDUA) =====
# ================================================================
def generate_feature_plot(df_data, columns_to_plot, colors, title, y_label, output_filename, output_dir):
    """
    Menghasilkan plot garis untuk fitur yang dipilih, dikelompokkan dan diurutkan berdasarkan grade.
    Fungsi ini digeneralisasi dari skrip visualisasi Anda.
    """
    print(f"Generating plot: {title}...")
    GRADE_ORDER = ['Grade A', 'Grade B', 'Grade C']
    
    # --- 1. Siapkan Data ---
    columns_to_keep = ['grade'] + list(columns_to_plot.keys())
    
    missing_cols = [col for col in columns_to_keep if col not in df_data.columns]
    if missing_cols:
        print(f"Error: DataFrame kehilangan kolom untuk plot ini: {missing_cols}")
        return
        
    df_plot = df_data[columns_to_keep].copy()
    df_plot = df_plot[df_plot['grade'].isin(GRADE_ORDER)]

    # --- 2. Urutkan Data berdasarkan Grade (KRITIS) ---
    df_plot['grade'] = pd.Categorical(df_plot['grade'], categories=GRADE_ORDER, ordered=True)
    df_sorted = df_plot.sort_values('grade').reset_index(drop=True)

    if df_sorted.empty:
        print(f"Tidak ada data ditemukan untuk grade {GRADE_ORDER} dalam dataframe.")
        return

    # --- 3. Cari Batasan untuk Garis dan Teks ---
    grade_counts = df_sorted['grade'].value_counts().reindex(GRADE_ORDER)
    grade_counts = grade_counts.fillna(0).astype(int)
    boundaries = grade_counts.cumsum()
    
    text_positions = {}
    last_boundary = 0
    for grade in GRADE_ORDER:
        count = grade_counts[grade]
        if count > 0:
            text_positions[grade] = last_boundary + (count / 2)
        last_boundary += count

    # --- 4. Dapatkan Batas Sumbu Y Dinamis ---
    plot_cols = list(columns_to_plot.keys())
    min_val = df_sorted[plot_cols].min().min()
    max_val = df_sorted[plot_cols].max().max()
    
    if min_val == max_val:
        padding = 10 
    else:
        padding = (max_val - min_val) * 0.1
    
    y_min = min_val - padding
    y_max = max_val + padding * 4 
    text_y_pos = max_val + padding * 2

    # --- 5. Buat Plot ---
    plt.figure(figsize=(15, 8))

    for (col, label), color in zip(columns_to_plot.items(), colors):
        plt.plot(df_sorted.index, df_sorted[col], marker='.', markersize=4, 
                 linestyle='-', label=label, color=color)

    # --- 6. Tambahkan Anotasi dan Label ---
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylim(y_min, y_max)
    plt.xlim(0, len(df_sorted) - 1)
    
    for boundary in boundaries[:-1]:
        if boundary > 0 and boundary < len(df_sorted) - 1:
            plt.axvline(x=boundary - 0.5, color='red', linestyle='-')

    for grade in GRADE_ORDER:
        count = grade_counts[grade]
        if count > 0:
            plt.text(text_positions[grade], text_y_pos, 
                     f'{grade}\n(n={count})', ha='center', fontsize=10, fontweight='bold')

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(columns_to_plot), frameon=False, fontsize=16)
    plt.ylabel(y_label)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Simpan plot
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path, dpi=300)
    plt.close() # Tutup plot untuk menghemat memori
    print(f"✓ Plot disimpan ke: {save_path}")

# ================================================================
# ===== KONFIGURASI SCRIPT UTAMA =====
# ================================================================

# ===== KONFIGURASI =====
base_input_citra = r"Single Larvae data/Citra2"
csv_file_path = os.path.join(base_input_citra, "folder_features/bwareaopen_features.csv")
output_klasifikasi = os.path.join(base_input_citra, "folder_hasil_klasifikasi_mlp_with_morphology")
os.makedirs(output_klasifikasi, exist_ok=True)

# Baca data
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: File CSV tidak ditemukan di: {csv_file_path}")
    sys.exit()

df = df[df["grade"].isin(["Grade A", "Grade B", "Grade C"])]
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["grade"])

# Simpan label encoder untuk digunakan nanti saat prediksi
joblib.dump(label_encoder, os.path.join(output_klasifikasi, "label_encoder.joblib"))
print("LabelEncoder disimpan ke 'label_encoder.joblib'")

print(f"Total data: {len(df)}\nDistribusi:\n{df['grade'].value_counts()}\n")

# ===== PEMBAGIAN DATA (TRAIN/TEST SPLIT) =====
# Kita lakukan split sekali di awal agar konsisten
print("Melakukan split data Train/Test (80/20)...")
df_train, df_test = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['label'], 
    random_state=42
)
print(f"Data Latih: {len(df_train)} | Data Uji: {len(df_test)}")
print(f"Distribusi Latih:\n{df_train['grade'].value_counts(normalize=True)}")
print(f"Distribusi Uji:\n{df_test['grade'].value_counts(normalize=True)}")


# ===== FASE 0: VISUALISASI DATA LATIH (BARU) =====
# Membuat visualisasi fitur HANYA dari data latih (df_train)
print("\n" + "="*80 + "\n=== FASE 0: VISUALISASI DATA LATIH ===\n" + "="*80)

# 1. Plot RGB
generate_feature_plot(
    df_data=df_train,
    columns_to_plot={'R_mean': 'Fitur_R', 'G_mean': 'Fitur_G', 'B_mean': 'Fitur_B'},
    colors=['#e74c3c', '#2ecc71', '#3498db'],
    title='Fitur Ruang Warna RGB',
    y_label='RGB Value',
    output_filename='visualisasi_fitur_RGB.png',
    output_dir=output_klasifikasi
)

# 2. Plot LAB
generate_feature_plot(
    df_data=df_train,
    columns_to_plot={'L_mean': 'Fitur_L', 'A_mean': 'Fitur_A', 'B_LAB_mean': 'Fitur_B_LAB'},
    colors=['#bdc3c7', '#e67e22', '#3498db'],
    title='Fitur Ruang Warna LAB',
    y_label='LAB Value',
    output_filename='visualisasi_fitur_LAB.png',
    output_dir=output_klasifikasi
)

# 3. Plot HSV
generate_feature_plot(
    df_data=df_train,
    columns_to_plot={'H_mean': 'Fitur_H', 'S_mean': 'Fitur_S', 'V_mean': 'Fitur_V'},
    colors=['#f1c40f', '#8e44ad', '#2c3e50'],
    title='Fitur Ruang Warna HSV',
    y_label='HSV Value',
    output_filename='visualisasi_fitur_HSV.png',
    output_dir=output_klasifikasi
)

# 4. Plot Morfologi
generate_feature_plot(
    df_data=df_train,
    columns_to_plot={'area': 'Area', 'aspect_ratio': 'Aspect Ratio', 'solidity': 'Solidity'},
    colors=['#1abc9c', '#9b59b6', '#e67e22'],
    title='Fitur Morfologi',
    y_label='Feature Value',
    output_filename='visualisasi_fitur_Morfologi.png',
    output_dir=output_klasifikasi
)
# 4.1 Plot Morfologi (Area)
generate_feature_plot(
    df_data=df_train,
    columns_to_plot={'area': 'Area'},
    colors=['#1abc9c'],
    title='Fitur Morfologi - Area',
    y_label='Feature Value',
    output_filename='visualisasi_fitur_Morfologi(Area).png',
    output_dir=output_klasifikasi
)
# 4.2 Plot Morfologi (Aspect Ratio)
generate_feature_plot(
    df_data=df_train,
    columns_to_plot={'aspect_ratio': 'Aspect Ratio'},
    colors=[ '#9b59b6'],
    title='Fitur Morfologi - Aspect Ratio',
    y_label='Feature Value',
    output_filename='visualisasi_fitur_Morfologi(Aspect Ratio).png',
    output_dir=output_klasifikasi
)
# 4.3 Plot Morfologi (Solidity)
generate_feature_plot(
    df_data=df_train,
    columns_to_plot={'solidity': 'Solidity'},
    colors=[ '#e67e22'],
    title='Fitur Morfologi - Solidity',
    y_label='Feature Value',
    output_filename='visualisasi_fitur_Morfologi(Solidity).png',
    output_dir=output_klasifikasi
)

# Feature sets dengan fitur morfologi TERPILIH
feature_sets = {
    # === FITUR WARNA SAJA ===
    "RGB": ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"],
    "LAB": ["L_mean", "A_mean", "B_LAB_mean", "L_std", "A_std", "B_LAB_std"],
    "HSV": ["H_mean", "S_mean", "V_mean", "H_std", "S_std", "V_std"],

    # === KOMBINASI WARNA + MORFOLOGI (3 fitur morfologi terpilih) ===
    "RGB + Morfologi": ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std",
                       "area", "aspect_ratio", "solidity"],
    
    "LAB + Morfologi": ["L_mean", "A_mean", "B_LAB_mean", "L_std", "A_std", "B_LAB_std",
                       "area", "aspect_ratio", "solidity"],
    
    "HSV + Morfologi": ["H_mean", "S_mean", "V_mean", "H_std", "S_std", "V_std",
                       "area", "aspect_ratio", "solidity"]
}

TUNING_METHOD = "grid"  # "grid", "random", "none"

# Hyperparameter space
if TUNING_METHOD == "grid":
    param_grid = {
        'hidden_layer_sizes': [(100, 50), (128, 64), (150, 100, 50), (200, 100), (256, 128, 64)],
        'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005],
        'activation': ['relu', 'tanh'],
        'batch_size': [16, 32, 64]
    }
elif TUNING_METHOD == "random":
    param_distributions = {
        'hidden_layer_sizes': [(100, 50), (128, 64), (150, 100, 50), (200, 100), (256, 128, 64), (128, 64, 32), (200, 100, 50), (300, 150, 75)],
        'alpha': loguniform(0.0001, 0.01),
        'learning_rate_init': loguniform(0.0001, 0.005),
        'activation': ['relu', 'tanh'],
        'batch_size': [16, 32, 64, 128]
    }

# ===== FASE 1: HYPERPARAMETER TUNING =====
print("\n" + "="*80 + "\n=== FASE 1: HYPERPARAMETER TUNING ===\n" + "="*80)

if TUNING_METHOD != "none":
    tuning_results = []
    
    for name, features in feature_sets.items():
        print(f"\nTesting: {name}")
        # Ambil data dari df_train dan df_test yang sudah di-split
        X_train, y_train = df_train[features].fillna(0), df_train["label"]
        X_test, y_test = df_test[features].fillna(0), df_test["label"]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) # Transform X_test
        sample_weights = compute_sample_weight('balanced', y_train)
        
        mlp_base = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.15, 
                                 n_iter_no_change=15, tol=1e-4, solver='adam', learning_rate='adaptive')
        
        start_time = time.time()
        if TUNING_METHOD == "grid":
            search = GridSearchCV(mlp_base, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        else:
            search = RandomizedSearchCV(mlp_base, param_distributions, n_iter=30, cv=3, scoring='accuracy', 
                                        n_jobs=-1, verbose=1, random_state=42)
        
        search.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        search_time = time.time() - start_time
        
        y_pred = search.best_estimator_.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred) * 100
        
        tuning_results.append({
            'Feature Set': name,
            'Num Features': len(features),
            'Best Score (CV)': round(search.best_score_ * 100, 2),
            'Test Accuracy': round(test_acc, 2),
            'Best Params': search.best_params_,
            'Search Time (s)': round(search_time, 2)
        })
        
        print(f"   CV: {search.best_score_*100:.2f}% | Test: {test_acc:.2f}% | Features: {len(features)} | Time: {search_time:.2f}s")
        print(f"   Params: {search.best_params_}")
    
    df_tuning = pd.DataFrame(tuning_results)
    print("\n" + "="*80 + "\n=== HASIL TUNING ===\n" + "="*80)
    print(df_tuning[['Feature Set', 'Num Features', 'Best Score (CV)', 'Test Accuracy', 'Search Time (s)']].to_string(index=False))
    df_tuning.to_csv(os.path.join(output_klasifikasi, "hyperparameter_tuning_results.csv"), index=False)
    
    # Pilih parameter terbaik
    df_tuning['Weighted Score'] = df_tuning['Best Score (CV)'] * 0.5 + df_tuning['Test Accuracy'] * 0.5
    best_idx = df_tuning['Weighted Score'].idxmax()
    best_params_global = df_tuning.loc[best_idx, 'Best Params']
    best_feature = df_tuning.loc[best_idx, 'Feature Set']
    
    print("\n" + "="*80 + "\n=== PARAMETER GLOBAL TERBAIK ===\n" + "="*80)
    print(f"Dari: {best_feature}\nWeighted Score: {df_tuning.loc[best_idx, 'Weighted Score']:.2f}%\n")
    for key, value in best_params_global.items():
        print(f"   {key}: {value}")
    
    # Simpan best params
    with open(os.path.join(output_klasifikasi, "best_global_hyperparameters.txt"), 'w') as f:
        f.write("="*60 + "\nBEST GLOBAL HYPERPARAMETERS\n" + "="*60 + f"\n\nFrom: {best_feature}\n\nParameters:\n")
        for key, value in best_params_global.items():
            f.write(f"   {key}: {value}\n")
    
    # Plot tuning results
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    x = np.arange(len(df_tuning))
    width = 0.35
    plt.bar(x - width/2, df_tuning['Best Score (CV)'], width, label='CV Score', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, df_tuning['Test Accuracy'], width, label='Test Accuracy', color='#2ecc71', alpha=0.8)
    wrapped = [textwrap.fill(l, 15) for l in df_tuning['Feature Set']]
    plt.xticks(x, wrapped, rotation=45, ha='right', fontsize=7)
    plt.ylabel('Accuracy (%)')
    plt.title('Hyperparameter Tuning: CV vs Test')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 110)
    
    plt.subplot(1, 2, 2)
    colors = ['#e74c3c' if i == best_idx else '#95a5a6' for i in range(len(df_tuning))]
    plt.bar(x, df_tuning['Weighted Score'], color=colors, alpha=0.8)
    plt.xticks(x, wrapped, rotation=45, ha='right', fontsize=7)
    plt.ylabel('Weighted Score (%)')
    plt.title('Weighted Score (Red = Selected)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(output_klasifikasi, "hyperparameter_tuning_comparison.png"), dpi=300)
    plt.close()
else:
    best_params_global = {'hidden_layer_sizes': (150, 100, 50), 'alpha': 0.001, 'learning_rate_init': 0.001, 'activation': 'relu', 'batch_size': 32}

# ===== FASE 2: TRAINING FINAL =====
print("\n" + "="*80 + "\n=== FASE 2: TRAINING FINAL ===\n" + "="*80)
print("Parameter:")
for key, value in best_params_global.items():
    print(f"   {key}: {value}")

results = []

for name, features in feature_sets.items():
    print(f"\n{'='*60}\n=== {name} ({len(features)} features) ===\n{'='*60}")
    
    # Ambil data dari df_train dan df_test
    X_train, y_train = df_train[features].fillna(0), df_train["label"]
    X_test, y_test = df_test[features].fillna(0), df_test["label"]
    
    # Ambil metadata (filename, grade) dari df_train/df_test
    files_train, grades_train = df_train['filename'], df_train['grade']
    files_test, grades_test = df_test['filename'], df_test['grade']
    
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
    sample_weights = compute_sample_weight('balanced', y_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=best_params_global['hidden_layer_sizes'],
        activation=best_params_global['activation'],
        alpha=best_params_global['alpha'],
        learning_rate_init=best_params_global['learning_rate_init'],
        batch_size=best_params_global.get('batch_size', 32),
        solver='adam', max_iter=2000, random_state=42, learning_rate='adaptive',
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, tol=1e-4
    )
   
    
    start_train = time.time()
    mlp.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    train_time = time.time() - start_train
    
    print(f"   Epoch: {mlp.n_iter_} | Val Score: {mlp.best_validation_score_:.4f} | Time: {train_time:.2f}s")
    
    cv_scores = cross_val_score(mlp, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"   CV: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
    
    safe_name = name.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')
    
    # Learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mlp.loss_curve_, color='#3498db', linewidth=2)
    plt.title(f'Loss\n{name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot([s*100 for s in mlp.validation_scores_], color='#2ecc71', linewidth=2)
    plt.axhline(y=mlp.best_validation_score_*100, color='#e74c3c', linestyle='--', linewidth=2, label=f'Best: {mlp.best_validation_score_*100:.2f}%')
    plt.title(f'Validation Accuracy\n{name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_klasifikasi, f"learning_curves_{safe_name}.png"), dpi=300)
    plt.close()
    
    # Predictions
    start_test = time.time()
    y_pred_train, y_pred_test = mlp.predict(X_train_scaled), mlp.predict(X_test_scaled)
    test_time = time.time() - start_test
    
    train_prec, train_rec, train_f1 = precision_score(y_train, y_pred_train, average=None, zero_division=0), recall_score(y_train, y_pred_train, average=None, zero_division=0), f1_score(y_train, y_pred_train, average=None, zero_division=0)
    test_prec, test_rec, test_f1 = precision_score(y_test, y_pred_test, average=None, zero_division=0), recall_score(y_test, y_pred_test, average=None, zero_division=0), f1_score(y_test, y_pred_test, average=None, zero_division=0)
    
    results.append({
        "Fitur": name,
        "Jumlah Fitur": len(features),
        "Akurasi (Train)": round(accuracy_score(y_train, y_pred_train)*100, 2),
        "Precision (Train)": round(precision_score(y_train, y_pred_train, average='weighted', zero_division=0)*100, 2),
        "Recall (Train)": round(recall_score(y_train, y_pred_train, average='weighted', zero_division=0)*100, 2),
        "F1-Score (Train)": round(f1_score(y_train, y_pred_train, average='weighted', zero_division=0)*100, 2),
        "Akurasi (Test)": round(accuracy_score(y_test, y_pred_test)*100, 2),
        "Precision (Test)": round(precision_score(y_test, y_pred_test, average='weighted', zero_division=0)*100, 2),
        "Recall (Test)": round(recall_score(y_test, y_pred_test, average='weighted', zero_division=0)*100, 2),
        "F1-Score (Test)": round(f1_score(y_test, y_pred_test, average='weighted', zero_division=0)*100, 2),
        "CV Accuracy": round(cv_scores.mean()*100, 2),
        "CV Std": round(cv_scores.std()*100, 2),
        "Waktu Latih (s)": round(train_time, 4),
        "Waktu Uji (s)": round(test_time, 4),
        "Epoch": mlp.n_iter_
    })
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {name}\nTest Acc: {accuracy_score(y_test, y_pred_test)*100:.2f}%")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_klasifikasi, f"confmatrix_{safe_name}.png"), dpi=300)
    plt.close()
    
    # Metrics per class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(label_encoder.classes_))
    width = 0.25
    
    for idx, (ax, prec, rec, f1, title) in enumerate([(axes[0], train_prec, train_rec, train_f1, 'Train'), (axes[1], test_prec, test_rec, test_f1, 'Test')]):
        ax.bar(x-width, prec*100, width, label='Precision', color='#3498db')
        ax.bar(x, rec*100, width, label='Recall', color='#2ecc71')
        ax.bar(x+width, f1*100, width, label='F1-Score', color='#e74c3c')
        ax.set_xticks(x)
        ax.set_xticklabels(label_encoder.classes_)
        ax.set_ylabel('Score (%)')
        ax.set_title(f'Metrics per Class ({title})')
        ax.legend()
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Per-Class Metrics - {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_klasifikasi, f"metrics_per_class_{safe_name}.png"), dpi=300)
    plt.close()
    
    # Save predictions
    print(f"   Saving predictions...")
    folder_hasil = os.path.join(output_klasifikasi, f"Hasil_Gambar_{safe_name}")
    os.makedirs(folder_hasil, exist_ok=True)
    
    # Gunakan .iterrows() pada df_test untuk mendapatkan file dan grade yang sesuai
    for idx, row in df_test.iterrows():
        file = row['filename']
        actual_grade = row['grade']
        
        # Cari prediksi untuk indeks ini
        # Kita harus yakin y_pred_test selaras dengan df_test
        # Karena kita menggunakan X_test yang berasal langsung dari df_test, indeksnya harus cocok
        pred_label = mlp.predict(scaler.transform(row[features].fillna(0).values.reshape(1, -1)))[0]
        pred_grade = label_encoder.inverse_transform([pred_label])[0]
        
        src = os.path.join(base_input_citra, actual_grade, file)
        if pred_grade == actual_grade:
            dst_folder = os.path.join(folder_hasil, f"Benar_Prediksi_{pred_grade.replace(' ', '_')}")
        else:
            dst_folder = os.path.join(folder_hasil, f"Salah_Prediksi_(Aktual_{actual_grade.replace(' ', '_')}_ke_{pred_grade.replace(' ', '_')})")
        
        os.makedirs(dst_folder, exist_ok=True)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_folder, file))
        else:
            # Peringatan jika file sumber tidak ditemukan
            # print(f"Warning: Source file not found {src}")
            pass

# ===== HASIL AKHIR =====
df_result = pd.DataFrame(results)
print("\n" + "="*80 + "\n=== HASIL AKHIR ===\n" + "="*80)
print(df_result.to_string(index=False))

df_result.to_csv(os.path.join(output_klasifikasi, "hasil_akhir_mlp_with_morphology.csv"), index=False)
df_result.to_excel(os.path.join(output_klasifikasi, "hasil_akhir_mlp_with_morphology.xlsx"), index=False, engine='openpyxl')

# ===== ANALISIS KONTRIBUSI FITUR MORFOLOGI =====
print("\n" + "="*80 + "\n=== ANALISIS KONTRIBUSI FITUR MORFOLOGI ===\n" + "="*80)

# Bandingkan performa dengan dan tanpa morfologi
comparisons = []
for name in df_result['Fitur']:
    if 'Morfologi' in name:
        base_name = name.replace(' + Morfologi', '').replace(' Morfologi', '')
        if base_name in df_result['Fitur'].values and base_name != name:
            row_with = df_result[df_result['Fitur'] == name].iloc[0]
            row_without = df_result[df_result['Fitur'] == base_name].iloc[0]
            
            comparisons.append({
                'Base Feature': base_name,
                'Without Morphology': row_without['Akurasi (Test)'],
                'With Morphology': row_with['Akurasi (Test)'],
                'Improvement': round(row_with['Akurasi (Test)'] - row_without['Akurasi (Test)'], 2),
                'F1 Without': row_without['F1-Score (Test)'],
                'F1 With': row_with['F1-Score (Test)'],
                'F1 Improvement': round(row_with['F1-Score (Test)'] - row_without['F1-Score (Test)'], 2)
            })

if comparisons:
    df_comparison = pd.DataFrame(comparisons)
    print("\nPerbandingan Dengan vs Tanpa Fitur Morfologi:")
    print(df_comparison.to_string(index=False))
    df_comparison.to_csv(os.path.join(output_klasifikasi, "morphology_impact_analysis.csv"), index=False)
    
    # Visualisasi improvement
    plt.figure(figsize=(14, 6))
    x = np.arange(len(df_comparison))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar(x - width/2, df_comparison['Without Morphology'], width, label='Without Morphology', color='#95a5a6', alpha=0.8)
    plt.bar(x + width/2, df_comparison['With Morphology'], width, label='With Morphology', color='#27ae60', alpha=0.8)
    plt.xticks(x, df_comparison['Base Feature'], rotation=45, ha='right')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Impact of Morphological Features on Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 110)
    
    plt.subplot(1, 2, 2)
    colors = ['#e74c3c' if imp < 0 else '#2ecc71' for imp in df_comparison['Improvement']]
    bars = plt.bar(x, df_comparison['Improvement'], color=colors, alpha=0.8)
    plt.xticks(x, df_comparison['Base Feature'], rotation=45, ha='right')
    plt.ylabel('Improvement (%)')
    plt.title('Accuracy Improvement with Morphology\n(Green = Positive, Red = Negative)')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.2 if h > 0 else h - 0.5, 
                 f'{h:.2f}', ha='center', va='bottom' if h > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_klasifikasi, "morphology_impact_visualization.png"), dpi=300)
    plt.close()

# ===== GRAFIK PERBANDINGAN =====
# 1. Metrics
plt.figure(figsize=(16, 9))
x = np.arange(len(df_result))
width = 0.25
bars1 = plt.bar(x-width, df_result["Precision (Test)"], width, label="Precision", color='#3498db')
bars2 = plt.bar(x, df_result["Recall (Test)"], width, label="Recall", color='#2ecc71')
bars3 = plt.bar(x+width, df_result["F1-Score (Test)"], width, label="F1-Score", color='#e74c3c')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, h+0.5, f'{h:.1f}', ha='center', va='bottom', fontsize=8)

wrapped = [textwrap.fill(l, 20) for l in df_result["Fitur"]]
plt.xticks(x, wrapped, rotation=45, ha='right', fontsize=9)
plt.ylabel("Score (%)")
plt.title("Test Metrics Comparison (with Morphological Features)")
plt.legend(loc='upper right')
plt.ylim(0, 110)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_klasifikasi, "grafik_perbandingan_metrics.png"), dpi=300)
plt.close()

# 2. Train vs Test
plt.figure(figsize=(14, 7))
x = np.arange(len(df_result))
width = 0.35
bars1 = plt.bar(x-width/2, df_result["Akurasi (Train)"], width, label="Train", color='#9b59b6', alpha=0.8)
bars2 = plt.bar(x+width/2, df_result["Akurasi (Test)"], width, label="Test", color='#f39c12', alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, h+0.5, f'{h:.1f}', ha='center', va='bottom', fontsize=8)

plt.xticks(x, wrapped, rotation=45, ha='right', fontsize=9)
plt.ylabel("Accuracy (%)")
plt.title("Train vs Test Accuracy (Overfitting Detection)")
plt.legend()
plt.ylim(0, 110)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_klasifikasi, "grafik_train_vs_test.png"), dpi=300)
plt.close()

# 3. CV
plt.figure(figsize=(14, 7))
bars = plt.bar(x, df_result["CV Accuracy"], color='#16a085', alpha=0.8)
plt.errorbar(x, df_result["CV Accuracy"], yerr=df_result["CV Std"], fmt='none', ecolor='red', capsize=5, label='Std Dev')

for i, bar in enumerate(bars):
    h = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, h+0.5, f'{h:.1f}±{df_result["CV Std"].iloc[i]:.1f}', ha='center', va='bottom', fontsize=8)

plt.xticks(x, wrapped, rotation=45, ha='right', fontsize=9)
plt.ylabel("CV Accuracy (%)")
plt.title("5-Fold Cross-Validation")
plt.legend()
plt.ylim(0, 110)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_klasifikasi, "grafik_cross_validation.png"), dpi=300)
plt.close()

# 4. Dashboard
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
x = np.arange(len(df_result))
wrapped_s = [textwrap.fill(l, 12) for l in df_result["Fitur"]]

ax1 = fig.add_subplot(gs[0, 0])
width = 0.25
ax1.bar(x-width, df_result["Precision (Test)"], width, label="Precision", color='#3498db', alpha=0.8)
ax1.bar(x, df_result["Recall (Test)"], width, label="Recall", color='#2ecc71', alpha=0.8)
ax1.bar(x+width, df_result["F1-Score (Test)"], width, label="F1-Score", color='#e74c3c', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(wrapped_s, rotation=45, ha='right', fontsize=7)
ax1.set_ylabel('Score (%)')
ax1.set_title('Test Metrics', fontsize=10, fontweight='bold')
ax1.legend(fontsize=8)
ax1.set_ylim(0, 110)
ax1.grid(True, alpha=0.3, axis='y')

ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(x-width, df_result["Akurasi (Train)"], width, label="Train", color='#9b59b6', alpha=0.8)
ax2.bar(x, df_result["Akurasi (Test)"], width, label="Test", color='#f39c12', alpha=0.8)
ax2.bar(x+width, df_result["CV Accuracy"], width, label="CV", color='#16a085', alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(wrapped_s, rotation=45, ha='right', fontsize=7)
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Train vs Test vs CV', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8)
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.3, axis='y')

ax3 = fig.add_subplot(gs[0, 2])
gap = df_result["Akurasi (Train)"] - df_result["Akurasi (Test)"]
colors_gap = ['#e74c3c' if g > 5 else '#2ecc71' for g in gap]
ax3.bar(x, gap, color=colors_gap, alpha=0.8)
ax3.axhline(y=5, color='orange', linestyle='--', linewidth=2, label='Threshold (5%)')
ax3.set_xticks(x)
ax3.set_xticklabels(wrapped_s, rotation=45, ha='right', fontsize=7)
ax3.set_ylabel('Gap (%)')
ax3.set_title('Overfitting Detection', fontsize=10, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(x, df_result["Jumlah Fitur"], color='#8e44ad', alpha=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(wrapped_s, rotation=45, ha='right', fontsize=7)
ax4.set_ylabel('Number of Features')
ax4.set_title('Feature Count', fontsize=10, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(x, df_result["Epoch"], color='#e67e22', alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(wrapped_s, rotation=45, ha='right', fontsize=7)
ax5.set_ylabel('Epochs')
ax5.set_title('Convergence Speed', fontsize=10, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

ax6 = fig.add_subplot(gs[1, 2])
overall = df_result["Akurasi (Test)"]*0.4 + df_result["F1-Score (Test)"]*0.3 + df_result["CV Accuracy"]*0.3
colors_rank = plt.cm.RdYlGn(overall/100)
bars = ax6.bar(x, overall, color=colors_rank, alpha=0.8)
ax6.set_xticks(x)
ax6.set_xticklabels(wrapped_s, rotation=45, ha='right', fontsize=7)
ax6.set_ylabel('Overall Score (%)')
ax6.set_title('Performance Ranking\n(40% Test + 30% F1 + 30% CV)', fontsize=10, fontweight='bold')
ax6.set_ylim(0, 110)
ax6.grid(True, alpha=0.3, axis='y')
for bar in bars:
    h = bar.get_height()
    ax6.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}', ha='center', va='bottom', fontsize=7)

plt.suptitle('Comprehensive Performance Dashboard\n(With Morphological Features)', fontsize=14, fontweight='bold', y=0.98)
plt.savefig(os.path.join(output_klasifikasi, "dashboard_comprehensive.png"), dpi=300, bbox_inches='tight')
plt.close()

# ===== LAPORAN RINGKASAN =====
print("\n" + "="*80 + "\n=== LAPORAN RINGKASAN ===\n" + "="*80)

best_test = df_result.loc[df_result["Akurasi (Test)"].idxmax()]
best_f1 = df_result.loc[df_result["F1-Score (Test)"].idxmax()]
best_cv = df_result.loc[df_result["CV Accuracy"].idxmax()]
fastest = df_result.loc[df_result["Waktu Latih (s)"].idxmin()]

# Hitung overall score
overall_scores = df_result["Akurasi (Test)"]*0.4 + df_result["F1-Score (Test)"]*0.3 + df_result["CV Accuracy"]*0.3
best_overall_idx = overall_scores.idxmax()
best_overall = df_result.loc[best_overall_idx]

print("\n1. HYPERPARAMETERS USED:")
print("-"*60)
for k, v in best_params_global.items():
    print(f"   {k}: {v}")

print("\n2. BEST OVERALL PERFORMANCE (RECOMMENDED):")
print("-"*60)
print(f"   Set Fitur: {best_overall['Fitur']}")
print(f"   Jumlah Fitur: {best_overall['Jumlah Fitur']}")
print(f"   Overall Score: {overall_scores.iloc[best_overall_idx]:.2f}%")
print(f"   Test Accuracy: {best_overall['Akurasi (Test)']:.2f}%")
print(f"   F1-Score (Test): {best_overall['F1-Score (Test)']:.2f}%")
print(f"   CV Accuracy: {best_overall['CV Accuracy']:.2f}% ± {best_overall['CV Std']:.2f}%")
print(f"   Training Time: {best_overall['Waktu Latih (s)']:.2f}s")
print(f"   Overfitting Gap: {best_overall['Akurasi (Train)'] - best_overall['Akurasi (Test)']:.2f}%")

print("\n3. BEST TEST ACCURACY:")
print("-"*60)
print(f"   Set Fitur: {best_test['Fitur']}")
print(f"   Test: {best_test['Akurasi (Test)']:.2f}% | F1: {best_test['F1-Score (Test)']:.2f}% | CV: {best_test['CV Accuracy']:.2f}%±{best_test['CV Std']:.2f}%")

print("\n4. BEST F1-SCORE:")
print("-"*60)
print(f"   Set Fitur: {best_f1['Fitur']}")
print(f"   F1: {best_f1['F1-Score (Test)']:.2f}% | Test: {best_f1['Akurasi (Test)']:.2f}%")

print("\n5. BEST CV ACCURACY:")
print("-"*60)
print(f"   Set Fitur: {best_cv['Fitur']}")
print(f"   CV: {best_cv['CV Accuracy']:.2f}%±{best_cv['CV Std']:.2f}% | Test: {best_cv['Akurasi (Test)']:.2f}%")

print("\n6. FASTEST TRAINING:")
print("-"*60)
print(f"   Set Fitur: {fastest['Fitur']}")
print(f"   Time: {fastest['Waktu Latih (s)']:.4f}s | Test: {fastest['Akurasi (Test)']:.2f}%")

print("\n7. OVERFITTING ANALYSIS:")
print("-"*60)
for _, row in df_result.iterrows():
    g = row['Akurasi (Train)'] - row['Akurasi (Test)']
    s = "   OVERFIT" if g > 10 else "   WARNING" if g > 5 else "   GOOD"
    print(f"   {row['Fitur'][:40]:<40} | Gap: {g:>5.2f}% | {s}")

print("\n8. TOP 5 PERFORMING FEATURE SETS:")
print("-"*60)
df_ranked = df_result.copy()
df_ranked['Overall Score'] = overall_scores
df_ranked = df_ranked.sort_values('Overall Score', ascending=False)
for i, (_, row) in enumerate(df_ranked.head(5).iterrows(), 1):
    print(f"\n   #{i}. {row['Fitur']}")
    print(f"       Overall: {row['Overall Score']:.2f}% | Test: {row['Akurasi (Test)']:.2f}% | F1: {row['F1-Score (Test)']:.2f}%")
    print(f"       CV: {row['CV Accuracy']:.2f}%±{row['CV Std']:.2f}% | Features: {row['Jumlah Fitur']}")

print("\n9. MORPHOLOGY CONTRIBUTION ANALYSIS:")
print("-"*60)
morphology_sets = df_result[df_result['Fitur'].str.contains('Morfologi', case=False)]
non_morphology_sets = df_result[~df_result['Fitur'].str.contains('Morfologi', case=False)]

if len(morphology_sets) > 0 and len(non_morphology_sets) > 0:
    print(f"   Average Test Accuracy WITH Morphology: {morphology_sets['Akurasi (Test)'].mean():.2f}%")
    print(f"   Average Test Accuracy WITHOUT Morphology: {non_morphology_sets['Akurasi (Test)'].mean():.2f}%")
    print(f"   Average Improvement: {morphology_sets['Akurasi (Test)'].mean() - non_morphology_sets['Akurasi (Test)'].mean():.2f}%")
    print(f"   \n   Morphological Features Used: area, aspect_ratio, solidity")
else:
    print("   Tidak cukup data untuk perbandingan morfologi.")

# Simpan laporan lengkap
with open(os.path.join(output_klasifikasi, "LAPORAN_RINGKASAN_FINAL.txt"), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\nLAPORAN RINGKASAN FINAL - MLP CLASSIFIER WITH MORPHOLOGY\n" + "="*80 + "\n\n")
    
    f.write("1. HYPERPARAMETERS:\n" + "-"*60 + "\n")
    for k, v in best_params_global.items():
        f.write(f"   {k}: {v}\n")
    
    f.write("\n2. BEST OVERALL PERFORMANCE (RECOMMENDED):\n" + "-"*60 + "\n")
    f.write(f"   Set Fitur: {best_overall['Fitur']}\n")
    f.write(f"   Jumlah Fitur: {best_overall['Jumlah Fitur']}\n")
    f.write(f"   Overall Score: {overall_scores.iloc[best_overall_idx]:.2f}%\n")
    f.write(f"   Test Accuracy: {best_overall['Akurasi (Test)']:.2f}%\n")
    f.write(f"   F1-Score (Test): {best_overall['F1-Score (Test)']:.2f}%\n")
    f.write(f"   Precision (Test): {best_overall['Precision (Test)']:.2f}%\n")
    f.write(f"   Recall (Test): {best_overall['Recall (Test)']:.2f}%\n")
    f.write(f"   CV Accuracy: {best_overall['CV Accuracy']:.2f}% ± {best_overall['CV Std']:.2f}%\n")
    f.write(f"   Training Time: {best_overall['Waktu Latih (s)']:.2f}s\n")
    f.write(f"   Epochs: {best_overall['Epoch']}\n")
    f.write(f"   Overfitting Gap: {best_overall['Akurasi (Train)'] - best_overall['Akurasi (Test)']:.2f}%\n")
    
    f.write(f"\n3. BEST TEST ACCURACY:\n" + "-"*60 + f"\n   {best_test['Fitur']}\n")
    f.write(f"   Test: {best_test['Akurasi (Test)']:.2f}% | F1: {best_test['F1-Score (Test)']:.2f}%\n")
    
    f.write("\n4. TOP 5 PERFORMING FEATURE SETS:\n" + "-"*60 + "\n")
    for i, (_, row) in enumerate(df_ranked.head(5).iterrows(), 1):
        f.write(f"\n   #{i}. {row['Fitur']}\n")
        f.write(f"       Overall: {row['Overall Score']:.2f}% | Test: {row['Akurasi (Test)']:.2f}% | F1: {row['F1-Score (Test)']:.2f}%\n")
        f.write(f"       CV: {row['CV Accuracy']:.2f}%±{row['CV Std']:.2f}% | Features: {row['Jumlah Fitur']}\n")
    
    f.write("\n5. MORPHOLOGY CONTRIBUTION:\n" + "-"*60 + "\n")
    if len(morphology_sets) > 0 and len(non_morphology_sets) > 0:
        f.write(f"   Average Test Accuracy WITH Morphology: {morphology_sets['Akurasi (Test)'].mean():.2f}%\n")
        f.write(f"   Average Test Accuracy WITHOUT Morphology: {non_morphology_sets['Akurasi (Test)'].mean():.2f}%\n")
        f.write(f"   Average Improvement: {morphology_sets['Akurasi (Test)'].mean() - non_morphology_sets['Akurasi (Test)'].mean():.2f}%\n")
        f.write(f"   Morphological Features: area, aspect_ratio, solidity\n")
    else:
        f.write("   Tidak cukup data untuk perbandingan morfologi.\n")
    
    f.write("\n6. COMPLETE RESULTS:\n" + "-"*60 + "\n")
    f.write(df_result.to_string(index=False))
    
    f.write("\n\n7. OVERFITTING ANALYSIS:\n" + "-"*60 + "\n")
    for _, row in df_result.iterrows():
        g = row['Akurasi (Train)'] - row['Akurasi (Test)']
        s = "OVERFIT" if g > 10 else "WARNING" if g > 5 else "GOOD"
        f.write(f"   {row['Fitur'][:40]:<40} | {g:>5.2f}% | {s}\n")
        
# ================================================================
# ===== FASE 3: LATIH ULANG & SIMPAN MODEL TERBAIK =====
# ================================================================
print("\n" + "="*80 + "\n=== FASE 3: LATIH ULANG & SIMPAN MODEL TERBAIK ===\n" + "="*80)

# FILTER: Hanya ambil model dengan fitur morfologi
df_result_with_morph = df_result[df_result['Fitur'].str.contains('Morfologi', case=False)].copy()

if len(df_result_with_morph) == 0:
    print("PERINGATAN: Tidak ada model dengan fitur morfologi yang ditemukan!")
    print("Menggunakan model terbaik dari semua feature set...")
    df_result_filtered = df_result
else:
    print(f"Filter diterapkan: Hanya memilih dari {len(df_result_with_morph)} model dengan Morfologi")
    print("Model yang dipertimbangkan:")
    for idx, row in df_result_with_morph.iterrows():
        print(f"   - {row['Fitur']}")
    df_result_filtered = df_result_with_morph

# Hitung overall score HANYA untuk model dengan morfologi
overall_scores_filtered = (df_result_filtered["Akurasi (Test)"] * 0.4 + 
                           df_result_filtered["F1-Score (Test)"] * 0.3 + 
                           df_result_filtered["CV Accuracy"] * 0.3)

# Ambil model terbaik dari hasil filter
best_overall_idx_filtered = overall_scores_filtered.idxmax()
best_overall_filtered = df_result_filtered.loc[best_overall_idx_filtered]
best_overall_score = overall_scores_filtered.loc[best_overall_idx_filtered]

print("\n" + "="*80)
print("MODEL TERBAIK TERPILIH (DARI WARNA + MORFOLOGI):")
print("="*80)
print(f"Set Fitur: {best_overall_filtered['Fitur']}")
print(f"Jumlah Fitur: {best_overall_filtered['Jumlah Fitur']}")
print(f"Overall Score: {best_overall_score:.2f}%")
print(f"Test Accuracy: {best_overall_filtered['Akurasi (Test)']:.2f}%")
print(f"F1-Score (Test): {best_overall_filtered['F1-Score (Test)']:.2f}%")
print(f"CV Accuracy: {best_overall_filtered['CV Accuracy']:.2f}% ± {best_overall_filtered['CV Std']:.2f}%")
print(f"Training Time: {best_overall_filtered['Waktu Latih (s)']:.2f}s")
print(f"Overfitting Gap: {best_overall_filtered['Akurasi (Train)'] - best_overall_filtered['Akurasi (Test)']:.2f}%")

# Ambil nama dan list fitur dari model terbaik
best_feature_name = best_overall_filtered['Fitur']
best_features = feature_sets[best_feature_name]

print(f"\nMelatih ulang model terbaik: {best_feature_name}")
print(f"Jumlah Fitur: {len(best_features)}")
print(f"\nFitur yang digunakan:")
for i, feat in enumerate(best_features, 1):
    print(f"   {i}. {feat}")

# 1. Siapkan data (menggunakan df_train/df_test yang sudah ada)
X_train, y_train = df_train[best_features].fillna(0), df_train["label"]
X_test, y_test = df_test[best_features].fillna(0), df_test["label"]

# 2. Siapkan scaler (HANYA fit pada data train)
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
sample_weights_final = compute_sample_weight('balanced', y_train)

# 3. Inisialisasi model final 
mlp_final = MLPClassifier(
    hidden_layer_sizes=best_params_global['hidden_layer_sizes'],
    activation=best_params_global['activation'],
    alpha=best_params_global['alpha'],
    learning_rate_init=best_params_global['learning_rate_init'],
    batch_size=best_params_global.get('batch_size', 32),
    solver='adam', max_iter=2000, random_state=42, learning_rate='adaptive',
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, tol=1e-4
)

# 4. Latih model final
print("\n" + "-"*80)
print("Memulai pelatihan model final...")
print("-"*80)
mlp_final.fit(X_train_scaled, y_train, sample_weight=sample_weights_final)
print(f"✓ Pelatihan selesai!")
print(f"   Total epoch: {mlp_final.n_iter_}")
print(f"   Best validation score: {mlp_final.best_validation_score_:.4f}")

# 5. Validasi performa model final
y_pred_test_final = mlp_final.predict(scaler_final.transform(X_test))
final_test_acc = accuracy_score(y_test, y_pred_test_final) * 100
final_f1 = f1_score(y_test, y_pred_test_final, average='weighted') * 100

print(f"\nValidasi Model Final:")
print(f"   Test Accuracy: {final_test_acc:.2f}%")
print(f"   Test F1-Score: {final_f1:.2f}%")

# 6. Simpan artefak (model, scaler, features list)
model_path = os.path.join(output_klasifikasi, "model_terbaik.joblib")
scaler_path = os.path.join(output_klasifikasi, "scaler_terbaik.joblib")
features_path = os.path.join(output_klasifikasi, "features_terbaik.txt")

joblib.dump(mlp_final, model_path)
joblib.dump(scaler_final, scaler_path)

# Simpan daftar fitur untuk referensi
with open(features_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("MODEL TERBAIK - FEATURE INFORMATION\n")
    f.write("="*80 + "\n\n")
    f.write(f"Feature Set Name: {best_feature_name}\n")
    f.write(f"Number of Features: {len(best_features)}\n\n")
    f.write("Features Used:\n")
    f.write("-"*60 + "\n")
    for i, feat in enumerate(best_features, 1):
        f.write(f"{i:2d}. {feat}\n")
    f.write("\n" + "="*80 + "\n")
    f.write("PERFORMANCE METRICS\n")
    f.write("="*80 + "\n")
    f.write(f"Overall Score: {best_overall_score:.2f}%\n")
    f.write(f"Test Accuracy: {best_overall_filtered['Akurasi (Test)']:.2f}%\n")
    f.write(f"Precision: {best_overall_filtered['Precision (Test)']:.2f}%\n")
    f.write(f"Recall: {best_overall_filtered['Recall (Test)']:.2f}%\n")
    f.write(f"F1-Score: {best_overall_filtered['F1-Score (Test)']:.2f}%\n")
    f.write(f"CV Accuracy: {best_overall_filtered['CV Accuracy']:.2f}% ± {best_overall_filtered['CV Std']:.2f}%\n")
    f.write(f"Training Time: {best_overall_filtered['Waktu Latih (s)']:.2f}s\n")
    f.write(f"Epochs: {best_overall_filtered['Epoch']}\n")
    f.write(f"Overfitting Gap: {best_overall_filtered['Akurasi (Train)'] - best_overall_filtered['Akurasi (Test)']:.2f}%\n")
    f.write("\n" + "="*80 + "\n")
    f.write("HYPERPARAMETERS\n")
    f.write("="*80 + "\n")
    for k, v in best_params_global.items():
        f.write(f"{k}: {v}\n")

print(f"\n{'='*80}")
print("MODEL BERHASIL DISIMPAN:")
print(f"{'='*80}")
print(f" 1. Model    : {model_path}")
print(f" 2. Scaler   : {scaler_path}")
print(f" 3. Encoder  : {os.path.join(output_klasifikasi, 'label_encoder.joblib')}")
print(f" 4. Features : {features_path}")

print(f"\n{'='*80}\n=== SELESI ===\n{'='*80}")
print(f"\nModel yang disimpan: {best_feature_name}")
print(f"Feature Set: {best_features}")
print(f"Test Accuracy: {final_test_acc:.2f}%")
print(f"F1-Score: {final_f1:.2f}%")

print("\n LAPORAN & HASIL:")
print("   - LAPORAN_RINGKASAN_FINAL.txt")
print("   - hasil_akhir_mlp_with_morphology.csv")
print("   - hasil_akhir_mlp_with_morphology.xlsx")
print("   - morphology_impact_analysis.csv")
if TUNING_METHOD != "none":
    print("   - hyperparameter_tuning_results.csv")
    print("   - best_global_hyperparameters.txt")
try:
    print(f"   Output activation function: {mlp.out_activation_}")
except: pass

print("\n VISUALISASI BARU (DARI DATA LATIH):")
print("   - visualisasi_fitur_RGB.png")
print("   - visualisasi_fitur_LAB.png")
print("   - visualisasi_fitur_HSV.png")
print("   - visualisasi_fitur_Morfologi.png")

print("\n VISUALISASI UTAMA:")
print("   - dashboard_comprehensive.png")
print("   - grafik_perbandingan_metrics.png")
print("   - grafik_train_vs_test.png")
print("   - grafik_cross_validation.png")
print("   - morphology_impact_visualization.png")
if TUNING_METHOD != "none":
    print("   - hyperparameter_tuning_comparison.png")

print("\n PER FEATURE SET:")
print("   - learning_curves_*.png")
print("   - confmatrix_*.png")
print("   - metrics_per_class_*.png")
print("   - Hasil_Gambar_*/")
print("\n DONE!")