# MLP CLASSIFIER - WITH MORPHOLOGICAL FEATURES 
import pandas as pd
import sys
import numpy as np
import time, os, shutil, textwrap, warnings
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import loguniform
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# Configuration Script
base_input_citra = r"Single Larvae data/Citra2"
csv_file_path = os.path.join(base_input_citra, "folder_features/bwareaopen_features.csv")
output_klasifikasi = os.path.join(base_input_citra, "folder_hasil_klasifikasi_mlp_with_morphology")
os.makedirs(output_klasifikasi, exist_ok=True)

# Read data
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: File CSV tidak ditemukan di: {csv_file_path}")
    sys.exit()

df = df[df["grade"].isin(["Grade A", "Grade B", "Grade C"])]
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["grade"])

# Save encoder label
joblib.dump(label_encoder, os.path.join(output_klasifikasi, "label_encoder.joblib"))
print("LabelEncoder disimpan ke 'label_encoder.joblib'")

print(f"Total data: {len(df)}\nDistribusi:\n{df['grade'].value_counts()}\n")

# Train/Test split 
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

#  Feature sets
feature_sets = {
    "RGB": ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"],
    "LAB": ["L_mean", "A_mean", "B_LAB_mean", "L_std", "A_std", "B_LAB_std"],
    "HSV": ["H_mean", "S_mean", "V_mean", "H_std", "S_std", "V_std"],

    "RGB + Morfologi": ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std",
                        "area", "aspect_ratio", "solidity"],
    "LAB + Morfologi": ["L_mean", "A_mean", "B_LAB_mean", "L_std", "A_std", "B_LAB_std",
                        "area", "aspect_ratio", "solidity"],
    "HSV + Morfologi": ["H_mean", "S_mean", "V_mean", "H_std", "S_std", "V_std",
                        "area", "aspect_ratio", "solidity"]
}

TUNING_METHOD = "grid"  # "grid", "random", "none"

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

#  Phase 1: Hyperparameter Tuning 
print("\n" + "="*80 + "\n=== Phase 1: HYPERPARAMETER TUNING ===\n" + "="*80)

if TUNING_METHOD != "none":
    tuning_results = []
    for name, features in feature_sets.items():
        print(f"\nTesting: {name}")
        X_train, y_train = df_train[features].fillna(0), df_train["label"]
        X_test, y_test   = df_test[features].fillna(0),  df_test["label"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        sample_weights = compute_sample_weight('balanced', y_train)

        mlp_base = MLPClassifier(
            max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, tol=1e-4, solver='adam', learning_rate='adaptive'
        )

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
    print("\n" + "="*80 + "\n=== Result Tuning ===\n" + "="*80)
    print(df_tuning[['Feature Set', 'Num Features', 'Best Score (CV)', 'Test Accuracy', 'Search Time (s)']].to_string(index=False))
    df_tuning.to_csv(os.path.join(output_klasifikasi, "hyperparameter_tuning_results.csv"), index=False)

    # Select the best weighted parameters
    df_tuning['Weighted Score'] = df_tuning['Best Score (CV)'] * 0.5 + df_tuning['Test Accuracy'] * 0.5
    best_idx = df_tuning['Weighted Score'].idxmax()
    best_params_global = df_tuning.loc[best_idx, 'Best Params']
    best_feature = df_tuning.loc[best_idx, 'Feature Set']

    print("\n" + "="*80 + "\n=== Best Global parameter ===\n" + "="*80)
    print(f"Dari: {best_feature}\nWeighted Score: {df_tuning.loc[best_idx, 'Weighted Score']:.2f}%\n")
    for key, value in best_params_global.items():
        print(f"   {key}: {value}")

    with open(os.path.join(output_klasifikasi, "best_global_hyperparameters.txt"), 'w') as f:
        f.write("="*60 + "\nBEST GLOBAL HYPERPARAMETERS\n" + "="*60 + f"\n\nFrom: {best_feature}\n\nParameters:\n")
        for key, value in best_params_global.items():
            f.write(f"   {key}: {value}\n")
else:
    best_params_global = {'hidden_layer_sizes': (150, 100, 50), 'alpha': 0.001, 'learning_rate_init': 0.001, 'activation': 'relu', 'batch_size': 32}

# FASE 2: TRAINING FINAL 
print("\n" + "="*80 + "\n=== Phase 2: Training Final ===\n" + "="*80)
print("Parameter:")
for key, value in best_params_global.items():
    print(f"   {key}: {value}")

results = []

for name, features in feature_sets.items():
    print(f"\n{'='*60}\n=== {name} ({len(features)} features) ===\n{'='*60}")

    X_train, y_train = df_train[features].fillna(0), df_train["label"]
    X_test,  y_test  = df_test[features].fillna(0),  df_test["label"]

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

    # Prediction
    start_test = time.time()
    y_pred_train = mlp.predict(X_train_scaled)
    y_pred_test  = mlp.predict(X_test_scaled)
    test_time = time.time() - start_test

    # Score
    train_prec = precision_score(y_train, y_pred_train, average=None, zero_division=0)
    train_rec  = recall_score(y_train, y_pred_train, average=None, zero_division=0)
    train_f1   = f1_score(y_train, y_pred_train, average=None, zero_division=0)

    test_prec = precision_score(y_test, y_pred_test, average=None, zero_division=0)
    test_rec  = recall_score(y_test, y_pred_test, average=None, zero_division=0)
    test_f1   = f1_score(y_test, y_pred_test, average=None, zero_division=0)

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

    # Save the numerical confusion matrix as a CSV file 
    cm = confusion_matrix(y_test, y_pred_test)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    cm_path = os.path.join(output_klasifikasi, f"confmatrix_numeric_{name.replace(' ', '_').replace('+','plus')}.csv")
    cm_df.to_csv(cm_path)

    # save prediction
    print(f"   Saving predictions ...")
    folder_hasil = os.path.join(output_klasifikasi, f"Hasil_Gambar_{name.replace(' ', '_').replace('+','plus')}")
    os.makedirs(folder_hasil, exist_ok=True)

    for idx, row in df_test.iterrows():
        file = row['filename']
        actual_grade = row['grade']
        # Row-by-row re-prediction for index synchronization
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

# Result
df_result = pd.DataFrame(results)
print("\n" + "="*80 + "\n=== Result ===\n" + "="*80)
print(df_result.to_string(index=False))

df_result.to_csv(os.path.join(output_klasifikasi, "hasil_akhir_mlp_with_morphology.csv"), index=False)
df_result.to_excel(os.path.join(output_klasifikasi, "hasil_akhir_mlp_with_morphology.xlsx"), index=False, engine='openpyxl')

# Analysis of morphological feature contributions 
print("\n" + "="*80 + "\n=== Analysis of morphological feature contributions  ===\n" + "="*80)

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

# Summary Report
print("\n" + "="*80 + "\n=== Summary Report ===\n" + "="*80)

best_test = df_result.loc[df_result["Akurasi (Test)"].idxmax()]
best_f1   = df_result.loc[df_result["F1-Score (Test)"].idxmax()]
best_cv   = df_result.loc[df_result["CV Accuracy"].idxmax()]
fastest   = df_result.loc[df_result["Waktu Latih (s)"].idxmin()]

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

# Save complete report (txt)
with open(os.path.join(output_klasifikasi, "LAPORAN_RINGKASAN_FINAL.txt"), 'w', encoding='utf-8') as f:
    f.write("="*80 + "\nLAPORAN RINGKASAN FINAL - MLP CLASSIFIER WITH MORPHOLOGY (NO-PLOT)\n" + "="*80 + "\n\n")
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


# Phase 3: Retrain & Save the best model
print("\n" + "="*80 + "\n=== FASE 3: LATIH ULANG & SIMPAN MODEL TERBAIK ===\n" + "="*80)

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

overall_scores_filtered = (df_result_filtered["Akurasi (Test)"] * 0.4 +
                           df_result_filtered["F1-Score (Test)"] * 0.3 +
                           df_result_filtered["CV Accuracy"] * 0.3)

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

best_feature_name = best_overall_filtered['Fitur']
best_features = feature_sets[best_feature_name]

print(f"\nMelatih ulang model terbaik: {best_feature_name}")
print(f"Jumlah Fitur: {len(best_features)}")
print(f"\nFitur yang digunakan:")
for i, feat in enumerate(best_features, 1):
    print(f"   {i}. {feat}")

X_train, y_train = df_train[best_features].fillna(0), df_train["label"]
X_test,  y_test  = df_test[best_features].fillna(0),  df_test["label"]

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
sample_weights_final = compute_sample_weight('balanced', y_train)

mlp_final = MLPClassifier(
    hidden_layer_sizes=best_params_global['hidden_layer_sizes'],
    activation=best_params_global['activation'],
    alpha=best_params_global['alpha'],
    learning_rate_init=best_params_global['learning_rate_init'],
    batch_size=best_params_global.get('batch_size', 32),
    solver='adam', max_iter=2000, random_state=42, learning_rate='adaptive',
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, tol=1e-4
)

print("\n" + "-"*80)
print("Memulai pelatihan model final...")
print("-"*80)
mlp_final.fit(X_train_scaled, y_train, sample_weight=sample_weights_final)
print(f"✓ Pelatihan selesai!")
print(f"   Total epoch: {mlp_final.n_iter_}")
print(f"   Best validation score: {mlp_final.best_validation_score_:.4f}")

y_pred_test_final = mlp_final.predict(scaler_final.transform(X_test))
final_test_acc = accuracy_score(y_test, y_pred_test_final) * 100
final_f1 = f1_score(y_test, y_pred_test_final, average='weighted') * 100
print(f"\nValidasi Model Final:")
print(f"   Test Accuracy: {final_test_acc:.2f}%")
print(f"   Test F1-Score: {final_f1:.2f}%")

# Save artifacts (model, scaler, features list)
model_path = os.path.join(output_klasifikasi, "model_terbaik.joblib")
scaler_path = os.path.join(output_klasifikasi, "scaler_terbaik.joblib")
features_path = os.path.join(output_klasifikasi, "features_terbaik.txt")

joblib.dump(mlp_final, model_path)
joblib.dump(scaler_final, scaler_path)

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

print("\n" + "="*80)
print("MODEL BERHASIL DISIMPAN:")
print("="*80)
print(f" 1. Model    : {model_path}")
print(f" 2. Scaler   : {scaler_path}")
print(f" 3. Encoder  : {os.path.join(output_klasifikasi, 'label_encoder.joblib')}")
print(f" 4. Features : {features_path}")

print("\n" + "="*80 + "\n=== SELESAI (NO-PLOT) ===\n" + "="*80)
print(f"\nModel yang disimpan: {best_feature_name}")
print(f"Feature Set: {best_features}")
print(f"Test Accuracy: {final_test_acc:.2f}%")
print(f"F1-Score: {final_f1:.2f}%")

print("\n OUTPUT (tanpa gambar):")
print("   - LAPORAN_RINGKASAN_FINAL.txt")
print("   - hasil_akhir_mlp_with_morphology.csv")
print("   - hasil_akhir_mlp_with_morphology.xlsx")
print("   - morphology_impact_analysis.csv (jika ada)")
print("   - hyperparameter_tuning_results.csv & best_global_hyperparameters.txt (jika tuning aktif)")
print("   - confmatrix_numeric_*.csv (confusion matrix numerik per feature set)")
