import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

try:
    result_df = pd.read_csv('passing-grade.csv')
except FileNotFoundError:
    print("File 'passing-grade.csv' tidak ditemukan. Pastikan file ada di direktori yang benar.")
    exit()

required_columns = ['RATAAN', 'PTN', 'KODE PRODI', 'NAMA PRODI', 'MIN', 'MAX']
if not all(col in result_df.columns for col in required_columns):
    print("Dataset tidak memiliki kolom yang diperlukan: RATAAN, PTN, KODE PRODI, NAMA PRODI, MIN, MAX.")
    exit()

data_rataan = result_df['RATAAN']

print("Statistik Data yang Dihasilkan:")
print(result_df[['RATAAN', 'MIN', 'MAX']].describe())

if data_rataan.nunique() <= 1:
    print("Data pada kolom 'RATAAN' tidak mencukupi untuk estimasi KDE.")
    exit()

try:
    user_grade = float(input("Masukkan nilai untuk menghitung peluang mendapatkan nilai tersebut atau lebih rendah: "))
except ValueError:
    print("Masukkan nilai numerik yang valid.")
    exit()

if user_grade < data_rataan.min() or user_grade > data_rataan.max():
    print(f"Nilai yang dimasukkan ({user_grade}) berada di luar rentang data ({data_rataan.min()} - {data_rataan.max()}).")
    exit()

kde = gaussian_kde(data_rataan)
x = np.linspace(data_rataan.min(), data_rataan.max(), 1000)
density = kde(x)

cumulative_probability = kde.integrate_box(data_rataan.min(), user_grade)
inverted_probability = 1 - cumulative_probability

print(f"\nProbabilitas kumulatif perkiraan untuk mendapatkan nilai kurang dari atau sama dengan {user_grade} adalah: {cumulative_probability:.4f} atau {cumulative_probability * 100:.2f}%")
print(f"Probabilitas kumulatif perkiraan untuk mendapatkan nilai lebih dari {user_grade} adalah: {inverted_probability:.4f} atau {inverted_probability * 100:.2f}%")

# Cari prodi yang paling dekat dengan nilai input
result_df['Jarak'] = (data_rataan - user_grade).abs()

# Menampilkan beberapa program studi yang paling mendekati nilai input pengguna
closest_prodi = result_df.sort_values(by='Jarak').head(5)  # Menampilkan 5 program studi terdekat
print("\nBerikut adalah beberapa program studi yang dapat Anda pilih berdasarkan nilai Anda:")
for idx, prodi in closest_prodi.iterrows():
    print(f"\nProgram Studi: {prodi['NAMA PRODI']}")
    print(f"PTN        : {prodi['PTN']}")
    print(f"Kode Prodi : {prodi['KODE PRODI']}")
    print(f"Rata-rata  : {prodi['RATAAN']:.2f}")
    print(f"Min        : {prodi['MIN']:.2f}")
    print(f"Max        : {prodi['MAX']:.2f}")
    print(f"Jarak dari nilai Anda: {prodi['Jarak']:.2f}")

# Membuat subplot untuk menampilkan grafik secara bersamaan
fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid untuk menampilkan 4 grafik

# Grafik Jarak Program Studi
axs[0, 0].barh(closest_prodi['NAMA PRODI'], closest_prodi['Jarak'], color='skyblue')
axs[0, 0].axvline(x=0, color='gray', linestyle='--')
axs[0, 0].set_title('Jarak Program Studi dari Nilai Pengguna')
axs[0, 0].set_xlabel('Jarak dari Nilai Pengguna')
axs[0, 0].set_ylabel('Program Studi')

# Grafik Kepadatan Probabilitas
axs[0, 1].plot(x, density, label='Fungsi Kepadatan Probabilitas')
axs[0, 1].axvline(user_grade, color='r', linestyle='--', label=f'Nilai Pengguna: {user_grade}')
axs[0, 1].set_title('Fungsi Kepadatan Probabilitas Berdasarkan RATAAN')
axs[0, 1].set_xlabel('Rata-rata Nilai')
axs[0, 1].set_ylabel('Kepadatan')
axs[0, 1].legend()

# Grafik Distribusi Kumulatif Terbalik
cumulative_density = np.cumsum(density) * (x[1] - x[0])
axs[1, 0].plot(x, 1 - cumulative_density, label='Fungsi Distribusi Kumulatif Terbalik', color='orange')
axs[1, 0].axvline(user_grade, color='r', linestyle='--', label=f'Nilai Pengguna: {user_grade}')
axs[1, 0].set_title('Fungsi Distribusi Kumulatif Terbalik Berdasarkan RATAAN')
axs[1, 0].set_xlabel('Rata-rata Nilai')
axs[1, 0].set_ylabel('Probabilitas Kumulatif')
axs[1, 0].legend()

# Kosongkan grafik terakhir (optional, jika Anda hanya ingin 3 grafik)
axs[1, 1].axis('off')

plt.tight_layout()  # Menyesuaikan layout agar grafik tidak tumpang tindih
plt.show()
