# Import modules and packages
from flask import (
    Flask,
    request,
    render_template,
    url_for
)
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from scipy.spatial import distance
import pandas as pd
import numpy as np
from sklearn import tree
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import os
from other_routes import other_blueprint
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score
from flask import Flask, request, redirect , Blueprint

app = Flask(__name__)


# Membuat blueprint baru
rmseroutes = Blueprint('rmse_routes', __name__)

@rmseroutes.route('/rmse',  methods=['POST'])
def other_rmse():
    
    KECAMATAN = ['PADEMANGAN', 'TAMBORA', 'KRAMAT JATI', 'JATINEGARA', 'CIPAYUNG',
       'MAMPANG PRAPATAN', 'PASAR REBO', 'TANAH ABANG', 'PESANGGRAHAN',
       'TEBET', 'SENEN', 'CAKUNG', 'KEMAYORAN', 'CEMPAKA PUTIH',
       'CENGKARENG', 'CIRACAS', 'GAMBIR', 'JAGAKARSA', 'MENTENG',
       'PANCORAN', 'CILANDAK', 'PASAR MINGGU', 'CILINCING',
       'KEBAYORAN BARU', 'PULO GADUNG', 'MAKASAR', 'KEBAYORAN LAMA',
       'DUREN SAWIT', 'KEBON JERUK', 'JOHAR BARU', 'TAMAN SARI',
       'GROGOL PETAMBURAN', 'SETIA BUDI', 'SAWAH BESAR', 'PALMERAH',
       'KEMBANGAN', 'KALI DERES', 'PENJARINGAN', 'MATRAMAN',
       'TANJUNG PRIOK', 'KELAPA GADING', 'KOJA',
       'KEP. SERIBU UTARA', 'KEP. SERIBU SELATAN']


    # Membuat variabel data1 sampai data30
    for i in range(1, 31):
        globals()[f'data{i}'] = None

    # Load dataset dan memberikan nilai pada variabel data1 sampai data30
    for i in range(1, 31):
        if i < 10:
            globals()[f'data{i}'] = pd.read_csv(f'C:\KMeans-Yonatan A.P.L Tobing\dataset\ekap-data-covid-19-per-kelurahan-provinsi-dki-jakarta-tanggal-0{i}-september-2020.csv')
        else:
            globals()[f'data{i}'] = pd.read_csv(f'C:\KMeans-Yonatan A.P.L Tobing\dataset\ekap-data-harian-covid-19-per-kelurahan-provinsi-dki-jakarta-tanggal-{i}-september-2020.csv')

    # Membuat array dengan jumlah anggota sebanyak 30
    data_arr = [None] * 30

    # Melakukan deklarasi
    for i in range(1,31):
        data_arr[i-1] = globals()[f'data{i}']


    # Menghilangkan row dengan nilai "nama_kecamatan" tidak kecamatan di Jakarta
    for i in range(30):
        data_arr[i] = data_arr[i].loc[data_arr[i]['nama_kecamatan'].isin(KECAMATAN)]

    for i in range (30):
        print(data_arr[i].isna().sum())
    

    # Menghapus kolom yang tidak diperlukan
    for i in range (30):
        columns_to_drop = ['id_kel', 'nama_provinsi', 'nama_kota', 'nama_kelurahan',
                            'probable', 'probable_meninggal', 'discarded', 'keterangan']

    for col in columns_to_drop:
        if col in data_arr[i].columns:
            data_arr[i] = data_arr[i].drop(columns=[col])

    data_arr[16] = data_arr[16].drop(columns='probable meninggal')

    data_arr[16].rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

    data_arr[16] = pd.DataFrame(data_arr[16])



    nama_kolom = data_arr[0].columns



    # inisialisasi kolom yang ingin dijumlahkan
    kolom_perawatan_rs = ['perawatan_rs', 'perawatan_rs.1', 'perawatan_rs.2']
    kolom_isolasi_di_rumah = ['isolasi_di_rumah', 'isolasi_di_rumah.1', 'isolasi_di_rumah.2', 'isolasi_di_rumah.3']
    kolom_selesai_isolasi = ['selesai_isolasi', 'selesai_isolasi.1', 'selesai_isolasi.2', 'selesai_isolasi.3', 'selesai_isolasi.4']
    kolom_meninggal = ['meninggal', 'meninggal.1']

    for i in range(len(data_arr)):
        data_arr[i]['total_perawatan_rs'] = data_arr[i][kolom_perawatan_rs].sum(axis=1)
        data_arr[i]['total_isolasi_di_rumah'] = data_arr[i][kolom_isolasi_di_rumah].sum(axis=1)
        data_arr[i]['total_selesai_isolasi'] = data_arr[i][kolom_selesai_isolasi].sum(axis=1)
        data_arr[i]['total_meninggal'] = data_arr[i][kolom_meninggal].sum(axis=1)


    for i in range(len(data_arr)):
        data_arr[i] = data_arr[i].drop(columns=['perawatan_rs', 'perawatan_rs.1', 'perawatan_rs.2',
                                                'isolasi_di_rumah', 'isolasi_di_rumah.1', 'isolasi_di_rumah.2', 'isolasi_di_rumah.3',
                                                'selesai_isolasi', 'selesai_isolasi.1', 'selesai_isolasi.2', 'selesai_isolasi.3', 'selesai_isolasi.4',
                                                'meninggal', 'meninggal.1'])
    

    # Menyiapkan list untuk menyimpan hasil pengecekan kolom pertama dan kedua
    results = []

    
    for i in range(30):
        data_arr[i] = data_arr[i].groupby('nama_kecamatan').agg({'suspek': 'sum',
                                                          'suspek_meninggal': 'sum', 'pelaku_perjalanan': 'sum',
                                                          'kontak_erat': 'sum', 'positif': 'sum', 'dirawat': 'sum', 'sembuh': 'sum',
                                                          'self_isolation': 'sum',
                                                           'total_perawatan_rs': 'sum', 'total_isolasi_di_rumah': 'sum','total_selesai_isolasi': 'sum', 'total_meninggal': 'sum'}).reset_index()

  

    df = pd.concat(data_arr).groupby('nama_kecamatan').agg({'suspek': 'sum',
                                                          'suspek_meninggal': 'sum', 'pelaku_perjalanan': 'sum',
                                                          'kontak_erat': 'sum', 'positif': 'sum', 'dirawat': 'sum', 'sembuh': 'sum',
                                                          'self_isolation': 'sum',
                                                           'total_perawatan_rs': 'sum', 'total_isolasi_di_rumah': 'sum','total_selesai_isolasi': 'sum', 'total_meninggal': 'sum'}).reset_index()


    # # Memisahkan kolom 'nama_kecamatan' sebagai index
    # df_index = df['nama_kecamatan']
    # df = df.drop('nama_kecamatan', axis=1)

    # Memisahkan kolom 'nama_kecamatan' dari dataset
    df_index = df['nama_kecamatan']
    df_without_entity = df.drop('nama_kecamatan', axis=1)

    plt.figure(figsize=(10,10))

    corr_matrix = df_without_entity.corr()

    # Membuat heatmap dari matriks korelasi
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f")

    # Menampilkan heatmap
    plt.title('Heatmap Korelasi')
    plt.savefig('static/images/heatmap.png')
    plt.show()

    # Hitung matriks korelasi untuk dataset
    corr_matrix = df_without_entity.corr()

    # Inisialisasi threshold
    threshold = 0.0

    # Inisialisasi list untuk menyimpan variabel yang akan dihapus
    cols_to_drop = []

    # Loop melalui setiap variabel dalam dataset
    for column in df_without_entity.columns:
        # Hitung jumlah variabel lain yang memiliki korelasi di bawah threshold dengan variabel saat ini
        num_below_threshold = (corr_matrix[column] < threshold).sum()

        # Jika jumlah variabel lain yang memiliki korelasi di bawah threshold lebih dari atau sama dengan 5 (termasuk variabel itu sendiri), masukkan variabel tersebut ke dalam cols_to_drop
        if num_below_threshold >= len(df_without_entity.columns) / 2:
            cols_to_drop.append(column)

    # Menghapus variabel dari dataset
    df_filtered = df.drop(cols_to_drop, axis=1)

    # Menampilkan dataset yang telah difilter
    # print(df_filtered)


    # Separates 'nama_kecamatan' columns from non-string columns
    df_country = df_filtered['nama_kecamatan']

    # Delete column 'nama_kecamatan'
    df_filtered = df_filtered.drop('nama_kecamatan', axis=1)

    ###############################################
    # Normalisasi
    ###############################################
    
    scaler = MinMaxScaler()

    normalized_data = scaler.fit_transform(df_without_entity)


    # 7. Skala data
    # Konversi hasil normalisasi menjadi DataFrame
    df_normalized = pd.DataFrame(normalized_data, columns=df_filtered.columns)



    ######################################################
    # Dimension reduction with PCA
    ######################################################

    # Shrink the dataset into 2 features
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(normalized_data)

    print(X_pca)

    kmeans_label_arr = [None] * 1000
    kmeans_centroid_arr = [None] * 1000

    for i in range(1000):
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(X_pca)

        kmeans_label_arr[i] = kmeans.labels_
        kmeans_centroid_arr[i] = kmeans.cluster_centers_

    cmeans_label_arr = [None] * 1000
    cmeans_centroid_arr = [None] * 1000

    n_clusters = 5

    for i in range(1000):
        # Perform clustering with Fuzzy C-Means (FCM)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_pca.T, c=5, m=2, error=0.005, maxiter=10, init=None)

        cluster_membership = np.argmax(u, axis=0)

        cmeans_label_arr[i] = cluster_membership
        cmeans_centroid_arr[i] = cntr

        # Tabel sesudah preprocessing
        df_combined = pd.concat([df_country, df_filtered], axis=1)

        # print(df_combined)

    # Inisialisasi nilai tertinggi Silhouette Score dan iterasinya
    best_silhouette_score = -1
    best_iteration = -1
    best_k_cluster = -1

    # Data dan label klaster
    data = X_pca  # Matriks data
    # labels = kmeans_label_arr  # Daftar hasil label klaster dari 10 iterasi

    for i, labels in enumerate(kmeans_label_arr):
        silhouette_avg = silhouette_score(data, labels)
        # print(f"Iterasi ke-{i+1}: Silhouette Score = {silhouette_avg}")

        # Memperbarui nilai tertinggi jika ditemukan nilai Silhouette Score yang lebih tinggi
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_iteration = i + 1

    best_k_cluster = best_iteration

    # print(f"Hasil terbaik ada pada iterasi ke-{best_iteration} dengan Silhouette Score = {best_silhouette_score}")

    # Inisialisasi nilai tertinggi Silhouette Score dan iterasinya
    best_silhouette_score = -1
    best_iteration = -1
    best_c_cluster = -1

    # Data dan label klaster
    data = X_pca  # Matriks data
    # labels = kmeans_label_arr  # Daftar hasil label klaster dari 10 iterasi

    for i, labels in enumerate(cmeans_label_arr):
        silhouette_avg = silhouette_score(data, labels)
        # print(f"Iterasi ke-{i+1}: Silhouette Score = {silhouette_avg}")

        # Memperbarui nilai tertinggi jika ditemukan nilai Silhouette Score yang lebih tinggi
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_iteration = i + 1

    best_c_cluster = best_iteration

    # print(f"Hasil terbaik ada pada iterasi ke-{best_iteration} dengan Silhouette Score = {best_silhouette_score}")

    data = X_pca  # Matriks data
    labels = cmeans_label_arr[6]  # Label klaster dari hasil clustering

    silhouette_avg = silhouette_score(data, labels)
    # print(silhouette_avg)

    ###############################################################
    # Visualisasi Hasil Kmeans dan CMeans terbaik
    ###############################################################

    
    # Visualisasi KMeans
    fig, ax = plt.subplots(figsize=(10, 10))

    # Ganti 'viridis' dengan 'coolwarm', 'tab10', atau palet warna lainnya sesuai selera Anda
    cmap = 'viridis' 

    # Membuat daftar warna sesuai dengan zona yang telah ditentukan
    color_map = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red', 4: 'black'}

    nama_kecamatan = df['nama_kecamatan']

    # Membuat scatter plot dengan warna berbeda untuk setiap zona (kluster) dan menambahkan label negara
    for zone, color in color_map.items():
        cluster_points = X_pca[kmeans_label_arr[best_k_cluster] == zone]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=f'Zone {zone}', cmap=cmap)
        # for i, (x, y) in enumerate(cluster_points):
        #     country_name = country[i] 
        #     ax.annotate(country_name, (x, y), textcoords="offset points", xytext=(5, 5), ha='center')

    # Menambahkan simbol bintang pada koordinat pusat kluster dengan warna merah
    ax.scatter(kmeans_centroid_arr[best_k_cluster][:, 0], kmeans_centroid_arr[best_k_cluster][:, 1], marker='*', c='red')

    # Menambahkan keterangan pada setiap cluster (zona)
    for zone, color in color_map.items():
        centroid_x, centroid_y = kmeans_centroid_arr[best_k_cluster][zone]
        ax.text(centroid_x, centroid_y, f'Cluster {zone}', fontsize=12, ha='center', va='center', color='black', fontweight='bold')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')  
    ax.set_title('Data')
    ax.legend()

    plt.tight_layout()
    plt.savefig('static/images/visualisasikmeans.png')
    plt.show()


    # # Visualisasi KMeans
    # fig, ax = plt.subplots(figsize=(10, 10))

    # ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_label_arr[best_k_cluster], cmap='viridis')
    # ax.scatter(kmeans_centroid_arr[best_k_cluster][:, 0], kmeans_centroid_arr[best_k_cluster][:, 1], marker='*', c='red')
    # ax.set_xlabel('Principal Component 1')
    # ax.set_ylabel('Principal Component 2')
    # ax.set_title('Data')

    # plt.tight_layout()
    # plt.savefig('static/images/visualisasikmeans.png')
    # plt.show()

    # Visualisasi CMeans
    fig, ax = plt.subplots(figsize=(10, 10))

    # Ganti 'viridis' dengan 'coolwarm', 'tab10', atau palet warna lainnya sesuai selera Anda
    cmap = 'viridis'

    # Membuat daftar warna sesuai dengan zona yang telah ditentukan
    color_map_cmeans = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red', 4: 'black'}

    nama_kecamatan = df['nama_kecamatan']

    # Membuat scatter plot dengan warna berbeda untuk setiap zona (kluster) pada CMeans dan menambahkan label negara
    for zone, color in color_map_cmeans.items():
        cluster_points_cmeans = X_pca[cmeans_label_arr[best_c_cluster] == zone]
        ax.scatter(cluster_points_cmeans[:, 0], cluster_points_cmeans[:, 1], c=color, label=f'Zone {zone}', cmap=cmap)
        # for i, (x, y) in enumerate(cluster_points_cmeans):
        #     country_name = country[i] 
        #     ax.annotate(country_name, (x, y), textcoords="offset points", xytext=(5, 5), ha='center')

    # Menambahkan simbol bintang pada koordinat pusat kluster dengan warna merah pada CMeans
    ax.scatter(cmeans_centroid_arr[best_c_cluster][:, 0], cmeans_centroid_arr[best_c_cluster][:, 1], marker='*', c='red')

    # Menambahkan keterangan pada setiap cluster (zona) pada CMeans
    for zone, color in color_map_cmeans.items():
        centroid_x, centroid_y = cmeans_centroid_arr[best_c_cluster][zone]
        ax.text(centroid_x, centroid_y, f'Cluster {zone}', fontsize=12, ha='center', va='center', color='black', fontweight='bold')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Data')
    ax.legend()

    plt.tight_layout()
    plt.savefig('static/images/visualisasicmeans.png')
    plt.show()

    ##################################################
    # Hasil Klaster KMeans
    ##################################################


    nama_kecamatans = df['nama_kecamatan']
    output_data1 = []

    for i, nama_kecamatanss in enumerate(nama_kecamatans):
        temps = kmeans_label_arr[best_k_cluster][i]
        output_data1.append({'Kecamatan': nama_kecamatanss, 'Label Klaster': temps})

    # # Mencetak nama_kecamatan dan label klaster dari kmeans_label_arr[best_k_cluster]
    # for i, negara in enumerate(nama_kecamatan):
    #     print(f"Kecamatan: {negara}, Label Klaster: {kmeans_label_arr[best_k_cluster][i]}")

    # Mengakses data asli dari variabel df dan hasil label klaster k-means dari variabel kmeans_label_arr
    entity_in_cluster_0 = df[kmeans_label_arr[best_k_cluster] == 0]['nama_kecamatan']

    # Mencetak negara-negara yang masuk ke klaster '0'
    # print(entity_in_cluster_0)

    # Mengakses data asli dari variabel df dan hasil label klaster k-means dari variabel kmeans_label_arr
    entity_in_cluster_1 = df[kmeans_label_arr[best_k_cluster] == 1]['nama_kecamatan']

    # Mencetak negara-negara yang masuk ke klaster '1'
    # print(entity_in_cluster_1)

    # Mengakses data asli dari variabel df dan hasil label klaster k-means dari variabel kmeans_label_arr
    entity_in_cluster_2 = df[kmeans_label_arr[best_k_cluster] == 2]['nama_kecamatan']

    # Mencetak negara-negara yang masuk ke klaster '2'
    # print(entity_in_cluster_2)

    #############################################
    # Hasil Klaster CMeans #
    #############################################

    nama_kecamatan = df['nama_kecamatan']
    output_data = []

    for i, nama_kecamatans in enumerate(nama_kecamatan):
        temp = cmeans_label_arr[best_c_cluster][i]
        output_data.append({'Kecamatan': nama_kecamatans, 'Label Klaster': temp})

    # # Mencetak nama_kecamatan dan label klaster dari cmeans_label_arr[best_c_cluster]
    # for i, negara in enumerate(country):
    #     print(f"Country: {negara}, Label Klaster: {cmeans_label_arr[best_c_cluster][i]}")
    
    # Mengakses data asli dari variabel df dan hasil label klaster fuzzy c-means dari variabel cmeans_label_arr
    entity_in_cluster_0 = df[cmeans_label_arr[best_c_cluster] == 0]['nama_kecamatan']

    # Mencetak negara-negara yang masuk ke klaster '0'
    # print(entity_in_cluster_0)

    # Mengakses data asli dari variabel df dan hasil label klaster fuzzy c-means dari variabel cmeans_label_arr
    entity_in_cluster_1 = df[cmeans_label_arr[best_c_cluster] == 1]['nama_kecamatan']

    # Mencetak negara-negara yang masuk ke klaster '1'
    # print(entity_in_cluster_1)

    # Mengakses data asli dari variabel df dan hasil label klaster fuzzy c-means dari variabel cmeans_label_arr
    entity_in_cluster_2 = df[cmeans_label_arr[best_c_cluster] == 2]['nama_kecamatan']

    # Mencetak negara-negara yang masuk ke klaster '2'
    # print(entity_in_cluster_2)

    return render_template('hasilakhir.html',plot='heatmap.png', best_iteration=best_iteration, best_c_cluster=best_c_cluster, best_silhouette_score=best_silhouette_score, label={kmeans_label_arr[best_k_cluster][i]}, data=output_data, data1=output_data1, kecamatan=nama_kecamatans, nama_kecamatansss=nama_kecamatanss)

if __name__ == '__main__':
    app.run()