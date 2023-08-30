# Import modules and packages
from flask import (
    Flask,
    redirect,
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
from rmse import rmseroutes
   



app = Flask(__name__)
# Gunakan route yang telah didefinisikan di dalam module route
app.register_blueprint(other_blueprint)

# Menambahkan blueprint ke aplikasi Flask utama
app.register_blueprint(rmseroutes)



@app.route('/')
def index():
    return render_template('index.html')


def read_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            csv_files.append(filename)
    return csv_files

def read_csv_data(csv_filename):
    df = pd.read_csv(csv_filename)
    return df

@app.route('/views')
def views():
    data_directory = 'C:/KMeans-Yonatan A.P.L Tobing/dataset'  # Ganti dengan path direktori tempat file CSV Anda berada

    csv_files = read_csv_files(data_directory)
    tables_html = {}
    for csv_file in csv_files:
        table_html = ""
        try:
            df = read_csv_data(os.path.join(data_directory, csv_file))
            table_html = df.to_html(classes='table table-bordered table-striped table-hover')
        except Exception as e:
            table_html = f"Error reading CSV: {e}"
        tables_html[csv_file] = table_html

    return render_template('view_csv.html', tables_html=tables_html)

@app.route('/delete/<csv_file>', methods=['POST'])
def delete_csv_data(csv_file):
    data_directory = 'C:/KMeans-Yonatan A.P.L Tobing/dataset'  # Ganti dengan path direktori tempat file CSV Anda berada

    try:
        csv_path = os.path.join(data_directory, csv_file)
        os.remove(csv_path)
    except Exception as e:
        return f"Error deleting CSV: {e}"

    return redirect(url_for('views'))

@app.route('/hasils')
def hasils():
    return render_template('hasilakhir.html')

@app.route('/preprocesspage')
def preprocesspage():
    return render_template('preprocesspage.html')


@app.route('/', methods=['POST'])
def get_input_values():
    val = request.form['my_form']

    
@app.route('/preprocessing', methods=['POST'])
def preprocessing():
    
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

    # Melihat variabel data_arr yang bernilai None
    print([i for i, val in enumerate(data_arr) if val is None])

    # Menghilangkan row dengan nilai "nama_kecamatan" tidak kecamatan di Jakarta
    for i in range(30):
        data_arr[i] = data_arr[i].loc[data_arr[i]['nama_kecamatan'].isin(KECAMATAN)]

    for i in range (30):
        print(data_arr[i].isna().sum())
    
    print(type(data_arr[16]))
    print(data_arr[16].columns)

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

    print(data_arr[16].head())

    nama_kolom = data_arr[0].columns

    print(nama_kolom)

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

    print(data_arr[0].columns)

    for i in range(len(data_arr)):
        data_arr[i] = data_arr[i].drop(columns=['perawatan_rs', 'perawatan_rs.1', 'perawatan_rs.2',
                                                'isolasi_di_rumah', 'isolasi_di_rumah.1', 'isolasi_di_rumah.2', 'isolasi_di_rumah.3',
                                                'selesai_isolasi', 'selesai_isolasi.1', 'selesai_isolasi.2', 'selesai_isolasi.3', 'selesai_isolasi.4',
                                                'meninggal', 'meninggal.1'])
    
    print(data_arr[0].head())
    print(data_arr[0].columns)

    # Menyiapkan list untuk menyimpan hasil pengecekan kolom pertama dan kedua
    results = []

    # Mengecek kecamatan yang datanya tidak terdapat pada dataset
    for i in range(len(data_arr)):
        print(f"DataFrame ke-{i+1}: {data_arr[i].shape}")
        print(set(KECAMATAN) - set(data_arr[i].iloc[:, 0].unique()))
    
    for i in range(30):
        data_arr[i] = data_arr[i].groupby('nama_kecamatan').agg({'suspek': 'sum',
                                                          'suspek_meninggal': 'sum', 'pelaku_perjalanan': 'sum',
                                                          'kontak_erat': 'sum', 'positif': 'sum', 'dirawat': 'sum', 'sembuh': 'sum',
                                                          'self_isolation': 'sum',
                                                           'total_perawatan_rs': 'sum', 'total_isolasi_di_rumah': 'sum','total_selesai_isolasi': 'sum', 'total_meninggal': 'sum'}).reset_index()

    print(data_arr[0].head())

    df = pd.concat(data_arr).groupby('nama_kecamatan').agg({'suspek': 'sum',
                                                          'suspek_meninggal': 'sum', 'pelaku_perjalanan': 'sum',
                                                          'kontak_erat': 'sum', 'positif': 'sum', 'dirawat': 'sum', 'sembuh': 'sum',
                                                          'self_isolation': 'sum',
                                                           'total_perawatan_rs': 'sum', 'total_isolasi_di_rumah': 'sum','total_selesai_isolasi': 'sum', 'total_meninggal': 'sum'}).reset_index()

    print(df.head())
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

    # Menampilkan variabel-variabel yang memiliki korelasi dibawah treshold dengan setengah atau lebih dari total variabel
    print(cols_to_drop)

    # Menampilkan variabel-variabel yang akan digunakan
    print(df_filtered.columns)

    # Separates 'nama_kecamatan' columns from non-string columns
    df_country = df_filtered['nama_kecamatan']

    # Delete column 'nama_kecamatan'
    df_filtered = df_filtered.drop('nama_kecamatan', axis=1)

    ##########################################################
    # Karakteristik Data #
    ##########################################################

    # 1. Ukuran data
    print("Ukuran dataset: ", df_filtered.shape)
    # 2. Tipe data
    print("Tipe data setiap kolom: ", df_filtered.dtypes)
    # 3. Missing Values
    print("Jumlah missing values setiap kolom: ")
    print(df_filtered.isnull().sum())
    # 4. Outliers
    # Menentukan nilai batas untuk mengidentifikasi outlier
    def count_outliers(col):
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        batas_bawah = Q1 - 1.5 * IQR
        batas_atas = Q3 + 1.5 * IQR
        return len(col[(col < batas_bawah) | (col > batas_atas)])

    # Loop untuk menghitung jumlah outlier pada setiap kolom
    jumlah_outliers_per_kolom = df_filtered.apply(count_outliers)

    print("Jumlah outlier per kolom:")
    print(jumlah_outliers_per_kolom)
    # 5. Distribusi Data
    distribusi_data = df_filtered.describe()

    print("Informasi distribusi data per kolom:")
    print(distribusi_data)
    # 6. Duplikasi data
    jumlah_duplikasi_per_kolom = df_filtered.duplicated().sum()

    print("Jumlah duplikasi data per kolom:")
    print(jumlah_duplikasi_per_kolom)

    ###############################################
    # Normalisasi
    ###############################################
    
    scaler = MinMaxScaler()

    normalized_data = scaler.fit_transform(df_without_entity)
    print(df_filtered.columns)

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

    # Menggabungkan hasil PCA dengan kolom kecamatan
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    result_df = pd.concat([df_country, pca_df], axis=1)

    print(result_df)
    # Pastikan Anda mengganti contoh data dan nilai-nilai fitur sesuai dengan dataset Anda. Kode di atas akan menggabungkan hasil PCA dengan kolom kecamatan dan menghasilkan DataFrame result_df yang berisi nama kecamatan, nilai PC1, dan nilai PC2.








    # kmeans_label_arr = [None] * 1000
    # kmeans_centroid_arr = [None] * 1000

    # for i in range(1000):
    #     kmeans = KMeans(n_clusters=5)
    #     kmeans.fit(X_pca)

    #     kmeans_label_arr[i] = kmeans.labels_
    #     kmeans_centroid_arr[i] = kmeans.cluster_centers_

    # cmeans_label_arr = [None] * 1000
    # cmeans_centroid_arr = [None] * 1000

    # n_clusters = 5

    # for i in range(1000):
    #     # Perform clustering with Fuzzy C-Means (FCM)
    #     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_pca.T, c=5, m=2, error=0.005, maxiter=10, init=None)

    #     cluster_membership = np.argmax(u, axis=0)

    #     cmeans_label_arr[i] = cluster_membership
    #     cmeans_centroid_arr[i] = cntr

    # Tabel sesudah preprocessing
    df_combined = pd.concat([df_country, df_filtered], axis=1)

    # Simpan hasil preprocessing ke dalam file CSV
    output_file_path = "static/hasil_preprocessing.csv"
    df_combined.to_csv(output_file_path, index=False)


    return render_template('preprocesspage.html',plot='heatmap.png', df_filteredshape = df_filtered.shape, df_filtereddtypes=df_filtered.dtypes, df_filteredisnull=df_filtered.isnull().sum(), jumlah_outliers_per_kolom=jumlah_outliers_per_kolom, df_filtereddescribe=df_filtered.describe, jumlah_duplikasi_per_kolom=jumlah_duplikasi_per_kolom, df_normalizeddescribe=df_normalized.describe, df_filteredcolumns=df_filtered.columns, table_data=df_combined.to_html(classes='table table-bordered table-striped'), xpca=X_pca)

    
if __name__ == '__main__':
    app.run()

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return 'The URL /predict is accessed directly. Go to the main page firstly'

    if request.method == 'POST':
        input_val = request.form

        if input_val != None:
            # collecting values
            vals = []
            for key, value in input_val.items():
                vals.append(float(value))

        # Calculate Euclidean distances to freezed centroids
        with open('freezed_centroids.pkl', 'rb') as file:
            freezed_centroids = pickle.load(file)

        assigned_clusters = []
        l = []  # list of distances

        for i, this_segment in enumerate(freezed_centroids):
            dist = distance.euclidean(*vals, this_segment)
            l.append(dist)
            index_min = np.argmin(l)
            assigned_clusters.append(index_min)

        return render_template(
            'predict.html', result_value=f'Segment = #{index_min}'
            )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
