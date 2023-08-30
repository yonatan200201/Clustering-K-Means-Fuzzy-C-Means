import os
from flask import Flask, request, redirect, Blueprint, render_template

app = Flask(__name__)
other_blueprint = Blueprint('other', __name__)

# Tentukan lokasi folder untuk menyimpan file
UPLOAD_FOLDER = 'dataset'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan direktori 'dataset' ada sebelum server berjalan
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@other_blueprint.route('/other', methods=['POST'])
def other_route():
    if request.method == "POST":
        files = request.files.getlist("file")

        # Periksa apakah ada file yang di-upload
        if not any(files):
            return "Tidak ada file yang di-upload."

        for file in files:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return redirect("/")

@other_blueprint.route('/', methods=['GET'])
def index():
    # Mendapatkan daftar file yang ada di folder UPLOAD_FOLDER
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])

    # Mengambil hanya file dengan ekstensi .csv
    csv_files = [file for file in uploaded_files if file.lower().endswith('.csv')]

    return render_template('index.html', uploaded_files=csv_files)


if __name__ == '_main_':
    app.run(debug=True)