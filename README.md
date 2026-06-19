# Inti Berita

RingkasKilat adalah aplikasi web berbasis *Machine Learning* yang dirancang untuk meringkas teks bahasa Indonesia secara otomatis. Aplikasi ini dibangun dengan framework Flask di sisi backend dan menggunakan model **mT5** (Multilingual T5) untuk memproses teks secara cerdas, memberikan hasil ringkasan yang padat dan akurat.

Cocok digunakan untuk keperluan akademik, pekerjaan kantor, maupun sekadar membaca artikel berita yang panjang dengan lebih cepat.

## Fitur Utama

- **Fleksibilitas Input**: Anda bisa memasukkan teks dengan cara menyalin (paste), memasukkan URL artikel berita, atau mengunggah dokumen langsung (mendukung format `.pdf` dan `.docx`).
- **Streaming Output**: Hasil ringkasan dimunculkan secara bertahap (*real-time typing effect*), sehingga antarmuka terasa jauh lebih responsif.
- **Mode Abstraktif & Ekstraktif**:
  - *Abstraktif*: Menulis ulang inti teks dengan kalimat baru yang natural.
  - *Ekstraktif*: Menyoroti dan mengambil kalimat-kalimat paling krusial dari teks asli.
- **Ekspor Dokumen**: Hasil ringkasan dapat dengan mudah diunduh menjadi file PDF maupun TXT.
- **Text-to-Speech**: Dilengkapi ikon *speaker* untuk membacakan hasil ringkasan secara lisan (menggunakan bawaan Web Speech API browser).

## Prasyarat

Sebelum menjalankan proyek ini, pastikan perangkat Anda sudah memiliki:
- Python (versi 3.8 ke atas disarankan)
- Koneksi internet (untuk mengunduh *library* dan model pertama kali jika diperlukan)

## Cara Menjalankan Aplikasi

1. **Clone repositori**
   ```bash
   git clone https://github.com/YuyunNailufar/Inti-Berita.git
   cd Inti-Berita
   ```

2. **Siapkan Virtual Environment (Direkomendasikan)**
   ```bash
   python -m venv venv
   
   # Aktivasi di Windows:
   .\venv\Scripts\activate
   
   # Aktivasi di Linux/Mac:
   source venv/bin/activate
   ```

3. **Instal pustaka yang dibutuhkan**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan server**
   ```bash
   python app.py
   ```

5. **Buka aplikasi**
   Buka *web browser* Anda dan kunjungi `http://127.0.0.1:5000`.

## Struktur Direktori

- `app.py`: File utama aplikasi yang mengatur server Flask dan logika model.
- `requirements.txt`: Daftar *dependencies* Python.
- `static/`: Tempat menyimpan file penunjang tampilan antarmuka (`styles.css`, `app.js`).
- `templates/`: Menyimpan file struktur kerangka antarmuka (`index.html`).

## Catatan Tambahan

Proyek ini merupakan tugas/implementasi untuk mata kuliah Pemrosesan Bahasa Alami (NLP). Penggunaan model mungkin membutuhkan waktu *loading* atau memori tambahan di awal pemanggilan. Jika menemui masalah terkait dependensi PDF/Word, pastikan pustaka `PyPDF2` dan `python-docx` telah terinstal dengan benar.
