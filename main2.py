import sklearn.linear_model as lm
import pandas as pd
import sklearn.model_selection as ms
import numpy as np

data = pd.read_csv("calonpembeli_ch5.csv")

data = data[data["Usia"] <= 100]


X = data[["Usia", "Status", "Kelamin", "Memiliki_Mobil", "Penghasilan"]]
y = data[["Beli_Mobil"]]

# Membagi Sebuah dataset ke training dan test dataset
X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=0.2, random_state=0)


# flatten
y_train = np.array(y_train).flatten()
y_test = np.array(y_test).flatten()


# Bikin sebuah model dengan logistic regresion
model = lm.LogisticRegression(solver="lbfgs")
model.fit(X_train, y_train)

# Mencari nilai koefisien dan intercept
# print(f"Koefisien = {model.coef_}")
# print(f"Intercept = {model.intercept_}")


# Memprediksi data
y_prediksi = model.predict(X_test)
# print(X_test.shape)
# print(y_prediksi)

# Melihat akurasi model dari data
akurasi_model = model.score(X_test, y_test)
# print(f"Akurasi= {akurasi_model}")

# print(data.describe())

print(60 * "=")
print("MEMPREDIKSI APAKAH ORANG TERSEBUT AKAN BISA MEMBELI MOBIL")
print(60 * "=")

loop = True
Usia = []
Status = []
Kelamin = []
JumlahMobil = []
Penghasilan = []

while loop:
    try:
        masukanUsia = int(input("Masukan Umur: ")) 
        masukanStatus = int(input("Masukan Status anda belumNikah(0), nikah(1), nikahDanPunyaAnak(2), duda/janda(3): "))
        masukanKelamin = int(input("Masukan jenis kelamin Laki-laki(0), perempuan(1): "))
        masukanJumlahMobil = int(input("Masukan jumlah mobil yang anda punya: "))
        masukanPenghasilan = int(input("Masukan Penghasilan anda pertahun: "))
        Usia.append(masukanUsia)
        Status.append(masukanStatus)
        Kelamin.append(masukanKelamin)
        JumlahMobil.append(masukanJumlahMobil)
        Penghasilan.append(masukanPenghasilan)
        loop = False
    except ValueError:
        print("yang anda masukan bukan angka")
        break


# Menggabungkan data yang telah di isi
dataLenkap = {
    "Usia": Usia,
    "Status": Status,
    "Kelamin": Kelamin,
    "Memiliki_Mobil": JumlahMobil,
    "Penghasilan": Penghasilan
}
dataLenkap2 = {
    "Status": {
        0: "Belum Menikah",
        1: "Nikah",
        2: "Nikah punya Anak",
        3: "Duda/Janda",
    },
    "Kelamin": {
        0: "Laki-laki",
        1: "Perempuan",
    },
}

dataLenkap = pd.DataFrame(dataLenkap)
dataLenkap2 = dataLenkap.replace(dataLenkap2)

print()
print()
print(60 * "=")
print("Daftar Data yang Sudah di Isi")
print(60 * "=")
print(dataLenkap2)
# print(dataLenkap.shape)
# print(type(X_test))
# print(type(dataLenkap))


# Memprediksi sebuah data
prediksi = model.predict(dataLenkap)
# print(prediksi)
print()
print(50 * "=")
print("Hasil Prediksi")
print(50 * "=")
if prediksi == 1:
    print("Orang tersebut akan beli mobil")
else:
    print("Orang tersebut tidak akan membeli mobil")

