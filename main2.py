import sklearn.linear_model as lm
import pandas as pd
import sklearn.model_selection as ms

data = pd.read_csv("calonpembeli_ch5.csv")

data = data[data["Usia"] <= 100]


X = data[["Usia", "Status", "Kelamin", "Memiliki_Mobil", "Penghasilan"]]
y = data[["Beli_Mobil"]]

# Membagi Sebuah dataset ke training dan test dataset
X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=0.2, random_state=0)


# Bikin sebuah model dengan logistic regresion
model = lm.LogisticRegression(solver="lbfgs")
model.fit(X_train, y_train)

# Mencari nilai koefisien dan intercept
# print(f"Koefisien = {model.coef_}")
# print(f"Intercept = {model.intercept_}")


# Memprediksi data
y_prediksi = model.predict([[32,1,0,0,240]])
# print(y_prediksi)

# Melihat akurasi model dari data
akurasi_model = model.score(X_test, y_test)
# print(f"Akurasi= {akurasi_model}")

# print(data.describe())

print("=========================================================")
print("MEMPREDIKSI APAKAH ORANG TERSEBUT AKAN BISA MEMBELI MOBIL")
print("=========================================================")

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

Usia = pd.DataFrame(Usia, columns=["Usia"])
Status = pd.DataFrame(Status, columns=["Status"])
Kelamin = pd.DataFrame(Kelamin, columns=["Kelamin"])
JumlahMobil = pd.DataFrame(JumlahMobil, columns=["JumlahMobil"])
Penghasilan = pd.DataFrame(Penghasilan, columns=["Penghasilan"])

data1 = pd.DataFrame.join(Usia, Status)
data2 = pd.DataFrame.join(Kelamin, JumlahMobil)
data3 = pd.DataFrame.join(data1, data2)


# Menggabungkan data yang telah di isi
dataLengkap = pd.DataFrame.join(data3, Penghasilan)
# print(dataLengkap)


# Memprediksi sebuah data
prediksi = model.predict(dataLengkap)
print("=====================")
print("Hasil Prediksi")
print("=====================")
if prediksi == 1:
    print("Orang tersebut akan beli mobil")
else:
    print("Orang tersebut tidak akan membeli mobil")

