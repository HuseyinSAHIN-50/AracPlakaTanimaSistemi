# 🚗 Web Tabanlı Araç Plaka Tanıma Sistemi (ALPR)

Bu proje, bilgisayarlı görü (Computer Vision) ve Derin Öğrenme (Deep Learning) teknikleri kullanılarak geliştirilmiş, web tabanlı canlı bir Araç Plaka Tanıma Sistemi (ALPR - Automatic License Plate Recognition) prototipidir. 

Sistem, kameradan gelen canlı video akışını işler, tespit edilen plakaları okur ve modern bir web arayüzü (Dashboard) üzerinden anlık olarak raporlar.

## ✨ Özellikler

* **Canlı Görüntü İşleme:** Web kamerası veya harici kamera üzerinden gerçek zamanlı video akışı.
* **Yazılımsal ROI (İlgi Alanı):** İşlem yükünü azaltmak için ekranın sadece belirli bir bölgesindeki (kırmızı dikdörtgen) plakalar aranır.
* **Yüksek Başarı Oranı:** YOLOv8 ve EasyOCR kullanılarak Türkiye plaka formatına (`^\d{2}[A-Z]{1,3}\d{2,4}$`) uygun okumalar filtrelenir.
* **Debouncing Mekanizması:** Aynı plakanın üst üste okunmasını engellemek için başarılı okumalarda sistem kendini 4 saniye beklemeye (uyku moduna değil) alır.
* **Canlı Web Dashboard:** Python Flask üzerinden sunulan asenkron web arayüzü sayesinde bariyer durumu ve son geçiş logları anlık takip edilir.

## 🛠️ Kullanılan Teknolojiler

**Backend & AI Engine:**
* Python
* OpenCV (Görüntü İşleme)
* Ultralytics YOLOv8 (Nesne ve Plaka Tespiti)
* EasyOCR (Optik Karakter Tanıma)
* Flask (Web Sunucusu ve API)

**Frontend:**
* HTML5, CSS3, JavaScript
* Tailwind CSS (Modern Arayüz Tasarımı)

## 🚀 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

**1. Depoyu Klonlayın**
```bash
git clone [https://github.com/](https://github.com/)[KULLANICI_ADIN]/[DEPO_ADIN].git
cd [DEPO_ADIN]
