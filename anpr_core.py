import time
import re
import cv2
from ultralytics import YOLO
import easyocr

# 1. Hazır Yapay Zeka Modellerini Yükleyelim
# Ultralytics, internetten hazır eğitilmiş nesne bulma modelini (yolov8n.pt) otomatik indirir.
yolo_model = YOLO("yolov8n.pt")
# Yazı okuyucu kütüphaneyi Türkçe ve İngilizce için başlatalım
ocr_okuyucu = easyocr.Reader(['tr', 'en'], gpu=False)  # GPU yoksa False kalsın

# 2. Türkiye Plaka Formatı İçin Regex Tanımı
# Örn: 34ABC123 veya 06A1234 formatlarına uyması gerekir
PLAKA_REGEX = r"^\d{2}[A-Z]{1,3}\d{2,4}$"

# 3. Kamerayı Başlatalım
kamera = cv2.VideoCapture(0)  # 0, bilgisayarın dahili veya ilk USB kamerasıdır

print("Sistem başlatıldı. Kameradan plaka aranıyor...")

while True:
    ret, kare = kamera.read()
    if not ret:
        print("Kameradan görüntü alınamadı!")
        break

    # --- YOL HARİTASI ADIM 2: YAZILIMSAL ROI (İlgi Alanı) ---
    # Ekranın tam ortasında plaka sığacak sabit bir kutu alanı belirliyoruz.
    # Tüm ekranda aramak yerine yapay zekaya sadece bu kutunun içini vereceğiz.
    yukseklik, genislik, _ = kare.shape
    x1, y1 = int(genislik * 0.3), int(yukseklik * 0.4)
    x2, y2 = int(genislik * 0.7), int(yukseklik * 0.6)

    # Ekrana kullanıcının görmesi için kırmızı bir ROI kutusu çiziyoruz
    cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Sadece o kutunun içindeki görüntüyü kesiyoruz (Crop)
    roi_alani = kare[y1:y2, x1:x2]

    # --- YOL HARİTASI ADIM 3: PLAKA TESPİTİ (YOLO) ---
    # Hazır nesne bulma modelimiz kesilen alanı tarıyor
    sonuclar = yolo_model(roi_alani, verbose=False)[0]

    for kutu in sonuclar.boxes:
        # Modelin bulduğu nesne bir araba plakası mı veya araba parçası mı kontrolü
        # Not: Genel model (yolov8n.pt) direkt plaka için eğitilmemiştir ancak araba/yazı alanlarını yakalayabilir.
        # Gerçek bir plaka ağırlığı (.pt dosyası) bulduğunda bu kısım doğrudan plakaya odaklanır.

        # Nesnenin koordinatlarını alıyoruz
        kx1, ky1, kx2, ky2 = map(int, kutu.xyxy[0])
        plaka_resmi = roi_alani[ky1:ky2, kx1:kx2]

        if plaka_resmi.size > 0:
            # --- YOL HARİTASI ADIM 3: KARAKTER OKUMA (OCR) ---
            ocr_sonuc = ocr_okuyucu.readtext(plaka_resmi)

            for (bbox, metin, olasilik) in ocr_sonuc:
                # Okunan metindeki boşlukları temizleyelim ve harfleri büyütelim
                temiz_metin = metin.replace(" ", "").upper()

                # --- YOL HARİTASI ADIM 4: REGEX VE DEBOUNCING ---
                if re.match(PLAKA_REGEX, temiz_metin) and olasilik > 0.5:
                    print(f"\n[!!!] PLAKA OKUNDU: {temiz_metin} (Başarı: %{int(olasilik * 100)})")
                    print("BARİYER AÇILDI!")

                    # Ekranda kutuyu yeşile çevirelim
                    cv2.rectangle(roi_alani, (kx1, ky1), (kx2, ky2), (0, 255, 0), 3)

                    # Aynı plakayı üst üste okumasın diye sistemi 3 saniye uyutuyoruz
                    time.sleep(3)

    # Canlı görüntüyü ekranda gösterelim
    cv2.imshow("Plaka Tanima Sistemi - Canli Akis", kare)

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()