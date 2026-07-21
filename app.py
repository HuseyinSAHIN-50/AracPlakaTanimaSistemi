import cv2
import time
import re
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import easyocr

app = Flask(__name__)

# --- YAPAY ZEKA MODELLERİ ---
yolo_model = YOLO("yolov8n.pt")
ocr_okuyucu = easyocr.Reader(['tr', 'en'], gpu=False)
PLAKA_REGEX = r"^\d{2}[A-Z]{1,3}\d{2,4}$"

# --- GLOBAL DEĞİŞKENLER (Hafıza) ---
son_okunan_plaka = ""
son_okuma_zamani = 0
gecmis_loglar = []  # Logları tutacağımız liste


def video_akis_olusturucu():
    global son_okunan_plaka, son_okuma_zamani, gecmis_loglar

    kamera = cv2.VideoCapture(0)

    while True:
        ret, kare = kamera.read()
        if not ret:
            break

        su_an = time.time()

        yukseklik, genislik, _ = kare.shape
        x1, y1 = int(genislik * 0.3), int(yukseklik * 0.4)
        x2, y2 = int(genislik * 0.7), int(yukseklik * 0.6)

        kutu_rengi = (0, 0, 255)

        if su_an - son_okuma_zamani < 4:
            kutu_rengi = (0, 255, 0)
        else:
            son_okunan_plaka = ""
            roi_alani = kare[y1:y2, x1:x2]
            sonuclar = yolo_model(roi_alani, verbose=False)[0]

            for kutu in sonuclar.boxes:
                kx1, ky1, kx2, ky2 = map(int, kutu.xyxy[0])
                plaka_resmi = roi_alani[ky1:ky2, kx1:kx2]

                if plaka_resmi.size > 0:
                    ocr_sonuc = ocr_okuyucu.readtext(plaka_resmi)
                    for (bbox, metin, olasilik) in ocr_sonuc:
                        temiz_metin = metin.replace(" ", "").upper()

                        if re.match(PLAKA_REGEX, temiz_metin) and olasilik > 0.5:
                            son_okunan_plaka = temiz_metin
                            son_okuma_zamani = su_an
                            kutu_rengi = (0, 255, 0)

                            # --- YENİ: LOGLARA EKLEME ---
                            zaman_damgasi = time.strftime("%H:%M:%S")
                            # En başa ekliyoruz ki son giren en üstte görünsün
                            gecmis_loglar.insert(0, {"plaka": temiz_metin, "zaman": zaman_damgasi})

                            # Log tablosu çok şişmesin diye son 10 kaydı tutuyoruz
                            if len(gecmis_loglar) > 10:
                                gecmis_loglar.pop()

                            break

        cv2.rectangle(kare, (x1, y1), (x2, y2), kutu_rengi, 2)

        ret, buffer = cv2.imencode('.jpg', kare)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def ana_sayfa():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(video_akis_olusturucu(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/durum')
def sistem_durumu():
    return jsonify({
        "plaka": son_okunan_plaka
    })


# --- YENİ: LOGLARI GETİREN ENDPOINT ---
@app.route('/api/logs')
def loglari_getir():
    return jsonify(gecmis_loglar)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)