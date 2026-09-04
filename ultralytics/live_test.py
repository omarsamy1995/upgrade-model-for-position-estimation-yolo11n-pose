import cv2
from ultralytics import YOLO

# تحميل أوزان KLYVERO التي تم تدريبها
model = YOLO(r'ultralytics\yolo11n-pose.pt')

# فتح كاميرا الجهاز (الرقم 0 يرمز للكاميرا الأساسية)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("تعذر الوصول للكاميرا.")
        break

    # تمرير إطار الكاميرا للموديل لاستخراج المفاصل
    results = model(frame)
    
    # رسم الصندوق والنقاط الـ 12 على الصورة
    annotated_frame = results[0].plot()

    # عرض النتيجة في نافذة حية
    cv2.imshow("KLYVERO Live Tracking", annotated_frame)

    # إغلاق النافذة فور الضغط على حرف 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()