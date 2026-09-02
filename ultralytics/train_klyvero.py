from ultralytics import YOLO

# تحميل المعمارية التي صممناها مع أوزان البداية لتسريع عملية التعلم
model = YOLO(r'D:\Upgrade model Yolo11n-pose\ultralytics\ultralytics\cfg\models\11\klyvero-pose.yaml').load('yolo11n-pose.pt')

# بدء عملية التدريب
results = model.train(
    data=r'D:\Upgrade model Yolo11n-pose\ultralytics\KLYVERO_Dataset\KLYVERO-Pose.v1i.yolov11-pose\data.yaml',
    epochs=100,          # عدد دورات التدريب
    imgsz=640,           # حجم الصور الذي حددناه في Roboflow
    batch=16,            # عدد الصور التي تتم معالجتها معاً في كل خطوة
    device='cpu',            # لتفعيل كارت الشاشة (Nvidia) في لاب توب Lenovo الخاص بك لتقليل وقت التدريب
    project='KLYVERO_Model',
    name='pose_v1'
)