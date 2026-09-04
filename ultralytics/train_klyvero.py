from ultralytics import YOLO

# تحميل المعمارية الأساسية
model = YOLO('yolo11n-pose.pt')

# بدء التدريب المخصص
results = model.train(
    data='data.yaml',            # اسم ملف الإعداد الخاص بك
    epochs=300,                  # الحد الأقصى للدورات
    imgsz=640,                   # دقة المعالجة
    device=0,                    # تشغيل على كارت الشاشة
    batch=16,                    
    
    # المعاملات الرياضية لمنع الـ Overfitting
    freeze=10,                   
    lr0=0.001,                   
    patience=50,                 
    weight_decay=0.0005,         
    
    # التعزيز البصري المخصص لوقوف الأشخاص
    degrees=10.0,                
    scale=0.2,                   
    hsv_s=0.7,                   
    hsv_v=0.4,                   
    close_mosaic=15,             
    
    project='KLYVERO_Engine',    
    name='yolo11n_body_12kpt'    
)