import os
from ultralytics import YOLO

def auto_annotate_body(images_dir, output_labels_dir):
    # 1. إنشاء مجلد المخرجات إذا لم يكن موجوداً
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # 2. تحميل النموذج الأساسي (الذي يفهم 17 نقطة)
    model = YOLO('yolo11n-pose.pt')
    
    print(f"بدء تحليل الصور في: {images_dir}")
    
    # 3. تمرير الصور للنموذج (stream=True تمنع استهلاك الذاكرة العشوائية RAM)
    results = model.predict(source=images_dir, stream=True, conf=0.5)
    
    for result in results:
        # استخراج اسم الصورة لإنشاء ملف نصي بنفس الاسم
        image_name = os.path.splitext(os.path.basename(result.path))[0]
        txt_path = os.path.join(output_labels_dir, f"{image_name}.txt")
        
        # إذا لم يكتشف النظام أي كائن أو لم يكتشف نقاط مفصلية، نتخطى الصورة
        if len(result.boxes) == 0 or result.keypoints is None:
            continue
            
        with open(txt_path, 'w') as f:
            # المرور على كل شخص تم اكتشافه في الصورة
            for i in range(len(result.boxes)):
                # التأكد من أن الكائن المكتشف هو "إنسان" (Class 0)
                class_id = int(result.boxes.cls[i].item())
                if class_id != 0:
                    continue
                
                # استخراج أبعاد الصندوق المحيط كنسبة مئوية من 0 إلى 1 (Normalized)
                # x_center, y_center, width, height
                box = result.boxes.xywhn[i].tolist()
                
                # استخراج إحداثيات النقاط المفصلية (x, y) ومستوى الثقة (Visibility/Conf)
                xyn = result.keypoints.xyn[i].tolist()   # الإحداثيات بنسبة مئوية
                conf = result.keypoints.conf[i].tolist() # احتمالية ظهور النقطة
                
                # -------------------------------------------------------------
                # التعديل المعماري: تجاهل أول 5 نقاط (الوجه) والاحتفاظ بـ 12 نقطة فقط
                # -------------------------------------------------------------
                body_xyn = xyn[5:]
                body_conf = conf[5:]
                
                # كتابة بيانات الصندوق المحيط في بداية السطر
                line = f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                
                # دمج إحداثيات الـ 12 نقطة مع الصندوق في نفس السطر
                for (x, y), c in zip(body_xyn, body_conf):
                    line += f" {x:.6f} {y:.6f} {c:.6f}"
                    
                # حفظ السطر في الملف النصي
                f.write(line + "\n")
                
    print(f"تم الانتهاء بنجاح! تم حفظ ملفات التدريب في: {output_labels_dir}")

# تشغيل الدالة (قم بتغيير مسارات المجلدات لتطابق جهازك)
auto_annotate_body(
    images_dir=r'D:\Upgrade model Yolo11n-pose\ultralytics\KLYVERO_Images',   # مسار مجلد الصور
    output_labels_dir=r'D:\Upgrade model Yolo11n-pose\ultralytics\KLYVERO_Labels' # المسار الذي سيتم حفظ ملفات txt فيه
)