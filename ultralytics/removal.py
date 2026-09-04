import os

def remove_facial_keypoints(labels_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(labels_dir):
        if not filename.endswith('.txt'): continue
            
        with open(os.path.join(labels_dir, filename), 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            # التأكد من أن الملف يحتوي على الصندوق المحيط + 17 نقطة (5 + 17*3 = 56 قيمة)
            if len(parts) == 56:
                # 1. الاحتفاظ ببيانات الصندوق المحيط (Class, x, y, w, h)
                bbox = parts[:5]
                # 2. تخطي أول 15 قيمة تعود لنقاط الوجه (5 نقاط × 3 قيم لكل نقطة)
                # 3. الاحتفاظ بالـ 12 نقطة الخاصة بالجسم (من الكتفين للقدمين)
                body_keypoints = parts[20:]
                
                # دمج السطر الجديد
                new_line = " ".join(bbox + body_keypoints)
                new_lines.append(new_line + "\n")
                
        # حفظ الملفات بالهيكل الجديد
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.writelines(new_lines)

# تنفيذ الكود على مجلد التدريب الخاص بك
remove_facial_keypoints(r'D:\CAR_PLATES_FINAL_PROJECT\code\datasets\coco8-pose\labels\train', r'D:\Upgrade model Yolo11n-pose\ultralytics\KLYVERO_Dataset\train_body_only')