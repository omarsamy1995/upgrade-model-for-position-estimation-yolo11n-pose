import os
import random
import shutil

def split_dataset(source_images_dir, source_labels_dir, output_base_dir):
    # إنشاء الهيكلة الرئيسية والفرعية
    splits = ['train', 'val', 'test']
    subfolders = ['images', 'labels']
    
    for split in splits:
        for sub in subfolders:
            os.makedirs(os.path.join(output_base_dir, split, sub), exist_ok=True)
            
    # جلب جميع أسماء الصور وخلطها عشوائياً لضمان تنوع البيانات
    all_images = [f for f in os.listdir(source_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_images)
    
    # حساب أرقام التقسيم بناءً على النسب
    total_images = len(all_images)
    train_end = int(total_images * 0.70)
    val_end = train_end + int(total_images * 0.20)
    
    # نقل الملفات إلى المجلدات الجديدة
    for index, image_name in enumerate(all_images):
        if index < train_end:
            current_split = 'train'
        elif index < val_end:
            current_split = 'val'
        else:
            current_split = 'test'
            
        # تحديد المسارات للملفات الأصلية
        img_src = os.path.join(source_images_dir, image_name)
        txt_name = os.path.splitext(image_name)[0] + '.txt'
        txt_src = os.path.join(source_labels_dir, txt_name)
        
        # تحديد المسارات الوجهة
        img_dst = os.path.join(output_base_dir, current_split, 'images', image_name)
        txt_dst = os.path.join(output_base_dir, current_split, 'labels', txt_name)
        
        # عملية النقل (Move) للصورة وملف النص المطابق لها
        shutil.move(img_src, img_dst)
        if os.path.exists(txt_src):
            shutil.move(txt_src, txt_dst)
            
    print(f"تم التقسيم والنقل بنجاح إلى: {output_base_dir}")
    print(f"Train: {train_end} | Val: {val_end - train_end} | Test: {total_images - val_end}")

# تشغيل الدالة مع وضع مسارات المجلدات التي قمت بإنشائها مسبقاً
split_dataset(
    source_images_dir=r'D:\Upgrade model Yolo11n-pose\ultralytics\KLYVERO_Images',
    source_labels_dir=r'D:\Upgrade model Yolo11n-pose\ultralytics\KLYVERO_Labels',
    output_base_dir=r'D:\Upgrade model Yolo11n-pose\ultralytics\KLYVERO_Dataset'
)