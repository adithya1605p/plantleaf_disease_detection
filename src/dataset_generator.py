"""
Data generator script for Multi-Crop Plant Disease Detection (PlantVillage Extended).
Creates realistic synthetic foliage with irregular lesions and valid YOLO labels
across 18 plant disease categories.
"""
import os
import random
import cv2
import numpy as np

CLASSES = [
    "Apple_Scab",
    "Corn_Common_Rust",
    "Potato_Early_Blight",
    "Potato_Late_Blight",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Healthy_Leaf",
    "Grape_Black_Rot",
    "Grape_Esca_Measles",
    "Pepper_Bacterial_Spot",
    "Tomato_Bacterial_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Cherry_Powdery_Mildew",
    "Peach_Bacterial_Spot",
    "Strawberry_Leaf_Scorch",
    "Corn_Northern_Leaf_Blight",
    "Citrus_Greening",
    "Squash_Powdery_Mildew",
]

def draw_realistic_leaf(img_size=(640, 640), class_idx=6):
    h, w = img_size
    # Soil/neutral agricultural background with subtle texture
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :] = [random.randint(35, 55), random.randint(45, 65), random.randint(30, 50)]
    noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Leaf center and axis
    center_x = w // 2 + random.randint(-30, 30)
    center_y = h // 2 + random.randint(-30, 30)

    # Adapt leaf morphology depending on crop family
    # Corn: elongated lanceolate leaf
    if class_idx in [1, 15]:
        axis_major = random.randint(240, 290)
        axis_minor = random.randint(65, 95)
    # Grape / Squash: broader palmate/round leaf
    elif class_idx in [7, 8, 17]:
        axis_major = random.randint(190, 230)
        axis_minor = random.randint(150, 190)
    # Strawberry / Peach / Cherry / Citrus: medium oval leaf
    else:
        axis_major = random.randint(180, 240)
        axis_minor = random.randint(100, 150)

    angle = random.randint(-35, 35)

    # Base leaf mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (center_x, center_y), (axis_major, axis_minor), angle, 0, 360, 255, -1)

    # Leaf foliage base color
    # Citrus / Peach: deeper dark green
    if class_idx in [13, 16]:
        leaf_color = np.array([random.randint(20, 35), random.randint(95, 130), random.randint(25, 55)], dtype=np.uint8)
    # Corn: brighter grassy green
    elif class_idx in [1, 15]:
        leaf_color = np.array([random.randint(30, 50), random.randint(140, 180), random.randint(60, 95)], dtype=np.uint8)
    else:
        leaf_color = np.array([random.randint(25, 45), random.randint(115, 160), random.randint(40, 75)], dtype=np.uint8)

    leaf_area = np.zeros_like(bg)
    leaf_area[:] = leaf_color
    
    # Vein structure
    cv2.line(leaf_area, (center_x - int(axis_major*0.8), center_y), (center_x + int(axis_major*0.8), center_y), (35, 185, 55), 2)
    for offset in range(-int(axis_major*0.6), int(axis_major*0.6), 35):
        vx = center_x + offset
        vy = center_y
        cv2.line(leaf_area, (vx, vy), (vx + random.randint(20, 50), vy - random.randint(25, 65)), (30, 170, 50), 1)
        cv2.line(leaf_area, (vx, vy), (vx - random.randint(20, 50), vy + random.randint(25, 65)), (30, 170, 50), 1)

    final_img = np.where(mask[:, :, None] == 255, leaf_area, bg)
    return final_img, mask, (center_x, center_y, axis_major, axis_minor)

def add_lesions(img, leaf_mask, class_idx, max_lesions=4):
    """
    Simulates disease lesions depending on class.
    Returns modified image and list of YOLO normalized bboxes: [class_idx, cx, cy, w, h]
    """
    h, w, _ = img.shape
    bboxes = []
    
    if class_idx == 6:  # Healthy_Leaf
        return img, bboxes

    valid_pts = np.argwhere(leaf_mask == 255)
    if len(valid_pts) == 0:
        return img, bboxes

    # Number of lesions
    num_lesions = random.randint(2, max_lesions if class_idx not in [9, 10, 13] else 6)

    # Pathology color mapping
    color_map = {
        0: (30, 60, 50),      # Apple_Scab (Olive-brown)
        1: (20, 70, 160),     # Corn_Common_Rust (Rust cinnamon)
        2: (25, 45, 75),      # Potato_Early_Blight (Dark target brown)
        3: (45, 55, 65),      # Potato_Late_Blight (Muted dark gray-brown)
        4: (30, 50, 80),      # Tomato_Early_Blight (Concentric dark brown)
        5: (40, 50, 60),      # Tomato_Late_Blight (Water-soaked dark brown)
        7: (20, 30, 45),      # Grape_Black_Rot (Sunken black-brown)
        8: (25, 35, 95),      # Grape_Esca_Measles (Tiger stripe reddish-brown)
        9: (20, 40, 50),      # Pepper_Bacterial_Spot (Dark brown necrotic spot)
        10: (20, 40, 50),     # Tomato_Bacterial_Spot (Dark brown water-soaked)
        11: (40, 190, 210),   # Tomato_Yellow_Leaf_Curl (Yellow chlorotic patch)
        12: (210, 215, 210),  # Cherry_Powdery_Mildew (Talc whitish-gray)
        13: (25, 45, 55),     # Peach_Bacterial_Spot (Angular dark spot)
        14: (30, 25, 85),     # Strawberry_Leaf_Scorch (Purplish dark brown)
        15: (70, 85, 95),     # Corn_Northern_Leaf_Blight (Cigar gray-tan)
        16: (50, 180, 200),   # Citrus_Greening (Asymmetric yellow mottle)
        17: (215, 220, 215),  # Squash_Powdery_Mildew (Talc whitish-gray)
    }

    base_color = color_map.get(class_idx, (30, 40, 70))

    for _ in range(num_lesions):
        pt_idx = random.randint(0, len(valid_pts) - 1)
        cy, cx = valid_pts[pt_idx]

        # Lesion geometry
        if class_idx == 15:  # Corn Northern Leaf Blight: elongated cigar shape
            rw = random.randint(40, 80)
            rh = random.randint(15, 25)
            angle = random.randint(-15, 15)
        elif class_idx in [9, 10, 13]:  # Bacterial spots: small punctate spots
            rw = random.randint(8, 20)
            rh = random.randint(8, 20)
            angle = 0
        elif class_idx in [12, 17]:  # Powdery mildew: broad powdery patches
            rw = random.randint(25, 55)
            rh = random.randint(25, 55)
            angle = random.randint(0, 180)
        else:
            rw = random.randint(15, 40)
            rh = random.randint(15, 40)
            angle = random.randint(0, 180)

        x1 = max(0, cx - rw)
        y1 = max(0, cy - rh)
        x2 = min(w - 1, cx + rw)
        y2 = min(h - 1, cy + rh)

        lesion_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(lesion_mask, (cx, cy), (rw, rh), angle, 0, 360, 255, -1)

        # Yellow halo for bacterial spots and early blights
        if class_idx in [2, 4, 9, 10, 13]:
            halo_mask = cv2.dilate(lesion_mask, np.ones((7, 7), np.uint8))
            halo_area = (halo_mask == 255) & (lesion_mask == 0) & (leaf_mask == 255)
            img[halo_area] = [random.randint(30, 60), random.randint(180, 210), random.randint(190, 220)]

        lesion_spots = (lesion_mask == 255) & (leaf_mask == 255)
        
        # Color with noise
        var_color = [
            min(255, max(0, base_color[0] + random.randint(-10, 10))),
            min(255, max(0, base_color[1] + random.randint(-10, 10))),
            min(255, max(0, base_color[2] + random.randint(-10, 10)))
        ]
        img[lesion_spots] = var_color

        # YOLO format box
        box_w = (x2 - x1) / float(w)
        box_h = (y2 - y1) / float(h)
        box_cx = ((x1 + x2) / 2.0) / float(w)
        box_cy = ((y1 + y2) / 2.0) / float(h)
        
        if box_w > 0.01 and box_h > 0.01:
            bboxes.append((class_idx, box_cx, box_cy, box_w, box_h))

    return img, bboxes

def generate_dataset(base_dir="data", num_train_per_class=3, num_val_per_class=1):
    train_img_dir = os.path.join(base_dir, "train", "images")
    train_lbl_dir = os.path.join(base_dir, "train", "labels")
    val_img_dir = os.path.join(base_dir, "val", "images")
    val_lbl_dir = os.path.join(base_dir, "val", "labels")

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Generating multi-crop dataset for {len(CLASSES)} classes: {num_train_per_class*len(CLASSES)} train, {num_val_per_class*len(CLASSES)} val images...")
    
    # Generate train
    idx = 0
    for cls_idx in range(len(CLASSES)):
        for _ in range(num_train_per_class):
            img, mask, _ = draw_realistic_leaf(class_idx=cls_idx)
            img, bboxes = add_lesions(img, mask, cls_idx)
            
            cls_name = CLASSES[cls_idx]
            img_name = f"leaf_train_{idx:04d}_{cls_name}.jpg"
            lbl_name = f"leaf_train_{idx:04d}_{cls_name}.txt"
            
            cv2.imwrite(os.path.join(train_img_dir, img_name), img)
            with open(os.path.join(train_lbl_dir, lbl_name), "w") as f:
                for b in bboxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
            idx += 1

    # Generate val
    idx = 0
    for cls_idx in range(len(CLASSES)):
        for _ in range(num_val_per_class):
            img, mask, _ = draw_realistic_leaf(class_idx=cls_idx)
            img, bboxes = add_lesions(img, mask, cls_idx)
            
            cls_name = CLASSES[cls_idx]
            img_name = f"leaf_val_{idx:04d}_{cls_name}.jpg"
            lbl_name = f"leaf_val_{idx:04d}_{cls_name}.txt"
            
            cv2.imwrite(os.path.join(val_img_dir, img_name), img)
            with open(os.path.join(val_lbl_dir, lbl_name), "w") as f:
                for b in bboxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
            idx += 1

    print(f"Multi-crop dataset successfully generated with {len(CLASSES)} classes at '{base_dir}'.")

if __name__ == "__main__":
    generate_dataset()
