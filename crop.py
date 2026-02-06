import os
import cv2
import numpy as np
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "Models/Segmentation/segment27.pt"
VIDEO_PATH = r"F:\DashCam\DashCam_pipelines\DashCam_pipleine_latest_12012026\DashCam_pipleine_latest_12012026\config_vectorDB\video\0000065_NO20251208_130931_F.MP4"

OUTPUT_DIR = "output"
CROP_PADDING = 20
CONF_THRESHOLD = 0.3

USE_TRACKING = False

# -------- Visualization toggles ----------
SHOW_VIDEO = True
SHOW_MASKS = True
SHOW_SPLIT_BBOX = True
MASK_ALPHA = 0.4
# =========================================


# ================= UTILS =================
def polygon_to_bbox(mask_xy):
    if mask_xy is None or len(mask_xy) < 3:
        return None

    x = mask_xy[:, 0]
    y = mask_xy[:, 1]

    if x.size == 0 or y.size == 0:
        return None

    return [
        int(np.min(x)),
        int(np.min(y)),
        int(np.max(x)),
        int(np.max(y)),
    ]


def find_near_far_points_from_bottom(mask, img_w, img_h, mode):
    if mask is None or len(mask) == 0:
        return None, None
    """Image co ordinate system has top left corner as 0,0 and right bottom corner as w-1,h-1"""
    if mode == 1:
        # right bottom corner
        ref = np.array([img_w - 1, img_h - 1])
    elif mode == 2:
        # bottom left corner
        ref = np.array([0, img_h - 1])
    else:
        ref = np.array([img_w // 2, img_h - 1])

    distances = np.linalg.norm(mask - ref, axis=1)
    return tuple(mask[np.argmin(distances)]), tuple(mask[np.argmax(distances)])


def get_separation_points(near, far):
    if near is None or far is None:
        return []

    dist = np.linalg.norm(np.array(near) - np.array(far))
    if dist < 400:
        return []

    n = int(dist // 400)
    points = []

    for i in range(1, n):
        r = i / n
        x = int(near[0] + r * (far[0] - near[0]))
        y = int(near[1] + r * (far[1] - near[1]))
        points.append((x, y))

    return points


def split_polygons(coords, split_points):
    if coords is None or len(coords) < 3:
        return []

    split_points = sorted(split_points)
    polys = [[] for _ in range(len(split_points) + 1)]

    for x, y in coords:
        placed = False
        for i, sp in enumerate(split_points):
            if x <= sp[0]:
                polys[i].append((x, y))
                placed = True
                break
        if not placed:
            polys[-1].append((x, y))

    return [np.array(p) for p in polys if len(p) >= 3]


def draw_mask(frame, polygon, color, alpha=0.4):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon.astype(np.int32)], color)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
# =========================================


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    crop_root = os.path.join(OUTPUT_DIR, "crops")
    os.makedirs(crop_root, exist_ok=True)

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        clean_frame = frame.copy()
        h, w = frame.shape[:2]

        results = (
            model.track(frame, persist=True, verbose=False)
            if USE_TRACKING
            else model.predict(frame, verbose=False, conf=CONF_THRESHOLD)
        )

        for res in results:
            if res.masks is None or res.boxes is None:
                continue

            masks = res.masks.xy
            boxes = res.boxes

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])

                if conf < CONF_THRESHOLD:
                    continue

                if i >= len(masks):
                    continue

                mask = masks[i].astype(np.int32)
                if mask is None or len(mask) < 3:
                    continue

                bbox = polygon_to_bbox(mask)
                if bbox is None:
                    continue

                cx = (bbox[0] + bbox[2]) // 2
                mode = 2 if cx < w // 2 else 1

                near, far = find_near_far_points_from_bottom(mask, w, h, mode)
                split_pts = get_separation_points(near, far)

                polygons = split_polygons(mask, split_pts) if split_pts else [mask]

                for poly in polygons:
                    if poly is None or len(poly) < 3:
                        continue

                    bx = polygon_to_bbox(poly)
                    if bx is None:
                        continue

                    x1, y1, x2, y2 = bx

                    # -------- Save CLEAN crop --------
                    px1 = max(0, x1 - CROP_PADDING)
                    py1 = max(0, y1 - CROP_PADDING)
                    px2 = min(w, x2 + CROP_PADDING)
                    py2 = min(h, y2 + CROP_PADDING)

                    crop = clean_frame[py1:py2, px1:px2]
                    if crop.size == 0:
                        continue

                    class_dir = os.path.join(crop_root, cls_name)
                    os.makedirs(class_dir, exist_ok=True)

                    crop_name = f"{frame_id}_{i}_{np.random.randint(9999)}_{cls_name}.jpg"
                    cv2.imwrite(os.path.join(class_dir, crop_name), crop)

                    # -------- Visualization --------
                    if SHOW_VIDEO:
                        color = (
                            (cls_id * 37) % 255,
                            (cls_id * 73) % 255,
                            (cls_id * 109) % 255,
                        )

                        if SHOW_MASKS:
                            frame = draw_mask(frame, poly, color, MASK_ALPHA)

                        if SHOW_SPLIT_BBOX:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                frame,
                                f"{cls_name} {conf:.2f}",
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                            )

        if SHOW_VIDEO:
            cv2.imshow("Segmentation + Split Polygons", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video processing completed safely")


if __name__ == "__main__":
    main()
