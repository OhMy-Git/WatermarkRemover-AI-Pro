import io
import os
import sys
import tempfile
import subprocess
import shutil
import warnings
# 強制設定 stdout 使用 UTF-8 避免 Windows 亂碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 忽略 timm 的 FutureWarning 讓輸出保持乾淨
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

import cv2
import time
import argparse
import numpy as np
import torch

# CLAHE mode constants
CLAHE_MODES = ["off", "auto", "mild", "aggressive"]
from pathlib import Path
from PIL import Image
from loguru import logger
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForZeroShotObjectDetection
from enum import Enum
from typing import List, Any

# SAM2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
# GroundingDINO imports
try:
    # 設定正確的路徑以支援 src/groundingdino 結構
    import sys
    import os
    gdino_path = os.path.join(os.getcwd(), "src", "groundingdino")
    if gdino_path not in sys.path:
        sys.path.append(gdino_path)
    
    from groundingdino.models import build_model  # type: ignore[import-not-found]
    from groundingdino.util import box_ops  # type: ignore[import-not-found]
    from groundingdino.util.inference import load_image, predict, load_model  # type: ignore[import-not-found]
    from groundingdino.util.slconfig import SLConfig  # type: ignore[import-not-found]
    import torchvision.transforms as T
except ImportError:
    pass

# 如果有安裝 ultralytics 則引入，若無則在 YOLO 模式下會報錯
try:
    from ultralytics.models.yolo import YOLO
except ImportError:
    YOLO = None

# =========================================================
# Geometry Mapper & NMS Utilities
# =========================================================
class GeometryMapper:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.center = (width // 2, height // 2)
    
    def get_transformed_image(self, image, angle: int):
        """旋轉圖像並返回變換矩陣"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        M_inv = cv2.invertAffineTransform(M)
        return rotated, M_inv
    
    def map_bbox_back(self, bbox, M_inv, offset, expand, rotate):
        """將旋轉後的 bbox 映射回原始座標"""
        x1, y1, x2, y2 = bbox
        pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
        transformed = cv2.transform(pts, M_inv)[0]
        x_min, y_min = np.min(transformed[:,0]), np.min(transformed[:,1])
        x_max, y_max = np.max(transformed[:,0]), np.max(transformed[:,1])
        
        # 如果有偏移，加上偏移量
        if offset != (0, 0):
            x_min += offset[0]
            y_min += offset[1]
            x_max += offset[0]
            y_max += offset[1]
        
        # 如果需要擴展
        if expand > 0:
            x_min = max(0, x_min - expand)
            y_min = max(0, y_min - expand)
            x_max = min(self.width, x_max + expand)
            y_max = min(self.height, y_max + expand)
        
        return [x_min, y_min, x_max, y_max]

def apply_nms(boxes, scores, iou_threshold=0.3):
    """非極大值抑制"""
    if len(boxes) == 0:
        return []
    
    # 使用 numpy 的 NMS 實現
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 如果沒有 OpenCV 的 NMS，使用自定義實現
    try:
        # Perform NMS using OpenCV; ensure proper typing for IntelliSense
        # Perform NMS using OpenCV; result is a list of indices
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold,
        )
        if len(indices) == 0:
            return []
        # Convert result to a flat list of ints, handling various possible return types from OpenCV
        if isinstance(indices, np.ndarray):
            # Numpy array: flatten to 1D list
            indices_list = indices.ravel().tolist()
        else:
            # Assume iterable (list/tuple) possibly containing nested structures
            indices_list = []
            for item in indices:
                if isinstance(item, (list, tuple, np.ndarray)):
                    # Take the first element if present (e.g., [0] or (0,))
                    if len(item) > 0:
                        indices_list.append(int(item[0]))
                else:
                    indices_list.append(int(item))
        return [int(x) for x in indices_list]
    except:
        # 如果無法使用 OpenCV NMS，使用簡單的 NMS
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            
            if order.size == 1:
                break
                
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep

# --- 全域設定 ---
IMAGE_EXTS = { ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
LAMA_MODEL_PATH = Path("models") / "big-lama.pt"

class TaskType(str, Enum):
    OPEN_VOC_DET = "<OPEN_VOCABULARY_DETECTION>"

# GroundingDINO model constants
GROUNDING_DINO_CONFIG_PATH = "weights/groundingdino_swinb_cfg.py"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swinb_cogcoor.pth"

# ================================================
# CLAHE preprocess for better watermark detection
# ================================================
def unsharp(image: np.ndarray, alpha: float):
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1 + alpha, blur, -alpha, 0)


def mild_preprocess(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return unsharp(img, 1.0)


def aggressive_preprocess(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return unsharp(img, 2.0)


def auto_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Adaptive sharpness controller
    根據圖像邊緣強度自動決定 mild 或 aggressive
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 數值可自行微調
    if lap_var < 30:
        return aggressive_preprocess(image)
    elif lap_var < 80:
        return mild_preprocess(image)
    else:
        return image


def preprocess_dispatch(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "off" or mode is None:
        return image
    elif mode == "mild":
        return mild_preprocess(image)
    elif mode == "aggressive":
        return aggressive_preprocess(image)
    elif mode == "auto":
        return auto_preprocess(image)
    else:
        return image
# ================================================

def rotate_image(image: np.ndarray, angle: float):
    """旋轉影像"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, M

def rotate_bbox(bbox, M_inv):
    """將 Bbox 坐標旋轉回原始位置"""
    x1, y1, x2, y2 = bbox
    pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    transformed = cv2.transform(pts, M_inv)[0]
    return [np.min(transformed[:,0]), np.min(transformed[:,1]), np.max(transformed[:,0]), np.max(transformed[:,1])]

def identify(task_prompt, image, text_input, model, processor, device):
    """執行 Florence-2 推理"""
    if processor is None:
        logger.error("Processor is None, cannot perform Florence-2 inference")
        # 返回空結果避免程式崩潰
        return {}
    prompt = task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    target_dtype = next(model.parameters()).dtype
    inputs = {k: v.to(device).to(target_dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(generated_text, task=task_prompt.value, image_size=(image.shape[1], image.shape[0]))

def get_mask_advanced(image: np.ndarray, model, processor, device: str, args, clahe_mode) -> np.ndarray:
    """Florence-2 專屬：進階 Tiling 偵測邏輯"""
    # 檢查處理器是否為 None
    if processor is None:
        logger.error("處理器為 None，無法進行 Florence-2 推理")
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    h, w = image.shape[:2] # 取得影像尺寸
    final_mask = np.zeros((h, w), dtype=np.uint8)

    tiles_per_axis = getattr(args, "tiles_per_axis", 5)  # 預設 5x5 切片
    ratio = getattr(args, "overlap_ratio", 0.15) #從 args 讀取 overlap 比例，若讀不到則預設 0.15
    tile_h, tile_w = h // tiles_per_axis, w // tiles_per_axis
    overlap = int(min(tile_h, tile_w) * ratio) # <-- 使用動態 overlap 比例

    tiles_coords = []
    for row in range(tiles_per_axis):
        for col in range(tiles_per_axis):
            y1 = max(0, row * tile_h - overlap)
            x1 = max(0, col * tile_w - overlap)
            y2 = min(h, (row + 1) * tile_h + overlap)
            x2 = min(w, (col + 1) * tile_w + overlap)
            tiles_coords.append((y1, x1, y2, x2))

    search_text = args.watermark_text if args.watermark_text else "watermark"
    # 根據 rotate_angle_step 參數生成角度列表
    rotate_angle_step = getattr(args, "rotate_angle_step", 0)
    if rotate_angle_step == 0:
        angles = [0]  # 不旋轉
    else:
        # 生成從 0 到 360 度的步進角度列表
        angles = list(range(0, 360, rotate_angle_step))

    logger.info(f"--- 啟動 Tiling 切片偵測 ({tiles_per_axis**2} 區塊) ---")

    for i, (y1, x1, y2, x2) in enumerate(tiles_coords):
        tile = image[int(y1):int(y2), int(x1):int(x2)]
        for angle in angles:
            rotated_img, M = rotate_image(tile, angle)
            M_inv = cv2.invertAffineTransform(M)
            processed = preprocess_dispatch(rotated_img, clahe_mode)
            
            # 檢查是否為 GroundingDINO 模型
            if hasattr(model, 'predict_with_caption'):
                # GroundingDINO 模型
                try:
                    import supervision as sv
                    detections, labels = model.predict_with_caption(
                        image=rotated_img,
                        caption=search_text,
                        box_threshold=args.groundingdino_conf_threshold,
                        text_threshold=0.25
                    )
                    bboxes = detections.xyxy
                    scores = detections.confidence
                    count = 0
                    # 確保 bboxes 和 scores 不為 None
                    if bboxes is not None and scores is not None:
                        for bbox, score in zip(bboxes, scores):
                            if score < args.groundingdino_conf_threshold:
                                continue
                            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            if (bw * bh) / (w * h) > (args.max_bbox_percent / 100.0):
                                continue
                                
                            # 將旋轉後的 bbox 轉換回原始座標
                            rb = rotate_bbox(bbox, M_inv)
                            ox1, oy1, ox2, oy2 = int(rb[0]+x1), int(rb[1]+y1), int(rb[2]+x1), int(rb[3]+y1)
                            cv2.rectangle(final_mask, (max(0, ox1-args.expand), max(0, oy1-args.expand)),
                                          (min(w, ox2+args.expand), min(h, oy2+args.expand)), 255, -1)
                            count += 1
                    if count > 0:
                        logger.info(f"區塊 {i+1} 角度 {angle:3d}°: 偵測到 {count} 個目標")
                except Exception as e:
                    logger.warning(f"GroundingDINO 模型處理失敗: {e}")
            else:
                # Florence-2 模型
                # 使用 DetectorEnsemble 的 forward 方法來處理，而不是直接調用 identify
                try:
                    # 確保 Florence 模型已載入
                    if model.active_type != "Florence-2":
                        model._load_florence()
                    
                    # 檢查處理器是否已載入
                    if model.processor is None:
                        # 如果處理器未載入，嘗試載入
                        model._load_florence()
                    
                    if model.processor is None or model.model is None:
                        logger.error("Florence-2 模型或處理器未正確載入")
                        continue
                    
                    # 使用模型的處理器進行物件偵測
                    # 由於我們在 get_mask_advanced 中已經處理過旋轉和預處理，直接使用模型的 forward 方法
                    # 但這裡我們需要一個不同的方法，因為我們已經有預處理過的圖像
                    # 因此，我們直接使用 Florence-2 的處理方式
                    # 創建一個臨時的 PIL 圖像
                    pil_img = Image.fromarray(processed)
                    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
                    text_input = f"{task_prompt} {search_text}"
                    
                    # 使用模型的處理器
                    inputs = model.processor(
                        text=text_input,
                        images=pil_img,
                        return_tensors="pt"
                    ).to(device)
                    
                    # 確保模型的 dtype 一致
                    if device == "cuda":
                        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
                    
                    with torch.inference_mode():
                        generated_ids = model.model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            num_beams=3
                        )
                    
                    generated_text = model.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=False
                    )[0]
                    
                    parsed = model.processor.post_process_generation(
                        generated_text,
                        task=task_prompt,
                        image_size=(pil_img.width, pil_img.height)
                    )
                except Exception as e:
                    logger.error(f"Florence-2 模型處理失敗: {e}")
                    continue
                
                if TaskType.OPEN_VOC_DET.value in parsed:
                    bboxes = parsed[TaskType.OPEN_VOC_DET.value].get("bboxes", [])
                    count = 0
                    for bbox in bboxes:
                        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        if (bw * bh) / (w * h) > (args.max_bbox_percent / 100.0):
                            continue
                            
                        rb = rotate_bbox(bbox, M_inv)
                        ox1, oy1, ox2, oy2 = int(rb[0]+x1), int(rb[1]+y1), int(rb[2]+x1), int(rb[3]+y1)
                        cv2.rectangle(final_mask, (max(0, ox1-args.expand), max(0, oy1-args.expand)),
                                      (min(w, ox2+args.expand), min(h, oy2+args.expand)), 255, -1)
                        count += 1
                    if count > 0:
                        logger.info(f"區塊 {i+1} 角度 {angle:3d}°: 偵測到 {count} 個目標")

    if np.any(final_mask):
        # 使用使用者傳入的 expand 參數作為基礎
        expand_size = int(args.expand) if 'args' in locals() else 5
        
        # 1. 為了不讓矩形直接相連，先做輕微的侵蝕 (Erosion) 讓框與框產生間隙
        # 這是為了解決浮水印太密集導致整個區域被塗白的問題
        kernel_sep = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        final_mask = cv2.erode(final_mask, kernel_sep, iterations=1)

        # 2. 進行「圓形」或「橢圓形」膨脹，而不是矩形
        # 這可以讓文字邊角比較圓潤，不會像方塊一樣生硬
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_size*2+1, expand_size*2+1))
        final_mask = cv2.dilate(final_mask, kernel_dilate, iterations=1)

        # 3. 強力羽化 (Blur): 這是讓 LaMa 修復自然的靈魂
        # 邊緣越模糊，AI 融合背景的能力越強
        #blur_val = max(5, expand_size * 2 + 1) # 確保為奇數
        #blur_val = max(1, 3)  # 確保為奇數
        #final_mask = cv2.GaussianBlur(final_mask, (blur_val, blur_val), 0)

    return final_mask

def process_image(path, model, processor, lama, args, clahe_mode):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None: return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 分支：偵測模式
    if args.model == 'yolo':
        if model is None:
            logger.error("YOLO 模型未正確載入，請檢查 ultralytics 是否安裝")
            return
        results = model.predict(img_rgb, imgsz=args.yolo_imgsz, conf=args.conf_threshold, iou=args.iou_threshold, verbose=False)
        mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if args.single_detection and len(boxes) > 0:
            boxes = boxes[:1]
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask, (x1-args.expand, y1-args.expand), (x2+args.expand, y2+args.expand), 255, -1)
    elif args.model == 'groundingdino':
        # 使用 DetectorEnsemble 進行 GroundingDINO 模型處理
        try:
            # 使用 DetectorEnsemble 進行偵測
            image_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            bboxes = model(
                image=image_tensor,
                prompt=args.watermark_text,
                detector_type="GroundingDINO",
                rotation_step=args.rotate_angle_step,
                conf_threshold=args.groundingdino_conf_threshold,
                iou_threshold=0.3
            )
            
            # 創建遮罩
            mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                bw, bh = x2 - x1, y2 - y1
                if (bw * bh) / (img_rgb.shape[0] * img_rgb.shape[1]) > (args.max_bbox_percent / 100.0):
                    continue
                cv2.rectangle(mask, (x1-args.expand, y1-args.expand), (x2+args.expand, y2+args.expand), 255, -1)
        except Exception as e:
            logger.error(f"GroundingDINO 模型處理失敗: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return
    else:
        # 當使用 Florence 模型時，確保使用正確的處理器
        if hasattr(model, 'processor') and model.processor is not None:
            mask = get_mask_advanced(img_rgb, model, model.processor, args.device, args, args.clahe_mode)
        else:
            # 如果處理器為 None，嘗試從 DetectorEnsemble 中獲取
            try:
                if hasattr(model, '_load_florence'):
                    model._load_florence()
                # 再次檢查處理器是否已正確載入
                if model.processor is not None:
                    mask = get_mask_advanced(img_rgb, model, model.processor, args.device, args, args.clahe_mode)
                else:
                    logger.error("無法載入 Florence 處理器: 處理器仍為 None")
                    # 創建一個空的遮罩避免錯誤
                    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
            except Exception as e:
                logger.error(f"無法載入 Florence 處理器: {e}")
                # 創建一個空的遮罩避免錯誤
                mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    
    # 如果啟用 SAM-2，則使用 SAM-2 進行遮罩優化
    if args.use_sam2 and SAM2_AVAILABLE:
        try:
            # 創建 SAM-2 segmentor
            sam2_segmentor = Sam2Segmentor(device=args.device)
            # 將 numpy mask 轉換為 PIL Image
            mask_pil = Image.fromarray(mask)
            # 創建一個簡單的 bounding box（使用 mask 的邊界框）
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                bboxes = [[x_min, y_min, x_max, y_max]]
                # 使用 SAM-2 進行 refine
                refined_mask = sam2_segmentor(Image.fromarray(img_rgb), bboxes, expand_pixels=1)
                # 保存 refine 後的遮罩
                mask_diag_path_sam2 = Path(args.output) / f"{path.stem}_mask_sam2.png"
                cv2.imwrite(str(mask_diag_path_sam2), refined_mask)
                # 使用 refine 後的遮罩
                mask = refined_mask
        except Exception as e:
            logger.warning(f"使用 SAM-2 遮罩優化失敗: {e}")
            # 如果 SAM-2 失敗，繼續使用原始遮罩
            pass
    
    # 診斷用 Mask 輸出 - 只在未使用 SAM-2 時產生 _mask.png
    if not args.use_sam2:
        mask_diag_path = Path(args.output) / f"{path.stem}_mask.png"
        cv2.imwrite(str(mask_diag_path), mask)

    # 產生遮罩標示圖檔 _mask_mark.png (紫色半透明)
    if np.any(mask):
        # 將遮罩轉換為彩色圖像，並設定為紫色
        mask_mark = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.uint8)
        mask_mark[:, :, 2] = mask  # 紅色通道
        mask_mark[:, :, 1] = 0     # 綠色通道
        mask_mark[:, :, 0] = mask  # 藍色通道
        
        # 設定遮罩圖像為半透明
        # 根據是否使用 SAM-2 決定檔名後綴
        if args.use_sam2:
            mask_mark_path = Path(args.output) / f"{path.stem}_mask_mark_sam2.png"
        else:
            mask_mark_path = Path(args.output) / f"{path.stem}_mask_mark.png"
        # 將原始圖像與遮罩圖像混合，產生半透明效果
        alpha = 0.5  # 透明度
        overlay = cv2.addWeighted(img_rgb, 1 - alpha, mask_mark, alpha, 0)
        cv2.imwrite(str(mask_mark_path), overlay)

    if np.any(mask):
        orig_h, orig_w = img_rgb.shape[:2]
        target_h = int(((min(orig_h, args.resize_limit) + 7) // 8) * 8 * args.lama_scale)
        target_w = int(((min(orig_w, args.resize_limit) + 7) // 8) * 8 * args.lama_scale)
        target_h, target_w = ((target_h+7)//8)*8, ((target_w+7)//8)*8

        img_pil = Image.fromarray(img_rgb).resize((target_w, target_h), Image.Resampling.LANCZOS)
        mask_pil = Image.fromarray(mask).resize((target_w, target_h), Image.Resampling.NEAREST)
        
        img_t = torch.from_numpy(np.array(img_pil).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(args.device)
        # Convert mask to tensor, threshold to binary, and move to device
        mask_tensor = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)
        mask_t = (mask_tensor.unsqueeze(0).unsqueeze(0) > 0.5).float().to(args.device)
        
        with torch.no_grad():
            res = lama(img_t * (1 - mask_t), mask_t)
            res = (np.clip(res[0].permute(1,2,0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            final = np.array(Image.fromarray(res).resize((orig_w, orig_h), Image.Resampling.LANCZOS))
    else:
        final = img_rgb
        logger.warning(f"{path.name} 未偵測到浮水印位置")

    # 確保只產生圖片格式的檔案
    save_path = Path(args.output) / (path.stem + "_no_watermark" + path.suffix)
    cv2.imwrite(str(save_path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    logger.info(f"已儲存圖片: {save_path}")

def process_video(path, detection_model, detection_processor, lama_model, args, clahe_mode):
    """處理影片 - 改進版"""
    logger.info(f"處理影片: {path.name}")
    
    # 打開影片
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error(f"無法打開影片: {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 創建臨時目錄和文件
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_video = tmp_dir / "tmp_no_audio.mp4"
    
    # 使用更兼容的編碼格式
    # 使用正確的數值方式避免類型檢查問題
    fourcc = 1983148141  # 'm', 'p', '4', 'v' 的正確數值表示 (0x7634706d)
    out = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))

    # 處理所有幀
    batch_frames_rgb = []
    last_mask = None
    last_frame = None
    same_frame_count = 0
    SKIP_SAME_FRAMES = 10  # 連續相同幀的最大跳過次數

    frame_count = 0
    while True:
        ret, frame_bgr = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # 記錄當前幀，不再跳過相同幀（避免掉幀）
            last_frame = frame_rgb.copy()
            batch_frames_rgb.append(frame_rgb)
        
        is_batch_full = len(batch_frames_rgb) == args.video_batch_size
        is_last_batch = (not ret) and (len(batch_frames_rgb) > 0)

        if is_batch_full or is_last_batch:
            # --- Batch Detection ---
            batch_masks = []
            if args.use_first_frame_detection and last_mask is not None:
                batch_masks = [last_mask] * len(batch_frames_rgb)
            else:
                if args.model == 'yolo':
                    # Run YOLO detection on the entire batch at once
                    # 降低YOLO檢測解析度，加速檢測
                    yolo_results = detection_model.predict(
                        batch_frames_rgb,
                        imgsz=args.yolo_imgsz,
                        conf=args.conf_threshold,
                        iou=args.iou_threshold,
                        verbose=False,
                        half=args.half_precision,  # 啟用半精度檢測
                        device=args.device
                    )

                    # Create masks from the batch results
                    for i, result in enumerate(yolo_results):
                        boxes = result.boxes.xyxy.cpu().numpy()
                        if args.single_detection and len(boxes) > 0:
                            boxes = boxes[:1]
                        
                        mask = np.zeros(batch_frames_rgb[i].shape[:2], dtype=np.uint8)
                        if boxes.size == 0:
                            batch_masks.append(mask)
                            continue
                        
                        img_area = batch_frames_rgb[i].shape[0] * batch_frames_rgb[i].shape[1]
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            if img_area > 0 and ((x2 - x1) * (y2 - y1) / img_area) * 100.0 > args.max_bbox_percent:
                                continue
                            x1, y1, x2, y2 = max(0, x1 - args.expand), max(0, y1 - args.expand), min(batch_frames_rgb[i].shape[1], x2 + args.expand), min(batch_frames_rgb[i].shape[0], y2 + args.expand)
                            mask[y1:y2, x1:x2] = 255
                        batch_masks.append(mask)
                    
                    if batch_masks:
                        last_mask = batch_masks[-1]
                    
                    # 當使用 --use-first-frame-detection 時，應該產生遮罩檔
                    # 但避免與後面的遮罩產生邏輯重複
                    # 由於這裡是 YOLO 模型處理，我們只在第一個批次時產生遮罩檔
                    if args.use_first_frame_detection and len(batch_frames_rgb) > 0 and last_mask is not None:
                        # 只在處理第一個批次時保存首幀遮罩
                        if frame_count == 0:
                            # 如果啟用 SAM-2，則使用 SAM-2 進行遮罩優化
                            if args.use_sam2 and SAM2_AVAILABLE:
                                try:
                                    # 創建 SAM-2 segmentor
                                    sam2_segmentor = Sam2Segmentor(device=args.device)
                                    # 創建一個簡單的 bounding box（使用 mask 的邊界框）
                                    coords = np.where(last_mask > 0)
                                    if len(coords[0]) > 0:
                                        y_min, y_max = coords[0].min(), coords[0].max()
                                        x_min, x_max = coords[1].min(), coords[1].max()
                                        bboxes = [[x_min, y_min, x_max, y_max]]
                                        # 使用 SAM-2 進行 refine
                                        refined_mask = sam2_segmentor(Image.fromarray(batch_frames_rgb[0]), bboxes, expand_pixels=1)
                                        # 使用 refine 後的遮罩
                                        last_mask = refined_mask
                                except Exception as e:
                                    logger.warning(f"使用 SAM-2 遮罩優化失敗: {e}")
                                    # 如果 SAM-2 失敗，繼續使用原始遮罩
                                    pass
                        
                            # 確保產生遮罩檔
                            if args.use_sam2:
                                mask_diag_path = Path(args.output) / f"{path.stem}_mask_sam2.png"
                            else:
                                mask_diag_path = Path(args.output) / f"{path.stem}_mask.png"
                            cv2.imwrite(str(mask_diag_path), last_mask)
                            logger.info(f"已儲存首幀遮罩: {mask_diag_path}")
                            
                            # 產生遮罩標示圖檔 _mask_mark.png (紫色半透明)
                            if np.any(last_mask):
                                # 將遮罩轉換為彩色圖像，並設定為紫色
                                mask_mark = np.zeros((last_mask.shape[0], last_mask.shape[1], 3), dtype=np.uint8)
                                mask_mark[:, :, 2] = last_mask  # 紅色通道
                                mask_mark[:, :, 1] = 0     # 綠色通道
                                mask_mark[:, :, 0] = last_mask  # 藍色通道
                                
                                # 設定遮罩圖像為半透明
                                # 根據是否使用 SAM-2 決定檔名後綴
                                if args.use_sam2:
                                    mask_mark_path = Path(args.output) / f"{path.stem}_mask_mark_sam2.png"
                                else:
                                    mask_mark_path = Path(args.output) / f"{path.stem}_mask_mark.png"
                                # 將原始圖像與遮罩圖像混合，產生半透明效果
                                alpha = 0.5  # 透明度
                                overlay = cv2.addWeighted(batch_frames_rgb[0], 1 - alpha, mask_mark, alpha, 0)
                                cv2.imwrite(str(mask_mark_path), overlay)
                elif args.model == 'groundingdino':
                    # 使用 DetectorEnsemble 進行 GroundingDINO 模型處理
                    try:
                        for frame_rgb in batch_frames_rgb:
                            # 使用 DetectorEnsemble 進行偵測
                            image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                            bboxes = detection_model(
                                image=image_tensor,
                                prompt=args.watermark_text,
                                detector_type="GroundingDINO",
                                rotation_step=args.rotate_angle_step,
                                conf_threshold=args.groundingdino_conf_threshold,
                                iou_threshold=0.3
                            )
                            
                            mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
                            for bbox in bboxes:
                                x1, y1, x2, y2 = map(int, bbox)
                                bw, bh = x2 - x1, y2 - y1
                                if (bw * bh) / (frame_rgb.shape[0] * frame_rgb.shape[1]) > (args.max_bbox_percent / 100.0):
                                    continue
                                cv2.rectangle(mask, (x1-args.expand, y1-args.expand), (x2+args.expand, y2+args.expand), 255, -1)
                            batch_masks.append(mask)
                            last_mask = mask
                            
                            # 當使用 --use-first-frame-detection 時，應該產生遮罩檔
                            # 但避免與後面的遮罩產生邏輯重複
                            # 由於這裡是 GroundingDINO 模型處理，我們只在第一個批次時產生遮罩檔
                            if args.use_first_frame_detection and frame_count + i == 0:
                                # 如果啟用 SAM-2，則使用 SAM-2 進行遮罩優化
                                if args.use_sam2 and SAM2_AVAILABLE:
                                    try:
                                        # 創建 SAM-2 segmentor
                                        sam2_segmentor = Sam2Segmentor(device=args.device)
                                        # 創建一個簡單的 bounding box（使用 mask 的邊界框）
                                        coords = np.where(mask > 0)
                                        if len(coords[0]) > 0:
                                            y_min, y_max = coords[0].min(), coords[0].max()
                                            x_min, x_max = coords[1].min(), coords[1].max()
                                            bboxes = [[x_min, y_min, x_max, y_max]]
                                            # 使用 SAM-2 進行 refine
                                            refined_mask = sam2_segmentor(Image.fromarray(frame_rgb), bboxes, expand_pixels=1)
                                            # 使用 refine 後的遮罩
                                            mask = refined_mask
                                    except Exception as e:
                                        logger.warning(f"使用 SAM-2 遮罩優化失敗: {e}")
                                        # 如果 SAM-2 失敗，繼續使用原始遮罩
                                        pass
                        
                                # 確保產生遮罩檔
                                if args.use_sam2:
                                    mask_diag_path = Path(args.output) / f"{path.stem}_mask_sam2.png"
                                else:
                                    mask_diag_path = Path(args.output) / f"{path.stem}_mask.png"
                                cv2.imwrite(str(mask_diag_path), mask)
                                logger.info(f"已儲存首幀遮罩: {mask_diag_path}")
                                
                                # 產生遮罩標示圖檔 _mask_mark.png (紫色半透明)
                                if np.any(mask):
                                    # 將遮罩轉換為彩色圖像，並設定為紫色
                                    mask_mark = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                                    mask_mark[:, :, 2] = mask  # 紅色通道
                                    mask_mark[:, :, 1] = 0     # 綠色通道
                                    mask_mark[:, :, 0] = mask  # 藍色通道
                       
                                    # 設定遮罩圖像為半透明
                                    # 根據是否使用 SAM-2 決定檔名後綴
                                    if args.use_sam2:
                                        mask_mark_path = Path(args.output) / f"{path.stem}_mask_mark_sam2.png"
                                    else:
                                        mask_mark_path = Path(args.output) / f"{path.stem}_mask_mark.png"
                                    # 將原始圖像與遮罩圖像混合，產生半透明效果
                                    alpha = 0.5  # 透明度
                                    overlay = cv2.addWeighted(frame_rgb, 1 - alpha, mask_mark, alpha, 0)
                                    cv2.imwrite(str(mask_mark_path), overlay)
                    except Exception as e:
                        logger.error(f"GroundingDINO 模型處理失敗: {e}")
                        return
                else: # Florence (remains single-frame processing in loop)
                    for i, frame_rgb in enumerate(batch_frames_rgb):
                        mask = get_mask_advanced(frame_rgb, detection_model, detection_processor, args.device, args, args.clahe_mode)
                        # 當使用 --use-first-frame-detection 時，應該產生遮罩檔
                        # 但避免與後面的遮罩產生邏輯重複
                        if args.use_first_frame_detection and frame_count + i == 0:
                            # 如果啟用 SAM-2，則使用 SAM-2 進行遮罩優化
                            if args.use_sam2 and SAM2_AVAILABLE:
                                try:
                                    # 創建 SAM-2 segmentor
                                    sam2_segmentor = Sam2Segmentor(device=args.device)
                                    # 創建一個簡單的 bounding box（使用 mask 的邊界框）
                                    coords = np.where(mask > 0)
                                    if len(coords[0]) > 0:
                                        y_min, y_max = coords[0].min(), coords[0].max()
                                        x_min, x_max = coords[1].min(), coords[1].max()
                                        bboxes = [[x_min, y_min, x_max, y_max]]
                                        # 使用 SAM-2 進行 refine
                                        refined_mask = sam2_segmentor(Image.fromarray(frame_rgb), bboxes, expand_pixels=1)
                                        # 使用 refine 後的遮罩
                                        mask = refined_mask
                                except Exception as e:
                                    logger.warning(f"使��� SAM-2 遮罩優化失敗: {e}")
                                    # 如果 SAM-2 失敗，繼續使用原始遮罩
                                    pass
                            
                            # 確保產生遮罩檔
                            if args.use_sam2:
                                mask_diag_path = Path(args.output) / f"{path.stem}_mask_sam2.png"
                            else:
                                mask_diag_path = Path(args.output) / f"{path.stem}_mask.png"
                            cv2.imwrite(str(mask_diag_path), mask)
                            logger.info(f"已儲存首幀遮罩: {mask_diag_path}")
      
                            # 產生遮罩標示圖檔 _mask_mark.png (紫色半透明)
                            if np.any(mask):
                                # 將遮罩轉換為彩色圖像，並設定為紫色
                                mask_mark = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                                mask_mark[:, :, 2] = mask  # 紅色通道
                                mask_mark[:, :, 1] = 0     # 綠色通道
                                mask_mark[:, :, 0] = mask  # 藍色通道
      
                                # 設定遮罩圖像為半透明
                                # 根據是否使用 SAM-2 決定檔名後綴
                                if args.use_sam2:
                                    mask_mark_path = Path(args.output) / f"{path.stem}_mask_mark_sam2.png"
                                else:
                                    mask_mark_path = Path(args.output) / f"{path.stem}_mask_mark.png"
                                # 將原始圖像與遮罩圖像混合，產生半透明效果
                                alpha = 0.5  # 透明度
                                overlay = cv2.addWeighted(frame_rgb, 1 - alpha, mask_mark, alpha, 0)
                                cv2.imwrite(str(mask_mark_path), overlay)
                        batch_masks.append(mask)
                        last_mask = mask
            
            # 保存首幀遮罩（如果啟用了首幀遮罩功能）
            # 由於前面的處理邏輯已經產生了遮罩檔，這裡不再重複處理
            # 避免重複產生遮罩檔
            # 這裡的邏輯實際上是多餘的，因為所有模型處理邏輯都已經產生了遮罩檔
            # 這個邏輯應該被移除，避免重複產生遮罩檔
            pass
            
            # --- Batch Inpainting ---
            frames_to_inpaint = []
            masks_to_inpaint = []
            inpaint_indices = []

            for i, (frame, mask) in enumerate(zip(batch_frames_rgb, batch_masks)):
                if np.any(mask):
                    frames_to_inpaint.append(frame)
                    masks_to_inpaint.append(mask)
                    inpaint_indices.append(i)
            
            final_batch_frames = list(batch_frames_rgb)
            if frames_to_inpaint:
                # 並行優化：如果有多个GPU，可以考慮數據並行
                # 這裡保持簡化實現，使用單線程處理
                for i, (frame, mask) in enumerate(zip(frames_to_inpaint, masks_to_inpaint)):
                    if np.any(mask):
                        orig_h, orig_w = frame.shape[:2]
                        target_h = int(((min(orig_h, args.resize_limit) + 7) // 8) * 8 * args.lama_scale)
                        target_w = int(((min(orig_w, args.resize_limit) + 7) // 8) * 8 * args.lama_scale)
                        target_h, target_w = ((target_h+7)//8)*8, ((target_w+7)//8)*8

                        img_pil = Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.LANCZOS)
                        mask_pil = Image.fromarray(mask).resize((target_w, target_h), Image.Resampling.NEAREST)
                        
                        img_t = torch.from_numpy(np.array(img_pil).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(args.device)
                        mask_t = (torch.from_numpy(np.array(mask_pil).astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0) > 0.5).float().to(args.device)
                        
                        with torch.no_grad():
                            res = lama_model(img_t * (1 - mask_t), mask_t)
                            res = (np.clip(res[0].permute(1,2,0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                            inpainted_frame = np.array(Image.fromarray(res).resize((orig_w, orig_h), Image.Resampling.LANCZOS))
                        
                        # 將修復後的幀放回對應位置
                        final_batch_frames[inpaint_indices[i]] = inpainted_frame

            # --- Batch Write ---
            for frame_rgb in final_batch_frames:
                out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            frame_count += len(batch_frames_rgb)
            logger.info(f"已處理 {frame_count} 幀")

            batch_frames_rgb.clear()

        if not ret:
            break

    cap.release()
    out.release()

    # Add suffix to output filename to avoid overwriting original
    out_filename = path.stem + "_no_watermark" + path.suffix
    output_path = Path(args.output) / out_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 檢查硬體加速
        video_codec = "libx264"
        if args.device == "cuda":
            try:
                subprocess.check_output(["ffmpeg", "-h", "encoder=h264_nvenc"], stderr=subprocess.STDOUT)
                video_codec = "h264_nvenc"
                logger.info("使用 NVIDIA GPU (h264_nvenc) 進行硬體加速視頻編碼。")
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("未找到 h264_nvenc 編碼器，將使用 CPU (libx264) 進行視頻編碼。")

        # 合併音頻
        cmd = ["ffmpeg", "-y", "-i", str(tmp_video), "-i", str(path), "-map", "0:v:0", "-map", "1:a?", "-c:v", video_codec, "-c:a", "copy", str(output_path)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"已合併音頻並輸出: {output_path.name}")
        except subprocess.SubprocessError as e:
            # 處理 ffmpeg 執行錯誤
            logger.warning(f"ffmpeg 執行錯誤: {e}. 將複製無音頻視頻。")
            shutil.copyfile(tmp_video, output_path)
    except Exception as e:
        logger.warning(f"合併音頻失敗: {e}. 將複製無音頻視頻。")
        shutil.copyfile(tmp_video, output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser()
    # 基礎參數
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", default="florence", choices=["yolo", "florence", "groundingdino"])
    parser.add_argument("--resize-limit", type=int, default=640)
    parser.add_argument("--expand", type=int, default=5)
    parser.add_argument("--max-bbox-percent", type=float, default=20.0)
    parser.add_argument("--lama-scale", type=float, default=0.6)
    parser.add_argument("--rotate-angle-step", type=int, default=0, help="Rotation step in degrees (0-90)")
    parser.add_argument("--tiles-per-axis", type=int, default=2)
    parser.add_argument("--overlap-ratio", type=float, default=0.10)
    parser.add_argument("--half-precision", action="store_true", default=True)
    parser.add_argument("--no-half-precision", dest="half_precision", action="store_false")
    parser.add_argument("--use-sam2", action="store_true", help="Use SAM-2 for mask refinement")
    parser.add_argument("--clahe-mode", choices=CLAHE_MODES, default="off", help="CLAHE preprocessing mode")

    # Florence 專用
    parser.add_argument("--model-size", default="large")
    parser.add_argument("--watermark-text", default="watermark")
    
    # GroundingDINO 專用
    parser.add_argument("--groundingdino-conf-threshold", type=float, default=0.35)
    parser.add_argument("--groundingdino-text-threshold", type=float, default=0.25)

    # YOLO 專用
    parser.add_argument("--yolo-model", default="models/yolo.pt")
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--single-detection", action="store_true")

    # 影片相關 (預留)
    parser.add_argument("--video-batch-size", type=int, default=None, help="Video batch size for processing")
    parser.add_argument("--use-first-frame-detection", action="store_true")

    args = parser.parse_args()

    # 載入偵測模型
    det_model, det_proc = None, None
    if args.model == 'yolo':
        logger.info(f"載入 YOLO 模型: {args.yolo_model}")
        if YOLO is not None:
            det_model = YOLO(args.yolo_model)
        else:
            logger.error("YOLO 類別未正確匯入，請檢查 ultralytics 套件是否正確安裝")
            return
    elif args.model == 'groundingdino':
        # 使用 DetectorEnsemble 來處理 GroundingDINO 模型
        logger.info("正在載入 GroundingDINO 模型")
        try:
            det_model = DetectorEnsemble(device=args.device)
            det_proc = None
        except Exception as e:
            logger.error(f"載入 GroundingDINO 模型失敗: {e}")
            return
    else:
        # 使用 DetectorEnsemble 來處理 Florence 模型
        logger.info("正在載入 Florence 模型")
        try:
            det_model = DetectorEnsemble(device=args.device, model_size=args.model_size)
            # 確保在 Florence 模式下正確初始化處理器
            if det_model.processor is None:
                det_model._load_florence()
            # 再次檢查處理器是否已正確載入
            if det_model.processor is None:
                logger.error("無法載入 Florence 處理器: 處理器仍為 None")
                # 創建一個空的處理器避免後續錯誤
                det_model.processor = None
            det_proc = det_model.processor
        except Exception as e:
            logger.error(f"載入 Florence 模型失敗: {e}")
            return
    
    # 載入修復模型
    try:
        lama = torch.jit.load(LAMA_MODEL_PATH, map_location=args.device)
    except:
        lama = torch.load(LAMA_MODEL_PATH, map_location=args.device)
    lama.to(args.device).eval()

    in_p = Path(args.input)
    files = [in_p] if in_p.is_file() else [f for f in in_p.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    
    # 檢查是否為影片格式
    if in_p.is_file() and in_p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}:
        logger.info("偵測到影片輸入，將使用影片處理模式")
        process_video(in_p, det_model, det_proc, lama, args, args.clahe_mode)
    else:
        # 只處理圖片檔案
        image_files = [f for f in files if f.suffix.lower() in IMAGE_EXTS]
        for f in image_files:
            process_image(f, det_model, det_proc, lama, args, args.clahe_mode)
    
    logger.info("✅ 處理完成！")

# =========================================================
# Detector Ensemble
# =========================================================
class DetectorEnsemble(torch.nn.Module):
    def __init__(self, device: str = "cuda", model_size: str = "large") -> None:
        super().__init__()

        self.device = device
        self.active_type: str | None = None
        self.model: Any = None
        self.processor: Any = None

        self.use_amp = device == "cuda"

        self.dino_id = "IDEA-Research/grounding-dino-base"
        self.florence_id = f"microsoft/Florence-2-{model_size}"

    # -------------------------
    # Model Loaders
    # -----------------------__
    def _clear_model(self):
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        self.model = None
        self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _load_dino(self):
        self.processor = AutoProcessor.from_pretrained(self.dino_id)

        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(self.dino_id)
            .to(self.device)
            .eval()
        )

        self.active_type = "GroundingDINO"

    def _load_florence(self):
        self.processor = AutoProcessor.from_pretrained(
            self.florence_id,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.florence_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="sdpa"
        ).to(self.device).eval()

        self.active_type = "Florence-2"

    # -----------------------__
    # Forward
    # -----------------------__
    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        prompt: str,
        detector_type: str = "GroundingDINO",
        rotation_step: int = 45,
        conf_threshold: float = 0.3,
        enhance_det: bool = False,
        iou_threshold: float = 0.3
    ) -> List[List[float]]:

        # 動態切換模型
        if self.active_type != detector_type:
            self._clear_model()
            if detector_type == "GroundingDINO":
                self._load_dino()
            else:
                self._load_florence()

        assert self.model is not None
        assert self.processor is not None

        # Tensor -> numpy
        img_np = (
            image.permute(1, 2, 0).float().cpu().numpy() * 255
        ).astype(np.uint8)

        h, w = img_np.shape[:2]
        mapper = GeometryMapper(w, h)

        # Collect all bounding boxes and scores without strict type annotations
        all_bboxes = []
        all_scores = []

        angles = [0] if rotation_step <= 0 else list(range(0, 360, rotation_step))

        # =================================================
        # Main Loop
        # =================================================
        for angle in angles:

            t_img, M_inv = mapper.get_transformed_image(img_np, angle=angle)

            # =========================
            # GroundingDINO
            # =========================
            if detector_type == "GroundingDINO":

                final_prompt = prompt.strip()
                if not final_prompt.endswith("."):
                    final_prompt += "."

                inputs = self.processor(
                    images=Image.fromarray(t_img),
                    text=final_prompt,
                    return_tensors="pt"
                ).to(self.device)

                try:
                    with torch.autocast(
                        device_type="cuda" if self.device == "cuda" else "cpu",
                        dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        enabled=self.use_amp and self.device == "cuda"
                    ):
                        outputs = self.model(**inputs)

                    # 直接使用位置參數調用以避免版本兼容性問題
                    res = self.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        conf_threshold,
                        target_sizes=[t_img.shape[:2]]
                    )[0]

                    current_bboxes = res["boxes"].detach().cpu().tolist()
                    current_scores = res["scores"].detach().cpu().tolist()
                except Exception as e:
                    # 如果 CUDA 編譯失敗，則使用 CPU 模式
                    if self.device == "cuda":
                        print(f"Warning: CUDA error in GroundingDINO, falling back to CPU mode: {e}")
                        # 嘗試在 CPU 上運行
                        inputs = inputs.to("cpu")
                        with torch.autocast(device_type="cpu", dtype=torch.float32):
                            outputs = self.model(**inputs)
                        res = self.processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            conf_threshold,
                            target_sizes=[t_img.shape[:2]]
                        )[0]
                        current_bboxes = res["boxes"].detach().cpu().tolist()
                        current_scores = res["scores"].detach().cpu().tolist()
                    else:
                        raise e

            # =========================
            # Florence-2
            # =========================
            else:

                task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
                text_input = f"{task_prompt} {prompt}"

                pil_img = Image.fromarray(t_img)

                inputs = self.processor(
                    text=text_input,
                    images=pil_img,
                    return_tensors="pt"
                ).to(self.device)

                # 🔥 保證 dtype 一致
                if self.device == "cuda":
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

                try:
                    with torch.autocast(
                        device_type="cuda" if self.device == "cuda" else "cpu",
                        dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        enabled=self.use_amp and self.device == "cuda"
                    ):
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            num_beams=3
                        )
                except Exception as e:
                    # 如果 CUDA 編譯失敗，則使用 CPU 模式
                    if self.device == "cuda":
                        print(f"Warning: CUDA error in Florence-2, falling back to CPU mode: {e}")
                        # 嘗試在 CPU 上運行
                        inputs = inputs.to("cpu")
                        with torch.autocast(device_type="cpu", dtype=torch.float32):
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=512,
                                num_beams=3
                            )
                    else:
                        raise e

                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=False
                )[0]

                parsed = self.processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(pil_img.width, pil_img.height)
                )

                current_bboxes = parsed.get(task_prompt, {}).get("bboxes", [])
                current_scores = [1.0] * len(current_bboxes)

            # =========================
            # Mapping Back
            # =========================
            for box, score in zip(current_bboxes, current_scores):

                mapped_box = mapper.map_bbox_back(
                    list(map(float, box)),
                    M_inv,
                    (0, 0),
                    False,
                    False
                )

                all_bboxes.append(mapped_box)
                all_scores.append(float(score))

        if not all_bboxes:
            return []

        keep = apply_nms(all_bboxes, all_scores, iou_threshold=iou_threshold)
        return [all_bboxes[i] for i in keep]


# =========================================================
# SAM2 Segmentor
# =========================================================
class Sam2Segmentor(torch.nn.Module):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()

        weights_dir = os.path.abspath("weights")
        checkpoint = os.path.join(weights_dir, "sam2_hiera_large.pt")
        config = os.path.join(weights_dir, "sam2_hiera_l.yaml")

        self.model = build_sam2(config, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.model)

    def forward(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
        expand_pixels: int = 1
    ) -> np.ndarray:

        self.predictor.set_image(image)

        w, h = image.size
        final_mask = np.zeros((h, w), dtype=np.uint8)

        for bbox in bboxes:
            # 確保 bbox 是 float 類型
            bbox_array = np.array(bbox, dtype=np.float32)
            masks, _, _ = self.predictor.predict(
                box=bbox_array[None, :],
                multimask_output=False
            )

            mask = (masks[0] > 0).astype(np.uint8) * 255
            final_mask = cv2.bitwise_or(final_mask, mask)

        if expand_pixels > 0:
            kernel = np.ones((expand_pixels, expand_pixels), np.uint8)
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        return final_mask


# =========================================================
# LaMa Inpaint
# =========================================================
class InpaintEngine:
    def __init__(self, device: str = "cuda") -> None:
        # 使用已存在的 lama 模型
        self.lama = None
        self.device = device

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        rescale_factor: float = 1.0
    ) -> Image.Image:

        # 這個方法在目前的程式碼中並未被使用，因為直接在 process_image 和 process_video 中使用了 lama 模型
        # 保留此方法以保持兼容性，但實際上不會被調用
        raise NotImplementedError("InpaintEngine.inpaint 方法未實現，請直接使用 lama 模型")


if __name__ == "__main__":
    main()