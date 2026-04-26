import gradio as gr
from pathlib import Path
import subprocess
import os
import sys
import warnings
import numpy as np  
import cv2


# 忽略 timm 的 FutureWarning 讓輸出保持乾淨
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

# 定義圖片和影片副檔名
IMAGE_EXTS = { ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = { ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}


# ================================================
# CLI PREVIEW BUILDER
# ================================================

# 引入 OpenCV 和 numpy 用於遮罩標示圖處理

def build_cli_preview(*args):
    (
        model, input_file,
        resize_limit, expand, max_bbox_percent, half_precision, lama_scale,
        rotate_angle_step, tiles_per_axis, overlap_ratio,
        conf_threshold, iou_threshold, single_detection, yolo_imgsz,
        florence_mode, florence_prompt, model_size,
        watermark_text_gd, groundingdino_conf_threshold,
        use_first_frame_mask, use_sam2, video_batch_size, clahe_mode
    ) = args

    if input_file is None:
        return "Waiting for input file..."

    cmd = [
        "python main.py",
        f"--model {model}",
        f"--resize-limit {resize_limit}",
        f"--expand {expand}",
        f"--max-bbox-percent {max_bbox_percent}",
        f"--lama-scale {lama_scale}",
        f"--rotate-angle-step {rotate_angle_step}",
        f"--tiles-per-axis {tiles_per_axis}",
        f"--overlap-ratio {overlap_ratio}",
        f"--video-batch-size {video_batch_size}",
    ]
    
    if use_first_frame_mask:
        cmd.append("--use-first-frame-detection")

    if half_precision:
        cmd.append("--half-precision")
    if use_sam2:
        cmd.append("--use-sam2")
    
    if clahe_mode != "off":
        cmd.append("--clahe-mode")
        cmd.append(clahe_mode)

    if model == "yolo":
        cmd.extend([
            f"--conf-threshold {conf_threshold}",
            f"--iou-threshold {iou_threshold}",
            f"--yolo-imgsz {yolo_imgsz}"
        ])
        if single_detection:
            cmd.append("--single-detection")

    elif model == "florence":
        cmd.append(f"--model-size {model_size}")

        if florence_mode == "Auto Text (OCR)":
            cmd.append("--florence-task OCR_with_region")
        elif florence_mode == "Targeted Text (Prompt-based)":
            cmd.append("--florence-task Referring_expression_segmentation")
            cmd.append(f"--watermark-text \"{florence_prompt}\"")
        elif florence_mode == "Visual Region Analysis (Dense Caption)":
            cmd.append("--florence-task Dense_region_caption")
        elif florence_mode == "Smart Mode (Auto Detect)":
            cmd.append("--florence-task Smart")

    elif model == "groundingdino":
        cmd.append(f"--watermark-text \"{watermark_text_gd}\"")
        cmd.append(f"--groundingdino-conf-threshold {groundingdino_conf_threshold}")

    cli_preview = "CLI Preview:\n\n" + " \\\n".join(cmd)
    return cli_preview


# ================================================
# HIGH SPEED / HIGH QUALITY PRESETS
# ================================================
def preset_speed():
    return (
        640,   # resize_limit (保留細節但仍快)
        5,     # expand
        20.0,  # max_bbox_percent
        0.6,   # lama_scale
        0,     # rotate_angle_step
        2,     # tiles_per_axis
        0.10,  # overlap_ratio
        True,  # fp16
        False  # use_sam2
    )

def preset_quality():
    return (
        2048,  # resize_limit
        12,    # expand
        30.0,  # max_bbox_percent
        1.0,   # lama_scale
        5,     # rotate_angle_step
        8,     # tiles_per_axis
        0.35,  # overlap_ratio
        False, # fp16 (全精度)
        True   # use_sam2
    )


# ================================================
# MAIN PROCESS
# ================================================
def process_media(*args):

    (
        model, input_file,
        resize_limit, expand, max_bbox_percent, half_precision, lama_scale,
        rotate_angle_step, tiles_per_axis, overlap_ratio,
        conf_threshold, iou_threshold, single_detection, yolo_imgsz,
        florence_mode, florence_prompt, model_size,
        watermark_text_gd, groundingdino_conf_threshold,
        use_first_frame_mask, use_sam2, video_batch_size, clahe_mode
    ) = args

    if input_file is None:
        return None, None, "Please upload a file.", None, None

    input_path = Path(input_file)
    output_dir = Path("output_webui")
    output_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable, "-u", "main.py",
        "--input", str(input_path),
        "--output", str(output_dir),
        "--model", model,
        "--resize-limit", str(resize_limit),
        "--expand", str(expand),
        "--max-bbox-percent", str(max_bbox_percent),
        "--lama-scale", str(lama_scale),
        "--rotate-angle-step", str(rotate_angle_step),
        "--tiles-per-axis", str(tiles_per_axis),
        "--overlap-ratio", str(overlap_ratio),
        "--video-batch-size", str(video_batch_size),
    ]
    
    if use_first_frame_mask:
        cmd.append("--use-first-frame-detection")

    if half_precision:
        cmd.append("--half-precision")
    else:
        cmd.append("--no-half-precision")
    if use_sam2:
        cmd.append("--use-sam2")
    
    if clahe_mode != "off":
        cmd.append("--clahe-mode")
        cmd.append(clahe_mode)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 獲取 CLI 預覽
    cli_preview = build_cli_preview(*args)
    
    log = cli_preview + "\n\n=== EXECUTION LOG ===\n"
    if process.stdout is not None:
        for line in process.stdout:
            log += line
            yield None, None, log, None, None

    process.wait()

    # 找到處理後的輸出檔案
    output_file = None

    if input_path.is_file():
        # 檢查輸出目錄中的檔案
        # 首先使用原始輸入檔名模式搜尋以"_no_watermark"結尾的檔案
        output_files = list(output_dir.glob(f"{input_path.stem}_no_watermark*"))
        
        # 如果找不到，再嘗試搜尋所有以"_no_watermark"結尾的檔案
        if not output_files:
            output_files = list(output_dir.glob(f"*_no_watermark*"))
        
        # 如果是圖片
        if input_path.suffix.lower() in IMAGE_EXTS:
            image_output_files = [f for f in output_files if f.suffix.lower() in IMAGE_EXTS]
            if image_output_files:
                output_file = str(image_output_files[0])
        # 如果是影片
        elif input_path.suffix.lower() in VIDEO_EXTS:
            video_output_files = [f for f in output_files if f.suffix.lower() in VIDEO_EXTS]
            if video_output_files:
                output_file = str(video_output_files[0])
        # 其他情況
        else:
            if output_files:
                output_file = str(output_files[0])

    # 產生遮罩標示圖檔 _mask_mark.png
    # 找到原始輸入檔案的遮罩圖檔
    mask_files = list(output_dir.glob(f"{input_path.stem}_mask.*"))
    if mask_files:
        mask_file = mask_files[0]
        
        # 讀取遮罩圖像
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # 讀取原始圖片
            original_img = cv2.imread(str(input_path))
            if original_img is not None:
                # 將遮罩轉換為彩色圖像，並設定為紫色
                mask_mark = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                mask_mark[:, :, 2] = mask  # 紅色通道
                mask_mark[:, :, 1] = 0     # 綠色通道
                mask_mark[:, :, 0] = mask  # 藍色通道
                
                # 設定遮罩圖像為半透明
                # 根據遮罩檔案名稱決定檔名後綴
                # 檢查是否有 _mask_sam2.png 檔案，如果有則使用 _sam2 後綴
                if mask_file.name.endswith("_mask_sam2.png"):
                    mask_mark_path = output_dir / f"{input_path.stem}_mask_mark_sam2.png"
                else:
                    mask_mark_path = output_dir / f"{input_path.stem}_mask_mark.png"
                # 將原始圖像與遮罩圖像混合，產生半透明效果
                alpha = 0.5  # 透明度
                overlay = cv2.addWeighted(original_img, 1 - alpha, mask_mark, alpha, 0)
                cv2.imwrite(str(mask_mark_path), overlay)
    # 如果是影片處理且使用了 --use-first-frame-detection，則在輸出目錄中尋找可能的遮罩檔
    # 這適用於使用 --use-first-frame-detection 時產生的遮罩
    elif input_path.suffix.lower() in VIDEO_EXTS and use_first_frame_mask:
        # 尋找可能的遮罩檔（不指定副檔名）
        mask_files = list(output_dir.glob(f"{input_path.stem}_mask*"))
        if mask_files:
            mask_file = mask_files[0]
            # 讀取遮罩圖像
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # 讀取原始圖片
                original_img = cv2.imread(str(input_path))
                if original_img is not None:
                    # 將遮罩轉換為彩色圖像，並設定為紫色
                    mask_mark = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                    mask_mark[:, :, 2] = mask  # 紅色通道
                    mask_mark[:, :, 1] = 0     # 綠色通道
                    mask_mark[:, :, 0] = mask  # 藍色通道
                    
                    # 設定遮罩圖像為半透明
                    # 根據遮罩檔案名稱決定檔名後綴
                    # 檢查是否有 _mask_sam2.png 檔案，如果有則使用 _sam2 後綴
                    if mask_file.name.endswith("_mask_sam2.png"):
                        mask_mark_path = output_dir / f"{input_path.stem}_mask_mark_sam2.png"
                    else:
                        mask_mark_path = output_dir / f"{input_path.stem}_mask_mark.png"
                    # 將原始圖像與遮罩圖像混合，產生半透明效果
                    alpha = 0.5  # 透明度
                    overlay = cv2.addWeighted(original_img, 1 - alpha, mask_mark, alpha, 0)
                    cv2.imwrite(str(mask_mark_path), overlay)
    
    # 新增：自動檢查並顯示遮罩圖像
    # 檢查 output_webui 資料夾中是否有產生的遮罩檔案
    # 如果有 _mask.png 或 _mask_sm2.png 檔案，則顯示在遮罩呈現區塊
    # 如果有 _mask_mark.png 或 _mask_mark_sm2.png 檔案，則顯示在遮罩呈現區塊
    mask_output_path = None
    mask_overlay_path = None
    
    # 檢查遮罩圖像
    mask_files = list(output_dir.glob(f"{input_path.stem}_mask*"))
    if mask_files:
        # 根據是否使用 SAM-2 決定檔名後綴
        if use_sam2:
            # 檢查是否有 _mask_sam2.png 檔案
            for mask_file in mask_files:
                if mask_file.name.endswith("_mask_sam2.png"):
                    mask_output_path = str(mask_file)
                    break
        else:
            # 檢查是否有 _mask.png 檔案
            for mask_file in mask_files:
                if mask_file.name.endswith("_mask.png"):
                    mask_output_path = str(mask_file)
                    break
    
    # 檢查遮罩標示圖像(紫色覆蓋)
    mask_mark_files = list(output_dir.glob(f"{input_path.stem}_mask_mark*"))
    if mask_mark_files:
        # 根據是否使用 SAM-2 決定檔名後綴
        if use_sam2:
            # 檢查是否有 _mask_mark_sam2.png 檔案
            for mask_mark_file in mask_mark_files:
                if mask_mark_file.name.endswith("_mask_mark_sam2.png"):
                    mask_overlay_path = str(mask_mark_file)
                    break
        else:
            # 檢查是否有 _mask_mark.png 檔案
            for mask_mark_file in mask_mark_files:
                if mask_mark_file.name.endswith("_mask_mark.png"):
                    mask_overlay_path = str(mask_mark_file)
                    break

    # 回傳處理結果
    if output_file:
        if input_path.suffix.lower() in IMAGE_EXTS:
            yield output_file, None, log, mask_output_path, mask_overlay_path
        elif input_path.suffix.lower() in VIDEO_EXTS:
            yield None, output_file, log, mask_output_path, mask_overlay_path
    else:
        yield None, None, log + "\n處理失敗，未找到輸出檔案", mask_output_path, mask_overlay_path


# ================================================
# UI
# ============================================
with gr.Blocks() as demo:

    gr.Markdown("# 🚀 Watermark Remover AI - Pro UI")

    model_selector = gr.Radio(
        ["yolo", "florence", "groundingdino"],
        value="yolo",
        label="Detection Model"
    )

    # 建立一個通用的檔案上傳元件
    input_file = gr.File(
    label="Upload Image / Video",
            type="filepath",
        height=300
    )


    # 建立圖片和影片的輸入/輸出元件
    with gr.Row():
        # 圖片處理區塊
        with gr.Column(visible=True) as image_block:
            gr.Markdown("## 🖼️ 圖片處理")
            with gr.Row():
                image_input = gr.Image(type="filepath", label="Input Image", height=300, width=400, interactive=False, show_label=True, scale=1)
                image_output = gr.Image(type="filepath", label="Output Image", height=300, width=400, interactive=False, show_label=True, scale=1)

    
    with gr.Row():
        # 影片處理區塊
        with gr.Column(visible=True) as video_block:
            gr.Markdown("## 🎬 影片處理")
            with gr.Row():
                video_input = gr.Video(label="Input Video", height=300, width=400, interactive=False, scale=1)
                video_output = gr.Video(label="Output Video", height=300, width=400, interactive=False, scale=1)
   
    with gr.Row():
        # 遮罩呈現區塊
        with gr.Column(visible=False) as mask_block:
            gr.Markdown("## 🎬 遮罩呈現")
            with gr.Row():
                mask_output = gr.Image(label="Mask Output", height=300, width=400, interactive=False, scale=1)
                mask_overlay = gr.Image(label="Mask Overlay", height=300, width=400, interactive=False, scale=1)


    # ========================
    # General Options
    # ========================

    with gr.Accordion("General Processing Options", open=True) as general_section:

        resize_limit = gr.Slider(256, 2048, 768, step=64, label="Resize Limit", info="控制輸入影像最大尺寸")

        expand = gr.Slider(0, 50, 0, step=1, label="BBox Expand Pixels", info="偵測框外擴像素")

        max_bbox_percent = gr.Slider(1.0, 50.0, 10.0, label="Max BBox Area (%)")

        lama_scale = gr.Slider(0.1, 1.0, 1.0, step=0.1, label="LaMa Scale")

        rotate_angle_step = gr.Slider(0, 90, 0, step=5, label="Rotation Step (°)")

        tiles_per_axis = gr.Slider(1, 10, 3, step=1, label="Tiles Per Axis")

        overlap_ratio = gr.Slider(0, 0.5, 0.25, step=0.05, label="Tile Overlap")
        
        half_precision = gr.Checkbox(True, label="Use FP16")
        
        use_sam2 = gr.Checkbox(False, label="Use SAM-2 Mask Refine")

        clahe_mode = gr.Radio(
                ["off", "auto", "mild", "aggressive"],
                value="off",
                label="CLAHE Mode"
            )

    # ========================
    # Model-specific Options
    # ========================
    
    # YOLO Options
    yolo_options = gr.Column(visible=True)
    with yolo_options:
        with gr.Accordion("YOLO Options", open=True):
            conf_threshold = gr.Slider(0.0, 1.0, 0.6, step=0.05,
                                       label="Confidence Threshold")
            iou_threshold = gr.Slider(0.0, 1.0, 0.45, step=0.05,
                                      label="IoU Threshold")
            single_detection = gr.Checkbox(True, label="Single Detection")
            yolo_imgsz = gr.Slider(320, 1280, 640, step=32,
                                   label="YOLO Image Size")

    # Florence Options
    florence_options = gr.Column(visible=False)
    with florence_options:
        with gr.Accordion("Florence Options", open=True):
            florence_mode = gr.Radio(
                ["Auto Text (OCR)", "Targeted Text (Prompt-based)",
                 "Visual Region Analysis (Dense Caption)", "Smart Mode (Auto Detect)"],
                value="Targeted Text (Prompt-based)",
                label="Florence Mode"
            )
            florence_prompt = gr.Textbox("watermark", label="Florence Prompt")
            model_size = gr.Radio(
                ["base", "large"],
                value="large",
                label="Model Size"
            )
            
            # Event handler to control florence_prompt textbox visibility and editability
            def update_florence_prompt_visibility(selected_mode):
                if selected_mode == "Targeted Text (Prompt-based)":
                    return gr.update(visible=True, interactive=True, label="Florence Prompt", value="watermark")
                else:
                    return gr.update(visible=True, interactive=False, label="不須輸入", value="")
            
            florence_mode.change(
                update_florence_prompt_visibility,
                inputs=florence_mode,
                outputs=florence_prompt
            )

    # GroundingDINO Options
    groundingdino_options = gr.Column(visible=False)
    with groundingdino_options:
        with gr.Accordion("GroundingDINO Options", open=True):
            watermark_text_gd = gr.Textbox("watermark", label="Watermark Text")
            groundingdino_conf_threshold = gr.Slider(0.0, 1.0, 0.35, step=0.05,
                                                     label="GroundingDINO Confidence Threshold")

    # Video Options
    video_options = gr.Column(visible=False)
    with video_options:
        with gr.Accordion("Video Options", open=True):
            use_first_frame_mask = gr.Checkbox(True, label="Use First Frame Mask")
            video_batch_size = gr.Slider(8, 64, 8, step=8, label="Video Batch Size")

    # ========================
    # Preset Buttons
    # ========================

    with gr.Row():
        speed_btn = gr.Button("⚡ High Speed Mode")
        quality_btn = gr.Button("🎯 High Quality Mode")

    speed_btn.click(
        preset_speed,
        outputs=[
            resize_limit,
            expand,
            max_bbox_percent,
            lama_scale,
            rotate_angle_step,
            tiles_per_axis,
            overlap_ratio,
            half_precision,
            use_sam2
        ]
    )

    quality_btn.click(
        preset_quality,
        outputs=[
            resize_limit,
            expand,
            max_bbox_percent,
            lama_scale,
            rotate_angle_step,
            tiles_per_axis,
            overlap_ratio,
            half_precision,
            use_sam2
        ]
    )

    run_btn = gr.Button("Run")

    # CLI Preview Auto Update
    all_inputs = [
        model_selector, input_file,
        resize_limit, expand, max_bbox_percent, half_precision, lama_scale,
        rotate_angle_step, tiles_per_axis, overlap_ratio,
        conf_threshold, iou_threshold, single_detection, yolo_imgsz,
        florence_mode, florence_prompt, model_size,
        watermark_text_gd, groundingdino_conf_threshold,
        use_first_frame_mask, use_sam2, video_batch_size, clahe_mode
    ]


# ========================
    # Outputs
    # ========================

    output_box = gr.Textbox(label="Log Outputs", lines=10)

    for comp in all_inputs[:10]:
        comp.change(build_cli_preview, all_inputs, output_box)

    # 修改 run_btn.click 以適應新的 UI 結構
    def run_process(*args):
        # 傳遞所有參數給 process_media 函數
        for result in process_media(*args):
            yield result
    
    
    run_btn.click(
        run_process,
        inputs=all_inputs,
        outputs=[image_output, video_output, output_box, mask_output, mask_overlay]
    )
    
    # 為遮罩呈現區塊添加事件處理
    def update_mask_on_file_change(input_file, model, use_first_frame_mask, use_sam2):
        return load_mask_images(input_file, model, use_first_frame_mask, use_sam2)
    
    # 為 run_btn 添加遮罩圖像更新功能
    def run_process_with_mask_update(*args):
        # 執行處理並取得結果
        results = list(process_media(*args))
        # 取得最後一個結果
        if results:
            result = results[-1]
            # 傳回處理結果
            yield result

    # 新增函數來根據條件顯示遮罩圖像
    def update_mask_display(input_file, model, use_first_frame_mask, use_sam2):

    # 如果沒有選擇檔案，隱藏遮罩區塊
        if input_file is None:
            return gr.update(visible=False)
    
    # 判斷是否為影片
        file_path = Path(input_file)
        file_ext = file_path.suffix.lower()
    
    # 根據輸入類型和參數決定遮罩區塊是否可見
    # 如果是圖片，總是顯示遮罩區塊
        if file_ext in IMAGE_EXTS:
            return gr.update(visible=True)
    # 如果是影片，只有在使用 first frame detection 時才顯示遮罩區塊
        elif file_ext in VIDEO_EXTS:
            if use_first_frame_mask:
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)
        else:
            return gr.update(visible=False)
    
    # 綁定遮罩圖像更新事件
    input_file.change(
        update_mask_on_file_change,
        inputs=[input_file, model_selector, use_first_frame_mask, use_sam2],
        outputs=[mask_output, mask_overlay]
    )
    
    # 綁定模型選擇事件
    model_selector.change(
        update_mask_on_file_change,
        inputs=[input_file, model_selector, use_first_frame_mask, use_sam2],
        outputs=[mask_output, mask_overlay]
    )
    
    # 綁定 use_first_frame_mask 事件
    use_first_frame_mask.change(
        update_mask_on_file_change,
        inputs=[input_file, model_selector, use_first_frame_mask, use_sam2],
        outputs=[mask_output, mask_overlay]
    )
    
    # 綁定 use_sam2 事件
    use_sam2.change(
        update_mask_on_file_change,
        inputs=[input_file, model_selector, use_first_frame_mask, use_sam2],
        outputs=[mask_output, mask_overlay]
    )
    
    # 綁定 input_file 變更事件來更新遮罩區塊可見性
    input_file.change(
        update_mask_display,
        inputs=[input_file, model_selector, use_first_frame_mask, use_sam2],
        outputs=mask_block
    )
    
    # 綁定模型選擇事件來更新遮罩區塊可見性
    model_selector.change(
        update_mask_display,
        inputs=[input_file, model_selector, use_first_frame_mask, use_sam2],
        outputs=mask_block
    )
    
    # 綁定 use_first_frame_mask 變更事件來更新遮罩區塊可見性
    use_first_frame_mask.change(
        update_mask_display,
        inputs=[input_file, model_selector, use_first_frame_mask, use_sam2],
        outputs=mask_block
    )
    
    # 綁定 use_sam2 變更事件來更新遮罩區塊可見性
    use_sam2.change(
        update_mask_display,
        inputs=[input_file, model_selector, use_first_frame_mask, use_sam2],
        outputs=mask_block
    )
 
       
    # Model selection event handler to control model options visibility
    def update_model_options(selected_model, input_file):
        # Return the updated visibility states for all model options
        model_updates = [
            gr.update(visible=(selected_model == "yolo")),
            gr.update(visible=(selected_model == "florence")),
            gr.update(visible=(selected_model == "groundingdino")),
            gr.update(visible=False)  # Video options hidden by default
        ]
        
        # 檢查是否之前有上傳影片檔案，如果有則保持 video_options 可見
        if input_file is not None:
            file_path = Path(input_file)
            file_ext = file_path.suffix.lower()
            if file_ext in VIDEO_EXTS:
                model_updates[3] = gr.update(visible=True)  # 設定 video_options 可見
                
        return model_updates[0], model_updates[1], model_updates[2], model_updates[3]

    # 副檔名判斷函數
    def update_video_options_visibility(input_file):
        # 如果沒有選擇檔案，預設隱藏 video options
        if input_file is None:
            return gr.update(visible=False)
        
        # 取得副檔名並轉為小寫
        file_path = Path(input_file)
        file_ext = file_path.suffix.lower()
        
        # 如果是圖片副檔名，隱藏 video options；否則顯示
        if file_ext in IMAGE_EXTS:
            return gr.update(visible=False)
        else:
            return gr.update(visible=True)

    # Attach the event handler to model selector
    model_selector.change(
        update_model_options,
        inputs=[model_selector, input_file],
        outputs=[yolo_options, florence_options, groundingdino_options, video_options]
    )
    
    # Attach the event handler to input_file to control video options visibility
    input_file.change(
        update_video_options_visibility,
        inputs=input_file,
        outputs=video_options
    )
    
    # 為了確保模型切換時也能正確處理 video_options，我們需要額外的處理
    def model_change_with_video_check(selected_model, input_file):
        # 先執行模型選項更新
        yolo_update, florence_update, groundingdino_update, video_update = update_model_options(selected_model, input_file)
        return yolo_update, florence_update, groundingdino_update, video_update
    
    # 重新綁定 model_selector 的事件處理
    model_selector.change(
        model_change_with_video_check,
        inputs=[model_selector, input_file],
        outputs=[yolo_options, florence_options, groundingdino_options, video_options]
    )
    
    
    # 新增函數來控制圖片和影片區塊的可見性
    def update_block_visibility(input_file):
        # 如果沒有選擇檔案，預設都隱藏
        if input_file is None:
            return [
                gr.update(visible=False),
                gr.update(visible=False)
            ]
        
        # 取得副檔名並轉為小寫
        file_path = Path(input_file)
        file_ext = file_path.suffix.lower()
        
        # 如果是圖片副檔名，顯示圖片區塊，隱藏影片區塊
        if file_ext in IMAGE_EXTS:
            return [
                gr.update(visible=True),
                gr.update(visible=False)
            ]
        else:
            # 否則顯示影片區塊，隱藏圖片區塊
            return [
                gr.update(visible=False),
                gr.update(visible=True)
            ]
    
    # Attach the event handler to input_file to control block visibility
    input_file.change(
        update_block_visibility,
        inputs=input_file,
        outputs=[image_block, video_block]
    )
    
    # 新增函數來根據檔案類型更新輸入元件
    def update_input_component(input_file):
        if input_file is None:
            return [
                gr.update(visible=False),
                gr.update(visible=False)
            ]
        
        file_path = Path(input_file)
        file_ext = file_path.suffix.lower()
        
        if file_ext in IMAGE_EXTS:
            return [
                gr.update(visible=True),
                gr.update(visible=False)
            ]
        else:
            return [
                gr.update(visible=False),
                gr.update(visible=True)
            ]

    # 當檔案上傳時，根據檔案類型更新輸入元件的可見性
    input_file.change(
        update_input_component,
        inputs=input_file,
        outputs=[image_input, video_input]
    )
    
    # 新增函數來根據檔案類型更新輸出元件
    def update_output_component(input_file):
        if input_file is None:
            return [
                gr.update(visible=False),
                gr.update(visible=False)
            ]
        
        file_path = Path(input_file)
        file_ext = file_path.suffix.lower()
        
        if file_ext in IMAGE_EXTS:
            return [
                gr.update(visible=True),
                gr.update(visible=False)
            ]
        else:
            return [
                gr.update(visible=False),
                gr.update(visible=True)
            ]

    # 當檔案上傳時，根據檔案類型更新輸出元件的可見性
    input_file.change(
        update_output_component,
        inputs=input_file,
        outputs=[image_output, video_output]
    )
    
    # 新增函數來處理圖片上傳並顯示
    def update_image_display(input_file):
        if input_file is None:
            return gr.update(visible=False)
        
        file_path = Path(input_file)
        file_ext = file_path.suffix.lower()
        
        if file_ext in IMAGE_EXTS:
            return gr.update(value=input_file, visible=True)
        else:
            return gr.update(visible=False)
    
    # 新增函數來處理影片上傳並顯示
    def update_video_display(input_file):
        if input_file is None:
            return gr.update(visible=False)
        
        file_path = Path(input_file)
        file_ext = file_path.suffix.lower()
        
        if file_ext in VIDEO_EXTS:
            return gr.update(value=input_file, visible=True)
        else:
            return gr.update(visible=False)
    
    # 當檔案上傳時，更新對應的輸入元件值
    input_file.change(
        update_image_display,
        inputs=input_file,
        outputs=image_input
    )
    
    input_file.change(
        update_video_display,
        inputs=input_file,
        outputs=video_input
    )


# 新增函數來讀取遮罩圖像
def load_mask_images(input_file, model, use_first_frame_mask, use_sam2):
    # 如果沒有選擇檔案，返回空圖像
    if input_file is None:
        return gr.update(value=None), gr.update(value=None)
    
    # 判斷是否為影片
    file_path = Path(input_file)
    file_ext = file_path.suffix.lower()
    
    # 如果是影片且未使用 first frame detection，返回空圖像
    if file_ext in VIDEO_EXTS and not use_first_frame_mask:
        return gr.update(value=None), gr.update(value=None)
    
    # 設定遮罩圖像路徑
    output_dir = Path("output_webui")
    mask_output_path = None
    mask_overlay_path = None
    
    # 根據是否使用 SAM-2 決定檔名後綴
    if use_sam2:
        mask_suffix = "_mask_sam2.png"
        mask_mark_suffix = "_mask_mark_sam2.png"
    else:
        mask_suffix = "_mask.png"
        mask_mark_suffix = "_mask_mark.png"
    
    # 根據檔案類型處理
    if file_ext in IMAGE_EXTS:
        # 圖片處理
        # 優先尋找指定後綴的檔案
        mask_path = output_dir / f"{file_path.stem}{mask_suffix}"
        mask_mark_path = output_dir / f"{file_path.stem}{mask_mark_suffix}"
        
        # 如果指定後綴的檔案不存在，則尋找所有以 _mask 為開頭的檔案
        if not mask_path.exists():
            mask_files = list(output_dir.glob(f"{file_path.stem}_mask*"))
            if mask_files:
                # 優先選擇 _mask_sam2.png 檔案（如果使用 SAM-2）
                if use_sam2:
                    for mask_file in mask_files:
                        if mask_file.name.endswith("_mask_sam2.png"):
                            mask_path = mask_file
                            break
                # 如果不是使用 SAM-2，則尋找 _mask.png 檔案
                if not mask_path.exists():
                    for mask_file in mask_files:
                        if mask_file.name.endswith("_mask.png"):
                            mask_path = mask_file
                            break
                # 如果都找不到，使用第一個找到的檔案
                if not mask_path.exists() and mask_files:
                    mask_path = mask_files[0]
        
        # 如果指定後綴的遮罩標示圖檔不存在，則尋找所有以 _mask_mark 為開頭的檔案
        if not mask_mark_path.exists():
            mask_mark_files = list(output_dir.glob(f"{file_path.stem}_mask_mark*"))
            if mask_mark_files:
                # 優先選擇 _mask_mark_sam2.png 檔案（如果使用 SAM-2）
                if use_sam2:
                    for mask_mark_file in mask_mark_files:
                        if mask_mark_file.name.endswith("_mask_mark_sam2.png"):
                            mask_mark_path = mask_mark_file
                            break
                # 如果不是使用 SAM-2，則尋找 _mask_mark.png 檔案
                if not mask_mark_path.exists():
                    for mask_mark_file in mask_mark_files:
                        if mask_mark_file.name.endswith("_mask_mark.png"):
                            mask_mark_path = mask_mark_file
                            break
                # 如果都找不到，使用第一個找到的檔案
                if not mask_mark_path.exists() and mask_mark_files:
                    mask_mark_path = mask_mark_files[0]
        
        if mask_path.exists():
            mask_output_path = str(mask_path)

        if mask_mark_path.exists():
            mask_overlay_path = str(mask_mark_path)
            
    elif file_ext in VIDEO_EXTS and use_first_frame_mask:
        # 影片處理且使用 first frame detection
        # 優先尋找指定後綴的檔案
        mask_path = output_dir / f"{file_path.stem}{mask_suffix}"
        mask_mark_path = output_dir / f"{file_path.stem}{mask_mark_suffix}"
        
        # 如果指定後綴的檔案不存在，則尋找所有以 _mask 為開頭的檔案
        if not mask_path.exists():
            mask_files = list(output_dir.glob(f"{file_path.stem}_mask*"))
            if mask_files:
                # 優先選擇 _mask_sam2.png 檔案（如果使用 SAM-2）
                if use_sam2:
                    for mask_file in mask_files:
                        if mask_file.name.endswith("_mask_sam2.png"):
                            mask_path = mask_file
                            break
                # 如果不是使用 SAM-2，則尋找 _mask.png 檔案
                if not mask_path.exists():
                    for mask_file in mask_files:
                        if mask_file.name.endswith("_mask.png"):
                            mask_path = mask_file
                            break
                # 如果都找不到，使用第一個找到的檔案
                if not mask_path.exists() and mask_files:
                    mask_path = mask_files[0]
        
        # 如果指定後綴的遮罩標示圖檔不存在，則尋找所有以 _mask_mark 為開頭的檔案
        if not mask_mark_path.exists():
            mask_mark_files = list(output_dir.glob(f"{file_path.stem}_mask_mark*"))
            if mask_mark_files:
                # 優先選擇 _mask_mark_sam2.png 檔案（如果使用 SAM-2）
                if use_sam2:
                    for mask_mark_file in mask_mark_files:
                        if mask_mark_file.name.endswith("_mask_mark_sam2.png"):
                            mask_mark_path = mask_mark_file
                            break
                # 如果不是使用 SAM-2，則尋找 _mask_mark.png 檔案
                if not mask_mark_path.exists():
                    for mask_mark_file in mask_mark_files:
                        if mask_mark_file.name.endswith("_mask_mark.png"):
                            mask_mark_path = mask_mark_file
                            break
                # 如果都找不到，使用第一個找到的檔案
                if not mask_mark_path.exists() and mask_mark_files:
                    mask_mark_path = mask_mark_files[0]
        
        if mask_path.exists():
            mask_output_path = str(mask_path)
        if mask_mark_path.exists():
            mask_overlay_path = str(mask_mark_path)
    
    # 返回遮罩圖像
    return gr.update(value=mask_output_path), gr.update(value=mask_overlay_path)

# 啟動 Gradio 界面
if __name__ == "__main__":
    demo.launch()