from ultralytics import YOLO

# This automatically downloads the 6MB yolov8n.pt model the first time it runs.
# We load it globally here so it stays cached in memory for fast inference.
detector_model = YOLO('yolov8n.pt')

def detect_objects(image_path: str) -> list:
    """Runs YOLO object detection and formats boxes for the SatQuery PDF trace."""
    # Run inference on the image
    results = detector_model(image_path)
    
    formatted_boxes = []
    
    # YOLO results is a list of Result objects (one per image)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Convert PyTorch tensors to standard Python integers and floats
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = result.names[cls_id]
            
            # Format exactly how report_generator.py expects it
            formatted_boxes.append({
                "label": label.capitalize(),
                "score": conf,
                "xmin": int(x1),
                "ymin": int(y1),
                "xmax": int(x2),
                "ymax": int(y2)
            })
            
    return formatted_boxes