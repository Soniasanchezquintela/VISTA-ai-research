from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(
    data="sku110k_product.yaml",
    epochs=50,
    patience=5,
    imgsz=768,
    batch=8,
    workers=8,
    single_cls=True,
    amp=True,
    name="sku110k_768_e20_pat5"
)

# imgsz=640, batch=8 => 3.2it/s, 1027 images/epoch => 1027/3.2 = 320segs/epoch GPU_mem: 9.39G
# imgsz=768, batch=8 => 2.3it/s, 1027 images/epoch => 1027/2.3 = 446segs/epoch GPU_mem: 8.86G
# imgsz=768, batch=4 => 4.4it/s, 2054 images/epoch => 2054/4.4 = 467segs/epoch GPU_mem: 5.22G
