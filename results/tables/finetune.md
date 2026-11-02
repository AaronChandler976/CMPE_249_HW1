| model                               | eval                  |   AP |   AP50 |   AP_S |   AP_person |   AP_car |   AP_truck |
|:------------------------------------|:----------------------|-----:|-------:|-------:|------------:|---------:|-----------:|
| YOLO11s zero-shot (COCO weights)    | kitti_cocomap (3 cls) | 24.6 |   48.2 |   14.0 |        19.5 |     45.3 |        9.0 |
| YOLO-World zero-shot (text prompts) | kitti_cocomap (3 cls) | 28.8 |   50.0 |   23.7 |        12.8 |     45.9 |       27.7 |
| YOLO-World zero-shot                | kitti_native (7 cls)  | 15.3 |   26.9 |   11.2 |       nan   |    nan   |      nan   |