| model                               | eval                  |   AP |   AP50 |   AP_S |   AP_person |   AP_car |   AP_truck |
|:------------------------------------|:----------------------|-----:|-------:|-------:|------------:|---------:|-----------:|
| YOLO11s zero-shot (COCO weights)    | kitti_cocomap (3 cls) | 33.9 |   53.7 |   16.8 |        32.4 |     49.5 |       19.9 |
| YOLO-World zero-shot (text prompts) | kitti_cocomap (3 cls) | 41.5 |   61.4 |   41.5 |        31.3 |     49.3 |       44.0 |
| YOLO11s fine-tuned on KITTI         | kitti_cocomap (3 cls) | 12.7 |   21.3 |   12.8 |         0.0 |     35.5 |        2.5 |
| YOLO-World zero-shot                | kitti_native (7 cls)  | 24.0 |   34.3 |   25.0 |       nan   |    nan   |      nan   |
| YOLO11s fine-tuned                  | kitti_native (7 cls)  |  6.3 |   10.7 |    7.7 |       nan   |    nan   |      nan   |