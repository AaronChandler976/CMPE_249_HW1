# Taxonomy sensitivity (KITTI val, 3-class AP@[.5:.95]) - NOT the main results

main = pipeline mapping; van_ign_trk = Van also ignored for truck; van_as_car = Van scored as car

| run                     | variant     |   AP |   AP_person |   AP_car |   AP_truck |
|:------------------------|:------------|-----:|------------:|---------:|-----------:|
| yolo11s__kitti          | main        | 24.6 |        19.5 |     45.3 |        9   |
| yolo11s__kitti          | van_ign_trk | 26.5 |        19.5 |     45.3 |       14.7 |
| yolo11s__kitti          | van_as_car  | 24.1 |        19.5 |     43.8 |        9   |
| rtdetr_l__kitti         | main        | 29   |        20.1 |     49.5 |       17.4 |
| rtdetr_l__kitti         | van_ign_trk | 32.7 |        20.1 |     49.5 |       28.5 |
| rtdetr_l__kitti         | van_as_car  | 28.5 |        20.1 |     48.1 |       17.4 |
| frcnn_r50_v2__kitti     | main        | 28.3 |        20.5 |     46.4 |       18.1 |
| frcnn_r50_v2__kitti     | van_ign_trk | 30.6 |        20.5 |     46.4 |       25   |
| frcnn_r50_v2__kitti     | van_as_car  | 27.7 |        20.5 |     44.6 |       18.1 |
| fcos_r50__kitti         | main        | 28.7 |        20.2 |     46.4 |       19.7 |
| fcos_r50__kitti         | van_ign_trk | 31.5 |        20.2 |     46.4 |       27.9 |
| fcos_r50__kitti         | van_as_car  | 28.4 |        20.2 |     45.3 |       19.7 |
| yoloworld_s__kitti      | main        | 28.8 |        12.8 |     45.9 |       27.7 |
| yoloworld_s__kitti      | van_ign_trk | 31.9 |        12.8 |     45.9 |       36.9 |
| yoloworld_s__kitti      | van_as_car  | 28.6 |        12.8 |     45.3 |       27.7 |
| yolo11s_kitti_ft__kitti | main        | 71.6 |        49.8 |     80.9 |       83.9 |
| yolo11s_kitti_ft__kitti | van_ign_trk | 71.6 |        49.8 |     80.9 |       83.9 |
| yolo11s_kitti_ft__kitti | van_as_car  | 71.5 |        49.8 |     80.8 |       83.9 |
