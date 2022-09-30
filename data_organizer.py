from pathlib import Path
import shutil
from tqdm import tqdm

output_path_images = Path(r"/media/access/New Volume/YOLOv6/data/custom_dataset/images/train")
output_path_labels = Path(r"/media/access/New Volume/YOLOv6/data/custom_dataset/labels/train")

for file in tqdm(Path(r"/media/access/New Volume/cv-asaf-research/asaf scoreboard detector/darknet/ready_to_go_scoreboard_data_including_tabletennis_11_08_2022_YOLOv4_TINY/train").iterdir(), total=len(list(Path(r"/media/access/New Volume/cv-asaf-research/asaf scoreboard detector/darknet/ready_to_go_scoreboard_data_including_tabletennis_11_08_2022_YOLOv4_TINY/train").iterdir()))):
    src = file.as_posix()
    if file.name.split('.')[1] == 'txt':
        dst = output_path_labels.as_posix() + '/' + file.name
    else:
        dst = output_path_images.as_posix() + '/' + file.name
    shutil.copyfile(src, dst)