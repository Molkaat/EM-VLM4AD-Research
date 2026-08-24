import os
import shutil
import sys

ROOT = os.path.dirname(__file__)

def check_paths():
    print('Checking expected dataset and checkpoint files...')
    checks = [
        ('data/multi_frame/multi_frame_train.json', False),
        ('data/multi_frame/multi_frame_val.json', False),
        ('data/multi_frame/multi_frame_test.json', False),
        ('data/multi_frame/multi_frame_test_coco.json', False),
        ('data/multi_frame/image_id.json', False),
        ('data/QA_dataset_nus/v1_0_train_nus.json', False),
        ('multi_frame_results/T5-Medium/latest_model.pth', False),
        ('multi_frame_results/T5-Large/latest_model.pth', False),
    ]
    missing = []
    for path, _ in checks:
        full = os.path.join(ROOT, path.replace('/', os.sep))
        ok = os.path.exists(full)
        print(f" - {path}: {'FOUND' if ok else 'MISSING'}")
        if not ok:
            missing.append(path)
    return missing


def print_instructions(missing):
    if not missing:
        print('\nAll files present.')
        return
    print('\nThe following files are missing:')
    for p in missing:
        print('  -', p)
    print('\nDownload instructions:')
    print('1) Dataset: Download the train/val/test split from the link in the README (Drive link). If you have gdown installed you can download via:')
    print('   pip install gdown')
    print("   gdown 'https://drive.google.com/uc?id=1isiXXTg46nl5SqMiEV4XjFD71KCCzezi' -O data.zip")
    print('   unzip data.zip -d .')
    print('\n2) Model weights: download the T5-Base and T5-Large folders from the links in the README and place them under `multi_frame_results/` as:')
    print('   multi_frame_results/T5-Medium/latest_model.pth')
    print('   multi_frame_results/T5-Large/latest_model.pth')
    print('\nIf you can provide the downloaded zip or the Drive links, I can help craft the exact gdown commands or move files into place.')


if __name__ == "__main__":
    missing = check_paths()
    print_instructions(missing)
    if missing:
        sys.exit(2)
    else:
        sys.exit(0)
