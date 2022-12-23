import cv2
import matplotlib.pyplot as plt
import os
from IPython import display

# '/content/drive/MyDrive' : google drive의 기본 path

def resize_images(orig, low_path, mid_path, high_path):
    r"""
    orig : 원본 이미지가 존재하는 경로
    low_path, mid_path, high_path : resizing된 이미지가 저장될 경로
    start_num : 저장될 이미지의 첫 번호
    """
    # orig에 존재하는 파일 이름 list화
    # listdir는 해당 path에 있는 모든 파일과 디렉토리의 이름을 리스트 형태로 리턴
    orig_name : list = os.listdir(orig)

    # data 각각의 path를 list로 저장
    # orig_files는 orig_name과 유사하지만, 해당 파일의 절대경로까지 포함한다는 점에서 차이점이 있음
    orig_files : list = [orig + '/' + name for name in orig_name]

    for step, path in enumerate(orig_files): # step : index, path : files with path

        # opencv로 이미지 불러오기
        file = cv2.imread(path)

        # hr, mr, lr로 이미지 크기 조정
        hr = cv2.resize(file, (640,480))
        mr = cv2.resize(file, (320,240))
        lr = cv2.resize(file, (160,120))

        # 정해진 path에 파일 저장
        # cv2.imwrite(path, 저장할 image)
        cv2.imwrite(high_path + '/' + str(step) + '.jpg', hr)
        cv2.imwrite(mid_path + '/' + str(step) + '.jpg', mr)
        cv2.imwrite(low_path + '/' + str(step) + '.jpg', lr)

        # 100 번 째 이미지마다 display library로 진행사항 보고
        if (step-1) % 100 == 0:
            display.clear_output(wait=True)
            plt.figure(figsize=(15,15))
            display_list = [hr, mr, lr]
            title = ["High resolution", "Mid resolution", "Low resolution"]

            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(title[i])
                plt.imshow(display_list[i])
                plt.axis('off')

            plt.show()
            print(f"Step: {step} / {len(orig_files)}")
            
train_origin = '/Users/healingmusic/Library/Mobile Documents/com~apple~CloudDocs/Programming/FLIR/dataset/train_orig'
test_origin = '/Users/healingmusic/Library/Mobile Documents/com~apple~CloudDocs/Programming/FLIR/dataset/test_orig'

train_low = '/Users/healingmusic/Library/Mobile Documents/com~apple~CloudDocs/Programming/FLIR/dataset/resized/train/low'
train_mid = '/Users/healingmusic/Library/Mobile Documents/com~apple~CloudDocs/Programming/FLIR/dataset/resized/train/mid'
train_high = '/Users/healingmusic/Library/Mobile Documents/com~apple~CloudDocs/Programming/FLIR/dataset/resized/train/high'

test_low = '/Users/healingmusic/Library/Mobile Documents/com~apple~CloudDocs/Programming/FLIR/dataset/resized/test/low'
test_mid = '/Users/healingmusic/Library/Mobile Documents/com~apple~CloudDocs/Programming/FLIR/dataset/resized/test/mid'
test_high = '/Users/healingmusic/Library/Mobile Documents/com~apple~CloudDocs/Programming/FLIR/dataset/resized/test/high'

resize_images(train_origin, train_low, train_mid, train_high)
resize_images(test_origin, test_low, test_mid, test_high)