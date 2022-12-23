# DataCapstonDesign
##### made by 임한결, 최지현
##### 2022.09.02 ~ 2022.12.31

> Project Info
>> 소방관은 연기로 가득찬 화재 진압 상황에서의 시야 확보를 위해 적외선 카메라를 휴대한다. 하지만 적외선 카메라는 센서 특성상 해상도가 낮아 고품질의 영상을 촬영하기가 어렵다. 따라서 휴대한 장비에서 Real-Time으로 촬영된 이미지를 Super Resolution 모델을 통해 해상도를 높여 제공할 수 있다면 소방관의 구조 활동에 도움이 될 것이다. 본 연구에서는 크기가 작으면서도 높은 성능을 가지는 Super Resolution 모델을 개발하기 위해 지식 증류 기법 (Knowledge Distillation)을 사용하여 학습을 진행한다. 지식 증류 기법을 적용하는 방법에서 Teacher 모델, Student 모델 및 학습 방법을 여러가지로 달리하며 학습을 진행하고, 각 방법으로 학습된 모델의 성능 측정 결과와 추후 연구 방향을 제시한다.


***
### Student Train
`python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 100 --batch 4 --KDloss L2 --lr 0.0001 --t0 30 --name n04`

***
### Teacher Train
`python IMDN_basecode_KD_branch_3ch_cosin.py --optimizer ADAM --epochs 200 --batch 2 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b2_L1_t30_v1`

***
### KD Train
`python FSRCNN_basecode_KD_branch.py --optimizer ADAM --epochs 200 --batch 2 --KDloss L1 --lr 0.0005 --t0 30 --name fsrcnn_KD_b2_Lre-55_nt200_nofe`

### Result
|Model|PSNR (dB)|SSIM| |
|------|---|---|---|
|Bicubic|27.797|0.728| |
|SwinIR|29.711|0.736|방법 1|
|FSRCNN(3채널)|27.760|0.714| |
|SwinIR + FSRCNN (3채널)|27.879|0.718| |
|IMDN|29.538|0.767|방법 2|
|FSRCNN(1채널)|28.874|0.759| |
|IMDN + FSRCNN (1채널)|29.120|0.760| |
