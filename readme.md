# DataCapstonDesign
##### made by 임한결, 최지현
##### 2022.09.02 ~ 2022.12.31

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
|SwinIR|29.711|0.736| |
|FSRCNN(3채널)|27.760|0.714|방법 1|
|SwinIR + FSRCNN (3채널)|27.879|0.718| |
|IMDN|29.538|0.767| |
|FSRCNN(1채널)|28.874|0.759|방법 2|
|IMDN + FSRCNN (1채널)|29.120|0.760| |
