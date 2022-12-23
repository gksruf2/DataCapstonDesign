# DataCapstonDesign
##### made by 임한결, 최지현
##### 2022.09.02 ~ 2022.12.31

***
### Student Train
'''
python FSRCNN_basecode_branch.py --optimizer ADAM --epochs 100 --batch 4 --KDloss L2 --lr 0.0001 --t0 30 --name n04
'''

***
### Teacher Train
'''
python IMDN_basecode_KD_branch_3ch_cosin.py --optimizer ADAM --epochs 200 --batch 2 --KDloss L1 --lr 0.0001 --t0 30 --name fsrcnn_b2_L1_t30_v1

***
### KD Train
'''
python FSRCNN_basecode_KD_branch.py --optimizer ADAM --epochs 200 --batch 2 --KDloss L1 --lr 0.0005 --t0 30 --name fsrcnn_KD_b2_Lre-55_nt200_nofe
'''
