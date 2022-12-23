import torch
import torch.nn as nn

class Loss(torch.nn.modules.loss._Loss):
    """ Loss 클래스 """
    def __init__(self, feature_loss_list, device):
        super(Loss, self).__init__()
        # Loss를 담을 리스트 선언
        self.loss = []
        self.feature_loss_module = torch.nn.ModuleList()
        
        # Device 설정
        self.feature_loss_module.to(device)
        
        # feature_loss 사용여부 설정
        if len(feature_loss_list) > 0:
            self.feature_loss_used = 1
        else:
            self.feature_loss_used = 0
        self.feature_loss_used = 0# feature loss 사용 안함.
        # SR Loss Weight 설정
        DS_weight = 0
        TS_weight = 1

        # Sr Loss 등록
        self.loss.append({'type': "DS", 'weight': DS_weight, 'function': torch.nn.L1Loss()})
        self.loss.append({'type': "TS", 'weight': TS_weight, 'function': torch.nn.L1Loss()})

        # feature loss 등록
        if self.feature_loss_used == 1:     
            for feature_loss in feature_loss_list:
                weight, feature_type = feature_loss.split('*')
                l = {'type': feature_type, 'weight': float(weight), 'function': FeatureLoss(loss=torch.nn.L1Loss())}
                self.loss.append(l)
                self.feature_loss_module.append(l['function'])
            
        # Total Loss 등록
        self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        
        

    def forward(self, student_sr, teacher_sr, hr, student_fms, teacher_fms):
        # DS Loss
        DS_loss = self.loss[0]['function'](student_sr, hr) * self.loss[0]['weight']
        #print("DS_loss =", DS_loss.item())
        
        # TS Loss
        TS_loss = self.loss[1]['function'](student_sr, teacher_sr) * self.loss[1]['weight']
        #print("TS_loss =", TS_loss.item())
        
        loss_sum = DS_loss + TS_loss
        
        if self.feature_loss_used == 0:
            pass
        elif self.feature_loss_used == 1:
            assert(len(student_fms) == len(teacher_fms))
            assert(len(student_fms) == len(self.feature_loss_module))

            for i in range(len(self.feature_loss_module)):
                feature_loss = self.feature_loss_module[i](student_fms[i], teacher_fms[i]) * self.loss[2+i]['weight']
                loss_sum += feature_loss
                print("feature_loss =", feature_loss.item())

        return loss_sum

class FeatureLoss(torch.nn.Module):
    """ Feature Loss 클래스 """
    def __init__(self, loss=torch.nn.L1Loss()):
        super(FeatureLoss, self).__init__()
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)        
        length = len(outputs)
        tmp = [self.loss(outputs[i], targets[i]) for i in range(length)]
        loss = sum(tmp)
        return loss

def spatial_similarity(fm): 
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
    s = norm_fm.transpose(1,2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s

def pooled_spatial_similarity(fm, k, pool_type):
    if pool_type == "max":
        pool = nn.MaxPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    elif pool_type == "avg":
        pool = nn.AvgPool2d(kernel_size=(k, k), stride=(k, k), padding=0, ceil_mode=True)
    fm = pool(fm)
    s = spatial_similarity(fm)
    return s