import math
from torch import nn
class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        """
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        """

        self.mid_part1 = nn.Sequential(nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s))
        self.mid_part2 = nn.Sequential(nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s))
        self.mid_part3 = nn.Sequential(nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s))
        self.mid_part4 = nn.Sequential(nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s))
        self.mid_part5 = nn.Sequential(nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s))
        self.mid_part6 = nn.Sequential(nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d))

        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        """
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        """
        
        for mid_part in [self.mid_part1, self.mid_part2, self.mid_part3, self.mid_part4, self.mid_part5, self.mid_part6]:
            for m in mid_part:
              if isinstance(m, nn.Conv2d):
                  nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                  nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        featuremap = []
        x = self.first_part(x)
        #x = self.mid_part(x)
        x = self.mid_part1(x)
        featuremap.append(x)
        x = self.mid_part2(x)
        #featuremap.append(x)
        x = self.mid_part3(x)
        #featuremap.append(x)
        x = self.mid_part4(x)
        featuremap.append(x)
        x = self.mid_part5(x)
        #featuremap.append(x)
        x = self.mid_part6(x)
        #featuremap.append(x)
        x = self.last_part(x)
        
        """for f in featuremap:
          print(f.shape)"""
        return featuremap, x