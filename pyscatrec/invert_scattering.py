import numpy as np
import imageio
import torch
from scattering.scattering2d import Scattering2D
from torch.autograd import Variable, grad
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX') # qt5agg')
else:
    print('OS',platform.system())
import matplotlib.pyplot as plt

# load image data to gpu
image_path = './data/fishingboat32.png'
image_gray = np.array(imageio.imread(image_path))
print (image_gray.shape)
N = image_gray.shape[0] # assert(image_gray.shape[1]==N)
x0 = torch.from_numpy(image_gray.reshape((1,1,N,N)))
x0 = x0.float()/256
x0 = x0.cuda() #print(x0.max(),x0.min())

# compute scattering of x0
J=2
scattering = Scattering2D(M=N, N=N, J=J).cuda()
#print (scattering(x0).size())
SJx0 = scattering(x0)

# recontruct x by matching || Sx - Sx0 ||^2
x = torch.Tensor(1, 1, N, N).normal_(std=0.1).cuda()
x = Variable(x, requires_grad=True)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam([x], lr=0.1)

nb_steps = 100
for step in range(1, nb_steps + 1):
    optimizer.zero_grad()
    SJx = scattering(x)
    loss = criterion(SJx, SJx0)
    loss.backward()
    optimizer.step()
    print('step',step,'loss',loss)
    
# plot x
plt.imshow(x.detach().cpu().numpy().reshape((N,N)))
plt.show()
