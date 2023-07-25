import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
from pytictoc import TicToc

plt.close('all')
t = TicToc() #create instance of class

# - -------------
N = 2**8
x = np.arange(N)
func1 = 2*np.cos( np.divide(8*np.pi*(x),N) ) + np.divide(4*np.pi*(x),N)
# func1 = 2*np.sin( np.divide(2*np.pi*(x),N) )
plt.figure(1)
plt.plot(2*np.pi*(x),func1)
plt.grid()
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.title('Orginal Signal')


# - -------------
noise = np.random.rand(N) 
func2 = func1 + noise
plt.figure(2)
plt.plot(np.pi*(x),func2,)
plt.grid()
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.title('Noisy Signal')

# - -------------
h1 , h2 = (1+np.sqrt(3))/(4*np.sqrt(2)) , (3+np.sqrt(3))/(4*np.sqrt(2)) 
h3 , h4 = (3-np.sqrt(3))/(4*np.sqrt(2)) , (1-np.sqrt(3))/(4*np.sqrt(2)) 
g1 , g2 = (1-np.sqrt(3))/(4*np.sqrt(2)) , (-3+np.sqrt(3))/(4*np.sqrt(2)) 
g3 , g4 = (3+np.sqrt(3))/(4*np.sqrt(2)) , (-1-np.sqrt(3))/(4*np.sqrt(2)) 
V = np.array([h1 ,h2 ,h3 ,h4])
W = np.array([g1 ,g2 ,g3 ,g4])
# np.dot(v,np.transpose(w))
# print(V)
dub4_list_w = []
dub4_list_v = []

dub4_list_Avrage = []
dub4_list_Detail = []

plt.figure(3)

for i in range(1,6,1):

  v = np.zeros((int(N/2**i),N))
  w = np.zeros((int(N/2**i),N))

  if i == 1:
    a = 1

    
    
    for j in range(0,int(N/2**i-1),1):
        
      v[j,2**i*j:2**i*(j+1)+2] = V
      w[j,2**i*j:2**i*(j+1)+2] = W

    v[j+1,0:2] = V[2:]
    v[j+1,N-2:] = V[0:2]

    w[j+1,0:2] = W[2:]
    w[j+1,N-2:] = W[0:2]

    # I = np.eye(2**i*int(N/2**i),int(N/2**i))
    # print(I.shape)
    I = np.tile(np.eye(int(N/2**i),int(N/2**i)),(2**i,1))
    v = np.dot(I,v)
    w = np.dot(I,w)
    
    dub4_list_v.append(v)
    dub4_list_w.append(w)

    # print(v[0:int(N/2**i-1),:])
    # print(np.reshape(np.dot(v[0:N/2**i-1,:],func2),(int(N/2**i),1)))
    Avrage = np.sum(np.multiply(np.dot(np.reshape(np.dot(v[0:int(N/2**i),:],func2),(int(N/2**i),1)),np.ones((1,N))),v[0:int(N/2**i),:]),axis = 0)
    Detail = np.sum(np.multiply(np.dot(np.reshape(np.dot(w[0:int(N/2**i),:],func2),(int(N/2**i),1)),np.ones((1,N))),w[0:int(N/2**i),:]),axis = 0)
    # 
    dub4_list_Avrage.append(Avrage)
    dub4_list_Detail.append(Detail)

    # plt.subplot(5, 2,2*i-1 )
    # plt.plot(2*np.pi*(x),Avrage)
    # plt.subplot(5, 2,2*i )
    # plt.plot(2*np.pi*(x),Detail)

  # elif i == 2 :
  else :

    a = 1
    v1 = dub4_list_v[i-2]
    w1 = dub4_list_w[i-2]
    j = 0
    
    for j in range(0,int(N/2**i),1):
      v[j,:] = np.sum(np.multiply(np.dot( np.transpose(np.reshape(V,(1,4))),np.ones((1,N))),v1[2*j:2*j+4,:]),axis = 0)
      w[j,:] = np.sum(np.multiply(np.dot( np.transpose(np.reshape(W,(1,4))),np.ones((1,N))),w1[2*j:2*j+4,:]),axis = 0)
    
    I = np.tile(np.eye(int(N/2**i),int(N/2**i)),(2**i,1))
    v = np.dot(I,v)
    w = np.dot(I,w)

    dub4_list_v.append(v)
    dub4_list_w.append(w)
    
    Avrage = np.sum(np.multiply(np.dot(np.reshape(np.dot(v[0:int(N/2**i),:],func2),(int(N/2**i),1)),np.ones((1,N))),v[0:int(N/2**i),:]),axis = 0)
    Detail = np.sum(np.multiply(np.dot(np.reshape(np.dot(w[0:int(N/2**i),:],func2),(int(N/2**i),1)),np.ones((1,N))),w[0:int(N/2**i),:]),axis = 0)
    # 
    dub4_list_Avrage.append(Avrage)
    dub4_list_Detail.append(Detail)

  ax = plt.subplot(5, 2,2*i-1 )
  plt.plot(2*np.pi*(x[10:N-10]),Avrage[10:N-10])
  plt.grid()
  if i == 5:
      plt.xlabel('Time(s)')   
  if i != 5:
    # plt.xticks([])
    ax.xaxis.set_tick_params(labelbottom = False)
  plt.ylabel('Amplitude')
  plt.title(f"Dub4 Average Smooth Level {i}")
  ax = plt.subplot(5, 2,2*i )
  plt.plot(2*np.pi*(x[10:N-10]),Detail[10:N-10])
  plt.grid()
  if i == 5:
      plt.xlabel('Time(s)')   
  if i != 5:
    # plt.xticks([])
    ax.xaxis.set_tick_params(labelbottom = False)
  plt.ylabel('Amplitude')
  plt.title(f"Dub4 Average Smooth Level {i}")
  
# - -------------
img = cv2.imread('lena_color.tiff')


img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (N, N))
plt.figure(4)
plt.imshow(img)
plt.title('Origin')
gauss = 255*random_noise(img, mode='gaussian', seed=None, clip=True)
sp = random_noise(img, mode='s&p', seed=None, clip=True)




plt.figure(5)
plt.subplot(131), plt.imshow(img), plt.title('Origin')
plt.subplot(132), plt.imshow(gauss.astype(np.uint8())), plt.title('Gaussian')
plt.subplot(133), plt.imshow(sp), plt.title('Salt & Pepper')




# - -------------

plt.figure(6)
GImg = gauss.astype(np.float32)
# print(GImg[:,:,2])
plt.subplot(2 , 3, 1)
plt.imshow(gauss.astype(np.uint8())), plt.title('Gaussian')

aa_list = []
ad_list = []
da_list = []
dd_list = []
# print(a)
for i in range(1,6,1):
    print(i)
    t.tic() #Start timer
    v = dub4_list_v[i-1]
    w = dub4_list_w[i-1]
    aa=np.zeros((N,N,3))
    ad=np.zeros((N,N,3))
    da=np.zeros((N,N,3))
    dd=np.zeros((N,N,3))

    
    
    for i1 in range(0,int(N/2**i),1):
      
        for j1 in range(0,int(N/2**i),1):
        

            Tvv = (np.dot(np.reshape(np.transpose(v[i1,:]),(N,1)),np.reshape(v[j1,:],(1,N))))
            # Tvv = np.dot(np.transpose(v[i1,:]),v[j1,:])
            Tvw = (np.dot(np.reshape(np.transpose(v[i1,:]),(N,1)),np.reshape(w[j1,:],(1,N))))
            # Tvw = np.dot(np.transpose(v[i1,:]),w[j1,:])
            Twv = (np.dot(np.reshape(np.transpose(w[i1,:]),(N,1)),np.reshape(v[j1,:],(1,N))))
            # Twv = np.dot(np.transpose(w[i1,:]),v[j1,:])
            Tww = (np.dot(np.reshape(np.transpose(w[i1,:]),(N,1)),np.reshape(w[j1,:],(1,N))))
            # Tww = np.dot(np.transpose(w[i1,:]),w[j1,:])
            
    
            # print(np.sum(np.multiply(GImg_b , Tvv)))
            aa[:,:,0] = aa[:,:,0] + np.dot(np.sum(np.multiply(GImg[:,:,0] , Tvv)) , Tvv)
            aa[:,:,1] = aa[:,:,1] + np.dot(np.sum(np.multiply(GImg[:,:,1] , Tvv)) , Tvv)
            aa[:,:,2] = aa[:,:,2] + np.dot(np.sum(np.multiply(GImg[:,:,2] , Tvv)) , Tvv)
            
            ad[:,:,0] = ad[:,:,0] + np.dot(np.sum(np.multiply(GImg[:,:,0] , Tvw)) , Tvw)
            ad[:,:,1] = ad[:,:,1] + np.dot(np.sum(np.multiply(GImg[:,:,1] , Tvv)) , Tvv)
            ad[:,:,2] = ad[:,:,2] + np.dot(np.sum(np.multiply(GImg[:,:,2] , Tvv)) , Tvv)
            
            da[:,:,0] = da[:,:,0] + np.dot(np.sum(np.multiply(GImg[:,:,0] , Twv)) , Twv)
            da[:,:,1] = da[:,:,1] + np.dot(np.sum(np.multiply(GImg[:,:,1] , Twv)) , Twv)
            da[:,:,2] = da[:,:,2] + np.dot(np.sum(np.multiply(GImg[:,:,2] , Twv)) , Twv)
            
            dd[:,:,0] = dd[:,:,0] + np.dot(np.sum(np.multiply(GImg[:,:,0] , Tww)) , Tww)
            dd[:,:,1] = dd[:,:,1] + np.dot(np.sum(np.multiply(GImg[:,:,1] , Tww)) , Tww)
            dd[:,:,2] = dd[:,:,2] + np.dot(np.sum(np.multiply(GImg[:,:,2] , Tww)) , Tww)
        
    aa_list.append(aa)
    ad_list.append(ad)
    da_list.append(da) 
    dd_list.append(dd)
    t.toc() #Time elapsed since t.tic()
    a1 = np.vstack((aa,ad))
    a2 = np.vstack((da,dd))
    a3 = np.hstack((a1,a2))
    plt.subplot(2 , 3, i+1)
    plt.imshow(a3.astype(np.uint8()))
    plt.title(f"Dub4 Smooth Level {i}")

e = []
for i in range(1,6,1):
    a = dub4_list_Avrage[i-1]
    b = np.sum(np.power(func1[15:N-15] - a[15:N-15],2))
    e.append(b)
    # plt.subplot(5, 2,2*i )
    # plt.plot(2*np.pi*(x[10:N-10]),Detail[10:N-10])
    # plt.grid()