import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import random_noise
from pytictoc import TicToc

plt.close('all')
t = TicToc() #create instance of class

# - ------------- True Signal ------------------
N = 2**8
x = np.arange(N)
func1 = 2*np.cos( np.divide(8*np.pi*(x),N) ) + np.divide(4*np.pi*(x),N)
plt.figure(1)
plt.plot(np.pi*(x),func1)
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

plt.figure(3)

harr_list_A = []
harr_list_D = []
v_harr = []
w_harr = []
for i in range(1,6,1):

  v = np.zeros((int(N/2**i),N))
  w = np.zeros((int(N/2**i),N))

  V = np.power(1/np.sqrt(2),i)*np.ones((1,int(2**i)))
  W = np.power(1/np.sqrt(2),i)*np.ones((1,int(2**i)))
  p = np.size(W)
  W[0,int(p/2):] = -W[0,int(p/2):]
  # print(p)
  # print(W.shape)
  # print(2**i*j,2**i*j+2)
  for j in range(0,int(N/2**i),1):
    v[j,2**i*j:2**i*(j+1)] = V
    w[j,2**i*j:2**i*(j+1)] = W
  
  # print(w)
  
  v_harr.append(v)
  w_harr.append(w)

  Avrage = np.sum(np.multiply(np.dot(np.reshape(np.dot(v,func2),(int(N/2**i),1)),np.ones((1,N))),v),axis = 0)
  Detail = np.sum(np.multiply(np.dot(np.reshape(np.dot(w,func2),(int(N/2**i),1)),np.ones((1,N))),w),axis = 0)
  # print(i)
  # print(Avrage.shape)
  harr_list_A.append(Avrage)
  harr_list_D.append(Detail)
  ax = plt.subplot(5, 2,2*i-1 )
  plt.plot(2*np.pi*(x[10:N-10]),Avrage[10:N-10])
  plt.grid()
  if i == 5:
      plt.xlabel('Time(s)')   
  if i != 5:
    # plt.xticks([])
    ax.xaxis.set_tick_params(labelbottom = False)
  plt.ylabel('Amplitude')
  plt.title(f"Harr Average Smooth Level {i}")
  
  ax = plt.subplot(5, 2,2*i )
  plt.plot(2*np.pi*(x[10:N-10]),Detail[10:N-10])
  plt.grid()
  if i == 5:
      plt.xlabel('Time(s)')   
  if i != 5:
    # plt.xticks([])
    ax.xaxis.set_tick_params(labelbottom = False)
  plt.ylabel('Amplitude')
  plt.title(f"Harr Details Smooth Level {i}")
  
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
    v = v_harr[i-1]
    w = w_harr[i-1]
    aa=np.zeros((N,N,3));
    ad=np.zeros((N,N,3));
    da=np.zeros((N,N,3));
    dd=np.zeros((N,N,3));

    
    
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
    plt.title(f"Harr Smooth Level {i}")

e = []
for i in range(1,6,1):
    a = harr_list_A[i-1]
    b = np.sum(np.power(func1[15:N-15] - a[15:N-15],2))
    e.append(b)   
    # plt.subplot(5, 2,2*i )
    # plt.plot(2*np.pi*(x[10:N-10]),Detail[10:N-10])
    # plt.grid()