#!/usr/bin/env python
# coding: utf-8

# # BASIC OPERATIONS ON IMAGE

# In[1]:


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv


# In[2]:


pil_img = Image.open('D:/kitten.jpg')


# In[3]:


np.array(pil_img)


# In[4]:


gb = np.array(pil_img)
plt.imshow(gb)


# In[5]:


A =gb


# In[6]:


gb.shape


# # Crop

# In[7]:


plt.imshow(gb[250:500,70:310,:])


# In[8]:


gb[250:500,70:310,:]=0


# In[9]:


plt.imshow(gb)


# In[10]:


gb[:,:,0]


# In[11]:


plt.imshow(gb[:,:,0])


# # FLIP
# 

# In[12]:


flipped_image = pil_img.transpose(Image.FLIP_TOP_BOTTOM)


# In[13]:


flipped_image


# In[14]:


flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)


# In[15]:


flipped


# # BLACK AND WHITE

# In[16]:


from PIL import ImageOps
img_gray = ImageOps.grayscale(pil_img)
img_gray


# In[17]:


flip = ImageOps.flip(pil_img)
flip


# In[18]:


mirror = ImageOps.mirror(pil_img)
mirror


# # Resizing the Image

# In[19]:


width = 512
height = 512
img = pil_img.resize((width, height), Image.BILINEAR)
img


# # ROTATE

# In[20]:


theta = 135
img_rotate =pil_img.rotate(theta)
img_rotate


# In[22]:


import cv2

cols,rows,_ = gb.shape
theta = 30
M= cv2.getRotationMatrix2D(center=(cols//2-1,rows//2-1),angle =theta,scale=1)
img =cv2.warpAffine(gb,M,(cols,rows))
plt.imshow(img)


# In[23]:


img_gray = cv2.imread("D:/kitten.jpg",cv2.IMREAD_GRAYSCALE)
depth = cv2.CV_16S
grad_x = cv2.Sobel(src= img_gray, ddepth= depth,dx=1,dy=0,ksize=3)
grad_y = cv2.Sobel(src= img_gray, ddepth= depth,dx=0,dy=1,ksize=3)


# In[24]:


plt.imshow(grad_x)


# In[25]:


plt.imshow(grad_y)


# In[26]:


abs_grad_x =cv2.convertScaleAbs(grad_x)
plt.imshow(abs_grad_x)


# In[27]:


abs_grad_y=cv2.convertScaleAbs(grad_y)
plt.imshow(abs_grad_y)


# In[28]:


grad= cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
plt.imshow(grad)

