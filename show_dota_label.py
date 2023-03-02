from PIL import Image,ImageFont,ImageDraw
import numpy as np
import os
from skimage import io
from pylab import *
# font = ImageFont.truetype('arial.tff',20)
images_path='/media/xaserver/DATA/zty/FoRDet/dataset/images/'
fillColor = (255,255,255)
label_path='/media/xaserver/DATA/zty/FoRDet/dataset/labelTxt/'
label_files= os.listdir(label_path)
label_files.sort()
#filename = 'show/labels/637.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
for label_name in label_files:
    print(label_name)
    image_name = label_name.replace('.txt', '.png')
    image = Image.open(images_path+image_name )
    filename= label_path + label_name
    print(filename)
    with open(filename, 'r') as file_to_read:

        lines = file_to_read.readlines() # 整行读取数据
        for line in lines:
          obj_infos = line.split(' ')
          if len(obj_infos) == 10:
            print('obj_infos', obj_infos)

            # conf = float(obj_infos[1])

            # x1 = float(obj_infos[2])
            # y1 = float(obj_infos[3])
            # x2 = float(obj_infos[4])
            # y2 = float(obj_infos[5])
            # x3 = float(obj_infos[6])
            # y3 = float(obj_infos[7])
            # x4 = float(obj_infos[8])
            # y4 = float(obj_infos[9]) 
            # class_name= obj_infos[0]


            x1 = float(obj_infos[0])
            y1 = float(obj_infos[1])
            x2 = float(obj_infos[2])
            y2 = float(obj_infos[3])
            x3 = float(obj_infos[4])
            y3 = float(obj_infos[5])
            x4 = float(obj_infos[6])
            y4 = float(obj_infos[7]) 
            class_name= obj_infos[8]
            conf = 1.0

            # text_info = class_name + ':' + str(conf)[0:5]
            outline_0 = (0, 255, 0)
            if class_name=="plane":
                outline_0 = (0, 255, 255)
            if class_name=="bridge":
              outline_0 = (255, 0, 0)
            if class_name == "ship":
                outline_0 = (255, 255, 0)
            if class_name == "tennis-court":
              outline_0 = (0,255, 0)
            if class_name=="harbor":
                outline_0 = (255, 0, 255)
            if class_name=="basketball-court":
                outline_0 = (128, 128, 105)
            if class_name=="baseball-dimond":
                outline_0 = (160, 32, 240)
            if class_name=="ground-track-field":
                outline_0 = (0, 255, 127)
            if class_name=="storage-tank":
                outline_0 = (0, 199, 140)
            if class_name=="soccer-ball-field":
                outline_0 = (255, 235, 205)
            if class_name=="small-vehicle":
                outline_0 = (65, 105, 225)
            if class_name=="large-vehicle":
                outline_0 = (227, 23, 13)
            if class_name=="swimming-pool":
                outline_0 = (210, 105, 30)
            if class_name=="roundabout":
                outline_0 = (240, 255, 255)
            if class_name=="helicopter":
                outline_0 = (221, 160, 221)
            # 创建一个可以在给定图像上绘图的对象
            draw = ImageDraw.Draw(image)  # 绘图句柄
            # offsetx, offsety = font.getoffset(class_name)  # 获得文字的offset位置
            # width, height = font.getsize(class_name)  # 获得文件的大小
            width = 2
            height = 2
            im = np.array(image)
            print(class_name, x1, y1, x2, y2, x3, y3, x4, y4)
            print(outline_0)
            if conf >=0.3:
                draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline=outline_0)
                draw.polygon([(x1 - 0.5, y1 - 0.5), (x2 - 0.5, y2 -  0.5), (x3 -  0.5, y3 -  0.5), (x4 -  0.5, y4 -  0.5)], outline=outline_0)
                draw.polygon([(x1-1, y1-1), (x2-1, y2-1), (x3-1, y3-1), (x4-1, y4-1)], outline=outline_0)
                #draw.polygon([(x1 - 1.5, y1 - 1.5), (x2 - 1.5, y2 - 1.5), (x3 - 1.5, y3 - 1.5), (x4 - 1.5, y4 - 1.5)], outline=outline_0)
                #draw.polygon([(x1-2, y1-2), (x2-2, y2-2), (x3-2, y3-2), (x4-2, y4-2)], outline=outline_0)
                draw.polygon([(x1 +0.5, y1 +0.5), (x2 +0.5, y2 +0.5), (x3 + 0.5, y3 + 0.5), (x4 + 0.5, y4 + 0.5)],outline=outline_0)
                draw.polygon([(x1+1, y1+1), (x2+1, y2+1), (x3+1, y3+1), (x4+1, y4+1)], outline=outline_0)
                #draw.polygon([(x1 +1.5, y1 +1.5), (x2 +1.5, y2 + 1.5), (x3 +1.5, y3 + 1.5), (x4 + 1.5, y4 + 1.5)],outline=outline_0)
                #draw.polygon([(x1+2, y1+2), (x2+2, y2+2), (x3+2, y3+2), (x4+2, y4+2)], outline=outline_0)
                #为了避免标签位置超出图片，需要对标签的顶点位置做一个判别
                if (x2 + 15 < 1024 and y2 + 25 < 1024):
                    pass
                elif(x3 + 15 < 1024 and y3 + 25 < 1024):
                    x2 = x3
                    y2 = y3
                elif (x4 + 15 < 1024 and y4 + 25 < 1024):
                    x2 = x4
                    y2 = y4
                else:
                    x2 = x1
                    y2 = y1

                draw.polygon([(x2, y2), (x2 + width, y2), (x2 + width, y2 + height), (x2, y2 + height)], outline_0, outline=outline_0)
                # draw.ink = 0
                # draw.text((x2, y2), class_name, font=font)  # 绘图
                # draw.text((x2, y2), text_info,fill=fillColor)  # 绘图
            else:
                pass
            #image.show()
        img=np.array(image)
        io.imsave('/media/xaserver/DATA/zty/FoRDet/1871/split/show/'+image_name,img)
        # assert 2==1
