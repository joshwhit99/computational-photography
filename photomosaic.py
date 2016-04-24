import json
import numpy as np
import cv2
import math
import sys
import copy


INDEX_PATH = "./index/"

def getAverageColor(image, index, bins):
    (h,w,_) = image.shape
    histogram = cv2.calcHist([image],[index], None, [bins],[0,bins])
    x = 0
    for i in range(0,len(histogram)):
        x += (int(histogram[i])*i)
    return x / (w*h)

def extractFeature(image):
    entry = {}
    entry["b"] = getAverageColor(image, 0, 256)
    entry["g"] = getAverageColor(image, 1, 256)
    entry["r"] = getAverageColor(image, 2, 256)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    entry["h"] = getAverageColor(image, 0, 180)
    entry["s"] = getAverageColor(image, 1, 256)
    entry["v"] = getAverageColor(image, 2, 256)
    return entry

def readIndex():
    json_data = open(INDEX_PATH + "histogram.index").read()
    return json.loads(json_data)

#changes the input image to a size that is a multiple of the tilesize.
def preparInputImage(path, tileSize):
    i = cv2.imread(path)
    (h, w, _) = i.shape
    i = cv2.resize(i, (w / tileSize * tileSize, h / tileSize * tileSize))
    return i

def preparePatch(path, tileSize):
    image = cv2.imread(INDEX_PATH + path)
    image = cv2.resize(image, (tileSize, tileSize))
    return image

def calcDistance(fts1, fts2, vectors):
    distance = 0
    for vec in vectors:
        distance += math.pow(fts1[vec] - fts2[vec], 2)
    return math.sqrt(distance)

#copmutes the mean squared error (mes)
def avepix(cc,ff):
    g4 = 0
    b4 = 0
    r4 = 0
    (h, w, _) = cc.shape
    pix = h * w
    for i in range(h):
        for j in range(w):
            r3= int(ff[i][j][0])-int(cc[i][j][0])
            g3 = int(ff[i][j][1])-int(cc[i][j][1])
            b3=int(cc[i][j][2])-int(ff[i][j][2])
            g4 += g3*g3
            b4 += b3*b3
            r4 += r3*r3
    mesg = g4/pix
    mesr = r4/pix
    mesb = b4/pix
    mes = mesb+mesg+mesr
    return mes
    cv2.get


def getIndexImage(fts, index, vectors):
    minDistance = sys.maxint
    imagefile = ""
    for item in index:
        distance = calcDistance(fts, item, vectors)
        if distance < minDistance:
            minDistance = distance
            imagefile = item["file"]
    return imagefile

def processLine(w,h, index, inputImage, tileSize, channels):
    for i in range(0, h / tileSize):
        for j in range(0, w / tileSize):
            roi = inputImage[i * tileSize:(i + 1) * tileSize, j * tileSize:(j + 1) * tileSize]
            fts = extractFeature(roi)
            patch = preparePatch(getIndexImage(fts, index, channels), tileSize)
            inputImage[i * tileSize:(i + 1) * tileSize, j * tileSize:(j + 1) * tileSize] = patch
            #cv2.imshow("Progress", inputImage)
            #cv2.waitKey(1)
    return inputImage



def mosaic(tileSize):
    inputImagePath = str('pictures/bigpicture/bigimages/starwars101.jpg')
    channels = list(str('rgb'))
    index = readIndex()
    inputImage = preparInputImage(inputImagePath, tileSize)
    (h, w, _) = inputImage.shape
    inputImage = cv2.resize(inputImage, (w / tileSize * tileSize, h / tileSize * tileSize))
    print inputImage.shape
    cc = copy.deepcopy(inputImage)
    ff = processLine(w,h, index, inputImage, tileSize, channels)
    #cv2.imshow("Progress", ff)
    #cv2.waitKey(1000)
    ff = blend2(ff,tileSize)
    #cv2.imshow("Progress", ff)
    #cv2.waitKey(1000)
    mes = avepix(cc,ff)
    print "Finished processing of image"
    return mes, inputImage

#simple alpha blending. http://stackoverflow.com/questions/29106702/blend-overlapping-images-in-python
def blend2(image, tilesize):
    for i in range(0,len(image)-tileSize,tileSize):
        image1 = image[i+tileSize-2:i+tileSize,:]
        image2 = image[i+tileSize:i+tileSize+2,:]
        alpha = 0.5
        out = image1 * (1.0 - alpha) + image2 *alpha
        image[i+tileSize-1:i+tileSize+1,:] = out
    for i in range(0,len(image[0])-tileSize,tileSize):
        image1 = image[:,i+tileSize-2:i+tileSize]
        image2 = image[:,i+tileSize:i+tileSize+2]
        alpha = 0.5
        out = image1 * (1.0 - alpha) + image2 *alpha
        image[:,i+tileSize-1:i+tileSize+1]=out
        b = 1
    return image

#gausian bleninding of the individual pictures does not currently work, http://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html#gsc.tab=0
def blend(image, tilesize):
    b=1
    a = len(image)
    l = len(image[0])
    p= image[0:tileSize,30:tileSize]
    for j in range(0,len(image)-tileSize,tileSize):
        for k in range(0,len(image[0])-tileSize,tileSize):

            A = image[j+tileSize-3:j+tileSize,k+tileSize-3:k+tileSize]
            G = A.copy()
            gpA = [G]
            for i in xrange(3):
                G = cv2.pyrDown(G)
                gpA.append(G)
            B = image[j:j+tileSize,k+tileSize:k+tileSize+tileSize]
            G = B.copy()
            gpB = [G]
            for i in xrange(3):
                G = cv2.pyrDown(G)
                gpB.append(G)
            lpA = [gpA[2]]
            for i in xrange(2,0,-1):
                GE = cv2.pyrUp(gpA[i])
                b = GE[0:len(gpA[i-1]),0:len(gpA[i-1])]
                L = cv2.subtract(gpA[i-1],b)
                lpA.append(L)
            lpB = [gpB[2]]
            for i in xrange(2,0,-1):
                GE = cv2.pyrUp(gpB[i])
                b = GE[0:len(gpB[i-1]),0:len(gpB[i-1])]
                L = cv2.subtract(gpB[i-1],b)
                lpB.append(L)
             # Now add left and right halves of images in each level
            LS = []
            for la,lb in zip(lpA,lpB):
                rows,cols,dpt = la.shape
                p =la[:,0:2]
                pp= lb[:,cols/2:]
                ls = np.hstack((la[:,0:2], lb[:,cols/2:]))
                LS.append(ls)
            ls_ = LS[0]
            # now reconstruct
            for i in xrange(1,3):
                ls_ = cv2.pyrUp(ls_)
                b = ls_[0:len(LS[i]),0:len(LS[i])]
                ls_ = cv2.add(b, LS[i])
             #cv2.imwrite('Pyramid_blending2.jpg',ls_)
            image[j+tileSize-2:j+tileSize-2+tileSize,k+tileSize-2:k+tileSize-2+tileSize] = ls_
    return image
if __name__ == "__main__":
    #start with a tiles size of 30, this is almost always too big to work
    tileSize = int(25)
    #the mosaic function returns the mosaice image and the Mean Squared Error
    mes, inputImage = mosaic(tileSize)
    i = 0
    print tileSize
    #cv2.imshow("Progress", inputImage)
    #cv2.waitKey(10000)
    cv2.imwrite(str('pictures/out/original.jpg'), inputImage)
    s = inputImage.size
    s = s/3
   # s=s/134000
    s = 0.000005 * s
    print s
    #if image blending is turned on a good mes is under 5000, if it is turned off then under 9000 is good. if it does not reach that threshold it will
    #subtract 4 from the tile size and try again, as long as the tile size is not less then 8.


    while i == 0:
        if mes < 5000 or tileSize < 8 or tileSize < s:
            i = 1

            cv2.imwrite(str('pictures/out/out.jpg'), inputImage)
            print "the optimal tile size is "
            print tileSize
            print mes
        else:
            tileSize -=4
            mes,inputImage = mosaic(tileSize)
            s = inputImage.size
            s = s/3
            #s=s/134000
            s = 0.000005 * s
            print tileSize
            print mes
            print s

