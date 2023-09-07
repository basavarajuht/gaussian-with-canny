import cv2
import numpy as np
import imutils
image = cv2.imread('C:\\Users\\Lenovo\\Desktop\\16.png')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3,3), 0)
edged = cv2.Canny(img, 40, 90)
dilate = cv2.dilate(edged, None, iterations=1)
mask = np.ones(img.shape[:2], dtype="uint8") * 255
cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
orig = img.copy()
newimage = cv2.bitwise_and(dilate.copy(), dilate.copy(), mask=mask)
img2 = cv2.erode(newimage, None, iterations=1)
img3 = cv2.erode(newimage, None, iterations=1)
cv2.imshow('Original image', image)
cv2.imshow('Dilated', dilate)
cv2.imshow('New Image', newimage)
cv2.imshow('New Image', img2)
#cv2.imshow('New Image3', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()


bin_thresh = (dilate == 0).astype(np.uint8)


def neighbours(x, y, image):
    """Return 8-neighbours of point p1 of picture, in clockwise order"""
    i = image
    x1, y1, x_1, y_1 = x+1, y-1, x-1, y+1
    return [i[y1][x],  i[y1][x1],   i[y][x1],  i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    n = neighbours + neighbours[0:1]    # P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))


def zhangSuen(image):
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        # Step 1
        changing1 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P4 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P6 == 0 and   # Condition 3
                    transitions(n) == 1 and  # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing1.append((x, y))
        for x, y in changing1:
            image[y][x] = 0
        # Step 2
        changing2 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P2 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P8 == 0 and   # Condition 3
                    transitions(n) == 1 and  # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing2.append((x, y))
        for x, y in changing2:
            image[y][x] = 0
    return image * 255


if __name__ == '__main__':
    after = zhangSuen(bin_thresh)
    cv2.imshow('after', after)
    cv2.waitKey()
    cv2.destroyAllWindows()
