import wrapper
import cv2

image = 0

for i in range(18,19,1):
    inputname =  "data/" + str(i)
    image = cv2.imread(inputname + '.jpg')
    print(image.shape)
    vps = wrapper.dealAImage(image,"data/result/" + str(i),True,True,True)
    vps = [[i[0], i[1], ] for i in vps]
    print("vps:")
    print(vps)
    for pt in vps:
         cv2.circle(image, (int(pt[0]), int(pt[1])), 20, (0,255,0), 3)
    while(True):
        cv2.imshow("frame", image)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
    

    
#while(True):
#    cv2.imshow("frame", image)
#    key = cv2.waitKey(1)
#    if key == 27:
#        break
    