import json
import os
import cv2
import numpy as np
path_to_json = 'JSON samples'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]


for kj in json_files:
    # print(i)
# Opening JSON file
# # f = open('/home/guru/anaan/Ak_99.json')
#
# #Testing double polygon image
    f = open(path_to_json+'/'+kj)
    R, C = 104, 104
    #Testing single polygon image
    #f = open('C:\\Users\\HP\\Desktop\\Single_annotation\\outputs\\Ak_435.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    poly_count = len(data['outputs']['object'])
    #print(poly_count)

    # Iterating through the json
    # list
    name_img = data['path'];
    split_obj = os.path.split(name_img)
    #print(split_obj[1])

    # image_curr = cv2.imread("/home/guru/anaan/Ak_99.jpg")


    image_segm = np.zeros([R, C])

    for k in range(poly_count):
        counter = 0
        prev = 0
        curr = 0
        count2 = 0
        countt = 0
        points = (np.zeros([int(len(data['outputs']['object'][k]['polygon']) / 2), 2]).astype(int))
        print(k)
        print(data['outputs']['object'][k]['polygon'])
        for i in data['outputs']['object'][0]['polygon']:
            counter += 1
            if counter % 2 == 0:
                points[count2, 0] = prev
                points[count2, 1] = data['outputs']['object'][0]['polygon'][i]
                # image_segm[prev,data['outputs']['object'][0]['polygon'][i]] = 1;
                count2 += 1
            # print(data['outputs']['object'][0]['polygon'][i])
            prev = data['outputs']['object'][0]['polygon'][i];
        # Closing file
        f.close()
        # print(counter)
        #print(points)
        # print(countt)
        contours = points
        # cv2.drawContours(image_segm,[points],0,(0,0,0),2)
        cv2.fillPoly(image_segm, [points], [1])

        # cv2.imshow('some', image_segm)

        (h, w) = image_segm.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated = cv2.warpAffine(image_segm, M, (w, h))
        # cv2.imshow("Rotated by 90 Degrees", rotated)

        # horizontal = 1 , vertical = 0,flipping image horizontally and vertically = -1
        flipped = cv2.flip(rotated, 0)

        # cv2.waitKey(-1)
    # cv2.imwrrite()
    cv2.imwrite( 'JSON 2 Masks/' + kj[:-5] + '.png', flipped*255)
    #cv2.imshow("la", flipped)
    #cv2.waitKey(2000)
