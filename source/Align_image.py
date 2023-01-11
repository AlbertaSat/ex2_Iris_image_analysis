### Image alignment software.
### Liam Droog, Iris Team Lead
### January, 2023
# adapted from https://thinkinfi.com/image-alignment-and-registration-with-opencv/

"""
    Usage: python3 Align_image.py -i inputfile.jpg -o outputfile.jpg -r referencefile.jpg 
           *Tested only on Ubuntu thus far * 
    Bugs: None yet
"""

import cv2
import numpy as np
import sys, getopt, os
def main(argv):
    # assert inputs:
    inputfile = ''
    referencefile = ''
    outputfile = ''
    help = 'Help: -r <referencefile> -i <inputfile> -o <outputfile> '
    debug = False
    try:    
      opts, args = getopt.getopt(argv,"hi:r:o:d:",["ifile=","ofile=", "rfile=", "debug="])
    except getopt.GetoptError:
      print(help)
      sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (help)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-r", "--rfile"):
            referencefile = arg
        elif opt in ("-d", "--debug"):
            if arg.lower() == "true":
                debug = True
    print('Input file is', inputfile)
    print('Output file is', outputfile)
    print('Reference file is', referencefile)

    # assert files exist
    for i in [inputfile, outputfile, referencefile]:
        if not os.path.exists(i):
            print(i, "is an invalid file")
            sys.exit(2)

    # load images
    # test image
    crop1 = cv2.imread(inputfile)

    # reference image
    crop2 = cv2.imread(referencefile)

    # Greyscale em
    imgTest_grey = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    imgRef_grey = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
    height, width = imgRef_grey.shape
    # find keypoints
    orb_detector = cv2.ORB_create(1000)
    keyPoint1, des1 = orb_detector.detectAndCompute(imgTest_grey, None)
    keyPoint2, des2 = orb_detector.detectAndCompute(imgRef_grey, None)
    # Display keypoints for reference image in green color
    if debug:
        imgKp_Ref = cv2.drawKeypoints(crop2, keyPoint1, 0, (0,222,0), None)
        imgKp_Ref = cv2.resize(imgKp_Ref, (width, height))
        
        cv2.imshow('Key Points', imgKp_Ref)
        cv2.waitKey(0)

    # Match features between two images using Brute Force matcher with Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match the two sets of descriptors.
    matches = matcher.match(des1, des2)
    
    # Sort matches on the basis of their Hamming distance.
    # matches.sort(key=lambda x: x.distance)
    list(matches).sort(key=lambda x: x.distance, reverse=False)
    
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)
    
    # Display only 100 best matches {good[:100}
    imgMatch = cv2.drawMatches(crop2, keyPoint2, crop1, keyPoint1, matches[:100], None, flags = 2)
    imgMatch = cv2.resize(imgMatch, (width, height))
    
    if debug:
        cv2.imshow('Image Match', imgMatch)
        cv2.waitKey(0)

    # Match images:
    # Define 2x2 empty matrices
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    # Storing values to the matrices
    for i in range(len(matches)):
        p1[i, :] = keyPoint1[matches[i].queryIdx].pt
        p2[i, :] = keyPoint2[matches[i].trainIdx].pt
    
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use homography matrix to transform the unaligned image wrt the reference image.
    aligned_img = cv2.warpPerspective(crop1, homography, (width, height))
    # Resizing the image to display in our screen (optional)
    
    # Copy of input image
    if debug:
        aligned_img = cv2.resize(aligned_img, (width, height))
        imgTest_cp = crop1.copy()
        imgTest_cp = cv2.resize(imgTest_cp, (width, height))
        cv2.imshow('Input Image', imgTest_cp)
        cv2.imshow('Output Image', aligned_img)
    
    cv2.imwrite(outputfile, aligned_img)
    print("Image written to ", outputfile)
    cv2.waitKey(0)
    return

if __name__ == "__main__":
    main(sys.argv[1:])