import numpy as np
import cv2
import math
import pandas as pd
import glob

# START: OWN CODE

labels = ["none", "fire"] #labels to use for checking which class
rbg_labels = ['b1', 'g1', 'r1', 'b2', 'g2', 'r2', 'b3', 'g3', 'r3', 'b4', 'g4', 'r4', 'b5', 'g5', 'r5', 'b6', 'g6', 'r6', 'b7', 'g7', 'r7', 'b8', 'g8', 'r8', 'b9', 'g9', 'r9', 'b10', 'g10', 'r10', 'b11', 'g11', 'r11', 'b12', 'g12', 'r12', 'b13', 'g13', 'r13', 'b14', 'g14', 'r14', 'b15', 'g15', 'r15', 'b16', 'g16', 'r16']
train_output4 = {'b1': [], 'g1': [], 'r1': [], 'b2': [], 'g2': [], 'r2': [], 'b3': [], 'g3': [], 'r3': [], 'b4': [], 'g4': [], 'r4': [], 'class': []}
test_output4 = {'b1': [], 'g1': [], 'r1': [], 'b2': [], 'g2': [], 'r2': [], 'b3': [], 'g3': [], 'r3': [], 'b4': [], 'g4': [], 'r4': [], 'class': []}
train_output8 = {'b1': [], 'g1': [], 'r1': [], 'b2': [], 'g2': [], 'r2': [], 'b3': [], 'g3': [], 'r3': [], 'b4': [], 'g4': [], 'r4': [], 'b5': [], 'g5': [], 'r5': [], 'b6': [], 'g6': [], 'r6': [], 'b7': [], 'g7': [], 'r7': [], 'b8': [], 'g8': [], 'r8': [], 'class': []}
test_output8 = {'b1': [], 'g1': [], 'r1': [], 'b2': [], 'g2': [], 'r2': [], 'b3': [], 'g3': [], 'r3': [], 'b4': [], 'g4': [], 'r4': [], 'b5': [], 'g5': [], 'r5': [], 'b6': [], 'g6': [], 'r6': [], 'b7': [], 'g7': [], 'r7': [], 'b8': [], 'g8': [], 'r8': [], 'class': []}
train_output16 = {'b1': [], 'g1': [], 'r1': [], 'b2': [], 'g2': [], 'r2': [], 'b3': [], 'g3': [], 'r3': [], 'b4': [], 'g4': [], 'r4': [], 'b5': [], 'g5': [], 'r5': [], 'b6': [], 'g6': [], 'r6': [], 'b7': [], 'g7': [], 'r7': [], 'b8': [], 'g8': [], 'r8': [], 'b9': [], 'g9': [], 'r9': [], 'b10': [], 'g10': [], 'r10': [], 'b11': [], 'g11': [], 'r11': [], 'b12': [], 'g12': [], 'r12': [], 'b13': [], 'g13': [], 'r13': [], 'b14': [], 'g14': [], 'r14': [], 'b15': [], 'g15': [], 'r15': [], 'b16': [], 'g16': [], 'r16': [], 'class': []}
test_output16 = {'b1': [], 'g1': [], 'r1': [], 'b2': [], 'g2': [], 'r2': [], 'b3': [], 'g3': [], 'r3': [], 'b4': [], 'g4': [], 'r4': [], 'b5': [], 'g5': [], 'r5': [], 'b6': [], 'g6': [], 'r6': [], 'b7': [], 'g7': [], 'r7': [], 'b8': [], 'g8': [], 'r8': [], 'b9': [], 'g9': [], 'r9': [], 'b10': [], 'g10': [], 'r10': [], 'b11': [], 'g11': [], 'r11': [], 'b12': [], 'g12': [], 'r12': [], 'b13': [], 'g13': [], 'r13': [], 'b14': [], 'g14': [], 'r14': [], 'b15': [], 'g15': [], 'r15': [], 'b16': [], 'g16': [], 'r16': [], 'class': []}

#Returns a histogram that contains number of pixels in the blue/green/red color channels in the bins
#Verifies that the pixels are counted exactly 3 times, otherwise returns None

def hist(pic, num_bins, bol):
    histogram = []
    for i in range(num_bins):
        histogram += [0, 0, 0]
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            for k in range(3):
                histogram[k*num_bins + int(pic[i][j][k] // (256/num_bins))] += 1
                
    #Verfication Step: makes sure that all pixels are counted exactly 3 times, once in each color channel
    if int(sum(histogram)/3) == len(pic)*len(pic[0]):
        count = 0
        for each in rbg_labels:
            if bol == 0:
                train_output16[each].append(histogram[count])
            else:
                test_output16[each].append(histogram[count])
            count+=1
        return histogram
    else:
        return None

#classifies the histograms using "knn" nearest neighbors (findest the prediction)
def threeNNhist(test_hist, train_hist, knn, num_bins):
    print("Results: (" + str(num_bins) + " bins, " + str(knn) +" nearest neighbors) ")
    num_right = 0
    for test in test_hist:
        # Find the hist in the train_hist array that has smallest dist to current img
        mindist = [[-1,0] for i in range(knn)]
        for train in train_hist:
            a = np.array(test[1])
            b = np.array(train[1])
            dist = np.linalg.norm(a-b) # distance function using numpy between train[1] and test[1]
            for k in range(knn):
                if dist < mindist[k][0] or mindist[k][0] == -1:
                    mindist.insert(k, [dist, train[0]]) #assigns to the test image the label of the training image that has the nearest representation
                    break
        
        label = []
        for i in range(knn):
            label += [mindist[i][1]]
        
        #check which is the best label for the image
        count = [0,0,0]
        for i in range(len(label)):
            for j in range(3):
                if label[i] == labels[j]:
                    count[j] += len(label)-i
        maxval = max(count)
        for i in range(3):
            if count[i] == maxval:
                best = labels[i]
                if best == test[0]:
                    num_right += 1
        #print statement to see classifications
        print("Test image " + test[2] + " of class " + test[0] + " has been assigned to class "+ best + ".")
    #print statement to see accuracy of classifer
    print("Accuracy of classifier: " + str(num_right) + "/12 right.")



if __name__ == "__main__":
    #gets all the images in the ImClass folder that have term "train" or the term "test" and store them
    train_pics = glob.glob('Dataset/Training/*.jpg')
    test_pics = glob.glob('Dataset/Testing/*.jpg')
    
    # For bins = 8 and nearest neighbors = 1 (change these numbers to desired bins # and nearest neighbors #)
    knn_arr = [3]
    bins_arr = [16]

    for knn in knn_arr:
            for num_bins in bins_arr:
                if num_bins == 8 or knn == 3:
                    train_hist = []
                    test_hist = []
                    
                    #for each training pics store histogram in train_hist array
                    for file in train_pics:
                        pic = cv2.imread(file)
                        hist1 = hist(pic, num_bins, 0)
                        for i in range(len(labels)):
                            if labels[i] in file:
                                if hist1 != None:
                                    train_output16["class"].append(i)

                    df = pd.DataFrame(train_output16, columns= ['b1', 'g1', 'r1', 'b2', 'g2', 'r2', 'b3', 'g3', 'r3', 'b4', 'g4', 'r4', 'b5', 'g5', 'r5', 'b6', 'g6', 'r6', 'b7', 'g7', 'r7', 'b8', 'g8', 'r8', 'b9', 'g9', 'r9', 'b10', 'g10', 'r10', 'b11', 'g11', 'r11', 'b12', 'g12', 'r12', 'b13', 'g13', 'r13', 'b14', 'g14', 'r14', 'b15', 'g15', 'r15', 'b16', 'g16', 'r16', 'class'])
                    df.to_csv (r'C:\Users\sypat\Documents\hw3\training' + str(num_bins) + '.csv', index = False, header=False)
                    
                    
                    #for each testing pics store histogram in test_hist array
                    for file in test_pics:
                        pic = cv2.imread(file)
                        hist2 = hist(pic, num_bins, 1)
                        for i in range(len(labels)):
                            if labels[i] in file:
                                if hist1 != None:
                                    test_output16["class"].append(i)

                    df = pd.DataFrame(test_output16, columns= ['b1', 'g1', 'r1', 'b2', 'g2', 'r2', 'b3', 'g3', 'r3', 'b4', 'g4', 'r4', 'b5', 'g5', 'r5', 'b6', 'g6', 'r6', 'b7', 'g7', 'r7', 'b8', 'g8', 'r8', 'b9', 'g9', 'r9', 'b10', 'g10', 'r10', 'b11', 'g11', 'r11', 'b12', 'g12', 'r12', 'b13', 'g13', 'r13', 'b14', 'g14', 'r14', 'b15', 'g15', 'r15', 'b16', 'g16', 'r16', 'class'])
                    df.to_csv (r'C:\Users\sypat\Documents\hw3\testing' + str(num_bins) + '.csv', index = False, header=False)
                    # print("-----------------------------------------------------------")
                    # threeNNhist(test_hist, train_hist, knn, num_bins)
                    # print("-----------------------------------------------------------")

# END: OWN CODE