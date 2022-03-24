import os 
import cv2 
import dlib 
import numpy as np
import torch 
import dgl
from matplotlib import pyplot as plt
from tqdm import tqdm 


#Create graphs from face pictures

#Get the coordinates of 68 points in the face using dlib landmark detector 
def get_face_landmarks(img):

    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    faces = face_detector(img, 1)  

    if len(faces) == 1:

        face_shape= landmark_detector(img, faces[0])
        landmarks = np.zeros(shape=(68, 2))

        for i in range(68):
            coordinate = face_shape.part(i)
            landmarks[i, :] = np.array([coordinate.x, coordinate.y])

        return landmarks.astype(np.int32)

    else:

        return None 
    

#Get the mean/average coordinate for each face feature 
def get_face_features_coordinates(landmarks):

    face_features =   ['left_eyebrow', 'right_eyebrow', 'nose', 'left_eye', 'right_eye', 'mouth']
    features_points = [    (17, 22),       (22, 27),    (27,36), (36, 42),   (42,48),    (48,68)]

    #Create face features dictionary and the mean value of the landmarks coordinates is their value 
    face_features_coordinates = dict()
    for feature,points in zip(face_features, features_points):
        face_features_coordinates[feature] = np.mean(landmarks[np.arange(*points)], axis = 0)

    return face_features_coordinates

#Create an ORB descriptor using the coordinate of each face feature
def get_descriptors_from_mean_coordinate(img, face_features_coordinates):

    orb = cv2.ORB_create()

    face_features_descriptors = dict()

    for feature in face_features_coordinates:

        kp = [cv2.KeyPoint(face_features_coordinates[feature][0],face_features_coordinates[feature][1],1)]
        _, des = orb.compute(img, kp)
        face_features_descriptors[feature] = des

    return face_features_descriptors


#Create tensor for the node features 
def create_node_features_tensor(face_features_coordinates):

    one_hot_face_nodes = np.eye(len(face_features_coordinates))

    tensor = np.zeros(shape=(6,8))

    for count, key in enumerate(face_features_coordinates):

        #one_hot_coordinates = np.concatenate((one_hot_face_nodes[count],face_features_coordinates[key], descriptors[key]), axis = None, dtype = float)
        one_hot_coordinates = np.concatenate((one_hot_face_nodes[count],face_features_coordinates[key]), axis = None, dtype = float)
        tensor[count] = one_hot_coordinates

        #print(one_hot_face_nodes[count])
        #print(face_features_coordinates[key])
        #print(face_features_descriptors[key])
            
    return tensor 


#Create tensor made of the distances of all the face features for edge data
def calculate_distance_face_features(img, face_features_coordinates):
    
    #Height and width of the image
    h, w, c = img.shape
    
    points = []
    for key in face_features_coordinates:
        points.append(face_features_coordinates[key])
    
    #Eucliad distance between points and normalize using widht and height's hypotenus 
    distances_matrix = np.array([np.linalg.norm(((item*np.ones((len(points),len(item))))-points)/np.hypot(h,w),axis=1) for item in points])

    return distances_matrix


#Create bidicrectional DGL graph, 1 node for each feature
# 0---1
# |   |
# 2---3
#   |
#   4
#   |
#   5
def create_graph(nodes_features, distances):

    graph = dgl.graph(([0,0,1,2,2,3,4],[1,2,3,3,4,4,5]), num_nodes=6)
    graph = dgl.to_bidirected(graph)

    edge_data = np.zeros(shape=(14,1))

    #Get the desired distances
    for i in range(len(graph.edges()[0])):
        edge_data[i] = distances[graph.edges()[0][i],graph.edges()[1][i]]

    graph.ndata['node_features'] = torch.from_numpy(nodes_features)
    graph.edata['edge_features'] = torch.from_numpy(edge_data)

    return graph

#Call this function to display the original image, the 68 points in the face and the mean/average point for each face feature
def visualize(img, landmarks, face_feature_coordinate):

    fig = plt.figure(figsize=(15, 5))

    #Show face image
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img)

    #Show numbered landamarks from the image
    features_landmarks = landmarks[np.arange(17,68)]
    
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(features_landmarks[:,0], -features_landmarks[:,1], alpha = 0.5)
    for i in range(features_landmarks.shape[0]):
        plt.text(features_landmarks[i,0]+1, -features_landmarks[i,1], str(i), size=8)

    #Show mean coordinate for every face feature
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(features_landmarks[:,0], -features_landmarks[:,1], alpha = 0.5)
    for key in face_feature_coordinate:
        plt.plot([face_feature_coordinate[key][0]], [-face_feature_coordinate[key][1]], marker='+', color='blue', markersize=10, mew=4)

    plt.show()



#Dataset creation 
MY_FACE = 'Faces/my_face' 
NOT_MYFACE = 'Faces/not_my_face'

LABELS = {MY_FACE: 0, NOT_MYFACE: 1}

graphs = []
labels = []

for label in LABELS:
    for filename in tqdm(os.listdir(label)):
        if filename.endswith('.jpg'):

                img_path = os.path.join(label, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For visualisation 

                face_landmarks = get_face_landmarks(img)

                if type(face_landmarks) is np.ndarray:

                    face_features_coordinates = get_face_features_coordinates(face_landmarks)
                    #face_features_descriptors = get_descriptors_from_mean_coordinate(img, face_features_coordinates)

                    ##--Uncomment to visualize
                    #visualize(img,face_landmarks,face_features_coordinates)

                    node_features = create_node_features_tensor(face_features_coordinates)
                    edge_features = calculate_distance_face_features(img, face_features_coordinates)
                    graph = create_graph(node_features, edge_features)

                    graphs.append(graph)
                    labels.append(np.array([0,1])[LABELS[label]])
                    #labels.append(np.array([LABELS[label]]))

                else:

                    print(f'Face not found in {filename}')



# Convert the label list to tensor for saving.
labels = torch.LongTensor(labels)

dgl.save_graphs('dgl_graphs-1.dgl', graphs, {'labels': labels})


