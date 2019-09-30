import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
from src.model import MobileFaceNet
from torchvision import transforms
import numpy as np
import os

class faceRecognizer:
    def __init__(self, threshold, model_path, facebank_path, embedding_size, device):
        self.threshold = threshold
        self.device = device
        
        # Model Loading
        self.model = MobileFaceNet(embedding_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize([112,112]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
        self.facebank_path = facebank_path

        self.embeddings = None
        self.names = None
    
    def set_threshold(self, thresh):
        self.threshold = thresh
    

    def extract_feature(self, img):
        emb = self.model(self.transform(img).to(self.device).unsqueeze(0))
        return emb
    
    def make_facebank(self, groupID):
        names = []
        root_path = Path(self.facebank_path)
        
        embeddings = []

        groups = []
        # If didn't get groupID, make facebank using all groups data
        if groupID == "all":
            for group in root_path.iterdir():
                if group.is_file():
                    continue
                else:
                    for obj in group.iterdir():
                        if obj.is_file():
                            continue
                        else:
                            embs = []
                            for file in obj.iterdir():
                                if not file.is_file():
                                    continue
                                else:
                                    print(group, obj, file)
                                    emb = np.load(file)
                                    emb = torch.from_numpy(emb).to(self.device)
                                    embs.append(emb)

                            if len(embs) == 0:
                                continue

                            embedding = torch.cat(embs).mean(0, keepdim=True)
                            embeddings.append(embedding)
                            names.append(obj.name)
                            groups.append(group)
        
        # If get groupID, make facebank using groupID's data
        else:
            groups.append(groupID)
            group_path = Path(self.facebank_path + groupID) 
            for obj in group_path.iterdir():
                if obj.is_file():
                    continue
                else:
                    embs = []
                    for file in obj.iterdir():
                        if not file.is_file():
                            continue
                        else:
                            try:
                                emb = np.load(file)
                            except:
                                continue
                            emb = torch.from_numpy(emb).to(self.device)
                            embs.append(emb)

                    if len(embs) == 0:
                        continue

                    embedding = torch.cat(embs).mean(0, keepdim=True)
                    embeddings.append(embedding)
                    names.append(obj.name)
                    groups.append(groupID)
        
        if len(list(group_path.iterdir())) > 0:
            embeddings = torch.cat(embeddings)
            names = np.array(names)
            groups = np.array(groups)
        
        return embeddings, names, groups

    def check_registration(self, face, groupID):
        
        result_dict = {}
        result_dict['objectID_list'] = []
        
        # make feature
        face_tensor = self.transform(face).to(self.device)
        face_tensor = face_tensor.unsqueeze(0)
        emb = self.model(face_tensor)

        if len(os.listdir(self.facebank_path + groupID)) > 0:
            facebank_embeddings, facebank_names, facebank_groups = self.make_facebank(groupID) 

            diff = emb.unsqueeze(-1) - facebank_embeddings.transpose(1,0).unsqueeze(0)
            
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            np_dist = np.squeeze(dist.detach().cpu().numpy(), axis=0)
            
            result_dist = np_dist[np.where(np_dist < self.threshold)]
            result_names = facebank_names[np.where(np_dist < self.threshold)]
            result_groups = facebank_groups[np.where(np_dist < self.threshold)]
            
            if result_dist.shape[0] > 0:
                for i in range(result_dist.shape[0]):
                    object_dict = {}
                    object_dict['groupID'] = result_groups[i]
                    object_dict['objectID'] = result_names[i]
                    object_dict['distance'] = str(result_dist[i])
                    result_dict['objectID_list'].append(object_dict)

            else:
                object_dict = {}
                object_dict['objectID'] = "Unknown"
                object_dict['distance'] = "-1"
                result_dict['objectID_list'].append(object_dict)

        else:
            object_dict = {}
            object_dict['objectID'] = "Unknown"
            object_dict['distance'] = "-1"
            result_dict['objectID_list'].append(object_dict)
        
        result_dict['objectID_list'] = sorted(result_dict['objectID_list'], key=lambda result: (result['distance']))
        return result_dict        
