from PIL import Image

from src.faceRecognizer import faceRecognizer
from src.utils import *
import torch

import time
import numpy as np
import os
import io
import sys
from flask import Flask, request, Response, jsonify
#from flask_restful import Resource, Api
import base64
import shutil
import re
import json
from datetime import datetime

app = Flask(__name__)

database_foldername = './database/'
make_dir(database_foldername)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
f_Recognizer = faceRecognizer(threshold=0.7, model_path='model_mobilefacenet.pth', 
                                facebank_path= database_foldername, embedding_size=512, device=device)


## 0. 등록 / 미등록 인원판단
## input  : (필수) img, groupID, threshold // (필수아님) count
## output : result, objectID_list, description
@app.route('/api/check-registration', methods=['GET', 'POST'])
def check_register():
    result_dict = {}
    
    # Image Converting

    json_data = json.loads(request.data)
    try:
        b64_bytes = json_data['img']
        b64_image = Image.open(io.BytesIO(base64.b64decode(b64_bytes))).convert("RGB")
    except:
        result_dict['result'] = str('False')
        result_dict['description'] = "base64 decoding error"
        return jsonify(result_dict)

    
    group_id = json_data['groupID'].lower()
    user_thresh = json_data['threshold']

    f_Recognizer.set_threshold(float(user_thresh))
    
    

    # GroupID에 해당하는 폴더가 없을 경우에, 에러값 반환
    if not os.path.isdir(database_foldername+group_id):
        result_dict['description'] = "groupID : "+ str(group_id) + " do not exist"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)
    
    #b64_image = align_face(b64_image)
    result_dict = f_Recognizer.check_registration(b64_image, group_id)   

    if 'count' in json_data.keys():
        count = json_data['count']
        while count < len(result_dict['objectID_list']):
            del(result_dict['objectID_list'][-1])
        result_dict['count'] = len(result_dict['objectID_list'])

    return jsonify(result_dict)

## 0. 인원 등록
## input  : img, objectID, groupID
## output : result, imgID, description
@app.route('/api/register', methods=['POST'])
def register():
    result_dict = {}
    try:
        json_data = json.loads(request.data)
    except:
        result_dict['result'] = str('False')
        result_dict['description'] = "json decoding error"
        activate = True
        return jsonify(result_dict)

    try: 
        b64_bytes = json_data['img']
        img = Image.open(io.BytesIO(base64.b64decode(b64_bytes))).convert("RGB")
    except:
        result_dict['result'] = str('False')
        result_dict['description'] = "base64 decoding error"
        activate = True
        return jsonify(result_dict)

    # handling objectID 
    object_id = json_data['objectID'].lower()
        
    # handling groupID
    group_id = json_data['groupID'].lower()
    group_db = os.listdir(database_foldername)
    if len(group_db) is 0:
        make_dir(database_foldername + group_id)

    object_db = os.path.isdir(database_foldername+object_id)

    if object_db:
        make_dir(database_foldername + group_id + '/' + object_id)
        imgID = set_imgID(database_foldername + group_id + '/' + object_id)
        data_dir = database_foldername + group_id + '/' + object_id + '/'
    else:
        make_dir(database_foldername + group_id + '/' + object_id)
        imgID = set_imgID(database_foldername + group_id + '/' + object_id)
        data_dir = database_foldername + group_id + '/' + object_id + '/'

    # extract feature
    #img = align_face(img)
    embedding = f_Recognizer.extract_feature(img)
    embedding = embedding.detach().cpu().numpy()
    
    # save feature
    img.save(data_dir+imgID+".png")
    np.save(data_dir+imgID+".npy", embedding)
    
    result_dict['result'] = str('True')
    activate = True
    return jsonify(result_dict)

## 1. 등록 / 미등록 인원판단
## input  : (필수) img, groupID, threshold // (필수아님) count
## output : result, objectID_list, description
@app.route('/api/check-registration-align', methods=['GET', 'POST'])
def check_register_align():
    global activate
    activate = False
    result_dict = {}
    
    # Image Converting
    try:
        json_data = json.loads(request.data)
    except:
        result_dict['result'] = str('False')
        result_dict['description'] = "json decoding error"
        activate = True
        return jsonify(result_dict)
    
    if 'img' not in json_data:
        result_dict['description'] = "In check-registration phase, you have to input img"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)
    if json_data['img'] is "":
        result_dict['description'] = "In check-registration phase, you have to input img"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)
 
    try:
        b64_bytes = json_data['img']
        b64_image = Image.open(io.BytesIO(base64.b64decode(b64_bytes))).convert("RGB")
    except:
        result_dict['result'] = str('False')
        result_dict['description'] = "base64 decoding error"
        activate = True
        return jsonify(result_dict)

    if 'groupID' not in json_data:
        result_dict['description'] = "In check-registration phase, you have to input groupID"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)
    elif json_data['groupID'] is "":
        result_dict['description'] = "In check-registration phase, you have to input groupID"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)
    else:
        group_id = json_data['groupID']
    
    if 'threshold' not in json_data:
        result_dict['description'] = "In check-registration phase, you have to input threshold"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)
    if json_data['threshold'] is "":
        result_dict['description'] = "In check-registration phase, you have to input threshold"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)

    user_thresh = json_data['threshold']

    f_Recognizer.set_threshold(float(user_thresh))
    
    group_db,db_affect = db.query("select * from groupID where groupID = (%s)",group_id)

    # GroupID에 해당하는 폴더가 없을 경우에, 에러값 반환
    if len(group_db) is 0:
        result_dict['description'] = "groupID : "+ str(group_id) + " do not exist"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)
    
    b64_image = align_face(b64_image)
    result_dict = f_Recognizer.check_registration(b64_image, group_db[0]['groupID'])
    

    if 'count' in json_data.keys():
        count = json_data['count']
        while count < len(result_dict['objectID_list']):
            del(result_dict['objectID_list'][-1])
        result_dict['count'] = len(result_dict['objectID_list'])


    activate = True
    return jsonify(result_dict)

## 2. 인원 등록
## input  : img, objectID, groupID
## output : result, imgID, description
@app.route('/api/register-align', methods=['POST'])
def register_align():
    result_dict = {}
    global activate
    activate = False
    # handling img data
    try:
        json_data = json.loads(request.data)
    except:
        result_dict['result'] = str('False')
        result_dict['description'] = "json decoding error"
        activate = True
        return jsonify(result_dict)


    if 'img' not in json_data:
        result_dict['description'] = "In registration phase, you have to input img"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)

    try: 
        b64_bytes = json_data['img']
        img = Image.open(io.BytesIO(base64.b64decode(b64_bytes))).convert("RGB")
    except:
        result_dict['result'] = str('False')
        result_dict['description'] = "base64 decoding error"
        activate = True
        return jsonify(result_dict)

    # handling objectID 
    if 'objectID' not in json_data:
        result_dict['description'] = "In registration phase, you have to input objectID"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)
    elif json_data['objectID'] is "":
        result_dict['description'] = "In registration phase, you have to input objectID"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)
    else:
        object_id = json_data['objectID']
        
    # handling groupID
    if 'groupID' not in json_data:
        result_dict['description'] = "In registration phase, you have to input groupID"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)
    elif json_data['groupID'] is "":
        result_dict['description'] = "In registration phase, you have to input groupID"
        result_dict['result'] = str('False')
        activate = True
        return jsonify(result_dict)
    else:
        group_id = json_data['groupID']
        group_db,db_affect = db.query("select * from groupID where groupID = (%s)",group_id)
        if len(group_db) is 0:
            result_dict['description'] = "groupID : "+group_id+" do not exist"
            result_dict['result'] = str('False')
            activate = True
            return jsonify(result_dict)
    
    # make directory for save feature map
    object_db,db_affect = db.query("select * from objectID where groupID = (%s) and objectID = (%s)",(group_id,object_id))
    if len(object_db) is 0:
        db.query("insert into objectID (groupID,objectID) values(%s,%s)",(group_db[0]['groupID'],object_id))
        object_db,db_affect = db.query("select * from objectID where groupID = (%s) and objectID = (%s)",(group_id,object_id))
        make_dir(database_foldername + group_db[0]['groupID'] + '/' + object_db[0]['objectID'])
        imgID = set_imgID(database_foldername + group_db[0]['groupID'] + '/' + object_db[0]['objectID'])
        data_dir = database_foldername + str(group_db[0]['groupID']) + '/' + str(object_db[0]['objectID']) + '/'
        db.query("insert into imgID (id,imgID,datadir) values(%s,%s,%s)",(object_db[0]['id'],imgID,data_dir))
    else:
        make_dir(database_foldername + group_db[0]['groupID'] + '/' + object_db[0]['objectID'])
        imgID = set_imgID(database_foldername + group_db[0]['groupID'] + '/' + object_db[0]['objectID'])
        data_dir = database_foldername + str(group_db[0]['groupID']) + '/' + str(object_db[0]['objectID']) + '/'
        db.query("insert into imgID (id,imgID,datadir) values(%s,%s,%s)",(object_db[0]['id'],imgID,data_dir))

    # extract feature
    img = align_face(img)
    embedding = f_Recognizer.extract_feature(img)
    embedding = embedding.detach().cpu().numpy()
    
    # save feature
    np.save(data_dir+imgID+".npy", embedding)
    
    result_dict['result'] = str('True')
    activate = True
    return jsonify(result_dict)

## 4. 서버 DB조회
## output : result(num,list)
#TODO: 조건식에 대한부분 추가 필요
@app.route('/api/check-id', methods=['GET'])
def check_id_list():
    result_dict = {}

    groupID = request.args.get('groupID').lower()

    if os.path.isdir(database_foldername+groupID):
        group_result = os.listdir(database_foldername+groupID)
    else:
        group_result = []
    
    result_dict['serched'] = str(len(group_result))
    result_dict['result'] = str('True')
    result_dict['objectID_list'] = group_result
    activate = True
    return jsonify(result_dict)

## 5. 등록된 인원 삭제
## input  : objectID, imgID, groupID
## output : result, description
@app.route('/api/delete', methods=['POST'])
def delete_data():
    result_dict = {}
    res = json.loads(request.data)
    mode = res['mode']
    group_id = res['groupID'].lower()
    object_id = res['objectID'].lower()
    img_id = res['imgID'].lower()
    if mode == "group":
        remove_dir(database_foldername + group_id)
    elif mode == "object":
        remove_dir(database_foldername + group_id + '/' + object_id)
    elif mode == "img":
        remove_file(database_foldername + group_id + '/' + object_id + '/' + img_id + '.png')
        remove_file(database_foldername + group_id + '/' + object_id + '/' + img_id + '.npy')

    result_dict['result'] = str('True')
    
    return jsonify(result_dict)

## 8. 등록된 그룹 ID값 변경
## input  : berforeID, afterID
## output : result , description
@app.route('/api/update-groupid', methods=['POST'])
def change_groupid():
    result_dict = {}
    global activate
    activate = False
    try:
        res = json.loads(request.data)
    except:
        result_dict['result'] = str('False')
        result_dict['description'] = "json decoding error"
        activate = True
        return jsonify(result_dict)


    if 'beforeID' not in res:
        result_dict['result'] = str('False')
        result_dict['description'] = "please input the parameter : beforeID"
        activate = True
        return jsonify(result_dict)
    elif res['beforeID'] is "":
        result_dict['result'] = str('False')
        result_dict['description'] = "please input the parameter : beforeID"
        activate = True
        return jsonify(result_dict)
    else:
        before_id = res['beforeID']

    if 'afterID' not in res:
        result_dict['result'] = str('False')
        result_dict['description'] = "please input the parameter : afterID"
        activate = True
        return jsonify(result_dict)
    elif res['afterID'] is "":
        result_dict['result'] = str('False')
        result_dict['description'] = "please input the parameter : afterID"
        activate = True
        return jsonify(result_dict)
    else:
        after_id = res['afterID']


    db_result,db_affect = db.query("select * from groupID where groupID = (%s)",before_id)
    if len(db_result) is 0:
        result_dict['result'] = str('False')
        result_dict['description'] = "groupID : " + before_id + " do not exist"
        activate = True
        return jsonify(result_dict)
    
    db_result,db_affect = db.query("select * from groupID where groupID = (%s)",after_id)
    if len(db_result) is not 0:
        result_dict['result'] = str('False')
        result_dict['description'] = "groupID : " + after_id + " is alreay exist"
        activate = True
        return jsonify(result_dict)
    
    db_result,db_affect = db.query("select * from objectID where groupID = (%s)",before_id)
    for i,v in enumerate(db_result):
        db.query("update imgID set datadir = (%s) where id = (%s)",(database_foldername+after_id+'/'+v['objectID']+'/',v['id']))
    
    group_db,temp = db.query("select * from groupID where groupID = (%s)",before_id)
    db_result,db_affect = db.query("update groupID set groupID = (%s) where groupID = (%s)",(after_id,before_id))
    
    
    
    changeName(database_foldername+group_db[0]['groupID'],database_foldername+after_id) 
    
    result_dict['result'] = str('True')
    activate = True
    return jsonify(result_dict)


## 9. 등록된 인원 ID값 변경
## input  : objectID, groupID, changeID
## output : result
@app.route('/api/update-objectid', methods=['POST'])
def change_id():
    result_dict = {}
    global activate
    activate = False
    try:
        res = json.loads(request.data) 
    except:
        result_dict['result'] = str('False')
        result_dict['description'] = "json decoding error"
        activate = True
        return jsonify(result_dict)


    if 'groupID' not in res:
        result_dict['result'] = str('False')
        result_dict['description'] = "please input the parameter : groupID"
        activate = True
        return jsonify(result_dict)
    elif res['groupID'] is "":
        result_dict['result'] = str('False')
        result_dict['description'] = "please input the parameter : groupID"
        activate = True 
        return jsonify(result_dict)
    else:
        group_id = res['groupID']
    
    if 'beforeID' not in res:
        result_dict['result'] = str('False')
        result_dict['description'] = "please input the parameter : beforeID"
        activate = True
        return jsonify(result_dict)
    elif res['beforeID'] is "":
        result_dict['result'] = str('False')
        result_dict['description'] = "please input the parameter : beforeID"
        activate = True
        return jsonify(result_dict)
    else:
        before_id = res['beforeID']

    if 'afterID' not in res:
        result_dict['result'] = str('False')
        result_dict['description'] = "please input the parameter : afterID"
        activate = True
        return jsonify(result_dict)
    elif res['afterID'] is "":
        result_dict['result'] = str('False')
        result_dict['description'] = "please input the parameter : afterID"
        activate = True
        return jsonify(result_dict)
    else:
        after_id = res['afterID']


    db_result,db_affect = db.query("select * from groupID where groupID = (%s)",group_id)
    if len(db_result) is 0:
        result_dict['result'] = str('False')
        result_dict['description'] = "groupID : " + group_id + " do not exist"
        activate = True
        return jsonify(result_dict)

    
    db_result,db_affect = db.query("select * from objectID where groupID = (%s) and objectID = (%s)",(group_id,before_id))
    if len(db_result) is 0:
        result_dict['result'] = str('False')
        result_dict['description'] = "objectID : " + before_id + " do not exist"
        activate = True
        return jsonify(result_dict)


    db_result,db_affect = db.query("select * from objectID where groupID = (%s) and objectID = (%s)",(group_id,after_id))
    if len(db_result) is not 0:
        result_dict['result'] = str('False')
        result_dict['description'] = "objectID : " + after_id + " is already exist"
        activate = True
        return jsonify(result_dict)

    
    db_result,db_affect = db.query("select * from objectID where groupID = (%s) and objectID = (%s)",(group_id,before_id))
    for i,v in enumerate(db_result):
        db.query("update imgID set datadir = (%s) where id = (%s)",(database_foldername+db_result[0]['groupID']+'/'+after_id+'/',v['id']))
    temp,temp2 = db.query("update objectID set objectID = (%s) where groupID = (%s) and objectID = (%s)",(after_id,group_id,before_id))
    
    changeName(database_foldername+db_result[0]['groupID']+'/'+db_result[0]['objectID'],database_foldername+db_result[0]['groupID']+'/'+after_id) 
    
    result_dict['result'] = str('True')
    activate = True

    return jsonify(result_dict)

if __name__ == '__main__':
    app.run(host='localhost', port=3000)

