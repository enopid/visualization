import numpy as np
import os, textwrap, json
import matplotlib.pyplot as plt

#시각화 폴더
folder_path="D:/다른 컴퓨터/내 노트북/2024_2학기/sg/visualization/visualization"
data_path=os.path.join(folder_path,"data")
figure_path=os.path.join(folder_path,"figures")

#인덱스에 해당하는 관계나 클래스를 가져오게끔 리스트 반환
def read_relationships(read_file):
    relationships = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            relationship = line.rstrip().lower() 
            relationships.append("\n".join(textwrap.wrap(relationship, 15, break_long_words=False, break_on_hyphens=False))) 
    return relationships 

#정수의 씬 번호를 id로 전환하는 리스트 반환
def get_scenenum_to_id(read_file_path):
    scenenum_to_id = [] 
    with open(read_file_path, "r") as read_file:
        data = json.load(read_file)
    for i in data["scans"]:
        scenenum_to_id.append(i["scan"])
    return scenenum_to_id

pth_relationship = os.path.join(folder_path, 'relationships.txt')
relationNames = read_relationships(pth_relationship)

pth_class = os.path.join(folder_path, 'classes.txt')
classNames = read_relationships(pth_class)

scenenum_to_id=get_scenenum_to_id(os.path.join(folder_path, 'relationships_validation.json'))

def count_obj_acc(topk):
    wrong_count=[0 for _ in range(160)]
    total_count=[0 for _ in range(160)]
    for i in range(548):
        save_gt_path=os.path.join(data_path,str(i))

        gt_class = np.load(os.path.join(save_gt_path,"gt_class.npy"))
        classses = np.load(os.path.join(save_gt_path,"class.npy"))
        edge_indices = np.load(os.path.join(save_gt_path,"edge_indices.npy"))
        rel_cls = np.load(os.path.join(save_gt_path,"rel_cls.npy"))
        gt_rel_cls = np.load(os.path.join(save_gt_path,"gt_rel_cls.npy"))
        
        for i, gt in enumerate(gt_class):
            if gt not in classses[i][:topk]:
                wrong_count[classses[i][0]]+=1
            total_count[gt]+=1
    
    indices=[i for i in range(160)]
    indices=sorted(indices, key=lambda i : -wrong_count[i]/total_count[i])
    for i in indices:
        print(f"{classNames[i]} : {wrong_count[i]/total_count[i]} / {total_count[i]}")
    
def count_wrong_predicate(topk):
    wrong_count=[[0,0,0] for _ in range(26)]
    total_count=[[0,0,0] for _ in range(26)]
    edge_topk=1
    for i in range(548):
        save_gt_path=os.path.join(data_path,str(i))

        gt_class = np.load(os.path.join(save_gt_path,"gt_class.npy"))
        classses = np.load(os.path.join(save_gt_path,"class.npy"))
        edge_indices = np.load(os.path.join(save_gt_path,"edge_indices.npy"))
        rel_cls = np.load(os.path.join(save_gt_path,"rel_cls.npy"))
        gt_rel_cls = np.load(os.path.join(save_gt_path,"gt_rel_cls.npy"))
        
        for i, edge in enumerate(edge_indices):    
            for v in rel_cls[i][:edge_topk]:
                if gt_rel_cls[i][v]!=1 and v!=-1:
                    if gt_class[edge[0]]in classses[edge[0]][:topk] and gt_class[edge[1]]in classses[edge[1]][:topk]:
                        wrong_count[v][0]+=1
                    elif gt_class[edge[0]] not in classses[edge[0]][:topk] and gt_class[edge[1]] not in classses[edge[1]][:topk]:
                        wrong_count[v][1]+=1
                    else:
                        wrong_count[v][2]+=1
                    
                if gt_class[edge[0]]in classses[edge[0]][:topk] and gt_class[edge[1]]in classses[edge[1]][:topk]:
                    total_count[v][0]+=1
                elif gt_class[edge[0]] not in classses[edge[0]][:topk] and gt_class[edge[1]] not in classses[edge[1]][:topk]:
                    total_count[v][1]+=1
                else:
                    total_count[v][2]+=1
                    
    for i,v in enumerate(wrong_count):
        print(f"{relationNames[i+1]} : {v[0]} / {v[2]} / {v[2]}")
    print()
    for i,v in enumerate(wrong_count):
        print(f"{relationNames[i+1]} : {v[0]/total_count[i][0] if total_count[i][0]!=0 else None} / {v[2]/total_count[i][2] if total_count[i][2]!=0 else None} / {v[1]/total_count[i][1] if total_count[i][1]!=0 else None}")
    print()
    for i,v in enumerate(wrong_count):
        print(f"{relationNames[i+1]} : {v[0]/sum(v) if sum(v)!=0 else None} / {v[2]/sum(v) if sum(v)!=0 else None} / {v[1]/sum(v) if sum(v)!=0 else None}")

def count_forget_predicate():
    forget_count=[0 for _ in range(26)]
    total_count=[0 for _ in range(26)]
    edge_topk=1
    for i in range(548):
        save_gt_path=os.path.join(data_path,str(i))

        gt_class = np.load(os.path.join(save_gt_path,"gt_class.npy"))
        classses = np.load(os.path.join(save_gt_path,"class.npy"))
        edge_indices = np.load(os.path.join(save_gt_path,"edge_indices.npy"))
        rel_cls = np.load(os.path.join(save_gt_path,"rel_cls.npy"))
        gt_rel_cls = np.load(os.path.join(save_gt_path,"gt_rel_cls.npy"))
        
        for i, edge in enumerate(edge_indices): 
            for v in range(26): 
                if gt_rel_cls[i][v]==1:
                    if rel_cls[i][0]==-1:  
                        forget_count[v]+=1
                    total_count[v]+=1
                    
    indices=[i for i in range(26)]
    indices=sorted(indices, key=lambda i : -forget_count[i])
    for i in indices:
        print(f"{relationNames[i+1]} : {forget_count[i]}")
    for i in indices:
        print(f"{relationNames[i+1]} : {forget_count[i]/sum(forget_count)}")
    print()
    indices=[i for i in range(26)]
    indices=sorted(indices, key=lambda i : -forget_count[i]/total_count[i] if total_count[i]!=0 else 0)
    for i in indices:
        print(f"{relationNames[i+1]} : {forget_count[i]/total_count[i] if total_count[i]!=0 else 0}")
    print()

def count_miss_predicate():
    miss_count=[0 for _ in range(26)]
    total_count=[0 for _ in range(26)]
    edge_topk=1
    for i in range(548):
        save_gt_path=os.path.join(data_path,str(i))

        gt_class = np.load(os.path.join(save_gt_path,"gt_class.npy"))
        classses = np.load(os.path.join(save_gt_path,"class.npy"))
        edge_indices = np.load(os.path.join(save_gt_path,"edge_indices.npy"))
        rel_cls = np.load(os.path.join(save_gt_path,"rel_cls.npy"))
        gt_rel_cls = np.load(os.path.join(save_gt_path,"gt_rel_cls.npy"))
        
        for i, edge in enumerate(edge_indices):  
            if 1 not in gt_rel_cls[i] and rel_cls[i][0]!=-1:  
                miss_count[rel_cls[i][0]]+=1
            for v in range(26): 
                if gt_rel_cls[i][v]==1:
                    total_count[v]+=1
                    
    indices=[i for i in range(26)]
    indices=sorted(indices, key=lambda i : -miss_count[i])
    for i in indices:
        print(f"{relationNames[i+1]} : {miss_count[i]}")
    for i in indices:
        print(f"{relationNames[i+1]} : {miss_count[i]/sum(miss_count)}")
    print()
    indices=[i for i in range(26)]
    indices=sorted(indices, key=lambda i : -miss_count[i]/total_count[i] if total_count[i]!=0 else 0)
    for i in indices:
        print(f"{relationNames[i+1]} : {miss_count[i]/total_count[i] if total_count[i]!=0 else 0}")
    
    print()

def Predicate_Object_Correlation(topk=1):
    edge_topk=1
    wrong_count=[0,0,0]
    total_count=[0,0,0]
    for i in range(548):
        save_gt_path=os.path.join(data_path,str(i))

        gt_class = np.load(os.path.join(save_gt_path,"gt_class.npy"))
        classses = np.load(os.path.join(save_gt_path,"class.npy"))
        edge_indices = np.load(os.path.join(save_gt_path,"edge_indices.npy"))
        rel_cls = np.load(os.path.join(save_gt_path,"rel_cls.npy"))
        gt_rel_cls = np.load(os.path.join(save_gt_path,"gt_rel_cls.npy"))
        
        for i, edge in enumerate(edge_indices):
            true_flag=False
            for v in rel_cls[i]:
                if gt_rel_cls[i][v]==1 and v!=-1:
                    true_flag=True
            if 1 not in gt_rel_cls[i] and rel_cls[i][0]==-1:
                true_flag=True
                
            if not true_flag:
                if gt_class[edge[0]]in classses[edge[0]][:topk] and gt_class[edge[1]]in classses[edge[1]][:topk]:
                    wrong_count[0]+=1
                elif gt_class[edge[0]] not in classses[edge[0]][:topk] and gt_class[edge[1]] not in classses[edge[1]][:topk]:
                    wrong_count[1]+=1
                else:
                    wrong_count[2]+=1
                    
            if gt_class[edge[0]]in classses[edge[0]][:topk] and gt_class[edge[1]]in classses[edge[1]][:topk]:
                total_count[0]+=1
            elif gt_class[edge[0]] not in classses[edge[0]][:topk] and gt_class[edge[1]] not in classses[edge[1]][:topk]:
                total_count[1]+=1
            else:
                total_count[2]+=1
    print(f"topk : {topk}")
    print(f"2 correct node : {wrong_count[0]/total_count[0]:0.2f} / 0 correct node : {wrong_count[2]/total_count[2]:0.2f} / 0 correct node : {wrong_count[1]/total_count[1]:0.2f}")

count_forget_predicate()
#count_miss_predicate()
#count_obj_acc(5)
#count_wrong_predicate(10)
#Predicate_Object_Correlation(1)
#Predicate_Object_Correlation(5)
#Predicate_Object_Correlation(10)
    