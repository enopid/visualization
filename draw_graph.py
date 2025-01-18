import networkx as nx
import numpy as np
import os, textwrap, json
import matplotlib.pyplot as plt

#시각화 폴더
folder_path="D:/다른 컴퓨터/내 노트북/2024_2학기/sg/visualization/visualization"
data_path=os.path.join(folder_path,"data")
figure_path=os.path.join(folder_path,"figure")

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

def draw_graph(G, axes, axes_index, node_colors=None, edge_colors=None, pos=None):
    #get_label
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    #set_node_pos
    if pos==None:
        pos = nx.spring_layout(G,k=10, seed=0, iterations=100)  # spring_layout은 노드 위치를 자동 배치
        #pos = nx.kamada_kawai_layout(G,scale=20)  # kamada_kawai_layout은 엣지 겹침 방지
    
    #draw_node
    nx.draw_networkx_nodes(G, pos, ax=axes[axes_index], node_size=1000, node_color=node_colors if node_colors is not None else 'lightblue')
    
    #draw_edge
    nx.draw_networkx_edges(G, pos, ax=axes[axes_index], edgelist=[(u, v) for u, v in G.edges() if (u>v) and (v,u) in  G.edges()],
                           edge_color=[edge_colors[(u, v)] for u, v in G.edges() if (u>v) and (v,u) in  G.edges()] if edge_colors is not None else 'Black',
                           connectionstyle="arc3,rad=0.1",arrows=True,arrowstyle="->", arrowsize=20,min_source_margin=15,min_target_margin=15)
    
    nx.draw_networkx_edges(G, pos, ax=axes[axes_index], edgelist=[(u, v) for u, v in G.edges() if (u<v) and (v,u) in  G.edges()],
                           edge_color=[edge_colors[(u, v)] for u, v in G.edges() if (u<v) and (v,u) in  G.edges()] if edge_colors is not None else 'Black',
                           arrows=True,arrowstyle="->", connectionstyle="arc3,rad=0.1",arrowsize=20,min_source_margin=15,min_target_margin=15)
    
    nx.draw_networkx_edges(G, pos, ax=axes[axes_index], edgelist=[(u, v) for u, v in G.edges() if (v,u) not in  G.edges()],
                           edge_color=[edge_colors[(u, v)] for u, v in G.edges() if (v,u) not in  G.edges()] if edge_colors is not None else 'Black',
                           connectionstyle="arc3",arrows=True,arrowstyle="->", arrowsize=20,min_source_margin=15,min_target_margin=15)
    
    #draw_node_label
    nx.draw_networkx_labels(G, pos, ax=axes[axes_index], labels=node_labels, font_size=8, font_color='black')
    
    #draw_edge_label
    edge_labels1 = {k: v for k, v in edge_labels.items() if (k[0] > k[1]) and  (k[1],k[0]) in edge_labels}
    edge_labels2 = {k: v for k, v in edge_labels.items() if (k[0] < k[1]) and  (k[1],k[0]) in edge_labels}
    edge_labels3 = {k: v for k, v in edge_labels.items() if (k[1],k[0]) not in edge_labels}
    
    nx.draw_networkx_edge_labels(G, pos, ax=axes[axes_index], edge_labels=edge_labels1, font_size=6, verticalalignment='top', horizontalalignment='center')
    nx.draw_networkx_edge_labels(G, pos, ax=axes[axes_index], edge_labels=edge_labels2, font_size=6, verticalalignment='bottom', horizontalalignment='center')
    nx.draw_networkx_edge_labels(G, pos, ax=axes[axes_index], edge_labels=edge_labels3, font_size=6, verticalalignment='center', horizontalalignment='center')
    
    return pos

def draw_GT(scene_number, axes, axes_index):
    save_gt_path=os.path.join(data_path,str(scene_number))
    
    # 유향 그래프 생성
    G = nx.DiGraph()

    # 노드 추가 (노드 라벨 포함)
    gt_class = np.load(os.path.join(save_gt_path,"gt_class.npy"))
    for i, cls in enumerate(gt_class):
        G.add_node(i, label=classNames[cls])

    # 엣지 추가 (엣지 라벨 포함)
    edge_indices = np.load(os.path.join(save_gt_path,"edge_indices.npy"))
    gt_rel_cls = np.load(os.path.join(save_gt_path,"gt_rel_cls.npy"))
    for i, edge in enumerate(edge_indices):
        rel_name=""
        for j, v in enumerate(gt_rel_cls[i]):
            if v==1:
                if not rel_name:
                    rel_name=relationNames[j+1]
                else:
                    rel_name+=" / "+relationNames[j+1]
        if rel_name:
            G.add_edge(edge[0], edge[1], label=rel_name)
    
    return draw_graph(G, axes, axes_index)

def draw_Prediction(scene_number, topk, axes, axes_index, pos):
    save_gt_path=os.path.join(data_path,str(scene_number))
    
    # 유향 그래프 생성
    G = nx.DiGraph()

    # 노드 추가 (노드 라벨 포함)
    include_node_flag=False
    red_nodes = []  # 빨간색으로 표시할 노드
    gt_class = np.load(os.path.join(save_gt_path,"gt_class.npy"))
    classses = np.load(os.path.join(save_gt_path,"class.npy"))
    for i, cls in enumerate(classses):
        include_node_flag=False
        for j in range(topk[0]):
            if cls[j]==gt_class[i]:
                G.add_node(i, label=classNames[cls[j]])
                include_node_flag=True
                break
        if not include_node_flag:
            G.add_node(i, label=classNames[cls[0]])
            red_nodes.append(i)

    # 엣지 추가 (엣지 라벨 포함)
    wrong_edges = []  # 빨간색으로 표시할 노드
    missed_edges = []
    include_predicate_flag=False
    edge_indices = np.load(os.path.join(save_gt_path,"edge_indices.npy"))
    rel_cls = np.load(os.path.join(save_gt_path,"rel_cls.npy"))
    gt_rel_cls = np.load(os.path.join(save_gt_path,"gt_rel_cls.npy"))
    print(rel_cls.shape, gt_rel_cls.shape)
    for i, edge in enumerate(edge_indices):
        include_predicate_flag=False
        rel_name=""
        for j in range(topk[1]):
            v=rel_cls[i][j]
            if v!=-1:
                if not rel_name:
                    rel_name=relationNames[v+1]
                else:
                    rel_name+=" / "+relationNames[v+1]
                    
                if gt_rel_cls[i][v]==1:
                    include_predicate_flag=True
            else:
                break
                if 1 in gt_rel_cls[i]:
                    for j, v in enumerate(gt_rel_cls[i]):
                        if v==1:
                            G.add_edge(edge[0], edge[1], label=relationNames[j+1])
                        missed_edges.append((edge[0], edge[1]))
        if not include_predicate_flag:
            wrong_edges.append((edge[0], edge[1]))
        if rel_name:
            G.add_edge(edge[0], edge[1], label=rel_name)
    
    node_colors = ["red" if node in red_nodes else "lightblue" for node in G.nodes]
    edge_colors = {}
    for edge in G.edges:
        if edge in wrong_edges:
            edge_colors[edge]="red"
        elif edge in missed_edges:
            edge_colors[edge]="blue"
        else:
            edge_colors[edge]="black"
            
    draw_graph(G, axes, axes_index, node_colors, edge_colors, pos)

#GT와 pred 동시에 시각화
def draw_GT_pred(scenenumber,logs,infos=[0,0,0],save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    pos=draw_GT(scenenumber, axes, 0)
    obj_topk=[1,5,10]
    pred_topk=[1,3,5]
    topk=[obj_topk[infos[0]],pred_topk[infos[1]]]
    draw_Prediction(scenenumber, topk, axes, 1,pos)
    
    log_text = ["Acc@1/obj_cls_acc",
        "Acc@1/obj_cls_2d_acc",
        "Acc@5/obj_cls_acc",
        "Acc@5/obj_cls_2d_acc",
        "Acc@10/obj_cls_acc",
        "Acc@10/obj_cls_2d_acc", 
        "Acc@1/rel_cls_acc",
        "Acc@1/rel_cls_2d_acc",
        "Acc@3/rel_cls_acc",
        "Acc@3/rel_cls_2d_acc",
        "Acc@5/rel_cls_acc",
        "Acc@5/rel_cls_2d_acc",
        "Acc@50/triplet_acc",
        "Acc@50/triplet_2d_acc", 
        "Acc@100/triplet_acc",
        "Acc@100/triplet_2d_acc"
        ]
    infos=[2*infos[0],2*infos[1]+6,2*infos[2]+12]
    text=[f"{log_text[i]} : {logs[scenenumber][i]:0.6f}"  for i in infos]
    text=f"ID : {scenenum_to_id[scenenumber]}"+'\n'+" || ".join(text)
    
    fig.suptitle(f"graph_{scenenumber}")
    fig.subplots_adjust(bottom=0.5)
    fig.text(0.5, 0.1,text,  ha="center", va="top")
    plt.tight_layout(rect=[0, 0.1, 1, 1])

#github내 figure내 파일들을 생성하는 코드
def main():
    logs = np.load(os.path.join(folder_path, 'logs.npy'))
    names=["ACC@5_obj_cls_lowest_20scene",
           "ACC@10_obj_cls_lowest_20scene",
           "ACC@1_rel_cls_lowest_20scene",
           "ACC@3_rel_cls_lowest_20scene",
           "ACC@5_rel_cls_lowest_20scene",
           "ACC@50_tri_cls_lowest_20scene",
           "ACC@100_tri_cls_lowest_20scene",
           ]
    
    #각 인덱스는 첫인덱스부터 obj, predicate, triplet의 topk를 결정
    #obj       || 0:ACC@1,  1:ACC@5,   2:ACC@10
    #predicate || 0:ACC@1,  1:ACC@3,   2:ACC@5
    #triplet   || 0:ACC@50, 1:ACC@100
    infolist=[[1,0,0],
              [2,0,0],
              [0,0,0],
              [0,1,0],
              [0,2,0],
              [0,0,0],
              [0,0,1],
              ]
    
    for i in range(7):
        sorted_obj_indices = np.argsort(logs[:, 2*infolist[i][0]])
        sorted_pred_indices = np.argsort(logs[:, 2*infolist[i][1]+6])
        sorted_triplet_indices = np.argsort(logs[:, 2*infolist[i][2]+12])
        
        if i<=2:
            temp=sorted_obj_indices
        elif i<=5:
            temp=sorted_pred_indices
        else:
            temp=sorted_triplet_indices
        
        issave=False
        for ind,j in enumerate(temp[:20]):
            draw_GT_pred(j, logs, infolist[i])
            if issave:
                plt.savefig(os.path.join(figure_path,names[i],f"{ind}.png"))
            else:
                plt.show()
    
if __name__ == '__main__':
    main()