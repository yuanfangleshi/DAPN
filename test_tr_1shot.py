import numpy as np
from io_utils import parse_args_test
import test_dataset
import ResNet10
import torch
from sklearn.linear_model import LogisticRegression
import random
import warnings
warnings.filterwarnings("ignore", category=Warning)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def Tr(support_set, query_set, pred, K):
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(pred)
        # print('pred',pred)
        score, index = pred.max(1)
        # print('score',score)
        # print('index',index)
        all_class_score = []
        all_class_index = []
        # 遍历所有查询样本（假设有75个），如果样本被预测为当前类别（即索引匹配），则收集该样本的预测得分和索引。
        for j in range(5):
            current_class_score = []
            current_class_index = []
            for i in range(75):
                if index[i]==j:
                    current_class_score.append(score[i])
                    current_class_index.append(i)
            all_class_score.append(current_class_score)
            all_class_index.append(current_class_index)
        # print('score',all_class_score)
        # print('index',all_class_index)

        # 该函数的作用是基于模型对查询集的预测，对每个类别的原型进行更新。它首先识别出每个类别在查询集中被预测到的样本，然后根据这些样本的数量（不超过
        # K）与支持集中的对应类别特征相结合，通过取均值得到新的类别原型。这种机制允许模型在少样本学习场景中动态调整类别表示，以更好地适应查询数据。
        prototype = []
        for i in range(5):
            current_class_index = all_class_index[i]
            if len(current_class_index) == 0:
                current_prototype = torch.mean(support_set[i],0) # (640)
            elif len(current_class_index) <= K:
                current_query_image = query_set[current_class_index] 
                current_prototype = torch.cat((support_set[i], current_query_image),0) 
                current_prototype = torch.mean(current_prototype, 0) # (640)
            else:
                current_class_score = all_class_score[i]
                current_class_score_index = np.argsort(current_class_score)
                current_class_index = np.array(current_class_index)[current_class_score_index[-K:].tolist()] 
                current_query_image = query_set[current_class_index] 
                current_prototype = torch.cat((support_set[i], current_query_image),0)
                current_prototype = torch.mean(current_prototype, 0) # (640)
            prototype.append(current_prototype)
        prototype = torch.stack(prototype, 0) #(5,640)
          
        return prototype
    
    
def test(novel_loader, model, params):
    iter_num = len(novel_loader) 
    acc_all_LR = []
    with torch.no_grad():
        for i, (x,_) in enumerate(novel_loader):
            x_query = x[:, params.n_support:,:,:,:].contiguous().view(params.n_way*params.n_query, *x.size()[2:]).cuda() 
            x_support = x[:,:params.n_support,:,:,:].contiguous().view(params.n_way*params.n_support, *x.size()[2:]).cuda() # (25, 3, 224, 224)
            out_support = model(x_support) # (way*shot,512)
            out_query = model(x_query) # (way*query,512)
            
            beta = 0.5
            out_support = torch.pow(out_support, beta) 
            out_query = torch.pow(out_query, beta) 
          
            _, c = out_support.size()
            
            out_support_LR = out_support.view(params.n_way, params.n_support, c) #(way,shot,512) 
            prototype_LR = torch.mean(out_support_LR, 1) # (way,512)
            prototype_LR = prototype_LR.cpu().numpy()


            out_query_LR = out_query.cpu().numpy()
            
            y = np.tile(range(params.n_way), 1)
            y.sort()
            classifier = LogisticRegression(max_iter=1000).fit(X=prototype_LR, y=y)
            pred = classifier.predict_proba(out_query_LR)
            # print('pred1',pred)

            for k in range(7):
                pred = torch.from_numpy(pred).cuda()
                current_prototype = Tr(out_support_LR, out_query, pred, 10)
                current_prototype = current_prototype.cpu().numpy()
                classifier = LogisticRegression(max_iter=1000).fit(X=current_prototype, y=y)
                pred = classifier.predict_proba(out_query_LR)
            
            y_query = np.repeat(range(params.n_way), params.n_query)
            pred = torch.from_numpy(pred).cuda()
            topk_scores, topk_labels = pred.data.topk(1, 1, True, True) # 获取最高预测得分和对应标签
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query)
            correct_this, count_this = float(top1_correct), len(y_query)
            acc_all_LR.append((correct_this/ count_this *100))

    
    acc_all  = np.asarray(acc_all_LR)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all) 
    print('acc : %4.2f%% +- %4.2f%%' %(acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    
    
    
if __name__=='__main__':
    
    params = parse_args_test()
    setup_seed(params.seed)
    
    datamgr = test_dataset.Eposide_DataManager(data_path=params.current_data_path, num_class=params.current_class, image_size=params.image_size, n_way=params.n_way, n_support=params.n_support, n_query=params.n_query, n_eposide=params.test_n_eposide)
    novel_loader = datamgr.get_data_loader(aug=False) 
    
    
    model = ResNet10.ResNet(list_of_out_dims=params.list_of_out_dims, list_of_stride=params.list_of_stride, list_of_dilated_rate=params.list_of_dilated_rate)

    tmp = torch.load(params.model_path)
    state_model = tmp['state_model']
    model.load_state_dict(state_model)
    model.cuda()
    model.eval()
    test(novel_loader, model, params)
    
    



