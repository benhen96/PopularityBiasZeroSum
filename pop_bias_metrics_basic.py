import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def pred_item_rank(model_here, test_data, sid_pop_total):
    data2 = test_data.copy()
    
    data2['uid'] = data2['uid'].astype(int)
    data2['sid'] = data2['sid'].astype(int)    
    
    # Filter out users with fewer than 2 interactions
    filter_users = data2['uid'].value_counts()[data2['uid'].value_counts() > 1].index
    data2 = data2[data2['uid'].isin(filter_users)]
    data2 = data2.reset_index(drop=True)[['uid', 'sid']]
    
    # Model evaluation setup
    model_here.eval()
    
    # Prediction and rank calculation
    predictions_list = []
    batch_size = 50
    
    for start in range(0, len(data2), batch_size):
        end = start + batch_size
        batch_data = data2.iloc[start:end]
        
        user = torch.from_numpy(batch_data['uid'].values).to(torch.int32)
        item = torch.from_numpy(batch_data['sid'].values).to(torch.int32)
        
        # No CUDA here, using CPU
        with torch.no_grad():
            predictions, _ = model_here(user, item, item)
        
        predictions_list += predictions.detach().cpu().tolist()
    
    # Add predictions to data2 DataFrame
    data2['pred'] = predictions_list
    
    # Calculate user-item ranks
    data2['user_item_rank'] = data2.groupby('uid')['pred'].rank('average', ascending=False) - 1
    
    # Calculate user counts and adjust ranks
    user_count = data2.groupby('uid').size().reset_index(name='user_count')
    user_count['user_count'] = user_count['user_count'] - 1
    user_count_dict = dict(zip(user_count['uid'], user_count['user_count']))
    
    data2['user_count'] = data2['uid'].map(user_count_dict)
    data2['user_item_rank2'] = data2['user_item_rank'] / data2['user_count']
    
    # Aggregate item ranks and popularity counts
    item_rank = data2.groupby('sid')['user_item_rank2'].mean().reset_index()
    item_rank.columns = ['sid', 'rank']
    
    # Add popularity counts
    sid_pop_dict = dict(sid_pop_total.values)
    item_rank['sid_pop_count'] = item_rank['sid'].map(sid_pop_dict)
    
    return item_rank

def pred_item_score(model_here, test_data, sid_pop_total):
    data = test_data.copy()
    
    data['uid'] = data['uid'].astype(int)
    data['sid'] = data['sid'].astype(int)
    
    # Filter out users with fewer than 2 interactions
    filter_users = data['uid'].value_counts()[data['uid'].value_counts() > 1].index
    data = data[data['uid'].isin(filter_users)]
    data = data.reset_index()[['uid', 'sid']]
    
    predictions_list = []
    
    # Move model to CPU if necessary (not required if model_here is already on CPU)
    model_here.eval()
    
    # Iterate over data in batches
    batch_size = 50
    num_batches = len(data) // batch_size
    
    for itr in range(num_batches):
        batch_data = data.iloc[itr * batch_size:(itr + 1) * batch_size]
        user = torch.from_numpy(batch_data['uid'].values).to(torch.int32)
        item = torch.from_numpy(batch_data['sid'].values).to(torch.int32)
        
        with torch.no_grad():
            predictions, _ = model_here(user, item, item)
        
        predictions_list += predictions.detach().cpu().tolist()
    
    # Process remaining data (last batch)
    if len(data) % batch_size != 0:
        batch_data = data.iloc[num_batches * batch_size:]
        user = torch.from_numpy(batch_data['uid'].values).to(torch.int32)
        item = torch.from_numpy(batch_data['sid'].values).to(torch.int32)
        
        with torch.no_grad():
            predictions, _ = model_here(user, item, item)
        
        predictions_list += predictions.detach().cpu().tolist()
    
    # Move model back to CPU (if moved to CUDA within the loop)
    model_here.cpu()
    
    # Calculate item scores
    data['pred'] = predictions_list
    item_score = data.groupby('sid')['pred'].mean().reset_index()
    
    # Merge with item popularity if necessary
    sid_pop_dict = dict(sid_pop_total.values)
    item_score['sid_pop_count'] = item_score['sid'].map(sid_pop_dict)
    
    return item_score

def pred_item_stdscore(model_here, test_data, sid_pop_total):
    
    data = test_data
    
    data['uid'] = data['uid'].apply(lambda x : int(x))
    data['sid'] = data['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 1밖에 안되는 user들을 먼저 제거해야함.
    # value가 1이면 std deviation이 계산 안되기 때문
    filter_users = data.uid.value_counts()[data.uid.value_counts() > 1].index
    data = data[data.uid.isin(filter_users)]
    data = data.reset_index()[['uid', 'sid']]
    
    user_num = len(data.uid.unique())
    data_len = data.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data.iloc[frac_user_num*itr:frac_user_num*(itr+1)].values    
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()

        predictions, _ = model_here(user, item, item)
        predictions_list += predictions.detach().cpu().tolist()
        
        if itr+1 == frac:
            tmp = data.iloc[frac_user_num*(itr+1):].values    
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()

            predictions, _ = model_here(user, item, item)
            predictions_list += predictions.detach().cpu().tolist()
    
    model_here.cpu()
    data['pred'] = predictions_list
    user_mean_dict = dict(data.groupby('uid')['pred'].mean().reset_index().values)
    user_std_dict = dict(data.groupby('sid')['pred'].std().reset_index().values)
    
    data['mean'] = data['uid'].map(user_mean_dict)
    data['std'] = data['uid'].map(user_std_dict)
    
    data['z'] = (data['pred'] - data['mean']) / data['std']
    item_z_score = data[['sid','z']].groupby('sid').mean().reset_index()
    
    sid_pop_dict = dict(sid_pop_total.values)
    item_z_score['sid_pop_count'] = item_z_score['sid'].map(sid_pop_dict)
    
    return item_z_score

def pred_item_rankdist(model_here, test_data, sid_pop_total):
    data = test_data.copy()
    
    data['uid'] = data['uid'].astype(int)
    data['sid'] = data['sid'].astype(int)
    
    # Filter out users with fewer than 5 interactions
    filter_users = data['uid'].value_counts()[data['uid'].value_counts() > 4].index
    data = data[data['uid'].isin(filter_users)]
    data = data.reset_index()[['uid', 'sid']]
    
    predictions_list = []
    
    # Move model to CPU if necessary (not required if model_here is already on CPU)
    model_here.eval()
    
    # Iterate over data in batches
    batch_size = 50
    num_batches = len(data) // batch_size
    
    for itr in range(num_batches):
        batch_data = data.iloc[itr * batch_size:(itr + 1) * batch_size]
        user = torch.from_numpy(batch_data['uid'].values).to(torch.int32)
        item = torch.from_numpy(batch_data['sid'].values).to(torch.int32)
        
        with torch.no_grad():
            predictions, _ = model_here(user, item, item)
        
        predictions_list += predictions.detach().cpu().tolist()
    
    # Process remaining data (last batch)
    if len(data) % batch_size != 0:
        batch_data = data.iloc[num_batches * batch_size:]
        user = torch.from_numpy(batch_data['uid'].values).to(torch.int32)
        item = torch.from_numpy(batch_data['sid'].values).to(torch.int32)
        
        with torch.no_grad():
            predictions, _ = model_here(user, item, item)
        
        predictions_list += predictions.detach().cpu().tolist()
    
    # Move model back to CPU (if moved to CUDA within the loop)
    model_here.cpu()
    
    # Add predictions to data
    data['pred'] = predictions_list
    
    # Calculate item popularity ranks
    sid_pop_dict = dict(sid_pop_total.values)
    data['sid_pop_count'] = data['sid'].map(sid_pop_dict)
    
    data['user_item_pop_rank'] = data.groupby('uid')['sid_pop_count'].rank('average', ascending=False) - 1
    data['user_item_score_rank'] = data.groupby('uid')['pred'].rank('average', ascending=False) - 1
    
    # Calculate user interaction count
    user_count = data.groupby('uid').size().reset_index(name='user_count')
    user_count['user_count'] = user_count['user_count'] - 1
    user_count_dict = dict(zip(user_count['uid'], user_count['user_count']))
    
    # Add user interaction count and calculate normalized ranks
    data['user_count'] = data['uid'].map(user_count_dict)
    data['user_item_pop_rank2'] = data['user_item_pop_rank'] / data['user_count']
    data['user_item_score_rank2'] = data['user_item_score_rank'] / data['user_count']
    
    # Sort data and extract item rank distribution
    data = data.sort_values(['uid', 'user_item_score_rank2'], ascending=(True, True))
    item_rankdist = data.groupby('uid')['user_item_pop_rank2'].head(1)
    
    return item_rankdist

def pred_item_rankdist_modified(model_here, test_data, sid_pop_total):
    data = test_data
    
    data['uid'] = data['uid'].apply(lambda x : int(x))
    data['sid'] = data['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 4밖에 안되는 user들을 먼저 제거해야함.
    # 4 이하면 quantile 예측이 너무 극단적일 수 있기 때문
    # ex : rate 한 positive item이 하나밖에 없다
    # 이 item의 순위(quantile)은 무조건 0.0 으로 계산됨    
    filter_users = data.uid.value_counts()[data.uid.value_counts() > 4].index
    data = data[data.uid.isin(filter_users)]
    data = data.reset_index()[['uid', 'sid']]
    
    user_num = len(data.uid.unique())
    data_len = data.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data.iloc[frac_user_num*itr:frac_user_num*(itr+1)].values    
        #tmp = tmp.values
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()
        #label = tmp[:, 2]

        predictions, _ = model_here(user, item, item)
        predictions_list += predictions.detach().cpu().tolist()
        
        if itr+1 == frac:
            tmp = data.iloc[frac_user_num*(itr+1):].values    
            #tmp = tmp.values
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()
            #label = tmp[:, 2]

            predictions, _ = model_here(user, item, item)
            predictions_list += predictions.detach().cpu().tolist()
    
    model_here.cpu()
    data['pred'] = predictions_list
    
    sid_pop_dict = dict(sid_pop_total.values)
    data['sid_pop_count'] = data['sid'].map(sid_pop_dict)    
    
    user_item_pop_rank = data.groupby('uid')['sid_pop_count'].rank('average', ascending = False)
    user_item_pop_rank = user_item_pop_rank - 1
    data['user_item_pop_rank'] = user_item_pop_rank
    
    user_item_score_rank = data.groupby('uid')['pred'].rank('average', ascending = False)
    user_item_score_rank = user_item_score_rank - 1 # 코드 오류
    data['user_item_score_rank'] = user_item_score_rank
    
    user_count = data.groupby('uid')['pred'].count().reset_index()
    user_count.columns = ['uid', 'user_count']
    user_count['user_count'] = user_count['user_count'] - 1
    user_count_dict = dict(user_count.values)
    
    data['user_count'] = data['uid'].map(user_count_dict)
    data['user_item_pop_rank2'] = data['user_item_pop_rank'] / data['user_count']
    data['user_item_score_rank2'] = data['user_item_score_rank'] / data['user_count']
    
    data = data.sort_values(['uid', 'user_item_score_rank2'], ascending = (True, True))
    item_rankdist = data.groupby('uid')['user_item_pop_rank2'].head(1)    
    return item_rankdist



def pred_item_rankdist2(model_here, test_data, sid_pop_total):
    data = test_data
    
    data['uid'] = data['uid'].apply(lambda x : int(x))
    data['sid'] = data['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 4밖에 안되는 user들을 먼저 제거해야함.
    # 4 이하면 quantile 예측이 너무 극단적일 수 있기 때문
    # ex : rate 한 positive item이 하나밖에 없다
    # 이 item의 순위(quantile)은 무조건 0.0 으로 계산됨    
    filter_users = data.uid.value_counts()[data.uid.value_counts() > 4].index
    data = data[data.uid.isin(filter_users)]
    data = data.reset_index()[['uid', 'sid']]
    
    user_num = len(data.uid.unique())
    data_len = data.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data.iloc[frac_user_num*itr:frac_user_num*(itr+1)].values    
        #tmp = tmp.values
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()
        #label = tmp[:, 2]

        predictions, _ = model_here(user, item, item)
        predictions_list += predictions.detach().cpu().tolist()
        
        if itr+1 == frac:
            tmp = data.iloc[frac_user_num*(itr+1):].values    
            #tmp = tmp.values
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()
            #label = tmp[:, 2]

            predictions, _ = model_here(user, item, item)
            predictions_list += predictions.detach().cpu().tolist()
    
    model_here.cpu()
    data['pred'] = predictions_list
    
    sid_pop_dict = dict(sid_pop_total.values)
    data['sid_pop_count'] = data['sid'].map(sid_pop_dict)    
    
    user_item_pop_rank = data.groupby('uid')['sid_pop_count'].rank('average', ascending = False)
    user_item_pop_rank = user_item_pop_rank - 1
    data['user_item_pop_rank'] = user_item_pop_rank
    
    user_item_score_rank = data.groupby('uid')['pred'].rank('average', ascending = False)
    user_item_score_rank = user_item_score_rank -1
    data['user_item_score_rank'] = user_item_score_rank
    
    user_count = data.groupby('uid')['pred'].count().reset_index()
    user_count.columns = ['uid', 'user_count']
    user_count['user_count'] = user_count['user_count'] - 1
    user_count_dict = dict(user_count.values)
    
    data['user_count'] = data['uid'].map(user_count_dict)
    data['user_item_pop_rank2'] = data['user_item_pop_rank'] / data['user_count']
    data['user_item_score_rank2'] = data['user_item_score_rank'] / data['user_count']
    
    data = data.sort_values(['uid', 'user_item_score_rank2'], ascending = (True, True))
    
    res = data[['user_item_pop_rank2', 'user_item_score_rank2']]
    res.columns = ['pop_rank', 'score_rank']

    
    bins = np.linspace(0, 1, 20)
    
    res['bins'] = pd.cut(res['pop_rank'], bins=bins, include_lowest=True)    
    
    
    
    return res




def raw_pred_score(model_here, test_data):
    data2 = test_data
    
    data2['uid'] = data2['uid'].apply(lambda x : int(x))
    data2['sid'] = data2['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 1밖에 안되는 user들을 먼저 제거해야함.
    #filter_users = data2.uid.value_counts()[data2.uid.value_counts() > 1].index
    #data2 = data2[data2.uid.isin(filter_users)]
    if 'type' in data2.columns:
        data2 = data2.reset_index()[['uid', 'sid', 'type']]
    else:
        data2 = data2.reset_index()[['uid', 'sid']]
    
    user_num = len(data2.uid.unique())
    data_len = data2.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data2.iloc[ (frac_user_num*itr) : (frac_user_num*(itr+1)) ].values    
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()
        predictions, _ = model_here(user, item, item)
        predictions_list += predictions.detach().cpu().tolist()        
        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()

            predictions, _ = model_here(user, item, item)
            predictions_list += predictions.detach().cpu().tolist()
    
    model_here.cpu()    
    data2['pred'] = predictions_list

    return data2

def uPO(model_here, without_neg_data, sid_pop_total):
    data2 = without_neg_data
    
    # Filter out users with fewer than 4 interactions
    filter_users = data2['uid'].value_counts()[data2['uid'].value_counts() > 3].index
    data2 = data2[data2['uid'].isin(filter_users)]
    data2 = data2.reset_index(drop=True)[['uid', 'sid']]
    
    # Convert columns to integers
    data2['uid'] = data2['uid'].astype(int)
    data2['sid'] = data2['sid'].astype(int)
    
    # Model evaluation setup
    model_here.eval()
    
    # Prediction collection
    predictions_list = []
    batch_size = 50
    
    for start in range(0, len(data2), batch_size):
        end = start + batch_size
        batch_data = data2.iloc[start:end]
        
        user = torch.from_numpy(batch_data['uid'].values).to(torch.int32)
        item = torch.from_numpy(batch_data['sid'].values).to(torch.int32)
        
        # No CUDA here, using CPU
        with torch.no_grad():
            predictions, _ = model_here(user, item, item)
        
        predictions_list += predictions.detach().cpu().tolist()
    
    # Add predictions to data2 DataFrame
    data2['pred'] = predictions_list
    
    # Sort data by uid and sid
    data2 = data2.sort_values(['uid', 'sid'], ascending=[True, False])
    
    # Map sid_pop_total to sid
    sid_pop_dict = dict(sid_pop_total.values)
    data2['sid_pop_count'] = data2['sid'].map(sid_pop_dict)
    
    # Calculate Spearman correlation between sid_pop_count and pred for each user
    result = data2.groupby('uid')[['sid_pop_count', 'pred']].corr(method='spearman')
    result2 = result.unstack().iloc[:, 1].fillna(0).values.mean()
    
    return result2

# pearson correlation coefficient 계산

def pcc_train(model_here, train_data, sid_pop, item_num):
    data2 = train_data
    
    data2['uid'] = data2['uid'].apply(lambda x : int(x))
    data2['sid'] = data2['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 1밖에 안되는 user들을 먼저 제거해야함.
    filter_users = data2.uid.value_counts()[data2.uid.value_counts() > 1].index
    data2 = data2[data2.uid.isin(filter_users)]
    data2 = data2.reset_index()[['uid', 'sid']]
    
    user_num = len(data2.uid.unique())
    data_len = data2.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = torch.tensor([]).cuda()
    
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data2.iloc[ (frac_user_num*itr) : (frac_user_num*(itr+1)) ].values    
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()
        predictions, _ = model_here(user, item, item)
        predictions_list = torch.hstack((predictions_list, predictions))

        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()

            predictions, _ = model_here(user, item, item)
            predictions_list = torch.hstack((predictions_list, predictions))

    sid_pop_dict = dict(sid_pop.values)
    data2['sid_pop_count'] = data2['sid'].map(sid_pop_dict)            
        
    values = predictions_list.reshape(-1, 1)
    sid_pop_count = data2.sid_pop_count.values
    sid_pop_count = sid_pop_count.astype(np.int32)
    sid_pop_count = torch.from_numpy(sid_pop_count).float().cuda()
    
    
    X = values
    Y = sid_pop_count # item pop
        
    pcc = ((X - X.mean())*(Y - Y.mean())).sum() / ((X - X.mean())*(X- X.mean())).sum().sqrt() / ((Y - Y.mean())*(Y- Y.mean())).sum().sqrt()    
    
    return pcc

def pcc_test(model_here, test_data, sid_pop, item_num):
    data2 = test_data.copy()
    
    data2['uid'] = data2['uid'].astype(int)
    data2['sid'] = data2['sid'].astype(int)
    
    # Filter out users with fewer than 2 interactions
    filter_users = data2['uid'].value_counts()[data2['uid'].value_counts() > 1].index
    data2 = data2[data2['uid'].isin(filter_users)]
    data2 = data2.reset_index(drop=True)
    
    predictions_list = []
    
    # Move model to CPU if necessary (not required if model_here is already on CPU)
    model_here.eval()
    
    # Iterate over data in batches
    batch_size = 50
    num_batches = len(data2) // batch_size
    
    for itr in range(num_batches):
        batch_data = data2.iloc[itr * batch_size:(itr + 1) * batch_size]
        user = torch.from_numpy(batch_data['uid'].values).to(torch.int32)
        item = torch.from_numpy(batch_data['sid'].values).to(torch.int32)
        
        with torch.no_grad():
            predictions, _ = model_here(user, item, item)
        
        predictions_list += predictions.detach().cpu().tolist()
    
    # Process remaining data (last batch)
    if len(data2) % batch_size != 0:
        batch_data = data2.iloc[num_batches * batch_size:]
        user = torch.from_numpy(batch_data['uid'].values).to(torch.int32)
        item = torch.from_numpy(batch_data['sid'].values).to(torch.int32)
        
        with torch.no_grad():
            predictions, _ = model_here(user, item, item)
        
        predictions_list += predictions.detach().cpu().tolist()
    
    # Add predictions to data2 DataFrame
    data2['pred'] = predictions_list
    
    # Retrieve item popularity counts
    sid_pop_dict = dict(sid_pop.values)
    data2['sid_pop_count'] = data2['sid'].map(sid_pop_dict)
    
    # Calculate PCC
    values = np.array(data2['pred'].values)
    sid_pop_count = np.array(data2['sid_pop_count'].values)
    
    X = values
    Y = sid_pop_count
    
    pcc = ((X - X.mean()) * (Y - Y.mean())).sum() / np.sqrt(((X - X.mean()) ** 2).sum() * ((Y - Y.mean()) ** 2).sum())
    
    return pcc

def pcc_test_check(model_here, without_neg_data, sid_pop_total):
    # https://www.statology.org/pandas-groupby-correlation/
    # https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.corr.html
    
    data2 = without_neg_data
    filter_users = data2.uid.value_counts()[data2.uid.value_counts() > 1].index
    data2 = data2[data2.uid.isin(filter_users)]
    data2 = data2.reset_index()[['uid', 'sid']]    
    
    
    data2['uid'] = data2['uid'].apply(lambda x : int(x))
    data2['sid'] = data2['sid'].apply(lambda x : int(x))    
    
    user_num = len(data2.uid.unique())
    data_len = data2.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data2.iloc[ (frac_user_num*itr) : (frac_user_num*(itr+1)) ].values    
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()
        predictions, _ = model_here(user, item, item)
        predictions_list += predictions.detach().cpu().tolist()        
        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()

            predictions, _ = model_here(user, item, item)
            predictions_list += predictions.detach().cpu().tolist()
    
    model_here.cpu()    
    data2['pred'] = predictions_list
    
    sid_pop_dict = dict(sid_pop_total.values)
    data2['sid_pop_count'] = data2['sid'].map(sid_pop_dict)    
        
    result = data2[['sid_pop_count', 'pred']].corr(method = 'pearson')
    
    return result