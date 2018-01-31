import numpy as np
import pandas as pd
import scipy.sparse

# Custom metric is implemented here
from scorer import scorer
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

# Use custom dtypes for efficiency
dtypes = {'id1': np.int16, 'id2': np.int16, 'id3': np.int16, 'user_id': np.int32, 'date': np.int16}

train = pd.read_csv('dataset/train_small', dtype=dtypes)
train.head(5)

date_validation_start = train.date.max() - 6


def calculate_target(data, date_test_start):
    '''
        This function returns a dictionary of type {user: items_list}
        Such that user viewed an item in testing period,
        but did not view it within the last 3 weeks of train period.
    '''

    test_mask = (data.date >= date_test_start) & (data.date < date_test_start + 7)
    last_3weeks_mask = (data.date >= date_test_start - 21 + 1) & (data.date < date_test_start)

    # Items that used viewed during test period
    items_test = data[test_mask].groupby('user_id').id3.apply(set)

    # Items, that user viewed in last 3 weeks
    user_last_3weeks = data[last_3weeks_mask].groupby('user_id').id3.apply(set)

    # Get table, where for each `user_id` we have both items from test period and 3 weeks
    joined = items_test.reset_index().merge(user_last_3weeks.reset_index(), on=['user_id'], how='left')
    joined.set_index('user_id', inplace=True)

    # Remove the items, which the user viewed during last 3 weeks
    target = {}
    for user_id, (id3_x, id3_y) in joined.iterrows():
        items = id3_x if id3_y is np.nan else id3_x - id3_y
        if items != set():
            target.update({user_id: items})

    return target


# # This function may take several minutes to finish
y_val_dict = calculate_target(train, date_validation_start)


# ML BENCHMARK EXAMPLE
ids = train.id3.unique()
users = train.user_id[train.date < date_validation_start].unique()
num_users = len(users)

mask_train = train.date < date_validation_start - 7
mask_test = (train.date < date_validation_start) & (train.date >= train.date.min() + 7)

# For the sake of speed select only first 10k users to train on
users_mask = train.user_id < 10000
mask_train &= users_mask


def get_feats(data):
    '''
        Builds sparse matrix using users' history.
    '''
    return scipy.sparse.coo_matrix(([1] * data.shape[0], (data.user_id, data.id3)),
                                    shape=[data.user_id.max()+1, data.id3.max()+1]).tocsr()


def get_target_matrix(X, target_dict):
    '''
        Builds sparse matrix using dictionary.
    '''
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}

    ks = []
    for k in tqdm(range(X.user_id.max()+1)):
        d = target_dict.get(k, [])
        for y in d:
            indices.append(y)
            data.append(1)
        indptr.append(len(indices))

    return scipy.sparse.csr_matrix((data, indices, indptr), dtype=int, shape =[X.user_id.max()+1, X.id3.max()+1])


# For each user count how many items he viewed
X_train = get_feats(train.loc[mask_train])
X_test = get_feats(train.loc[mask_test])

y_train_dict = calculate_target(train.loc[users_mask], date_validation_start - 7)
y_train = get_target_matrix(train.loc[mask_train], y_train_dict)
y_test = get_target_matrix(train.loc[mask_test], y_val_dict)


def fit(i):
    target = y_train[:, i].toarray().ravel()

    if target.mean() == 0:
        return np.zeros((X_test.shape[0],)) - 1

    d = LogisticRegression(max_iter=10)
    d.fit(X_train, target)
    return (d.predict_proba(X_test)[:, 1])

preds = Parallel(n_jobs=8, verbose=50)(delayed(fit)(i) for i in range(y_train.shape[1]))
preds = np.vstack(preds).T

# To reduce memory usage
preds = preds.astype(np.float16)

num = int(np.ceil(num_users * 0.05))

# Let's take not random users, but the ones who viewed a lot
users = train.loc[mask_test].user_id.value_counts().index[:num]
ans_inds =  np.argsort(preds[users])

test_inds_dict =  {k: list(ans_inds[i, -5:]) for i,k in enumerate(users)}
scorer(y_val_dict, test_inds_dict, num_users=num_users)

# For each user find the categories, which we do not want to predict
last_3weeks = train.loc[mask_test].loc[train.loc[mask_test].date >= train.loc[mask_test].date.max() - 21 + 1]
y_not = last_3weeks.groupby('user_id').id3.apply(set)

y_pred = {}

for u_idx, user in tqdm(enumerate(users)):
    items_not = y_not.get(user, [])
    items_pred = []
    i = 1
    while len(items_pred) < 5:
        if not ans_inds[u_idx, -i] in items_not:
            items_pred += [ans_inds[u_idx, -i]]

        i += 1
    y_pred.update({user: items_pred})

print(scorer(y_val_dict, y_pred, num_users))

y_pred_df = pd.DataFrame.from_records(y_pred).T.reset_index()
y_pred_df.columns = ['user_id', 'id3_1', 'id3_2', 'id3_3', 'id3_4', 'id3_5']

y_pred_df.to_csv('y_pred.csv', index=False)