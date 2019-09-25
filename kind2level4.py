import pandas as pd
import numpy as np
import jieba_fast as jieba
import psycopg2 as pgsql
import math
import time
import json
import re
import os
import codecs
import datetime
import gc
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score




model_path = r'/home/sunxianwei/python/result/kind2level4/test_model_path'
bs_kind_cat4_system_path = r'/home/sunxianwei/python/data/kind/test_config_filepath/kind_system_allcat_20180601.csv'
stopwords_path = r'/home/sunxianwei/python/data/kind/test_config_filepath/filter_char.txt'

dataseg_path = r'/home/sunxianwei/python/data/kind/test_config_filepath/prod_clothes20180601_to_keywords.csv'
userdict_path = False
tfidf_max_features_config_dict = {1:None,3:None,5:None,7:None,10:None}

if userdict_path:
    print('加载自定义词典')
    jieba.load_userdict(userdict_path)

bs_kind_cat4_system = pd.read_csv(bs_kind_cat4_system_path,encoding='utf8')
bs_kind_cat4_system.fillna(0,inplace=True)
bs_kind_cat4_system.parent_catid=bs_kind_cat4_system.parent_catid.astype(int)
bs_kind_cat4_system=[{'catid':row['cat_id'],'catname':row['cat_name'],'parent_catid':row['parent_catid']} for index,row in bs_kind_cat4_system.iterrows()]




def readdata_from_pgsql(username,passwd,queryStr):
    '''
    两个重点字段需要规范，其它字段保留
    prod_name:产品名称
    catid:品类ID
    '''
    time_start = time.time()
    try:
        with pgsql.connect(host='116.62.168.87',port=5432,user=username,password=passwd,database='warehouse') as conn:
            df = pd.read_sql(queryStr,conn)
            if 'prod_name' not in df.columns or 'catid' not in df.columns:
                print('数据读取结果不规范，请检查查询语句')
                return False
            print('数据读取耗时：{0}s'.format(round(time.time()-time_start,2)).center(80,'*'))
            return df
    except:
        print('数据库查询失败,请检查语句')
        return False

def readdata_from_csv(filepath,encodeType='utf8'):
    '''
    两个重点字段需要规范，其它字段保留
    prod_name:产品名称
    catid:品类ID
    '''
    time_start = time.time()
    try:
        df = pd.read_csv(filepath,encoding=encodeType)
        if 'prod_name' not in df.columns or 'catid' not in df.columns:
            print('数据读取结果不规范，请检查查询语句')
            return False
        print('数据读取耗时：{0}s'.format(round(time.time()-time_start,2)).center(80,'*'))
        return df
    except:
        print('csv文件读取失败,请检查文件')
        return False
    
def translate_extendcat_for_level(input_data):
    time_start = time.time()
    input_data['catid'] = input_data['catid'].astype(str)
    input_data['catid_level0'] = input_data['catid'].apply(lambda x:None if x==None or x is np.nan or len(x)<1 else x[:1])
    input_data['catid_level1'] = input_data['catid'].apply(lambda x:None if x==None or x is np.nan or len(x)<3 else x[:3])
    input_data['catid_level2'] = input_data['catid'].apply(lambda x:None if x==None or x is np.nan or len(x)<5 else x[:5])
    input_data['catid_level3'] = input_data['catid'].apply(lambda x:None if x==None or x is np.nan or len(x)<7 else x[:7])
    input_data['catid_level4'] = input_data['catid'].apply(lambda x:None if x==None or x is np.nan or len(x)<10 else x[:10])
    print('数据集分级映射标签耗时：{0}s'.format(round(time.time()-time_start,2)).center(80,'*'))
    return input_data

def extract_keyword_from_prodname(prod_name,stopwords=stopwords_path):
    try:
        word_list = jieba.cut(prod_name, cut_all=True)
        stop_words_list = get_stopwords(stopwords)
        if stop_words_list:
            word_list = [word.strip() for word in word_list if word.strip() not in stop_words_list]
        word_list = ' '.join(word_list)
    except:
        word_list = 'None'
    return word_list

def get_stopwords(stopwords):
    if stopwords:
        stop_words_list = set()
        for word in codecs.open(stopwords, 'r', 'utf-8', 'ignore'):
            stop_words_list.add(word.strip())
        return stop_words_list
    else:
        return stopwords

def multiextract_keyword_from_prodname(input_data,stopwords=stopwords_path,saveResult = False):
    stop_words_list = get_stopwords(stopwords)
    time_start = time.time()
    if stop_words_list:
        input_data['prod_name_keywords'] = input_data['prod_name'].apply(lambda x:' '.join([word.strip() for word in jieba.cut(x,cut_all=True) if word.strip() not in stop_words_list]))
    else:
        input_data['prod_name_keywords'] = input_data['prod_name'].apply(lambda x:' '.join([word.strip() for word in jieba.cut(x,cut_all=True)]))
    print('产品名称批量分词总数：{0}；耗时：{1}s'.format(input_data['prod_name_keywords'].count(),round(time.time()-time_start,2)).center(80,'*'))
    if saveResult:
        input_data.to_csv(dataseg_path,sep=',',encoding = 'utf8')
    return input_data

def balance_sample_count(input_data,min_nsample,total_nsample,balanceType=2,samples_rate=False):
    time_start = time.time()
    diff_case = pd.DataFrame(pd.Series(input_data['catid']).value_counts())
    diff_case.columns = ['cnt']
    diff_case  = pd.DataFrame(diff_case['cnt']).sort_values(by = 'cnt',axis = 0,ascending = True).reset_index()
    diff_case.columns = ['id', 'cnt']
    result = pd.DataFrame()
    if samples_rate:
        samples_num = [int(i/sum(samples_rate)*sum(diff_case['cnt'])) for i in samples_rate]
        for i in range(len(diff_case)):
            result = result.append(input_data[input_data['catid'] == diff_case.iloc[i,0]].sample(n=samples_num[i], replace=True, random_state=22))
    else:
        if min_nsample*len(diff_case) >= total_nsample:
            print('最低类数总量大于最高上限')
        else:
            samples_num=diff_case['cnt']
            if balanceType == 1:
                if min(samples_num) < min_nsample:
                    samples_num = [ int(min_nsample/min(samples_num)*i) if i<=min_nsample else i for i in diff_case['cnt']]
                else:
                    samples_num = [i for i in samples_num]
                samples_num = [int(math.log(i)*max(samples_num)/math.log(max(samples_num))) for i in samples_num]

                if sum(samples_num) >total_nsample:
                    samples_num = [int(total_nsample/sum(samples_num)*i) for i in samples_num]
            elif balanceType == 2:
                while min(samples_num) < min_nsample or sum(samples_num) >total_nsample:
                    if min(samples_num) < min_nsample:
                        samples_num = [ int(min_nsample/min(samples_num)*i) if i<=min_nsample else i for i in samples_num]
                    else:
                        samples_num = [i for i in samples_num]
                    if sum(samples_num) >total_nsample:
                        samples_num = [int(total_nsample/sum(samples_num)*i) for i in samples_num]
            for i in range(len(diff_case)):
                result = result.append(input_data[input_data['catid'] == diff_case.iloc[i,0]].sample(n=samples_num[i], replace=True, random_state=22))
    print('样本均衡耗时：{0}s'.format(round(time.time()-time_start,2)).center(80,'*'))
    return result




def convert_tojson(catid):
    result_id = 0
    if catid ==0 or catid ==1:
        result_id = 0
    elif catid >= 100 and catid < 10000000:
        result_id = catid//100
    elif catid >= 1000000000 and catid < 10000000000:
        result_id = catid//1000
    return json.dumps(get_children(result_id),ensure_ascii=False)

def get_children(id=0):
    result=[]
    for obj in bs_kind_cat4_system:
        if obj["parent_catid"] ==id:
            result.append({"catid":obj["catid"],"catname":obj["catname"],"children":get_children(obj["catid"])})
    return result

def hierarchical_classifier(catid,input_data,version='v1.0',model_alpha=1.0,model_fit_prior=True,tfidf_min_df=1,tfidf_max_df=1.0,inplace = True):
    """分层分类逻辑模块
    Args:
        catid:待训练分类根行业id
        input_data:待训练分类数据文件，包含prod_name_keywords，catid_level0~catid_level4
        cat_system:行业体系
    Returns:
        None
    """
    my_queue = []
    bs_kind_cat_system = json.loads(convert_tojson(catid))
    if isinstance(bs_kind_cat_system,list):
        my_queue.extend(bs_kind_cat_system)
    else:
        my_queue.append(bs_kind_cat_system)

    while len(my_queue)>0:
        for queue in my_queue:
            #sample_classifier(queue.get('catid'),)
            current_catid = queue.get('catid')
            current_catid_filter = []
            for obj_type in input_data[['catid_level0','catid_level1','catid_level2','catid_level3','catid_level4']].dtypes:
                if obj_type == np.int64:
                    current_catid_filter.append(int(current_catid))
                elif obj_type == np.float64:
                    current_catid_filter.append(float(current_catid))
                else:
                    current_catid_filter.append(str(current_catid))
            
            if current_catid == 0:
                current_input_data = input_data[['prod_name_keywords','catid_level0']]
            elif current_catid >= 1 and current_catid < 10:
                current_input_data = input_data[(input_data.catid_level0==current_catid_filter[0])&(input_data.catid_level1.notna())][['prod_name_keywords','catid_level1']]
            elif current_catid >= 100 and current_catid < 1000:
                current_input_data = input_data[(input_data.catid_level1==current_catid_filter[1])&(input_data.catid_level2.notna())][['prod_name_keywords','catid_level2']]
            elif current_catid >= 10000 and current_catid < 100000:
                current_input_data = input_data[(input_data.catid_level2==current_catid_filter[2])&(input_data.catid_level3.notna())][['prod_name_keywords','catid_level3']]
            elif current_catid >= 1000000 and current_catid < 10000000:
                current_input_data = input_data[(input_data.catid_level3==current_catid_filter[3])&(input_data.catid_level4.notna())][['prod_name_keywords','catid_level4']]
            else:
                continue
            
            current_path_list = []
            temp_current_catid = current_catid
            while True:
                current_path_list.append(str(temp_current_catid))
                if temp_current_catid == catid:
                    break
                elif temp_current_catid ==0 or temp_current_catid ==1:
                    temp_current_catid = 0
                elif temp_current_catid >= 100 and temp_current_catid < 10000000:
                    temp_current_catid = temp_current_catid//100
                elif temp_current_catid >= 1000000000 and temp_current_catid < 10000000000:
                    temp_current_catid = temp_current_catid//1000
            
            if max(current_input_data.count()) > 0:
                current_path = os.path.join(model_path,'/'.join(sorted(current_path_list,reverse=False)))
                current_input_data.columns = ['prod_name_keywords','catid']
                
                #total_count = current_input_data['prod_name_keywords'].count()
                #min_count = int(0.3*total_count//len(current_input_data['catid'].unique()))
                
                #current_input_data = balance_sample_count(current_input_data,min_count,total_count,balanceType=2)
                if os.path.exists(current_path) == False:
                    os.mkdir(current_path)
                sample_classifier(current_catid,current_path,current_input_data,version,
                                  tfidf_max_features=tfidf_max_features_config_dict.get(len(str(current_catid))),inplace=inplace)
        children_queue = []
        for queue in my_queue:
            children_queue.extend(queue.get('children'))
        my_queue = children_queue

def sample_classifier(catid,model_path_for_save,input_data,version,model_alpha=1.0,model_fit_prior=True,
                      tfidf_min_df=1,tfidf_max_df=1.0,tfidf_max_features=None,inplace = True):
    model_name = str(catid)
    if len(version)>0:
        model_name=model_name+'_'+version
    text2tfidf = build_text2tfidf(input_data.prod_name_keywords.tolist(),model_path_for_save,model_name+'_tfidf',tfidf_min_df,tfidf_max_df,
                                  tfidf_max_features)
    classifier = build_classifier(text2tfidf,input_data.catid,model_path_for_save,model_name+'_model',model_alpha,model_fit_prior)
    
def build_text2tfidf(namelist,model_path_for_save,model_name,tfidf_min_df,tfidf_max_df,tfidf_max_features,inplace = True):
    time_start = time.time()
    if os.path.exists(os.path.join(model_path_for_save,model_name)):
        if inplace:
            os.remove(os.path.join(model_path_for_save,model_name))
            text2tfidf=TfidfVectorizer(encoding='utf8',max_df=tfidf_max_df,min_df=tfidf_min_df, binary=False,max_features=tfidf_max_features)
            result_tfidf=text2tfidf.fit_transform(namelist)
            joblib.dump(text2tfidf,os.path.join(model_path_for_save,model_name))
        else:
            text2tfidf = joblib.dump(text2tfidf,os.path.join(model_path_for_save,model_name))
            result_tfidf = text2tfidf.transform(namelist)
    else:
        text2tfidf=TfidfVectorizer(encoding='utf8',max_df=tfidf_max_df,min_df=tfidf_min_df, binary=False,max_features=tfidf_max_features)
        result_tfidf=text2tfidf.fit_transform(namelist)
        joblib.dump(text2tfidf,os.path.join(model_path_for_save,model_name))
    print('文本向量化{0}耗时：{1}s'.format(model_name,round(time.time()-time_start,2)).center(80,'*'))
    return result_tfidf

def build_classifier(name_tfidf,cat_label,model_path_for_save,model_name,model_alpha,model_fit_prior):
    time_start = time.time()
    classifier=MultinomialNB(alpha=model_alpha,fit_prior=model_fit_prior)
    classifier.fit(name_tfidf,cat_label)
    if os.path.exists(os.path.join(model_path_for_save,model_name)):
        os.remove(os.path.join(model_path_for_save,model_name))
    joblib.dump(classifier,os.path.join(model_path_for_save,model_name))
    print('模型{0}训练耗时：{1}s'.format(model_name,round(time.time()-time_start,2)).center(80,'*'))
    return classifier

def listdirs(path,model_name):
    return [os.path.join(path,i) for i in os.listdir(path) if os.path.isdir(os.path.join(path,i)) and str(model_name) == i]

def listfiles(path,model_name):
    return [os.path.join(path,i) for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and str(model_name) == i]

def model_check(path,model_name,model_filter,tfidf_filter):
    model = listfiles(path,str(model_name)+'_'+model_filter)
    tfidf = listfiles(path,str(model_name)+'_'+tfidf_filter)
    if len(model) == 1 and len(tfidf) == 1:
        if model[0].replace(model_filter,'') == tfidf[0].replace(tfidf_filter,''):
            return model[0],tfidf[0]
    elif len(model) == 0 and len(tfidf) == 0:
        model_dir_list = listdirs(path,model_name)
        if len(model_dir_list) == 1:
            return model_check(model_dir_list[0],model_name,model_filter,tfidf_filter)
    return False,False

def predict_multi(prod_id_list,prod_name_keywords_list,version='v1.0',model_name=1,model_path_for_load=model_path):
    model_file,tfidf_file = model_check(model_path_for_load,model_name,version+'_model',version+'_tfidf')
    if model_file == False:
        print('模型路径错误，请重新训练')
        return
    
    time_start = time.time()
    
    pre_classifier = joblib.load(model_file)
    text2tfidf = joblib.load(tfidf_file)
    model_path_for_load = os.path.abspath(os.path.dirname(model_file)+os.path.sep+".")
    
    pre_text2tfidf = text2tfidf.transform(prod_name_keywords_list)
    
    result_total = pd.DataFrame({'prod_id':prod_id_list,'prod_name_keywords':prod_name_keywords_list,'catid_y':pre_classifier.predict(pre_text2tfidf)})
    result_total['catid_y'] = result_total.catid_y.astype(np.int64)
    
    result_final = pd.DataFrame(columns=['prod_id','prod_name_keywords','catid_y'])
    
    for catid in result_total['catid_y'].unique():
        if str(catid) in [i for i in os.listdir(model_path_for_load) if os.path.isdir(os.path.join(model_path_for_load,i))]:
            result_final = result_final.append(predict_multi(result_total[result_total['catid_y']==catid]['prod_id'].tolist(),
                          result_total[result_total['catid_y']==catid]['prod_name_keywords'].tolist(),
                          version,catid,os.path.join(model_path_for_load,str(catid))))
        else:
            result_final = result_final.append(result_total[result_total['catid_y']==catid])
    print('模型{0}数据预测耗时：{1}s'.format(model_name,round(time.time()-time_start,2)).center(80,'*'))
    return result_final

def score_all(y_real,y_predict):
    return accuracy_score(y_real, y_predict)

def score_categories(y_real,y_predict):
    return classification_report(y_real,y_predict)


# In[5]:


#参数为用户名、密码，username,password
def connectPostgreSQL(username,passwd):
    db = pgsql.connect(host='116.62.168.87',port=5432,user=username,password=passwd,database='warehouse')
    print("Connected Successful!")
    return db

#写入数据库
def write_to_table(df, table_name,schema_name,username,password, if_exists='fail'):
    from sqlalchemy import create_engine
    import io
    db_engine = create_engine('postgresql://{0}:{1}@116.62.168.87:5432/warehouse'.format(username,password))
    string_data_io = io.StringIO()
    df.to_csv(string_data_io, sep='|', index=False) 
    pd_sql_engine = pd.io.sql.pandasSQL_builder(db_engine)
    table = pd.io.sql.SQLTable(table_name, pd_sql_engine, frame=df,
                                   index=False, if_exists=if_exists,schema = schema_name)
    table.create()
    string_data_io.seek(0)  
    string_data_io.readline()
    with db_engine.connect() as connection:
        with connection.connection.cursor() as cursor:  
            copy_cmd = "COPY {0}.{1} FROM STDIN HEADER DELIMITER '|' CSV".format(schema_name,table_name)
            cursor.copy_expert(copy_cmd, string_data_io)
        connection.connection.commit()





#训练模块
if __name__ == '__main__':
    #数据读取，readdata_from_pgsql(username='',passwd='',queryStr='')
    train_data_for_clothes = readdata_from_pgsql(username='username',passwd='password',
                                                 queryStr='SELECT prod_id,prod_name,cat_id as catid FROM mining_original_datadb.original_traindata_clothes_jd_repair20180524;')
    #train_data_for_clothes = readdata_from_csv(r'/home/sunxianwei/python/data/kind/test_config_filepath/train_data_for_clothes_20180601.csv',encodeType='utf8')
    #train_data_for_clothes = train_data_for_clothes[train_data_for_clothes.catid.notna()]
    #del train_data_for_clothes['Unnamed: 0']
    #数据集分标签映射
    train_data_for_clothes = translate_extendcat_for_level(input_data = train_data_for_clothes)
    #数据集分词
    train_data_for_clothes = multiextract_keyword_from_prodname(input_data = train_data_for_clothes)
    #train_data_for_clothes.to_csv(r'/home/sunxianwei/python/data/kind/test_config_filepath/train_data_for_clothes_20180601.csv',encoding='utf8')
    #分层分类训练
    hierarchical_classifier(catid=1,input_data=train_data_for_clothes,version='v1.0')
    #预测
    testdata = train_data_for_clothes[train_data_for_clothes.catid.notna()]
    testdata['catid'] = testdata.catid.astype(np.int64)
    #testdata = multiextract_keyword_from_prodname(input_data = testdata)
    result_pre = predict_multi(testdata.prod_id.tolist(),testdata.prod_name_keywords.tolist()).merge(testdata,on=['prod_id','prod_name_keywords'])
    print(score_all(result_pre.catid.astype(str),result_pre.catid_y.astype(str)))
    print(score_categories(result_pre.catid.astype(str),result_pre.catid_y.astype(str)))


# In[ ]:


#解析模块
db = connectPostgreSQL('username','password')
taobao_data_query = '''SELECT prod_id,prod_name FROM gather_taobao.ir_taobao_product_trade_china_2017 
                            where prod_name NOTNULL AND length(prod_name)>10  AND "id" BETWEEN {0} AND {1};'''
for i in range(0,2170000000,10000000):
    start_time = time.time()
    #数据读取
    traindata_taobao = pd.read_sql(taobao_data_query.format(i,i+10000000),db)
    print('数据读取耗时:{0}'.format(round(time.time()-start_time,2)))
    
    #数据分词
    traindata_taobao = multiextract_keyword_from_prodname(input_data = traindata_taobao)
    
    start_time = time.time()
    #初步分类提取
    first_classifier = joblib.load(r'/home/sunxianwei/python/result/kind2level4/test_model_path/1/1_v0.0_model')
    text2tfidf = joblib.load(r'/home/sunxianwei/python/result/kind2level4/test_model_path/1/1_v0.0_tfidf')
    first_text2tfidf = text2tfidf.transform(traindata_taobao.prod_name_keywords.values)
    traindata_taobao['first_catid'] = first_classifier.predict(first_text2tfidf)
    print('初步分类耗时:{0}'.format(round(time.time()-start_time,2)))
    
    traindata_taobao = traindata_taobao[traindata_taobao['first_catid'].isin([101,106,107,108,199])][['prod_id','prod_name_keywords']]
    
    start_time = time.time()
    #重新预测
    traindata_taobao = predict_multi(traindata_taobao.prod_id.tolist(),traindata_taobao.prod_name_keywords.tolist()).merge(traindata_taobao,on=['prod_id','prod_name_keywords'])
    filename = datetime.datetime.now()
    traindata_taobao.to_csv(r'/home/sunxianwei/python/result/kind2level4/test_result_path/taobao_result_{0}.csv'.
                            format('-'.join([str(filename.year),str(filename.month),str(filename.day),str(filename.hour),str(filename.minute),str(filename.second)])),encoding = 'utf8')
    print('二次分类耗时:{0}'.format(round(time.time()-start_time,2)))
    del traindata_taobao
    del filename
    del first_classifier
    del text2tfidf
    del first_text2tfidf
    print('数据批次{0}'.format(i))
    gc.collect()

