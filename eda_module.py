import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier



def check_cat_features(data : pd.DataFrame , feature : str):
    """
    parameters:
    data : The data for which we need to evaluate the variables
    feature: column to verify whether it contains non numerical values or not    
    """
    converted = pd.to_numeric(data.replace({feature:{'XNA':np.nan}})[feature],errors='coerce')
    non_numerical_cnt = data[((converted.isna()) & (data.replace({feature:{'XNA':np.nan}})[feature].notna()))].count().iloc[0]
    return non_numerical_cnt



def variables_basic_info(data : pd.DataFrame,columns_remove : list =[]) -> pd.DataFrame :

    """
    parameters:
    data: The data for which we need to evaluate the variables
    columns_remove: The columns which need not be evaluated from the provided data. If no input is provided, then all the columns would be evaluated

    Return:
    Returns a dictionary with the unique count, data type, data type after evaluating the variables in the column, min, max, mode, null rate

    """

    categorical_variable_list = list(data.dtypes[data.dtypes == 'object'].index)
    numerical_variable_list = list(data.dtypes[(data.dtypes == 'int64') | (data.dtypes == 'float64')].index)
    date_variable_list = list(data.dtypes[data.dtypes == 'datetime64[ns]'].index)

    for i in columns_remove:
        try:
            categorical_variable_list.pop(categorical_variable_list.index(i))
        except :
            pass
        try:
            numerical_variable_list.pop(numerical_variable_list.index(i))
        except:
            pass
        try:
            date_variable_list.pop(date_variable_list.index(i))
        except:
            pass

    variable_dict = {}
    variable_dict['variables'] = categorical_variable_list
    variable_dict['unique_count'] = list(data[categorical_variable_list].nunique().values)
    variable_dict['variable_table_dtype'] = ['cat' for i in categorical_variable_list]

    variable_dict['variables'] = variable_dict['variables'] + numerical_variable_list
    variable_dict['unique_count'] = variable_dict['unique_count'] + list(data[numerical_variable_list].nunique().values)
    variable_dict['variable_table_dtype'] = variable_dict['variable_table_dtype'] + ['num' for i in numerical_variable_list]

    variable_dict['variables'] = variable_dict['variables'] + date_variable_list
    variable_dict['unique_count'] = variable_dict['unique_count'] + list(data[date_variable_list].nunique().values)
    variable_dict['variable_table_dtype'] = variable_dict['variable_table_dtype'] + ['date' for i in date_variable_list]

    variable_dtype = []
    min_value = []
    max_value = []
    mode_list=[]
    null_percentage = []
    for i in range(0,len(variable_dict['variables'])):

        if variable_dict['unique_count'][i]<=30:
            variable_dtype.append('cat')
        elif check_cat_features(data=data,feature=variable_dict['variables'][i]) == 0:
            variable_dtype.append('num')
        else:
            variable_dtype.append(variable_dict['variable_table_dtype'][i])

        # min and max values for numerical variables
        if check_cat_features(data=data,feature=variable_dict['variables'][i]) == 0:
            min_value.append(round(pd.to_numeric(data[variable_dict['variables'][i]],errors='coerce').min(),4))
            max_value.append(round(pd.to_numeric(data[variable_dict['variables'][i]],errors='coerce').max(),4))
        else:
            min_value.append(0)
            max_value.append(0)
        
        #mode for all variables
        mode_list.append(data[variable_dict['variables'][i]].mode()[0])
        null_percentage.append(data[variable_dict['variables'][i]].isnull().mean()*100)

        #null percentage

    min_value = [str(i) for i in min_value]
    max_value = [str(i) for i in max_value]

    variable_dict['variable_dtype'] = variable_dtype
    variable_dict['Minimum'] = min_value
    variable_dict['Maximum'] = max_value
    variable_dict['Mode'] = mode_list
    variable_dict['Null Rate'] = null_percentage
    return pd.DataFrame(variable_dict)



#distribution plot for categorical variables seperately for default and non-default
def distribution_plot_cat(data : pd.DataFrame,target:str = 'RISK_FPD30',unique_cnt =10, columns_remove : list =[]):
    print(f"All variables with unique count less than {unique_cnt+1} and is categorical in nature are plotted below")
    basic_info = variables_basic_info(data,columns_remove)
    ncol=2
    nrow=max(2,math.ceil(len(basic_info[(basic_info['unique_count']<=unique_cnt) & (basic_info['unique_count']>1)]['variables'].values)/ncol))
    fig, axs = plt.subplots(nrow, ncol,figsize=(15, nrow*3.5))
    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    for i in range(len(basic_info[(basic_info['unique_count']<=unique_cnt) & (basic_info['unique_count']>1)]['variables'].values)):
        
        variable = basic_info[(basic_info['unique_count']<=unique_cnt) & (basic_info['unique_count']>1)]['variables'].values[i]

        df_dist=pd.merge(left = data[data[target]==0].value_counts(subset=[variable],normalize=True),
            right = data[data[target]==1].value_counts(subset=[variable],normalize=True),
            how='left',right_on=variable,left_on=variable,suffixes=('_non_default','_default'))*100
        
        df_dist.reset_index(inplace=True)

        df_dist_unstacked = pd.melt(frame=df_dist,id_vars=[variable],value_vars=['proportion_non_default','proportion_default'],
                var_name = target,value_name='Distribution')
        df_dist_unstacked.replace({'proportion_non_default':'0','proportion_default':'1'},inplace=True)
        
        col = math.ceil((i-1)/2)

        if i%2 ==0:
            sns.barplot(ax=axs[col,0],x = variable, y='Distribution',hue = target,data=df_dist_unstacked)
            axs[col,0].tick_params(axis='x',labelrotation=15)
            plt.tight_layout()

        else:
            sns.barplot(ax=axs[col,1],x = variable, y='Distribution',hue = target,data=df_dist_unstacked)
            axs[col,1].tick_params(axis='x',labelrotation=15)
            plt.tight_layout()  

#box plot for numerical variables drawn seperately for default and non-default
def box_plot_num(data : pd.DataFrame,target: str = 'RISK_FPD30',columns_remove : list =[]):
    print(f"all variables with the nature as numerical are plotted below")
    basic_info = variables_basic_info(data,columns_remove)
    ncol=2
    nrow=max(2,math.ceil(len(basic_info[basic_info['variable_dtype']=='num']['variables'].values)/ncol))
    fig, axs = plt.subplots(nrow, ncol,figsize=(15, nrow*3.5))
    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    
    for i in range(len(basic_info[basic_info['variable_dtype']=='num']['variables'].values)):

        variable = basic_info[basic_info['variable_dtype']=='num']['variables'].values[i]
        col = math.ceil((i-1)/2)

        feature_table=data[[variable,target]].copy()
        feature_table[variable] = feature_table[variable].replace('XNA',0)
        feature_table[variable] = pd.to_numeric(feature_table[variable])
        feature_table.dropna(inplace=True)
        feature_table=feature_table[(feature_table[variable]<np.percentile(feature_table[variable].values,q=99)) 
                                    & (feature_table[variable]>np.percentile(feature_table[variable].values,q=1))][[variable,target]]
    
    

        if i%2 ==0:
            sns.boxplot(ax=axs[col,0],x = variable,hue = target,data=feature_table)
            axs[col,0].tick_params(axis='x',labelrotation=15)
            axs[col,0].legend(loc='upper left')
            plt.tight_layout()

        else:
            sns.boxplot(ax=axs[col,1],x = variable,hue = target,data=feature_table)
            axs[col,1].tick_params(axis='x',labelrotation=15)
            axs[col,1].legend(loc='upper left')
            plt.tight_layout() 

#kde density plot for numerical variables drawn seperately for default and non-default
def kde_plot_num(data : pd.DataFrame,target:str='RISK_FPD30',columns_remove : list =[]):
    print("All variables with the nature as numerical are plotted below")
    print(" Note: When the graph shows that the kde plot is higher in altitude for default than non-default: the segment has higher risk and vice-versa")
    basic_info = variables_basic_info(data,columns_remove)
    ncol=2
    nrow=max(2,math.ceil(len(basic_info[basic_info['variable_dtype']=='num']['variables'].values)/ncol))
    fig, axs = plt.subplots(nrow, ncol,figsize=(15, nrow*3.5))
    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    for i in range(len(basic_info[basic_info['variable_dtype']=='num']['variables'].values)):

        variable = basic_info[basic_info['variable_dtype']=='num']['variables'].values[i]
        col = math.ceil((i-1)/2)

        feature_table=data[[variable,target]].copy()
        feature_table[variable] = feature_table[variable].replace('XNA',0)
        feature_table[variable] = pd.to_numeric(feature_table[variable])
        feature_table.dropna(inplace=True)
        feature_table=feature_table[(feature_table[variable]<np.percentile(feature_table[variable].values,q=99)) & 
                                    (feature_table[variable]>np.percentile(feature_table[variable].values,q=1))][[variable,target]]
        
        
        if i%2 ==0:
            sns.kdeplot(ax=axs[col,0],x = variable,data=feature_table[feature_table[target]==0],color='b')
            sns.kdeplot(ax=axs[col,0],x = variable,data=feature_table[feature_table[target]==1],color='r')
            axs[col,0].legend([0,1])
            plt.tight_layout()
            

        else:
            sns.kdeplot(ax=axs[col,1],x = variable,data=feature_table[feature_table[target]==0],color='b')
            sns.kdeplot(ax=axs[col,1],x = variable,data=feature_table[feature_table[target]==1],color='r')
            axs[col,1].legend([0,1])
            plt.tight_layout()



#for univariant gini score
def univariant_gini(data : pd.DataFrame,target:str='RISK_FPD30',columns_remove : list =[]):
    basic_info = variables_basic_info(data,columns_remove)
    gini_dict = {'Variables': [], 'Gini_score': [] }
    for i in range(len(basic_info[basic_info['variable_dtype']=='num']['variables'].values)):

        variable = basic_info[basic_info['variable_dtype']=='num']['variables'].values[i]
        feature_table=data[[variable,target]].copy()
        feature_table[variable] = feature_table[variable].replace('XNA',0)
        feature_table[variable] = pd.to_numeric(feature_table[variable])
        feature_table.dropna(inplace=True)
        feature_table=feature_table[(feature_table[variable]<np.percentile(feature_table[variable].values,q=99)) & (feature_table[variable]>np.percentile(feature_table[variable].values,q=1))][[variable,target]]
        feature_table.sort_values(by=variable,ascending=True)
        gini_score = 1-2*roc_auc_score(y_true=feature_table[target],y_score=feature_table[variable])
        gini_dict['Variables'].append(variable)
        gini_dict['Gini_score'].append(gini_score)
    
    return pd.DataFrame(gini_dict).sort_values(by='Gini_score',ascending=False)



#Decision Tree function for finding Bands of a single Vaiable
def features_DecisionTree(feature:str,data:pd.DataFrame,target:str='RISK_1_5PD30',max_depth:int=10,min_samples_split:float=0.05,
                          min_samples_leaf:float=0.1):
    
    """
    Brief on the function: The function uses a single variable to run a Decision Tree Algorithm and provide the potential splits for the variable.
    The function works for all numerical variables and the splits can be changed by tuning the hyperparameters of the Decision Tree.
    
    Parameters:
    feature: The feature for which the porential bands needs to be identified. 
    data: The sample data on which the alogorithm would be run
    target: Target used by the Decision Tree.
    max_depth: max_depth of the Decision Tree.
    min_samples_split: Minimum sample required for a potential split in the Decision Tree
    min_samples_leaf: Minimum sample to be present in any leaf

    Return: 
    Returns the output of the Decision Tree with the node and leaf split and the gini impurity at each split.

    Example:
    features_DecisionTree(feature='TIMESINCEPANUPDATEDDAYS_PRED_VALUE',data=df_eda,target='RISK_1_5PD30',
    max_depth=10,min_samples_split=0.05,min_samples_leaf=0.1)
    
    -- the features lies between 0 and 4000

    Output:
    
    node_type	threshold	gini_impurity
0	leaf_node	180.0	0.265325
1	split_node	-2.0	0.336103
2	leaf_node	736.0	0.248729
3	leaf_node	362.0	0.273199
4	split_node	-2.0	0.286074
5	split_node	-2.0	0.264598
6	leaf_node	2118.0	0.232021
7	leaf_node	1350.0	0.237274
8	leaf_node	1008.0	0.241281
9	split_node	-2.0	0.233396
10	split_node	-2.0	0.251932
11	split_node	-2.0	0.229915
12	split_node	-2.0	0.210677

    """
    df_feat=data[[feature,target]].copy()
    df_feat = df_feat.drop_duplicates()
    df_feat[feature] = df_feat[feature].replace('XNA',np.nan)
    df_feat.dropna(inplace=True)
    dt_output={'node_type':[],'threshold':[],'gini_impurity':[]}
    clf=DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,criterion='gini')
    clf=clf.fit(pd.DataFrame(df_feat[feature]),df_feat[target])
    
    for j in range(len(clf.tree_.children_left)):
        if clf.tree_.children_left[j]!=clf.tree_.children_right[j]:
            dt_output['node_type'].append('leaf_node')
            if clf.tree_.threshold[j]>2 or clf.tree_.threshold[j]<-2:
                dt_output['threshold'].append(round(clf.tree_.threshold[j],0))
            else:
                dt_output['threshold'].append(round(clf.tree_.threshold[j],3))
            dt_output['gini_impurity'].append(clf.tree_.impurity[j])
        if clf.tree_.children_left[j]==clf.tree_.children_right[j]:
            dt_output['node_type'].append('split_node')
            dt_output['threshold'].append(clf.tree_.threshold[j])
            dt_output['gini_impurity'].append(clf.tree_.impurity[j])
    return pd.DataFrame(dt_output)


#for creating bands in discrete and continous variables
def band_creation(data:pd.DataFrame,table_id:str='SKP_CREDIT_CASE',target:str='RISK_1_5PD30',features_list:list=['TIMESINCEPANUPDATEDDAYS_PRED_VALUE',
                                                                                       'POSFSTQPD30LMONTH_V2_PRED_VALUE'],
                  max_depth:int=10,min_samples_split:float=0.05,min_samples_leaf:float=0.1):
    """
    Brief of the function: To create banding for multiple features from a sample data. We use the function features_DecisionTree here.

    Parameters:
    data: The sample data on which the alogorithm would be run
    target: Target used by the Decision Tree.
    features_list: The list of all the feature for which the porential bands needs to be identified. 
    The features should be present in the provided data.
    max_depth: max_depth of the Decision Tree.
    min_samples_split: Minimum sample required for a potential split in the Decision Tree
    min_samples_leaf: Minimum sample to be present in any leaf

    Returns:
    The functions returns a data frame with the table id and the banding mapped to each of features in the list.
    """ 
    bins={}
    num_list=features_list
    df_num_band=pd.DataFrame(data[[table_id]]).copy()
    df_num_band = df_num_band.drop_duplicates()
    for i in num_list:

        df_feat=data[[table_id,i]].copy()
        df_feat[i] = df_feat[i].replace('XNA',np.nan)
        df_feat.dropna(inplace=True)
        df_feat[i] = pd.to_numeric(df_feat[i])

        dt_output = features_DecisionTree(data=data,feature=i,target=target,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)

        threshold=list(dt_output[dt_output['node_type']=='leaf_node'].sort_values(by='threshold').reset_index(drop=True)['threshold'])
        threshold=list(set(threshold))
        threshold.insert(0,-np.inf)
        threshold.append(np.inf)
        threshold.sort()
        df_feat[i]=pd.cut(x=df_feat[i],bins=threshold)
        bins[i]=threshold
        df_num_band = pd.merge(left=df_num_band,right=df_feat,on=table_id,how='left')
    return df_num_band


#graph with x-axis for distribution and y-axis for risk
def dist_and_target_plot(data:pd.DataFrame,target:str='RISK_1_5PD30',features_list:list=[],table_id:str='SKP_CREDIT_CASE',dist_variable:str='RISK_AGRF30',risk_variable:str='RISK_FPD30',
                  max_depth:int=10,min_samples_split:float=0.05,min_samples_leaf:float=0.1):
    
    if len(features_list )==0:
        basic_info = variables_basic_info(data=data,columns_remove=[table_id,target])
        features_list = list(basic_info[basic_info['variable_dtype']=='num']['variables'])

    df_num_band = band_creation(data=data,target=target,features_list=features_list,max_depth=max_depth,
                                min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    
    ncol=2
    nrow=max(2,math.ceil(len(features_list)/ncol))
    fig, axs = plt.subplots(nrow, ncol,figsize=(15, nrow*3.5))
    
    for i in range(len(features_list)):

        grouped = pd.merge(left=df_num_band,right=data[[table_id,dist_variable,risk_variable]]).groupby(by=features_list[i],
                                                                                                        dropna=False,observed=True).aggregate(
        {dist_variable:'sum',risk_variable:'sum'}).reset_index()
        grouped["RISK_RATE"]=(grouped[risk_variable]*100/grouped[dist_variable]).round(2)
        grouped["DIST"]=(grouped[dist_variable]*100/grouped[dist_variable].sum()).round(0)    
        grouped[features_list[i]] = grouped[features_list[i]].astype('str')   
        grouped[features_list[i]].replace('nan','XNA')

        col = math.ceil((i-1)/2)

        if i%2 ==0:
            sns.barplot(ax=axs[col,0],x=grouped[features_list[i]],y=grouped['DIST'])
            sns.lineplot(ax=axs[col,0].twinx(),x=grouped[features_list[i]].astype(str),y=grouped['RISK_RATE'],color='y')
            axs[col,0].tick_params(axis='x',labelrotation=15)
            plt.tight_layout()

        else:
            sns.barplot(ax=axs[col,1],x=grouped[features_list[i]],y=grouped['DIST'])
            sns.lineplot(ax=axs[col,1].twinx(),x=grouped[features_list[i]].astype(str),y=grouped['RISK_RATE'],color='y')
            axs[col,1].tick_params(axis='x',labelrotation=15)
            plt.tight_layout()

    plt.show()
