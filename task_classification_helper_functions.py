# All helper functions used in O*NET and ORS task classification. 
# Some classifier function and pd.DataFrame re-shaping methods.
# Author: Rebecca Hu
# Last Updated: July 22, 2019

import numpy as numpy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# a list containing all unique GWA
all_gwa = ['Analyzing Data or Information',
 'Provide Consultation and Advice to Others',
 'Guiding, Directing, and Motivating Subordinates',
 'Communicating with Supervisors, Peers, or Subordinates',
 'Making Decisions and Solving Problems',
 'Developing Objectives and Strategies',
 'Resolving Conflicts and Negotiating with Others',
 'Documenting/Recording Information',
 'Communicating with Persons Outside Organization',
 'Interpreting the Meaning of Information for Others',
 'Selling or Influencing Others',
 'Identifying Objects, Actions, and Events',
 'Judging the Qualities of Things, Services, or People',
 'Scheduling Work and Activities',
 'Evaluating Information to Determine Compliance with Standards',
 'Thinking Creatively',
 'Getting Information',
 'Monitor Processes, Materials, or Surroundings',
 'Training and Teaching Others',
 'Staffing Organizational Units',
 'Updating and Using Relevant Knowledge',
 'Performing for or Working Directly with the Public',
 'Establishing and Maintaining Interpersonal Relationships',
 'Coaching and Developing Others',
 'Processing Information',
 'Monitoring and Controlling Resources',
 'Inspecting Equipment, Structures, or Material',
 'Repairing and Maintaining Mechanical Equipment',
 'Controlling Machines and Processes',
 'Performing General Physical Activities',
 'Estimating the Quantifiable Characteristics of Products, Events, or Information',
 'Performing Administrative Activities',
 'Interacting With Computers',
 'Organizing, Planning, and Prioritizing Work',
 'Handling and Moving Objects',
 'Assisting and Caring for Others',
 'Operating Vehicles, Mechanized Devices, or Equipment']

def stemmed_words(doc):
    '''
    Remove punctuation & stopwords & numbers, lowercase, use stemming to remove inflections, create n-grams, calculate TF-IDF matrix

    Parameters:
    ---------------
        doc: str
            the text of a work task

    Returns:
    ---------------
        generator object
    '''
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    analyzer = TfidfVectorizer(
        lowercase = True, 
        stop_words = 'english', 
        ngram_range = (1, 5), 
        token_pattern=u'(?ui)[a-z]*[a-z]+[a-z]*').build_analyzer()
    return (stemmer.stem(w) for w in analyzer(doc))

def logit_pipe(X, y):
    '''
    fits the logistic regression pipeline

    Parameters:
    ---------------
        X: array-like
            a list that contains the data the model should be fit on. The training data should be the text of work tasks.
        
        y: array-like
            a list that contains the labels of the the data. The length of y should be the same as the length of X. 
            The labels should be 1 if the task is in the target class, 0 otherwise. 

    Returns:
    ---------------
        sklearn.pipeline.Pipeline 
    '''
    lr_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer=stemmed_words)),
        ('logit', LogisticRegression())])

    return lr_pipe.fit(X, y)

def linearsvc_pipe(X, y):
    '''
    fits the support vector machine with linear kernel pipeline

    Parameters:
    ---------------
        X: array-like
            a list that contains the data the model should be fit on. The training data should be the text of work tasks.
        
        y: array-like
            a list that contains the labels of the the data. The length of y should be the same as the length of X. 
            The labels should be 1 if the task is in the target class, 0 otherwise. 

    Returns:
    ---------------
        sklearn.pipeline.Pipeline 
    '''
    svc_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer=stemmed_words)),
        ('linear svc', SVC(C = 0.4, kernel = 'linear'))])

    return svc_pipe.fit(X, y)

def randomforest_pipe(X, y):
    '''
    fits the random forest pipeline

    Parameters:
    ---------------
        X: array-like
            a list that contains the data the model should be fit on. The training data should be the text of work tasks.
        
        y: array-like
            a list that contains the labels of the the data. The length of y should be the same as the length of X. 
            The labels should be 1 if the task is in the target class, 0 otherwise. 

    Returns:
    ---------------
        sklearn.pipeline.Pipeline 
    '''
    rf_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer=stemmed_words)),
        ('linear svc', RandomForestClassifier(n_estimators = 1000, min_samples_leaf = 3, max_depth = 100))])

    return rf_pipe.fit(X, y)

def generate_gwa_column_names(df, is_prediction = False):
    '''
    This function is pretty much only necessary because it gets difficult to index pd.Dataframes by column when the column headers are just numbers. 

    Parameters:
    ---------------
        df: pd.DataFrame
            a pandas dataframe that you want to generate new column headers for

        is_prediction: bool
            a Boolean value that determines whether the resulting column headers are "GWA 1", "GWA 2", etc. or "Predicted GWA 1", "Predicted GWA 2", etc. 
            True if the target dataframe contains predicted values, False if the df contains true labels

    Returns:
    ---------------
        list: new column headers
    '''
    new_headers = []
    for i in range(df.shape[1]):
        if is_prediction:
            new_headers.append('Predicted GWA' + ' ' + str(i + 1))
        else:
            new_headers.append('GWA' + ' ' + str(i + 1))
    return new_headers

def reformat_to_ez(df, is_prediction = False):
    '''
    reformats the original data into a pd.DataFrame that is easier to read. 
    In the original data, each task has a new row for each additional label it has. 
    This function creates a dataframe that displays each unique task in a row, and presents each of its labels clearly.

    Parameters:
    ---------------
        df: pd.DataFrame
            a pandas dataframe with columns "Task" and "GWA". 
            Task column contains text describing an occupational task, each task is not necessarily unique. 
            GWA column contains text that describes the General Work Activity that the associated task was labeled in the O*NET data.
        
        is_prediction: bool
            a Boolean value that determines whether the resulting column headers are "GWA 1", "GWA 2", etc. or "Predicted GWA 1", "Predicted GWA 2", etc. 
            True if the target dataframe contains predicted values, False if the df contains true labels

    Returns:
    ---------------
        pd.DataFrame: reformatted dataframe
    '''
    rf_dict = {}
    for task in df.Task.unique():
        rf_dict[task] = list(df[df['Task'] == task].GWA)
        
    rf_df = pd.DataFrame.from_dict(rf_dict, orient='index')
    rf_df.columns = generate_gwa_column_names(rf_df, is_prediction)
    return rf_df

def ez_to_onehot(df, is_prediction = False):
    '''
    reformats a pd.DataFrame in the easy-to-read format to a one-hot encoded format where each GWA has its own column. 
    The resulting dataframe is a sparse matrix with the task texts as the index.

     Parameters:
    ---------------
        df: pd.DataFrame
            a pandas dataframe in "easy-to-read" format, with one row for each unique task in the index and each row contains the task's associated GWAs (One GWA in each column). 
            Not all tasks have the same number of GWAs, so there are Nan values.

        is_prediction: bool
            a Boolean value that determines whether the resulting column headers are "GWA 1", "GWA 2", etc. or "Predicted GWA 1", "Predicted GWA 2", etc. 
            True if the target dataframe contains predicted values, False if the df contains true labels

    Returns:
    ---------------
        pd.DataFrame: reformatted dataframe
    '''
    dummies = {}

    for gwa in all_gwa:
        one_hot = []
        for i, row in df.iterrows():
            if gwa in [row[col] for col in df.columns]:
                one_hot.append(1)
            else:
                one_hot.append(0)
        dummies[gwa] = one_hot
    onehot_df = pd.DataFrame(dummies, index = df.index)
    return onehot_df

def onehot_to_ez(df, is_prediction = False):
    '''
    reformats a sparse one-hot encoded pd.DataFrame into "easy-to-read" format.
    Resulting dataframe has one row per unique task, and each row contains the task's associated GWAs (one GWA per column)
    
    Parameters:
    ------------
        df: pd.DataFrame
            a sparse pandas dataframe with one row per task and one column for each unique GWA.

       is_prediction: bool
            a Boolean value that determines whether the resulting column headers are "GWA 1", "GWA 2", etc. or "Predicted GWA 1", "Predicted GWA 2", etc. 
            True if the target dataframe contains predicted values, False if the df contains true labels
        
    Returns:
    ------------
        pd.DataFrame: reformatted dataframe
    '''
    d = {}
    for index, row in df.iterrows():
        preds = []
        for col in df.columns:
            if row[col] == 1:
                preds.append(col)
        d[index] = preds   

    ez_df = pd.DataFrame.from_dict(d, orient='index')
    ez_df.columns = generate_gwa_column_names(ez_df, is_prediction)
    return ez_df

def train_classifier(df, gwa, pipeline_method):
    '''
    oversamples target gwa, undersamples rest, then fits models based on balanced dataset
    
    Parameters:
    -----------
        df: pd.DataFrame
            a dataframe containing the one-hot encoded training data (index holds task, one column per GWA, column headers are GWA)
        
        gwa: string
            the name of the target gwa

        pipeline: method
            one of the pipeline methods: logit_pipe, linearsvc_pipe, randomforest_pipe
        
    Returns: 
    -----------
        sklearn.pipeline.Pipeline 
    '''
    target_size = str(df[df[gwa] == 1].shape[0])
    
    if int(target_size) < 100: #If the class is small (less than 100)
        pos_sample_size = int(target_size)
        neg_sample_size = int(target_size) + 100
    
    pos_sample_size = int(target_size[0] + ('0' * (len(target_size) - 1)))
    neg_sample_size = pos_sample_size + int('1' +'0' * (len(target_size) - 1))
    
    #Oversample target class and undersample rest
    pos_sample = df[df[gwa] == 1][[gwa]].sample(n = pos_sample_size)
    neg_sample = df[df[gwa] == 0][[gwa]].sample(n = neg_sample_size)
    
    balanced = pd.concat([pos_sample, neg_sample]).sample(frac=1)

    return pipeline_method(balanced.index, balanced[gwa])
