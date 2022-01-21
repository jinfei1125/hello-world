'''
Reference: 

*Twint Package*
Pacage Introduction: 
    https://github.com/twintproject/twint
Configuration Options: 
    https://github.com/twintproject/twint/wiki/Configuration

*Other information*
Debugging: 
    https://github.com/twintproject/twint/issues/1121#issuecomment-773521415
Combine json files into one json: 
    https://www.freecodecamp.org/news/
    how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/

Notes: Now using Twitter Data requires a Twitter Developer Account, 
    which takes several days to get approved.
    So we use Twint package that can help us scrape Twitter data 
    without Twitter Developer Account.

'''
import os
import pandas as pd 

import twint
import nest_asyncio
nest_asyncio.apply()

def get_tweets(username=None, search=None, since=None, 
                until=None, output=None):
    '''
    Get Tweets with Twint package.
    Input:
        username(str): User of interest
        search(str): search terms
        since(str): filter Tweets sent since date 'yyyy-mm-dd'
        until(str): filter Tweets sent until date 'yyyy-mm-dd'
        output(str): the file name of the output json file
    Output:
        A json file containing tweets.
    '''
    # Configure
    c = twint.Config()
    if username:
        c.Username = username
    if search:
        c.Search = search
    if since:
        c.Since = since
    if until:
        c.Until = until
    if output:
        c.Output = output
    c.Store_json = True 

    # Run
    twint.run.Search(c)


def get_tweets_from_multiple_users(df, folder, search=None, 
                                since=None, until=None):
    '''
    Get Tweets from ultiple users
    Inputs:
        df(dataframe): a dataframe containing governors' state, 
                name, github handle
        search(str): search terms
        since(str): filter Tweets sent since date 'yyyy-mm-dd'
        until(str): filter Tweets sent until date 'yyyy-mm-dd'
        output(str): the file name of the output json file
        folder(str): a folder name to store multiple json files
    Output:
        A folder of json files containing all Tweets from multiple users
    '''
    # crete a list to store the output file names
    state_lst = list(df['State'])
    handle_lst = list(df['Twitter Handle'])

    for i in range(len(handle_lst)):
        state = state_lst[i]
        handle = handle_lst[i][1:] # delete @
        output_path = r'data//{}//{}.json'.format(folder, state)

        get_tweets(handle, search, since, until, output_path)




        

