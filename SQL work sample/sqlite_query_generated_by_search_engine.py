'''
Based on the input of search engine, constructed sql code to query database

Jinfei Zhu
'''

from math import radians, cos, sin, asin, sqrt, ceil
import sqlite3
import os


# Use this filename for the database
DATA_DIR = os.path.dirname(__file__)
DATABASE_FILENAME = os.path.join(DATA_DIR, 'course_information.sqlite3')


def find_courses(args_from_ui):
    '''
    Takes a dictionary containing search criteria and returns courses
    that match the criteria.  The dictionary will contain some of the
    following fields:

      - dept a string
      - day is list of strings
           -> ["'MWF'", "'TR'", etc.]
      - time_start is an integer in the range 0-2359
      - time_end is an integer an integer in the range 0-2359
      - enrollment is a pair of integers
      - walking_time is an integer
      - building_code ia string
      - terms is a list of strings string: ["quantum", "plato"]

    Returns a pair: an ordered list of attribute names and a list the
     containing query results.  Returns ([], []) when the dictionary
     is empty.
    '''

    assert_valid_input(args_from_ui)

    if len(args_from_ui) == 0:
        return ([], [])

    else:
        conn = sqlite3.connect(DATABASE_FILENAME)
        c = conn.cursor()
        conn.create_function("time_between", 4, compute_time_between)
        query, args = construct_query(args_from_ui)
        query_result = c.execute(query, args).fetchall()
        header = get_header(c.execute(query, args))
        return header, query_result


def construct_query(args_from_ui):
    '''
    Determine the output attributes given the keys of the input dictionary.

    Input: 
        args_from_ui(dic): input dictionary
    Return:
        (query, args)(tpl): a tuple store the query, args
    '''

    if 'building_code' in args_from_ui or 'walking_time' in args_from_ui:
        output_attr = ['courses.dept',
                       'courses.course_num',
                       'courses.title',
                       'sections.section_num',
                       'meeting_patterns.day',
                       'meeting_patterns.time_start',
                       'meeting_patterns.time_end',
                       'sections.enrollment',
                       'a.building_code',
                       'time_between(a.lon, a.lat, b.lon, b.lat) AS walking_time'] 
        # table a is the survey result, table b is reference       
        table = '''
                FROM courses JOIN sections JOIN meeting_patterns 
                JOIN gps a JOIN gps b
                ON courses.course_id = sections.course_id 
                AND sections.meeting_pattern_id = meeting_patterns.meeting_pattern_id
                AND sections.building_code = a.building_code
                '''
        table = add_catalog_index_if_items(args_from_ui, table)
        sec_bool = True

    elif 'day' in args_from_ui \
         or 'enrollment' in args_from_ui \
         or 'time_start' in args_from_ui \
         or 'time_end' in args_from_ui\
         and 'building_code' not in args_from_ui \
         and 'walking_time' not in args_from_ui:
        output_attr = ['courses.dept',
                       'courses.course_num',
                       'courses.title',
                       'sections.section_num',
                       'meeting_patterns.day',
                       'meeting_patterns.time_start',
                       'meeting_patterns.time_end',
                       'sections.enrollment']

        table = '''
                FROM courses JOIN sections JOIN meeting_patterns 
                ON courses.course_id = sections.course_id 
                AND sections.meeting_pattern_id = meeting_patterns.meeting_pattern_id
                '''
        table = add_catalog_index_if_items(args_from_ui, table)
        sec_bool = True


    elif 'terms' in args_from_ui or 'dept' in args_from_ui \
         and 'day' not in args_from_ui \
         and 'enrollment' not in args_from_ui \
         and 'time_start' not in args_from_ui \
         and 'time_end' not in args_from_ui \
         and 'building_code' not in args_from_ui \
         and 'walking_time' not in args_from_ui:
        output_attr = ['courses.dept',
                       'courses.course_num',
                       'courses.title']
        sec_bool = False

        if len(args_from_ui) == 1 and 'dept' in args_from_ui:
            table = "FROM courses"

        elif (len(args_from_ui) == 1 and 'terms' in args_from_ui) \
            or (len(args_from_ui) == 2 and 'terms' in args_from_ui \
            and 'dept' in args_from_ui):
            table = '''FROM courses JOIN catalog_index 
                    ON courses.course_id = catalog_index.course_id'''


    select_clause = "SELECT " + ", ".join(output_attr)

    where_clause, args = create_where(args_from_ui)

    group_by_clause, args  = create_group_by(args_from_ui, args, sec_bool)

    query = select_clause + " " + table + " " + where_clause + " " + group_by_clause     

    return query, args

def create_where(args_from_ui):
    '''
    Create where_clause for the sql query and args to execute.
    
    Input:
        args_from_ui(dic): input dictionary
    Returns:
        where_clause, args (tpl): the where clause and arguments for sql query

    '''
    where_lst = []
    args = ()

    if 'terms' in args_from_ui:

        ques = '?'
        if len(args_from_ui['terms']) > 1:
            ques = '?, '* (len(args_from_ui['terms'])-1) + ques
        where_lst.append('catalog_index.word in ({})'.format(ques))
        word_tpl = ()
        for i in range(len(args_from_ui['terms'])):
            word_tpl += (args_from_ui['terms'][i],)
        args = args + word_tpl

    if 'dept' in args_from_ui:
        where_lst.append('courses.dept = ? ')
        args = args + (args_from_ui['dept'],)

    if 'day' in args_from_ui:
        ques = '?'
        if len(args_from_ui['day']) > 1:
            ques = '?, '* (len(args_from_ui['day'])-1) + ques
        where_lst.append('meeting_patterns.day in ({})'.format(ques))
        day_tpl = ()
        for i in range(len(args_from_ui['day'])):
            day_tpl += (args_from_ui['day'][i],)
        args = args + day_tpl

    if 'enrollment' in args_from_ui:
        where_lst.append('sections.enrollment BETWEEN ? AND ? ')
        [enroll_min, enroll_max] = args_from_ui['enrollment']
        args = args + (enroll_min, enroll_max)

    if 'time_start' in args_from_ui:
        where_lst.append('meeting_patterns.time_start >= ? ')
        args = args + (args_from_ui['time_start'],)

    if 'time_end' in args_from_ui:
        where_lst.append('meeting_patterns.time_end <= ? ')
        args = args + (args_from_ui['time_end'],)

    if 'building_code' in args_from_ui and 'walking_time' in args_from_ui:
        where_lst.append('b.building_code = ?')
        where_lst.append('walking_time <= ?')
        args = args + (args_from_ui['building_code'],args_from_ui['walking_time'])


    where_clause = "WHERE " + "AND ".join(where_lst) 

    return where_clause, args


def add_catalog_index_if_items(args_from_ui, table):
    '''
    Add the table 'catalog_index' to table clause if items in args_from_ui
    Inputs:
        args_from_ui(dic): input dictionary
        table(str): the table of sql query
    Return:
        table(str): updated table of sql quer
    '''
    if 'terms' in args_from_ui:
    # add 'catalog_index' to the query
        table = table[:table.find("ON") ] + " JOIN catalog_index " \
                    + table[table.find("ON"):] \
                    + "AND courses.course_id = catalog_index.course_id"

    return table

def create_group_by(args_from_ui, args, sec_bool):
    '''
    Create group_by clause and add attributes to args of sql query
    Inputs:
        args_from_ui(dic): input dictionary
        args(tpl): the tuple stored all parameters for execute method
        sec_bool(bool): the boolean value indicating if section_num is in output
    Returns:
        group_by_clause, args (tpl): the group_by clause and arguments for sql query
    '''

    group_by_clause = ''
    if 'terms' in args_from_ui:
        group_by_clause += 'GROUP BY courses.course_id ' 
        if sec_bool:
            group_by_clause += ', sections.section_num '

        group_by_clause += 'HAVING COUNT(DISTINCT catalog_index.word) = ?'
        args += (len(args_from_ui['terms']),)
    return group_by_clause, args


########### auxiliary functions #################

def assert_valid_input(args_from_ui):
    '''
    Verify that the input conforms to the standards set in the
    assignment.
    '''

    assert isinstance(args_from_ui, dict)

    acceptable_keys = set(['time_start', 'time_end', 'enrollment', 'dept',
                           'terms', 'day', 'building_code', 'walking_time'])
    assert set(args_from_ui.keys()).issubset(acceptable_keys)

    # get both buiding_code and walking_time or neither
    has_building = ("building_code" in args_from_ui and
                    "walking_time" in args_from_ui)
    does_not_have_building = ("building_code" not in args_from_ui and
                              "walking_time" not in args_from_ui)

    assert has_building or does_not_have_building

    assert isinstance(args_from_ui.get("building_code", ""), str)
    assert isinstance(args_from_ui.get("walking_time", 0), int)

    # day is a list of strings, if it exists
    assert isinstance(args_from_ui.get("day", []), (list, tuple))
    assert all([isinstance(s, str) for s in args_from_ui.get("day", [])])

    assert isinstance(args_from_ui.get("dept", ""), str)

    # terms is a non-empty list of strings, if it exists
    terms = args_from_ui.get("terms", [""])
    assert terms
    assert isinstance(terms, (list, tuple))
    assert all([isinstance(s, str) for s in terms])

    assert isinstance(args_from_ui.get("time_start", 0), int)
    assert args_from_ui.get("time_start", 0) >= 0

    assert isinstance(args_from_ui.get("time_end", 0), int)
    assert args_from_ui.get("time_end", 0) < 2400

    # enrollment is a pair of integers, if it exists
    enrollment_val = args_from_ui.get("enrollment", [0, 0])
    assert isinstance(enrollment_val, (list, tuple))
    assert len(enrollment_val) == 2
    assert all([isinstance(i, int) for i in enrollment_val])
    assert enrollment_val[0] <= enrollment_val[1]


def compute_time_between(lon1, lat1, lon2, lat2):
    '''
    Converts the output of the haversine formula to walking time in minutes
    '''
    meters = haversine(lon1, lat1, lon2, lat2)

    # adjusted downwards to account for manhattan distance
    walk_speed_m_per_sec = 1.1
    mins = meters / (walk_speed_m_per_sec * 60)

    return int(ceil(mins))


def haversine(lon1, lat1, lon2, lat2):
    '''
    Calculate the circle distance between two points
    on the earth (specified in decimal degrees)
    '''
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))

    # 6367 km is the radius of the Earth
    km = 6367 * c
    m = km * 1000
    return m


def get_header(cursor):
    '''
    Given a cursor object, returns the appropriate header (column names)
    '''
    header = []

    for i in cursor.description:
        s = i[0]
        if "." in s:
            s = s[s.find(".")+1:]
        header.append(s)

    return header
