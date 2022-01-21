import sqlite3

conn = sqlite3.connect('course_information.sqlite3')
c = conn.cursor()

#1
s1 ='''
    SELECT title
    FROM courses
    WHERE dept = ?
    '''
args1 = ['CMSC']

# 2
s2 = '''
    SELECT dept, course_num, section_id
    FROM courses c INNER JOIN sections s 
    ON c.course_id = s.course_id
    INNER JOIN meeting_patterns mp
    ON s.meeting_pattern_id = mp.meeting_pattern_id
    WHERE mp.day = ? AND mp.time_start = ?'''

args2 = ['MWF',1030]

# 3
s3 ='''
    SELECT dept, course_num, section_num
    FROM courses c INNER JOIN sections s
    ON c.course_id = s.course_id
    INNER JOIN meeting_patterns mp
    ON s.meeting_pattern_id = mp.meeting_pattern_id
    WHERE s.building_code = ? 
        AND mp.time_start >= ?
        AND mp.time_end <= ?'''
args3 = ['RY', 1030, 1500]

# 4
s4 ='''
    SELECT dept, course_num, title
    FROM courses c 
    INNER JOIN catalog_index ci 
    ON c.course_id = ci.course_id
    WHERE ci.word = ? OR ci.word = ?'''
args4 = ['programming','abstraction']


c.execute(s1, args1).fetchall()

c.execute(s2, args2).fetchall()

c.execute(s3, args3).fetchall()

c.execute(s4, args4).fetchall()