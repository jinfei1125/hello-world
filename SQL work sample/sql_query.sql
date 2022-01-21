# 1
SELECT title
FROM courses
WHERE dept = 'CMSC';

# 2
SELECT dept, course_num, section_id
FROM courses c INNER JOIN sections s 
ON c.course_id = s.course_id
INNER JOIN meeting_patterns mp
ON s.meeting_pattern_id = mp.meeting_pattern_id
WHERE mp.day = 'MWF' AND mp.time_start = 1030;

# 3
SELECT dept, course_num, section_num
FROM courses c INNER JOIN sections s
ON c.course_id = s.course_id
INNER JOIN meeting_patterns mp
ON s.meeting_pattern_id = mp.meeting_pattern_id
WHERE s.building_code = 'RY' 
    AND mp.time_start >= 1030
    AND mp.time_end <= 1500;

# 4
SELECT dept, course_num, title
FROM courses c 
INNER JOIN catalog_index ci 
ON c.course_id = ci.course_id
WHERE ci.word = 'programming' OR ci.word = 'abstraction';