SELECT  province_state, country_region, date, SUM(confirmed) AS confirmed_state, SUM(deaths) AS deaths_state
FROM `bigquery-public-data.covid19_jhu_csse.summary`
WHERE country_region = 'US'
GROUP BY province_state, date, country_region