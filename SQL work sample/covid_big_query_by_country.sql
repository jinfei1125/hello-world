SELECT country_region, date, SUM(confirmed) AS confirmed_us, SUM(deaths) AS deaths_us
FROM `bigquery-public-data.covid19_jhu_csse.summary`
WHERE country_region = 'US'
GROUP BY date, country_region