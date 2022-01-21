# Case When
SELECT order_number, product_code, price_item,
CASE WHEN price_item <= 50 THEN 'low'
    WHEN price_item > 50 AND price_item <= 100 THEN 'medium'
    WHEN price_item > 100 THEN 'high'
    ELSE 'undefined'
END AS cost_level
FROM order_details
ORDER BY order_number, product_code
LIMIT 10;


# Subquery
SELECT show_id, rating
FROM (SELECT * FROM ratings WHERE rating > 8)
AS subquery
ORDER BY show_id
LIMIT 10;

# UNION example to set type
SELECT 'Customer' AS Type, ContactName, City, Country
FROM Customers
UNION
SELECT 'Supplier', ContactName, City, Country
FROM Suppliers;

# IN 
SELECT AVG(Quantity)
FROM OrderDetails
GROUP BY ProductID
HAVING ProductID IN (SELECT ProductID FROM OrderDetails WHERE Quantity = 10);

