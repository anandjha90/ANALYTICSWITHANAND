Here are some challenging SQL interview questions:

1. How can you find the second-lowest salary in a table without using MIN?
  
SELECT salary FROM employees 
WHERE salary > (SELECT DISTINCT salary FROM employees ORDER BY salary ASC LIMIT 1) 
ORDER BY salary ASC LIMIT 1;

2. Write a query to find employees who report to the same manager and have the exact same salary.
  
SELECT e1.* FROM employees e1 
JOIN employees e2 ON e1.manager_id = e2.manager_id 
WHERE e1.salary = e2.salary AND e1.employee_id <> e2.employee_id;
  
3. Identify rows where a specific column has missing or null values and replace them with the column’s average.
  
UPDATE table_name 
SET column_name = (SELECT AVG(column_name) FROM table_name WHERE column_name IS NOT NULL) 
WHERE column_name IS NULL;

4. Write a query to find the employees whose salaries rank in the top 5% without using percentile functions.
  
SELECT * FROM employees 
WHERE salary >= (SELECT salary FROM employees ORDER BY salary DESC LIMIT (SELECT COUNT(*) FROM employees) * 5 / 100);
  
5. Calculate a running total of sales in a table.

SELECT id, sales, SUM(sales) OVER (ORDER BY id) AS running_total 
FROM sales_table;

6. Find employees who have worked in more than one department.

SELECT employee_id FROM employee_department 
GROUP BY employee_id 
HAVING COUNT(DISTINCT department_id) > 1;

  
7. Write a query to compare each employee’s salary with the average salary of their department.

SELECT employee_id, salary, 
AVG(salary) OVER (PARTITION BY department_id) AS avg_dept_salary 
FROM employees;
  
  
8. Identify departments that currently have no employees.
SELECT department_id FROM departments 
WHERE department_id NOT IN (SELECT DISTINCT department_id FROM employees);

9. Write a query to display the total sales for each month in a pivot-like format.
  
SELECT 
 SUM(CASE WHEN month = 'January' THEN sales END) AS January, 
 SUM(CASE WHEN month = 'February' THEN sales END) AS February, 
 SUM(CASE WHEN month = 'March' THEN sales END) AS March 
FROM sales_table;
  
10. Determine employees whose salaries increased by more than 20% compared to their last salary.
  
SELECT e.employee_id FROM employees e 
JOIN salary_history s ON e.employee_id = s.employee_id 
WHERE e.salary > 1.2 * s.previous_salary;

11. Find employees who have worked in the most departments.

SELECT employee_id, COUNT(DISTINCT department_id) AS dept_count 
FROM employee_department_history 
GROUP BY employee_id 
ORDER BY dept_count DESC LIMIT 1;
  
12. Identify customers who made purchases in consecutive months.

SELECT customer_id 
FROM orders 
GROUP BY customer_id 
HAVING COUNT(DISTINCT DATE_FORMAT(order_date, '%Y-%m')) = 
(SELECT COUNT(DISTINCT DATE_FORMAT(order_date, '%Y-%m')) FROM orders);
  
13. Calculate the average session duration from login/logout timestamps.

SELECT user_id, AVG(TIMESTAMPDIFF(MINUTE, login_time, logout_time)) AS avg_session_duration 
FROM session_logs 
GROUP BY user_id;
  
14. Retrieve the least sold product in each category.

SELECT category_id, product_id, SUM(sales) AS total_sales 
FROM products p 
JOIN sales s ON p.product_id = s.product_id 
GROUP BY category_id, product_id 
HAVING total_sales = (SELECT MIN(SUM(sales)) FROM sales s2 WHERE p2.category_id = p.category_id);
  
15. Show the cumulative sales for each product by month.

SELECT product_id, month, 
SUM(sales) OVER (PARTITION BY product_id ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_sales 
FROM sales;
  
16. Find missing order IDs in a sequential orders table.

SELECT order_id + 1 AS missing_order_id 
FROM orders o1 
WHERE NOT EXISTS (SELECT order_id FROM orders o2 WHERE o2.order_id = o1.order_id + 1);  
  
17. Identify employees who received the same bonus for three years.

SELECT employee_id 
FROM bonus_history 
GROUP BY employee_id, bonus_amount 
HAVING COUNT(*) = 3 AND MAX(year) - MIN(year) = 2;
  
18. Find the product with the lowest sales-to-stock ratio.

SELECT product_id, (SUM(sales) / stock) AS sales_to_stock_ratio 
FROM products p 
JOIN sales s ON p.product_id = s.product_id 
GROUP BY product_id 
ORDER BY sales_to_stock_ratio ASC LIMIT 1;
  
19. Show the top and bottom-performing sales regions.

SELECT region, SUM(sales) AS total_sales 
FROM sales 
GROUP BY region 
ORDER BY total_sales DESC LIMIT 1 

UNION ALL 

SELECT region, SUM(sales) AS total_sales 
FROM sales 
GROUP BY region 
ORDER BY total_sales ASC LIMIT 1;
  
20. Rank employees by revenue contribution within each team.

SELECT team_id, employee_id, 
RANK() OVER (PARTITION BY team_id ORDER BY SUM(sales) DESC) AS rank 
FROM employees e 
JOIN sales s ON e.employee_id = s.employee_id 
GROUP BY team_id, employee_id;

21. Find the first purchase of each customer (excluding their first-ever order).

SELECT customer_id, order_id, order_date 
FROM ( 
 SELECT customer_id, order_id, order_date, 
 RANK() OVER (PARTITION BY customer_id ORDER BY order_date) AS rnk 
 FROM orders 
) ranked_orders 
WHERE rnk = 2; 

22. Identify employees who never reported to a manager but are still in the system.

SELECT employee_id, name 
FROM employees 
WHERE manager_id IS NULL AND employee_id NOT IN (SELECT manager_id FROM employees WHERE manager_id IS NOT NULL); 

23. Retrieve products that have been sold in every quarter of the last year.

SELECT product_id 
FROM sales 
WHERE YEAR(sale_date) = YEAR(CURRENT_DATE) - 1 
GROUP BY product_id 
HAVING COUNT(DISTINCT QUARTER(sale_date)) = 4; 

24. Find departments where every employee earns above the department’s average salary.

SELECT department_id 
FROM employees e1 
WHERE NOT EXISTS ( 
 SELECT 1 FROM employees e2 
 WHERE e1.department_id = e2.department_id 
 AND e2.salary < (SELECT AVG(salary) FROM employees e3 WHERE e3.department_id = e1.department_id) 
); 

25. Show the three most consistent customers (who made purchases every month for the past year).

SELECT customer_id 
FROM orders 
WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR) 
GROUP BY customer_id 
HAVING COUNT(DISTINCT DATE_FORMAT(order_date, '%Y-%m')) = 12 
ORDER BY customer_id 
LIMIT 3; 

26. Find pairs of customers who have ordered the exact same products in the last three months.

SELECT o1.customer_id AS customer_1, o2.customer_id AS customer_2 
FROM orders o1 
JOIN orders o2 ON o1.product_id = o2.product_id AND o1.customer_id < o2.customer_id 
WHERE o1.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 3 MONTH) 
GROUP BY o1.customer_id, o2.customer_id 
HAVING COUNT(DISTINCT o1.product_id) = (SELECT COUNT(DISTINCT product_id) 
FROM orders 
WHERE customer_id = o1.customer_id 
AND order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 3 MONTH)); 

27. Calculate the moving average of sales for the last three months for each product.

SELECT product_id, sale_date, 
AVG(total_sales) OVER (PARTITION BY product_id ORDER BY sale_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg 
FROM ( 
 SELECT product_id, DATE_FORMAT(sale_date, '%Y-%m') AS sale_date, SUM(sales) AS total_sales 
 FROM sales 
 GROUP BY product_id, sale_date 
) sales_data; 

28. Rank employees based on the total revenue they generated, but reset the ranking each year.

SELECT employee_id, YEAR(sale_date) AS year, 
RANK() OVER (PARTITION BY YEAR(sale_date) ORDER BY SUM(revenue) DESC) AS rank 
FROM sales 
GROUP BY employee_id, year;

29. Find the most active user by the longest total session duration.

SELECT user_id, SUM(session_duration) AS total_duration 
FROM user_sessions GROUP BY user_id 
ORDER BY total_duration DESC LIMIT 1;

30. Identify products that were only purchased on weekends.

SELECT DISTINCT product_id FROM orders 
WHERE WEEKDAY(order_date) IN (5,6);

31. Retrieve employees who have the same salary progression pattern.
  
SELECT employee_id FROM ( 
 SELECT employee_id, GROUP_CONCAT(salary ORDER BY year) AS salary_pattern 
 FROM salary_history GROUP BY employee_id 
) t GROUP BY salary_pattern HAVING COUNT(*) > 1;  

32. Find customers who placed exactly the same order more than once.

SELECT customer_id, order_details FROM orders 
GROUP BY customer_id, order_details 
HAVING COUNT(*) > 1;
 

33. List top 3 customers who have spent the highest amount every year.

SELECT year(order_date), customer_id, SUM(amount) AS total_spent 
FROM orders GROUP BY year(order_date), customer_id 
ORDER BY year(order_date), total_spent DESC LIMIT 3;

34. Identify consecutive absent days for employees exceeding 3 days.

SELECT employee_id FROM ( 
 SELECT employee_id, 
 DATEDIFF(end_date, start_date) AS absent_days 
 FROM attendance WHERE status = 'Absent' 
) t WHERE absent_days > 3;

35. Find the average order value change compared to the previous month.

SELECT month, AVG(order_value) - LAG(AVG(order_value)) OVER (ORDER BY month) AS value_change 
FROM orders GROUP BY month;
  
36. Retrieve categories where at least one product has never been sold.

SELECT category_id FROM products 
WHERE category_id NOT IN (SELECT DISTINCT category_id FROM orders);

37. Calculate the average number of items per order, excluding outliers.

SELECT ROUND(AVG(item_count), 2) AS avg_items 
FROM (SELECT order_id, COUNT(*) AS item_count FROM order_items GROUP BY order_id) t 
WHERE item_count BETWEEN 5 AND 95;
  
38. Find employees whose total working hours are below the team average.

SELECT employee_id FROM work_hours 
WHERE total_hours < (SELECT AVG(total_hours) FROM work_hours);

39. Identify products that have never been purchased by any customer.

SELECT product_id 
FROM products 
WHERE product_id NOT IN (SELECT DISTINCT product_id FROM orders);

40. Retrieve the second-highest salary in each department.

SELECT department_id, MAX(salary) AS second_highest_salary 
FROM (SELECT department_id, salary, 
 RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS salary_rank 
 FROM employees) subquery 
WHERE salary_rank = 2 
GROUP BY department_id;
  
41. Find customers who made purchases in consecutive months this year.
  
SELECT customer_id 
FROM (SELECT customer_id, MONTH(order_date) AS order_month, 
 LAG(MONTH(order_date)) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_month 
 FROM orders 
 WHERE YEAR(order_date) = YEAR(CURDATE())) subquery 
WHERE order_month - prev_month = 1 
GROUP BY customer_id;

42. Calculate the average session duration for users, excluding their first login.
  
SELECT user_id, AVG(session_duration) AS avg_session_duration 
FROM (SELECT user_id, session_duration, 
 ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_time) AS session_rank 
 FROM user_sessions) subquery 
WHERE session_rank > 1 
GROUP BY user_id;
  
43. Show the top 3 performing products based on sales revenue for each quarter.
  
SELECT quarter, product_id, SUM(sales) AS total_sales 
FROM (SELECT product_id, sales, 
 CASE 
 WHEN MONTH(sale_date) BETWEEN 1 AND 3 THEN 'Q1' 
 WHEN MONTH(sale_date) BETWEEN 4 AND 6 THEN 'Q2' 
 WHEN MONTH(sale_date) BETWEEN 7 AND 9 THEN 'Q3' 
 ELSE 'Q4' 
 END AS quarter 
 FROM sales) subquery 
GROUP BY quarter, product_id 
ORDER BY quarter, total_sales DESC LIMIT 3;
 
44. Generate a report of employees who have never received a promotion.

SELECT employee_id 
FROM employees 
WHERE employee_id NOT IN (SELECT DISTINCT employee_id FROM promotions);
  
45. Identify the months with the highest and lowest revenue in the past year.
SELECT MONTH(sale_date) AS sale_month, SUM(sales) AS monthly_revenue 
FROM sales 
WHERE sale_date > DATE_SUB(CURDATE(), INTERVAL 1 YEAR) 
GROUP BY sale_month 
ORDER BY monthly_revenue DESC LIMIT 1 

UNION ALL 

SELECT MONTH(sale_date), SUM(sales) 
FROM sales 
WHERE sale_date > DATE_SUB(CURDATE(), INTERVAL 1 YEAR) 
GROUP BY MONTH(sale_date) 
ORDER BY monthly_revenue ASC LIMIT 1;

46. Find the percentage of orders that were delivered late in the last 6 months.

SELECT ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE order_date > DATE_SUB(CURDATE(), INTERVAL 6 MONTH)), 2) AS late_percentage 
FROM orders 
WHERE delivery_date > expected_delivery_date AND order_date > DATE_SUB(CURDATE(), INTERVAL 6 MONTH);

47. Find the employees who have worked for the longest consecutive period without any leave.

SELECT employee_id, MAX(end_date - start_date) AS longest_period 
FROM attendance_logs 
WHERE status = 'Present' 
GROUP BY employee_id;

48. Identify the top 5 customers who have made the highest number of unique product purchases in the last quarter.

SELECT customer_id, COUNT(DISTINCT product_id) AS unique_products 
FROM orders 
WHERE order_date BETWEEN DATE_SUB(CURDATE(), INTERVAL 3 MONTH) AND CURDATE() 
GROUP BY customer_id 
ORDER BY unique_products DESC LIMIT 5;
  
49. Calculate the average time between two consecutive logins for users in the past week.

SELECT user_id, AVG(TIMESTAMPDIFF(SECOND, login_time, LEAD(login_time) OVER (PARTITION BY user_id ORDER BY login_time))) AS avg_time_diff 
FROM login_logs 
WHERE login_time > DATE_SUB(CURDATE(), INTERVAL 7 DAY) 
GROUP BY user_id;
 
50. Retrieve the product with the highest number of returns in the last month.

SELECT product_id, COUNT(*) AS return_count 
FROM product_returns 
WHERE return_date BETWEEN DATE_SUB(CURDATE(), INTERVAL 1 MONTH) AND CURDATE() 
GROUP BY product_id 
ORDER BY return_count DESC LIMIT 1;
  
51. Show the percentage change in sales for each product category from the previous year.

SELECT category_id, 
 (SUM(CASE WHEN YEAR(sale_date) = YEAR(CURDATE()) - 1 THEN sales END) - 
 SUM(CASE WHEN YEAR(sale_date) = YEAR(CURDATE()) - 2 THEN sales END)) / 
 SUM(CASE WHEN YEAR(sale_date) = YEAR(CURDATE()) - 2 THEN sales END) * 100 AS sales_percentage_change 
FROM sales 
GROUP BY category_id;

52. Find the employees who made the most number of overtime hours in the last 3 months.
  
SELECT employee_id, SUM(overtime_hours) AS total_overtime 
FROM employee_overtime 
WHERE overtime_date BETWEEN DATE_SUB(CURDATE(), INTERVAL 3 MONTH) AND CURDATE() 
GROUP BY employee_id 
ORDER BY total_overtime DESC LIMIT 1;

53. Retrieve the top-performing departments based on employee retention rates.

SELECT department_id, 
COUNT(DISTINCT employee_id) / (SELECT COUNT(DISTINCT employee_id) FROM employees WHERE department_id = d.department_id) * 100 AS retention_rate 
FROM employee_departments d 
WHERE status = 'Active' 
GROUP BY department_id 
ORDER BY retention_rate DESC LIMIT 1;
  
54. Find the product with the largest discount given, relative to its original price.

SELECT product_id, 
MAX((original_price - discounted_price) / original_price) AS max_discount_ratio 
FROM products 
GROUP BY product_id 
ORDER BY max_discount_ratio DESC LIMIT 1;

55. Find employees who worked in the same department the longest.

SELECT employee_id, department_id, MIN(start_date) AS start_date 
FROM employee_department_history 
GROUP BY employee_id, department_id 
ORDER BY DATEDIFF(CURRENT_DATE, start_date) DESC LIMIT 1; 

56. Identify customers who purchased all available products.

SELECT customer_id 
FROM orders 
GROUP BY customer_id 
HAVING COUNT(DISTINCT product_id) = (SELECT COUNT(*) FROM products); 

57. Calculate total working days for employees from attendance logs.

SELECT employee_id, COUNT(DISTINCT date) AS total_working_days 
FROM attendance_logs 
WHERE status = 'Present' GROUP BY employee_id; 

58. Retrieve the top-selling product in each category.

SELECT category_id, product_id, SUM(sales) AS total_sales 
FROM products p 
JOIN sales s ON p.product_id = s.product_id 
GROUP BY category_id, product_id 
HAVING total_sales = (SELECT MAX(SUM(sales)) FROM sales s2 
WHERE p2.category_id = p.category_id); 

59. Show the sales difference between the current and previous month for each product.

SELECT product_id, month, 
 SUM(sales) AS current_sales, 
 LAG(SUM(sales)) OVER (PARTITION BY product_id ORDER BY month) AS prev_sales, 
 SUM(sales) - LAG(SUM(sales)) OVER (PARTITION BY product_id ORDER BY month) AS sales_diff 
FROM sales GROUP BY product_id, month; 

60. Generate a list of missing dates from a daily sales table.

SELECT date + INTERVAL 1 DAY AS missing_date 
FROM daily_sales d1 
WHERE NOT EXISTS (SELECT date FROM daily_sales d2 WHERE d2.date = d1.date + INTERVAL 1 DAY); 

61. Identify employees with the same salary for more than two consecutive years.

SELECT employee_id 
FROM salary_history 
GROUP BY employee_id, salary 
HAVING COUNT(*) > 2 AND MAX(year) - MIN(year) = COUNT(*) - 1; 

62. Find the product with the highest price-to-sales ratio.

SELECT product_id, (price / SUM(sales)) AS price_to_sales_ratio 
FROM products p 
JOIN sales s ON p.product_id = s.product_id 
GROUP BY product_id 
ORDER BY price_to_sales_ratio DESC LIMIT 1; 

63. Show departments with the highest and lowest average salaries.

SELECT department_id, AVG(salary) AS avg_salary 
FROM employees 
GROUP BY department_id 
ORDER BY avg_salary DESC LIMIT 1 

UNION ALL 

SELECT department_id, AVG(salary) AS avg_salary 
FROM employees 
GROUP BY department_id 
ORDER BY avg_salary ASC LIMIT 1; 

64. Create a leaderboard of employees by sales, ranking within each department.

SELECT department_id, employee_id, 
RANK() OVER (PARTITION BY department_id ORDER BY SUM(sales) DESC) AS rank 
FROM employees e 
JOIN sales s ON e.employee_id = s.employee_id 
GROUP BY department_id, employee_id; 

65. Write a query to calculate the median salary of employees in a table.

SELECT AVG(salary) AS median_salary 
FROM ( 
 SELECT salary 
 FROM employees 
 ORDER BY salary 
 LIMIT 2 - (SELECT COUNT(*) FROM employees) % 2 
 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM employees) 
) subquery; 

66. Identify products that were sold in all regions.

SELECT product_id 
FROM sales 
GROUP BY product_id 
HAVING COUNT(DISTINCT region_id) = (SELECT COUNT(*) FROM regions); 

67. Retrieve the name of the manager who supervises the most employees.

SELECT manager_id, COUNT(*) AS num_employees 
FROM employees 
GROUP BY manager_id 
ORDER BY num_employees DESC 
LIMIT 1; 

68. Write a query to group employees by age ranges (e.g., 20–30, 31–40) and count the number of employees in each group.

SELECT 
  CASE 
      WHEN age BETWEEN 20 AND 30 THEN '20-30' 
      WHEN age BETWEEN 31 AND 40 THEN '31-40' 
      WHEN age BETWEEN 41 AND 50 THEN '41-50' 
  ELSE '50+' 
  END AS age_range, 
COUNT(*) AS num_employees 
FROM employees 
GROUP BY age_range; 

69. Display the cumulative percentage of total sales for each product.

SELECT product_id, 
 SUM(sales) AS product_sales, 
 SUM(SUM(sales)) OVER (ORDER BY SUM(sales) DESC) * 100.0 / SUM(SUM(sales)) OVER () AS cumulative_percentage 
FROM sales_table 
GROUP BY product_id; 
 
70. Write a query to retrieve the first order placed by each customer.

SELECT customer_id, MIN(order_date) AS first_order_date 
FROM orders 
GROUP BY customer_id; 
  
71. Identify employees who have never received a performance review.

SELECT * 
FROM employees 
WHERE employee_id NOT IN (SELECT employee_id FROM performance_reviews); 
  
72. Find the most common value (mode) in a specific column.

SELECT column_name, COUNT(*) AS frequency 
FROM table_name 
GROUP BY column_name 
ORDER BY frequency DESC 
LIMIT 1; 
  
73. Display all months where sales exceeded the average monthly sales.

SELECT month, SUM(sales) AS monthly_sales 
FROM sales 
GROUP BY month 
HAVING monthly_sales > (SELECT AVG(SUM(sales)) FROM sales GROUP BY month); 
  
74. Write a query to identify the employee(s) whose salary is closest to the average salary of the company.

SELECT employee_id, salary 
FROM employees 
ORDER BY ABS(salary - (SELECT AVG(salary) FROM employees)) ASC 
LIMIT 1; 

75. Write a query to fetch the third-highest salary without using LIMIT or ROW_NUMBER().

SELECT DISTINCT salary FROM employees e1 
WHERE 2 = (SELECT COUNT(DISTINCT salary) FROM employees e2 WHERE e2.salary > e1.salary); 

76. Find employees whose salaries are above the average salary of their department but below the average salary of the entire company.

SELECT e.* FROM employees e 
JOIN (SELECT department_id, AVG(salary) AS dept_avg_salary FROM employees GROUP BY department_id) d 
ON e.department_id = d.department_id 
WHERE e.salary > d.dept_avg_salary AND e.salary < (SELECT AVG(salary) FROM employees); 
  
77. Identify duplicate records in a table and write a query to delete all but one instance of each duplicate.

-- Find duplicates 
SELECT column1, column2, COUNT(*) FROM table_name 
GROUP BY column1, column2 
HAVING COUNT(*) > 1; 

-- Delete duplicates 
DELETE FROM table_name 
WHERE id NOT IN ( 
 SELECT MIN(id) FROM table_name GROUP BY column1, column2 
); 

78. Retrieve the top three highest-paid employees in each department.

SELECT * FROM ( 
 SELECT employee_id, department_id, salary, 
 RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank 
 FROM employees 
) ranked 
WHERE rank <= 3; 

79. Write a query to calculate the percentage contribution of each product to the total sales.
SELECT product_id, 
 SUM(sales) AS product_sales, 
 (SUM(sales) * 100.0 / (SELECT SUM(sales) FROM sales_table)) AS percentage_contribution 
FROM sales_table 
GROUP BY product_id; 

80. Display employees who joined within the last 6 months.

SELECT * FROM employees 
WHERE join_date >= DATE_ADD(CURRENT_DATE, INTERVAL -6 MONTH); 

81. Identify the employee(s) with the longest tenure in the company.

SELECT employee_id, DATEDIFF(CURRENT_DATE, join_date) AS tenure 
FROM employees 
ORDER BY tenure DESC 
LIMIT 1; 
  
82. Create a query to find gaps in a sequence of IDs in a table.

SELECT id + 1 AS missing_id 
FROM table_name t1 
WHERE NOT EXISTS (SELECT id FROM table_name t2 WHERE t2.id = t1.id + 1); 

83. Retrieve all records in a table that have a matching record in another table based on two or more columns.

SELECT t1.* FROM table1 t1 
JOIN table2 t2 ON t1.column1 = t2.column1 AND t1.column2 = t2.column2; 

84. Write a query to identify customers who placed more orders this year compared to last year.

SELECT customer_id FROM ( 
 SELECT customer_id, 
 SUM(CASE WHEN YEAR(order_date) = YEAR(CURRENT_DATE) THEN 1 ELSE 0 END) AS this_year_orders, 
 SUM(CASE WHEN YEAR(order_date) = YEAR(CURRENT_DATE) - 1 THEN 1 ELSE 0 END) AS last_year_orders 
 FROM orders 
 GROUP BY customer_id 
) order_summary 
WHERE this_year_orders > last_year_orders;

85. How can you find the second-highest salary in a table without using LIMIT or TOP?

SELECT MAX(salary) FROM table WHERE salary NOT IN (SELECT MAX (salary) FROM table)
  
86. Write a query to find employees whose salaries are higher than their managers’.

SELECT e1.* FROM employees e1 JOIN employees e2 
ON el.manager_id = e2.employee_id 
WHERE e1.salary > e2.salary

87. Identify duplicate rows in a table without using GROUP BY.

SELECT * FROM table WHERE rowid IN (SELECT rowid FROM table GROUP BY column HAVING COUNT (*) > 1)
  
88. Write a query to find the top 10% earners in a table.

SELECT * FROM table WHERE salary > (SELECT PERCENTILE_CONT (0.9) WITHIN GROUP (ORDER BY salary) FROM table)

89. Calculate the cumulative sum of a column in a table.

SELECT column, SUM(column) OVER (ORDER BY rowid) FROM table

90. Find all employees who have never taken any leave.

SELECT * FROM employees WHERE id NOT IN (SELECT employee_id FROM leaves)
  
91. Write a query to calculate the difference between the current row and the next row in a table.

SELECT *, column - LEAD(column) OVER (ORDER BY rowid) FROM table
  
92. Find departments that have more than one employee.

SELECT department FROM employees GROUP BY department HAVING COUNT (*) > 1

93. Determine the maximum value of a column for each group without using GROUP BY.

SELECT MAX(column) FROM table WHERE column NOT IN (SELECT MAX (column) FROM table GROUP BY group_column)
  
94. Identify employees who have taken more than 3 leaves in a single month.

SELECT * FROM employees WHERE id IN (SELECT
employee_id FROM leaves GROUP BY employee_id HAVING COUNT(*) > 3)

  
  








