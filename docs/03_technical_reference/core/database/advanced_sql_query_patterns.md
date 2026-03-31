# Advanced SQL Query Patterns

This comprehensive guide explores sophisticated SQL query techniques that enable complex data analysis and manipulation beyond basic CRUD operations. Master these patterns to unlock the full power of relational databases for analytics, hierarchical data processing, and optimized query execution.

## Window Functions and Analytics

Window functions represent one of the most powerful features in modern SQL, enabling calculations across sets of rows that are related to the current row without collapsing the result set. Unlike aggregate functions that group rows together, window functions preserve individual row identities while computing values across a specified window frame.

### Understanding Window Function Syntax

The fundamental syntax of a window function involves three key components: the function itself, the OVER clause that defines the window, and optional PARTITION BY and ORDER BY specifications within the window. The PARTITION BY clause divides the result set into partitions where the window function operates independently, while ORDER BY determines the row ordering within each partition. This combination allows for sophisticated analytical computations that would otherwise require complex self-joins or application-level processing.

```sql
SELECT 
    employee_id,
    department,
    salary,
    AVG(salary) OVER (PARTITION BY department) AS dept_avg_salary,
    salary - AVG(salary) OVER (PARTITION BY department) AS diff_from_avg,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_salary_rank,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS overall_salary_rank,
    NTILE(4) OVER (ORDER BY salary DESC) AS salary_quartile,
    LAG(salary, 1) OVER (PARTITION BY department ORDER BY hire_date) AS prev_dept_salary,
    LEAD(salary, 1) OVER (PARTITION BY department ORDER BY hire_date) AS next_dept_salary,
    SUM(salary) OVER (PARTITION BY department ORDER BY hire_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_dept_total
FROM employees
ORDER BY department, hire_date;
```

This query demonstrates multiple window functions operating simultaneously, each providing different analytical perspectives. The department average calculation shows how each employee's salary compares to their departmental peers, while the ranking functions provide positional context both within departments and across the entire organization.

### Ranking Functions Deep Dive

The three ranking functions—ROW_NUMBER, RANK, and DENSE_RANK—serve different use cases depending on how you want to handle ties. ROW_NUMBER assigns unique sequential integers to each row regardless of ties, making it ideal for pagination and deduplication. RANK leaves gaps after tied values (1, 2, 2, 4), while DENSE_RANK maintains consecutive ranking (1, 2, 2, 3). Understanding these distinctions is crucial when implementing business logic that depends on ranking positions.

```sql
-- Find top 3 highest-paid employees per department
WITH ranked_employees AS (
    SELECT 
        department,
        employee_name,
        salary,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rn
    FROM employees
)
SELECT department, employee_name, salary
FROM ranked_employees
WHERE rn <= 3;

-- Handle ties in ranking for bonus allocation
SELECT 
    employee_name,
    salary,
    RANK() OVER (ORDER BY salary DESC) AS rank_with_gaps,
    DENSE_RANK() AS dense_rank_no_gaps,
    PERCENT_RANK() OVER (ORDER BY salary DESC) AS percentile_position,
    CUME_DIST() OVER (ORDER BY salary DESC) AS cumulative_distribution
FROM employees;
```

### Frame Specifications and Running Calculations

Window frames define the set of rows relative to the current row for aggregate-like window functions. The frame can be specified using ROWS, RANGE, or GROUPS clauses with modifiers like UNBOUNDED PRECEDING, CURRENT ROW, and UNBOUNDED FOLLOWING. Running totals, moving averages, and cumulative percentages all rely on precise frame specifications to achieve the desired calculation behavior.

```sql
-- Moving average with 7-day window
SELECT 
    transaction_date,
    amount,
    AVG(amount) OVER (
        ORDER BY transaction_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7day,
    SUM(amount) OVER (
        ORDER BY transaction_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total,
    COUNT(*) OVER (
        ORDER BY transaction_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS transaction_count_7day
FROM transactions;

-- First and last value in partition
SELECT 
    department,
    employee_name,
    hire_date,
    FIRST_VALUE(salary) OVER (PARTITION BY department ORDER BY hire_date) AS first_hired_salary,
    LAST_VALUE(salary) OVER (
        PARTITION BY department ORDER BY hire_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS latest_hired_salary
FROM employees;
```

### Analytical Patterns for Business Intelligence

Window functions excel at solving common business intelligence challenges that would be cumbersome to implement with traditional SQL. Calculating period-over-period growth, identifying customer purchase patterns, and detecting anomalies all become straightforward with the right window function approach. The key is to think in terms of comparing rows within their defined windows rather than writing complex self-joins.

```sql
-- Year-over-year growth analysis
SELECT 
    year,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY year) AS prev_year_revenue,
    ROUND(
        (revenue - LAG(revenue, 1) OVER (ORDER BY year)) * 100.0 / 
        NULLIF(LAG(revenue, 1) OVER (ORDER BY year), 0), 
    2) AS yoy_growth_pct,
    SUM(revenue) OVER (ORDER BY year ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS rolling_4yr_total
FROM annual_revenue;

-- Nth highest value in group (find 2nd highest salary per department)
SELECT department, employee_name, salary
FROM (
    SELECT 
        department,
        employee_name,
        salary,
        DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS salary_rank
    FROM employees
) ranked
WHERE salary_rank = 2;
```

## Recursive CTEs for Hierarchical Data

Recursive Common Table Expressions provide SQL with the ability to traverse hierarchical structures, perform graph traversals, and generate sequences without explicit loop constructs in application code. The recursive CTE consists of two parts: an anchor member that provides the starting point, and a recursive member that references the CTE itself to build upon previous iterations.

### Basic Recursive CTE Structure

The recursive CTE follows a specific execution model where the anchor member runs first, producing initial results. These results then feed into the recursive member, which produces additional rows. This cycle continues until no new rows are generated, at which point the final result set is returned. Understanding this execution model is essential for writing efficient recursive queries and avoiding infinite loops.

```sql
-- Generate a sequence of numbers from 1 to 100
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 100
)
SELECT n FROM numbers;

-- Calculate factorial using recursive CTE
WITH RECURSIVE factorial(n, fact) AS (
    SELECT 0, 1
    UNION ALL
    SELECT n + 1, (n + 1) * fact FROM factorial WHERE n < 10
)
SELECT n, fact FROM factorial;
```

### Organizational Hierarchy traversal

One of the most common applications of recursive CTEs is traversing organizational charts or category hierarchies. The employee-manager relationship forms a classic tree structure that can be efficiently queried using recursion to find reporting chains, team sizes, or organizational depth.

```sql
-- Find all direct and indirect reports for a manager
WITH RECURSIVE org_chart AS (
    -- Anchor: direct reports
    SELECT 
        employee_id,
        manager_id,
        employee_name,
        1 AS level,
        CAST(employee_name AS VARCHAR(500)) AS path
    FROM employees
    WHERE manager_id = 100
    
    UNION ALL
    
    -- Recursive: reports of reports
    SELECT 
        e.employee_id,
        e.manager_id,
        e.employee_name,
        oc.level + 1,
        oc.path || ' -> ' || e.employee_name
    FROM employees e
    INNER JOIN org_chart oc ON e.manager_id = oc.employee_id
)
SELECT employee_id, employee_name, level, path
FROM org_chart
ORDER BY level, employee_name;

-- Calculate organizational depth from CEO down
WITH RECURSIVE hierarchy AS (
    SELECT 
        employee_id,
        manager_id,
        employee_name,
        0 AS depth,
        ARRAY[employee_id] AS ancestor_chain
    FROM employees
    WHERE manager_id IS NULL  -- CEO has no manager
    
    UNION ALL
    
    SELECT 
        e.employee_id,
        e.manager_id,
        e.employee_name,
        h.depth + 1,
        h.ancestor_chain || e.employee_id
    FROM employees e
    INNER JOIN hierarchy h ON e.manager_id = h.employee_id
    WHERE array_length(h.ancestor_chain, 1) < 20  -- Prevent infinite loops
)
SELECT 
    employee_name,
    depth,
    repeat('  ', depth) || employee_name AS indented_name,
    ancestor_chain
FROM hierarchy
ORDER BY depth, employee_name;
```

### Graph Traversals and Network Analysis

Beyond simple hierarchies, recursive CTEs can handle more complex graph structures with multiple connections between nodes. Social network analysis, transportation routes, and dependency graphs all benefit from recursive querying capabilities.

```sql
-- Find all reachable nodes in a directed graph (friends of friends)
WITH RECURSIVE social_network AS (
    -- Anchor: direct friends
    SELECT 
        user_id,
        friend_id,
        1 AS depth,
        ARRAY[user_id, friend_id] AS visited
    FROM friendships
    WHERE user_id = 1
    
    UNION ALL
    
    -- Recursive: friends of friends
    SELECT 
        f.user_id,
        f.friend_id,
        sn.depth + 1,
        sn.visited || f.friend_id
    FROM friendships f
    INNER JOIN social_network sn ON f.user_id = sn.friend_id
    WHERE f.friend_id != ALL(sn.visited)  -- Avoid cycles
    AND sn.depth < 5  -- Limit recursion depth
)
SELECT DISTINCT friend_id, MIN(depth) AS shortest_path
FROM social_network
GROUP BY friend_id
ORDER BY shortest_path;

-- Find shortest path between two nodes (route finding)
WITH RECURSIVE shortest_path AS (
    SELECT 
        source,
        target,
        cost,
        ARRAY[source] AS path,
        cost AS total_cost
    FROM routes
    WHERE source = 'A'
    
    UNION ALL
    
    SELECT 
        r.source,
        r.target,
        r.cost,
        sp.path || r.target,
        sp.total_cost + r.cost
    FROM routes r
    INNER JOIN shortest_path sp ON r.source = sp.target
    WHERE r.target != ALL(sp.path)
)
SELECT path, total_cost
FROM shortest_path
WHERE target = 'Z'
ORDER BY total_cost
LIMIT 1;
```

### Materialized Path and Nested Set Patterns

For frequently queried hierarchies, materialized path and nested set models provide alternative representations that can be more efficient than pure recursive queries. These patterns store the hierarchical structure in a way that enables simpler queries at the cost of more complex updates.

```sql
-- Materialized path pattern for category tree
SELECT * FROM categories
WHERE path LIKE '001/002/%'  -- All descendants of category 002
ORDER BY path;

-- Convert adjacency list to materialized path
WITH RECURSIVE path_builder AS (
    SELECT 
        category_id,
        parent_id,
        CAST(category_id AS VARCHAR(1000)) AS path,
        1 AS depth
    FROM categories
    WHERE parent_id IS NULL
    
    UNION ALL
    
    SELECT 
        c.category_id,
        c.parent_id,
        pb.path || '/' || c.category_id,
        pb.depth + 1
    FROM categories c
    INNER JOIN path_builder pb ON c.parent_id = pb.category_id
)
UPDATE categories c
SET materialized_path = pb.path
FROM path_builder pb
WHERE c.category_id = pb.category_id;
```

## Advanced JOIN Patterns

Advanced JOIN patterns extend beyond simple inner and outer joins to handle complex data relationships, optimize query performance, and express sophisticated business logic directly in SQL. Mastering these patterns reduces the need for multiple queries and application-level data manipulation.

### Self-Joins and Inequality Conditions

Self-joins enable comparison between rows within the same table, essential for comparing employees in the same department, finding duplicate records, or identifying temporal relationships. Inequality joins (using <, >, <=, >= rather than =) unlock additional analytical possibilities.

```sql
-- Compare employee salaries within department
SELECT 
    e1.employee_name AS employee_a,
    e2.employee_name AS employee_b,
    e1.department,
    e1.salary AS salary_a,
    e2.salary AS salary_b,
    e1.salary - e2.salary AS salary_diff
FROM employees e1
JOIN employees e2 
    ON e1.department = e2.department 
    AND e1.employee_id < e2.employee_id;

-- Find consecutive date pairs for inventory analysis
SELECT 
    inv1.inventory_date AS date_start,
    inv2.inventory_date AS date_end,
    inv1.product_id
FROM inventory inv1
JOIN inventory inv2 
    ON inv1.product_id = inv2.product_id
    AND inv2.inventory_date = inv1.inventory_date + INTERVAL '1 day';

-- Anti-join pattern: find orders without shipped items
SELECT o.*
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id AND oi.ship_date IS NOT NULL
WHERE oi.order_item_id IS NULL;
```

### Lateral Joins and Cross Apply

The LATERAL keyword and CROSS APPLY operator enable correlated subqueries that reference columns from preceding tables in the query. This pattern is invaluable for top-N per group problems, running calculations that depend on row context, and any scenario where subquery results need to reference outer query columns.

```sql
-- Top 3 products per category by sales (PostgreSQL)
SELECT c.category_name, p.product_name, p.total_sales
FROM categories c
CROSS JOIN LATERAL (
    SELECT product_name, total_sales
    FROM products p
    WHERE p.category_id = c.category_id
    ORDER BY total_sales DESC
    LIMIT 3
) p;

-- Running total per customer using lateral
SELECT 
    t.transaction_id,
    t.customer_id,
    t.amount,
    t.transaction_date,
    running.total AS running_balance
FROM transactions t
CROSS JOIN LATERAL (
    SELECT SUM(amount) AS total
    FROM transactions t2
    WHERE t2.customer_id = t.customer_id
    AND t2.transaction_date <= t.transaction_date
) running
ORDER BY t.customer_id, t.transaction_date;

-- Multiple columns from correlated subquery
SELECT 
    o.order_id,
    o.order_date,
    latest_item.*,
    total_items.*
FROM orders o
CROSS APPLY (
    SELECT item_name, quantity 
    FROM order_items 
    WHERE order_id = o.order_id 
    ORDER BY quantity DESC 
    LIMIT 1
) latest_item
CROSS APPLY (
    SELECT COUNT(*) AS total_items, SUM(quantity) AS total_quantity
    FROM order_items
    WHERE order_id = o.order_id
) total_items;
```

### Complex Join Patterns for Data Warehousing

Data warehousing scenarios often require complex join patterns that combine multiple fact tables, handle slowly changing dimensions, and implement conformed dimension logic. These patterns form the backbone of efficient analytical queries.

```sql
-- Star schema join with multiple fact tables
SELECT 
    d.calendar_date,
    p.product_name,
    c.customer_name,
    f.sales_amount,
    f.quantity_sold,
    f2.return_amount,
    f2.return_quantity
FROM dim_date d
CROSS JOIN dim_product p
CROSS JOIN dim_customer c
LEFT JOIN fact_sales f 
    ON f.date_key = d.date_key 
    AND f.product_key = p.product_key 
    AND f.customer_key = c.customer_key
LEFT JOIN fact_returns f2
    ON f2.date_key = d.date_key
    AND f2.product_key = p.product_key
    AND f2.customer_key = c.customer_key
WHERE d.calendar_date >= '2024-01-01';

-- Handle Type 2 Slowly Changing Dimension
SELECT 
    o.order_id,
    o.order_date,
    p.product_name,
    p.product_version,
    p.category_name AS category_at_order_time
FROM fact_orders o
JOIN dim_product p 
    ON o.product_key = p.product_key
    AND o.order_date >= p.effective_date
    AND o.order_date < p.expiration_date;
```

### Non-Equi Joins and Range Matching

Non-equi joins using BETWEEN or range conditions enable interval-based matching, a common requirement in temporal data processing and statistical analysis.

```sql
-- Assign transactions to fiscal periods
SELECT 
    t.transaction_id,
    t.transaction_date,
    t.amount,
    fp.period_name,
    fp.start_date,
    fp.end_date
FROM transactions t
JOIN fiscal_periods fp 
    ON t.transaction_date BETWEEN fp.start_date AND fp.end_date;

-- Range-based customer segmentation
SELECT 
    c.customer_id,
    c.total_purchases,
    s.segment_name,
    s.min_purchases,
    s.max_purchases
FROM customers c
JOIN customer_segments s 
    ON c.total_purchases BETWEEN s.min_purchases AND s.max_purchases;
```

## Subquery Optimization

Subqueries, when properly written and indexed, can be extremely powerful. However, poorly optimized subqueries often become performance bottlenecks. Understanding when to use correlated versus non-correlated subqueries, how to convert them to joins, and when to materialize them is essential for query optimization.

### Correlated vs Non-Correlated Subqueries

A correlated subquery references columns from the outer query and is executed once for each row processed by the outer query. Non-correlated subqueries execute once and their results are used for the entire outer query. The choice between them significantly impacts performance.

```sql
-- Correlated subquery: executed for each outer row
SELECT employee_name, salary, department
FROM employees e
WHERE salary > (
    SELECT AVG(salary) 
    FROM employees 
    WHERE department = e.department
);

-- Non-correlated subquery: executed once
SELECT employee_name, salary
FROM employees
WHERE department IN (
    SELECT department 
    FROM departments 
    WHERE location = 'New York'
);

-- Convert correlated to JOIN for better performance (often)
SELECT e.employee_name, e.salary, e.department
FROM employees e
INNER JOIN (
    SELECT department, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department
) dept_avg ON e.department = dept_avg.department
WHERE e.salary > dept_avg.avg_salary;
```

### Subquery Types and Performance Implications

Different SQL constructs have varying performance characteristics. IN, EXISTS, ANY/SOME, and ALL each have optimal use cases depending on data distribution and indexes.

```sql
-- EXISTS: efficient for existence checks, stops at first match
SELECT *
FROM customers c
WHERE EXISTS (
    SELECT 1 
    FROM orders o 
    WHERE o.customer_id = c.customer_id 
    AND o.order_date > '2024-01-01'
);

-- IN with subquery: can be inefficient with large result sets
SELECT *
FROM products p
WHERE category_id IN (
    SELECT category_id 
    FROM categories 
    WHERE active = true
);

-- ANY/SOME for comparison with set
SELECT *
FROM employees
WHERE salary > ANY (
    SELECT salary 
    FROM employees 
    WHERE department = 'Sales'
);

-- Set operation with ALL
SELECT *
FROM employees
WHERE salary > ALL (
    SELECT salary 
    FROM interns
);
```

### Scalar Subquery Optimization

Scalar subqueries returning a single value can appear in SELECT, WHERE, and other clauses. Their placement affects execution strategy and performance.

```sql
-- Scalar subquery in SELECT (often expensive)
SELECT 
    order_id,
    (SELECT MAX(amount) FROM order_items WHERE order_id = orders.order_id) AS max_item_price
FROM orders;

-- Better: JOIN approach or window function
SELECT 
    o.order_id,
    oi.max_amount AS max_item_price
FROM orders o
LEFT JOIN (
    SELECT order_id, MAX(amount) AS max_amount
    FROM order_items
    GROUP BY order_id
) oi ON o.order_id = oi.order_id;

-- Using window function (most efficient in many cases)
SELECT 
    order_id,
    MAX(amount) OVER (PARTITION BY order_id) AS max_item_price
FROM order_items;
```

### Common Table Expressions vs Subqueries

CTEs often provide better optimization opportunities than subqueries because the optimizer can materialize them and reference them multiple times. They also improve query readability and maintainability.

```sql
-- Complex subquery pattern
SELECT *
FROM (
    SELECT 
        department,
        AVG(salary) AS avg_salary,
        COUNT(*) AS employee_count
    FROM employees
    WHERE hire_date > '2023-01-01'
    GROUP BY department
) dept_stats
WHERE avg_salary > (
    SELECT AVG(avg_salary) 
    FROM (
        SELECT AVG(salary) AS avg_salary
        FROM employees
        GROUP BY department
    ) overall_avg
);

-- Refactored with CTEs for clarity and potential optimization
WITH dept_salaries AS (
    SELECT 
        department,
        AVG(salary) AS avg_salary,
        COUNT(*) AS employee_count
    FROM employees
    WHERE hire_date > '2023-01-01'
    GROUP BY department
),
overall_avg AS (
    SELECT AVG(avg_salary) AS global_avg
    FROM dept_salaries
)
SELECT ds.*
FROM dept_salaries ds
CROSS JOIN overall_avg oa
WHERE ds.avg_salary > oa.global_avg;
```

## Query Plan Analysis

Understanding and analyzing query execution plans is fundamental to SQL optimization. The query plan reveals how the database engine intends to execute your query, including which indexes are used, join order, and estimated versus actual row counts. Mastering plan analysis enables targeted optimization efforts.

### Reading Query Execution Plans

Modern databases provide detailed execution plans that show each operation, its cost, and estimated row counts. Understanding these outputs requires familiarity with common operations like table scans, index scans, joins, and aggregations.

```sql
-- PostgreSQL EXPLAIN ANALYZE
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT e.employee_name, d.department_name, e.salary
FROM employees e
JOIN departments d ON e.department_id = d.department_id
WHERE e.salary > 50000 AND d.location = 'New York';

-- Example output interpretation:
-- Seq Scan on employees (cost=0.00..154.25) actual time=0.015..2.342 rows=1500
--   Filter: (salary > 50000)
--   Buffers: shared hit=45
-- Index Scan using idx_dept_location on departments (cost=0.42..12.50) actual time=0.125..0.450 rows=8
--   Index Cond: (location = 'New York'::text)
-- Hash Join (cost=45.25..250.50) actual time=2.500..4.125 rows=1200
--   Hash Cond: (e.department_id = d.department_id)
```

### Identifying Performance Issues

Common performance indicators in execution plans include sequential scans on large tables, nested loop joins with high row counts, missing indexes leading to full table scans, and inaccurate row estimates causing suboptimal join orders.

```sql
-- Problem: Full table scan on large table
-- Bad: No index on filter column
EXPLAIN SELECT * FROM orders WHERE order_date = '2024-01-15';

-- Solution: Create appropriate index
CREATE INDEX idx_orders_date ON orders(order_date);

-- Problem: Nested loop with large inner relation
-- Bad: Joining without proper index
EXPLAIN SELECT * FROM orders o 
JOIN customers c ON c.customer_id = o.customer_id
WHERE c.region = 'EMEA';

-- Solution: Ensure join and filter columns are indexed
CREATE INDEX idx_customers_region ON customers(region);
CREATE INDEX idx_orders_customer ON orders(customer_id);
```

### Query Plan Hints and Optimizer Control

Most database systems provide mechanisms to influence optimizer decisions when the default plan is suboptimal. Hints should be used judiciously after thorough analysis.

```sql
-- PostgreSQL: Use specific index
SELECT * FROM orders
WHERE order_date > '2024-01-01'
ORDER BY order_date DESC
LIMIT 100;

-- Oracle: Hint for parallel execution
SELECT /*+ PARALLEL(orders, 8) */ *
FROM orders
WHERE order_date > '2024-01-01';

-- SQL Server: Force hash join
SELECT * FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
OPTION (HASH JOIN);

-- MySQL: Use index
SELECT * FROM orders USE INDEX (idx_order_date)
WHERE order_date > '2024-01-01';
```

### Advanced Plan Analysis Techniques

Advanced analysis involves comparing plans before and after changes, identifying parameter sniffing issues, and understanding statistics quality.

```sql
-- Compare execution times with different approaches
\timing on

-- Test 1: Subquery approach
SELECT * FROM orders 
WHERE customer_id IN (SELECT customer_id FROM customers WHERE country = 'USA');

-- Test 2: JOIN approach
SELECT DISTINCT o.* FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
WHERE c.country = 'USA';

-- Identify parameter sniffing issues (same query, different parameters)
-- Plan cached for first parameter value may be suboptimal for others
-- Solution: Plan guides or query rewriting

-- Statistics analysis
-- PostgreSQL
SELECT schemaname, relname, n_live_tup, n_dead_tup, last_autovacuum
FROM pg_stat_user_tables
WHERE relname = 'orders';

-- SQL Server
DBCC SHOW_STATISTICS('orders', 'idx_order_date');
```

### Cost Model Interpretation

Understanding the cost model helps predict how changes in data volume affect query performance and when optimization efforts will have the most impact.

```sql
-- PostgreSQL cost units:
-- seq_page_cost = 1.0 (default)
-- random_page_cost = 4.0 (default, SSD: 1.1-1.5)
-- cpu_tuple_cost = 0.01
-- cpu_index_tuple_cost = 0.005
-- cpu_operator_cost = 0.0025

-- Lower random page cost for SSD
SET random_page_cost = 1.1;

-- Check current settings
SHOW random_page_cost;
SHOW seq_page_cost;

-- Estimate query I/O
EXPLAIN (ANALYZE, COSTS, BUFFERS)
SELECT * FROM large_table WHERE indexed_column = 'value';
```

This comprehensive guide provides the foundation for advanced SQL query development. Practice these patterns with real datasets to develop intuition for when each technique is most appropriate, and always measure the impact of optimizations in your specific environment.
