# Specialized Database Queries

This comprehensive guide covers advanced database querying techniques for specialized data types and use cases. Modern applications require sophisticated querying capabilities beyond standard SQL, including full-text search, geospatial data, hierarchical JSON structures, arrays, and time-series analysis. Master these specialized query patterns to build powerful data-driven applications.

## Full-Text Search Implementations

Full-text search enables searching through large volumes of text data efficiently, ranking results by relevance, and supporting complex linguistic features like stemming and stop words. While traditional SQL LIKE queries are inadequate for production search requirements, specialized full-text search implementations provide the performance and features needed for modern applications.

### PostgreSQL Full-Text Search

PostgreSQL provides native full-text search capabilities through tsvector and tsquery types, eliminating the need for external search engines in many applications. The GIN (Generalized Inverted Index) index type provides excellent read performance at the cost of slower writes, while GiST (Generalized Search Tree) offers a balance suitable for dynamic data.

```sql
-- Create a tsvector column with weights for ranking
ALTER TABLE articles 
ADD COLUMN search_vector tsvector
GENERATED ALWAYS AS (
    setweight(to_tsvector('english', title), 'A') ||
    setweight(to_tsvector('english', COALESCE(abstract, '')), 'B') ||
    setweight(to_tsvector('english', COALESCE(body, '')), 'C')
) STORED;

-- Create GIN index for fast searching
CREATE INDEX idx_articles_search ON articles USING GIN(search_vector);

-- Basic full-text search query
SELECT 
    article_id,
    title,
    ts_rank(search_vector, query) AS rank
FROM articles, plainto_tsquery('english', 'artificial intelligence') query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- Advanced search with multiple terms and boolean operators
SELECT 
    article_id,
    title,
    snippet,
    ts_rank(search_vector, query) AS rank
FROM articles,
    to_tsquery('english', 'machine & (learning | intelligence)') query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- Highlight matching terms in results
SELECT 
    title,
    ts_headline('english', body, query, 
        'StartSel=<mark>, StopSel=</mark>, MaxWords=50, MinWords=25'
    ) AS highlighted_text,
    ts_rank(search_vector, query) AS rank
FROM articles,
    to_tsquery('english', 'database & optimization') query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- Search with prefix matching
SELECT title
FROM articles
WHERE search_vector @@ to_tsquery('english', 'data:*');

-- Full-text search with filtering
SELECT article_id, title, published_date
FROM articles
WHERE search_vector @@ to_tsquery('english', 'postgresql')
AND status = 'published'
AND published_date > '2024-01-01'
ORDER BY ts_rank(search_vector, to_tsquery('english', 'postgresql')) DESC;
```

### PostgreSQL Custom Dictionaries and Configuration

PostgreSQL's full-text search can be customized with custom dictionaries, stop words, and linguistic configurations for domain-specific applications.

```sql
-- Create custom text search configuration
CREATE TEXT SEARCH CONFIGURATION technical (
    PARSER = pg_catalog.default
);

-- Add dictionary mappings
ALTER TEXT SEARCH CONFIGURATION technical
    ADD MAPPING FOR asciiword WITH english_stem, english_stop;

-- Create custom stop word list
CREATE TEXT SEARCH DICTIONARY technical_stop (
    TYPE = STOPLIST,
    STOPWORDS = technical
);

-- Create domain-specific thesaurus
CREATE TEXT SEARCH THESAURUS tech_thesaurus (
    ('ai', 'ml', 'artificial intelligence') -> ('artificial_intelligence'),
    ('rdbms', 'relational database') -> ('relational_database')
);

-- Use custom configuration in queries
SELECT title
FROM articles
WHERE to_tsvector('technical', title || ' ' || abstract) @@ 
      to_tsquery('technical', 'ai | machine learning');
```

### MySQL Full-Text Search

MySQL provides full-text search through the InnoDB and MyISAM storage engines, with natural language and boolean mode search capabilities.

```sql
-- Create full-text index (InnoDB)
ALTER TABLE articles 
ADD FULLTEXT INDEX idx_search (title, body);

-- Natural language search (default)
SELECT 
    article_id,
    title,
    MATCH(title, body) AGAINST('artificial intelligence' IN NATURAL LANGUAGE MODE) AS relevance
FROM articles
WHERE MATCH(title, body) AGAINST('artificial intelligence' IN NATURAL LANGUAGE MODE)
ORDER BY relevance DESC;

-- Boolean mode search with operators
SELECT 
    article_id,
    title,
    MATCH(title, body) AGAINST('+artificial +intelligence -machine' IN BOOLEAN MODE) AS relevance
FROM articles
WHERE MATCH(title, body) AGAINST('+artificial +intelligence -machine' IN BOOLEAN MODE);

-- Boolean mode query examples
-- '+' must contain term
-- '-' must NOT contain term
-- '*' wildcard at end
-- '"' phrase search
SELECT title
FROM articles
WHERE MATCH(title, body) AGAINST('"+artificial intelligence" +machine* -learning' IN BOOLEAN MODE);

-- Query expansion mode for broader results
SELECT 
    article_id,
    title,
    MATCH(title, body) AGAINST('database' WITH QUERY EXPANSION) AS relevance
FROM articles
WHERE MATCH(title, body) AGAINST('database' WITH QUERY EXPANSION);

-- Search with relevance ranking and highlighting
SELECT 
    article_id,
    title,
    MATCH(title, body) AGAINST('optimization' IN NATURAL LANGUAGE MODE) AS relevance,
    SUBSTRING_INDEX(body, 'optimization', 1) AS context
FROM articles
WHERE MATCH(title, body) AGAINST('optimization' IN NATURAL LANGUAGE MODE)
ORDER BY relevance DESC
LIMIT 10;
```

### Full-Text Search Performance Considerations

Optimizing full-text search requires attention to index maintenance, query structure, and underlying data modeling decisions.

```sql
-- Partition large text data for better performance
CREATE TABLE articles_2024 (
    PRIMARY KEY (article_id)
) PARTITION BY RANGE (YEAR(published_date)) (
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026)
);

-- Use prefix indexes for large text columns in MySQL
ALTER TABLE articles ADD INDEX idx_title_prefix(title(50));

-- PostgreSQL: Combine full-text with traditional indexes
CREATE INDEX idx_date_status ON articles(published_date, status);

-- Query that uses both index types efficiently
SELECT *
FROM articles
WHERE status = 'published'
AND published_date > '2024-06-01'
AND search_vector @@ to_tsquery('english', 'postgresql')
ORDER BY published_date DESC
LIMIT 20;
```

## Geospatial Queries

Geospatial data processing enables location-aware finding nearby users to calculating applications, from delivery routes. PostGIS for PostgreSQL and MongoDB's 2dsphere index provide powerful geospatial query capabilities that go beyond simple distance calculations.

### PostGIS Fundamentals and Queries

PostGIS extends PostgreSQL with geospatial data types, functions, and indexes. Understanding coordinate reference systems, spatial relationships, and appropriate indexes is essential for geospatial applications.

```sql
-- Enable PostGIS extension
CREATE EXTENSION postgis;

-- Create spatial table
CREATE TABLE locations (
    location_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    coordinates GEOGRAPHY(POINT, 4326),  -- WGS84 CRS
    location_type VARCHAR(50),
    metadata JSONB
);

-- Create spatial index for geography type
CREATE INDEX idx_locations_coords ON locations USING GIST(coordinates);

-- Insert sample data
INSERT INTO locations (name, coordinates, location_type) VALUES
('Times Square', ST_SetSRID(ST_MakePoint(-73.9855, 40.7580), 4326), 'landmark'),
('Central Park', ST_SetSRID(ST_MakePoint(-73.9654, 40.7829), 4326), 'park'),
('Statue of Liberty', ST_SetSRID(ST_MakePoint(-74.0445, 40.6892), 4326), 'monument');

-- Find locations within radius (distance in meters)
SELECT 
    l1.name AS location_name,
    l2.name AS nearby_name,
    ST_Distance(l1.coordinates, l2.coordinates) AS distance_meters
FROM locations l1
CROSS JOIN locations l2
WHERE l1.name = 'Times Square'
AND l1.location_id != l2.location_id
AND ST_DWithin(l1.coordinates, l2.coordinates, 5000)  -- 5km radius
ORDER BY distance_meters;

-- Find nearest N locations using KNN
SELECT 
    name,
    coordinates <-> ST_SetSRID(ST_MakePoint(-73.9855, 40.7580), 4326) AS distance
FROM locations
ORDER BY coordinates <-> ST_SetSRID(ST_MakePoint(-73.9855, 40.7580), 4326)
LIMIT 5;

-- Calculate distance along Earth's surface using geography
SELECT 
    ST_Distance(
        ST_GeogFromText('POINT(-73.9855 40.7580)'),
        ST_GeogFromText('POINT(-74.0445 40.6892)'),
        use_spheroid => true
    ) AS distance_meters;

-- Spatial joins: find all points in polygon
CREATE TABLE neighborhoods (
    neighborhood_id SERIAL PRIMARY WAY,
    name VARCHAR(100),
    boundary GEOMETRY(POLYGON, 4326)
);

SELECT 
    n.name AS neighborhood,
    COUNT(l.location_id) AS location_count
FROM neighborhoods n
JOIN locations l ON ST_Contains(n.boundary, l.coordinates)
GROUP BY n.name
ORDER BY location_count DESC;

-- Advanced: Find locations along a route
SELECT 
    ST_AsText( ST_Buffer(
        ST_GeomFromText('LINESTRING(-73.9855 40.7580, -73.9654 40.7829)', 4326),
        0.01
    ));

-- Calculate area of polygons
SELECT 
    name,
    ST_Area(boundary::geography) AS area_sq_meters,
    ST_Area(boundary::geography) / 1000000 AS area_sq_km
FROM neighborhoods;
```

### PostGIS Advanced Patterns

Advanced PostGIS patterns include complex spatial aggregations, trajectory analysis, and raster operations for specialized applications.

```sql
-- Cluster nearby locations using ST_ClusterDBSCAN
SELECT 
    cluster_id,
    ST_Collect(coordinates) AS cluster_points,
    COUNT(*) AS point_count,
    ST_Centroid(ST_Collect(coordinates)) AS cluster_center
FROM (
    SELECT 
        location_id,
        coordinates,
        ST_ClusterDBSCAN(coordinates, eps := 5000, minpoints := 1) OVER() AS cluster_id
    FROM locations
) clustered
GROUP BY cluster_id
ORDER BY cluster_id;

-- Generate isochrone (reachable area in given time)
SELECT ST_AsText(
    ST_Transform(
        ST_Buffer(
            ST_Transform(coordinates, 3857),  -- Web Mercator
            5000  -- 5km radius
        ),
        4326
    )
)
FROM locations
WHERE name = 'Times Square';

-- Point-in-polygon with multiple polygons (spatial join with aggregation)
SELECT 
    n.name AS neighborhood,
    array_agg(l.name) AS locations,
    COUNT(*) AS count
FROM neighborhoods n
JOIN locations l ON ST_Contains(n.boundary, l.coordinates)
GROUP BY n.name;

-- Voronoi diagram for location analysis
WITH points AS (
    SELECT ST_Collect(coordinates) AS geom FROM locations
)
SELECT (ST_Dump(ST_VoronoiPolygons(geom))).geom
FROM points;

-- Street network analysis preparation
ALTER TABLE roads ADD COLUMN geom geometry(LineString, 4326);
CREATE INDEX idx_roads_geom ON roads USING GIST(geom);

-- Find nearest road segment to a point
SELECT 
    road_id,
    road_name,
    ST_Distance(
        ST_SetSRID(ST_MakePoint(-73.9855, 40.7580), 4326),
        geom
    ) AS distance
FROM roads
ORDER BY distance
LIMIT 1;
```

### MongoDB 2dsphere Queries

MongoDB's 2dsphere index supports queries on geospatial data represented as GeoJSON or legacy coordinate pairs. This is particularly useful for applications built on document databases.

```javascript
// Create 2dsphere index
db.locations.createIndex({ coordinates: "2dsphere" });

// Insert GeoJSON data
db.locations.insertMany([
  {
    name: "Times Square",
    coordinates: { type: "Point", coordinates: [-73.9855, 40.7580] },
    locationType: "landmark",
    tags: ["nyc", "tourist", "entertainment"]
  },
  {
    name: "Central Park",
    coordinates: { type: "Point", coordinates: [-73.9654, 40.7829] },
    locationType: "park",
    tags: ["nyc", "nature", "recreation"]
  },
  {
    name: "Brooklyn Bridge",
    coordinates: { type: "Point", coordinates: [-73.9969, 40.7061] },
    locationType: "landmark",
    tags: ["nyc", "architecture", "historic"]
  }
]);

// Find locations within distance (in meters)
db.locations.find({
  coordinates: {
    $near: {
      $geometry: { type: "Point", coordinates: [-73.9855, 40.7580] },
      $maxDistance: 5000
    }
  }
});

// Alternative syntax using $geoWithin
db.locations.find({
  coordinates: {
    $geoWithin: {
      $centerSphere: [
        [-73.9855, 40.7580],
        5000 / 6378100  // radius in radians
      ]
    }
  }
});

// Find locations within polygon (GeoJSON)
db.locations.find({
  coordinates: {
    $geoWithin: {
      $polygon: [
        [-74, 40],
        [-73, 40],
        [-73, 41],
        [-74, 41],
        [-74, 40]
      ]
    }
  }
});

// Aggregate with geospatial $geoNear stage
db.locations.aggregate([
  {
    $geoNear: {
      near: { type: "Point", coordinates: [-73.9855, 40.7580] },
      distanceField: "distance",
      maxDistance: 10000,
      includeLocs: "coordinates",
      spherical: true
    }
  },
  {
    $sort: { distance: 1 }
  },
  {
    $limit: 10
  }
]);

// Geospatial query with additional filters
db.locations.find({
  coordinates: {
    $near: {
      $geometry: { type: "Point", coordinates: [-73.9855, 40.7580] },
      $maxDistance: 5000
    }
  },
  locationType: { $in: ["landmark", "park"] }
});

// Find locations within GeoJSON Polygon
db.locations.find({
  coordinates: {
    $geoWithin: {
      $geometry: {
        type: "Polygon",
        coordinates: [[
          [-74.1, 40.7],
          [-73.9, 40.7],
          [-73.9, 40.9],
          [-74.1, 40.9],
          [-74.1, 40.7]
        ]]
      }
    }
  }
});

// Calculate distance in aggregation pipeline
db.locations.aggregate([
  {
    $addFields: {
      distanceFromTimesSquare: {
        $divide: [
          {
            $sqrt: {
              $add: [
                { $pow: [{ $subtract: [{ $arrayElemAt: ["$coordinates.coordinates", 0] }, -73.9855] }, 2] },
                { $pow: [{ $subtract: [{ $arrayElemAt: ["$coordinates.coordinates", 1] }, 40.7580] }, 2] }
              ]
            }
          },
          111000  // meters per degree approximately
        ]
      }
    }
  },
  { $match: { distanceFromTimesSquare: { $lt: 5000 } } },
  { $sort: { distanceFromTimesSquare: 1 } }
]);
```

## JSON and JSONB Advanced Querying

JSON data handling has become essential as applications exchange semi-structured data. PostgreSQL's JSONB type provides both flexibility and performance through specialized indexing and query capabilities that rival document databases while maintaining relational integrity.

### JSONB Query Basics

JSONB stores JSON data in decomposed binary format, enabling efficient querying, indexing, and key existence checks. Unlike JSON type, JSONB supports indexing and has predictable key ordering.

```sql
-- Create table with JSONB column
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT,
    order_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert JSON data
INSERT INTO orders (customer_id, order_data) VALUES
(1, '{"items": [{"product_id": 101, "qty": 2, "price": 29.99}], 
      "shipping_address": {"city": "New York", "zip": "10001"}, 
      "status": "pending"}'),
(2, '{"items": [{"product_id": 102, "qty": 1, "price": 49.99}], 
      "shipping_address": {"city": "Boston", "zip": "02101"}, 
      "status": "shipped"}');

-- Query top-level keys
SELECT order_id, order_data->>'status' AS status
FROM orders;

-- Query nested objects
SELECT 
    order_id,
    order_data->'shipping_address'->>'city' AS city,
    order_data->'items'->0->>'product_id' AS first_product
FROM orders;

-- Query arrays
SELECT 
    order_id,
    jsonb_array_elements(order_data->'items') AS item
FROM orders;

-- Extract multiple array elements
SELECT 
    order_id,
    item->>'product_id' AS product_id,
    item->>'qty' AS quantity
FROM orders,
    jsonb_array_elements(order_data->'items') AS item;

-- Query with containment operator (@>) - find orders with pending status
SELECT *
FROM orders
WHERE order_data @> '{"status": "pending"}';

-- Query with existence operator (?) - find orders with specific key
SELECT *
FROM orders
WHERE order_data ? 'shipping_address';

-- Query with path existence (??) - find orders with specific value in key
SELECT *
FROM orders
WHERE order_data->'shipping_address' ? 'city';
```

### JSONB Indexing Strategies

Proper indexing dramatically improves JSONB query performance. GIN indexes support containment and existence operators, while expression indexes optimize specific key access patterns.

```sql
-- Create GIN index for JSONB (general purpose)
CREATE INDEX idx_orders_data ON orders USING GIN(order_data);

-- Create specific expression index for frequently queried paths
CREATE INDEX idx_orders_status ON orders ((order_data->>'status'));
CREATE INDEX idx_orders_city ON orders ((order_data->'shipping_address'->>'city'));

-- Create index for array contents
CREATE INDEX idx_orders_items ON orders USING GIN(order_data->'items');

-- Partial index for active orders
CREATE INDEX idx_pending_orders ON orders ((order_data->>'status'))
WHERE order_data->>'status' = 'pending';

-- Query using index (containment)
EXPLAIN ANALYZE
SELECT *
FROM orders
WHERE order_data @> '{"status": "pending"}';

-- Query using expression index
EXPLAIN ANALYZE
SELECT *
FROM orders
WHERE order_data->>'status' = 'pending';

-- JSONB path queries with filters
SELECT 
    order_id,
    item->>'product_id' AS product_id,
    (item->>'price')::numeric * (item->>'qty')::numeric AS line_total
FROM orders,
    jsonb_array_elements(order_data->'items') AS item
WHERE (item->>'qty')::int >= 2;
```

### JSONB Transformation and Aggregation

JSONB excels at transforming relational data to JSON and aggregating relational data into JSON structures for API responses.

```sql
-- Convert rows to JSON objects
SELECT jsonb_build_object(
    'order_id', order_id,
    'customer_id', customer_id,
    'status', order_data->>'status',
    'total', (
        SELECT sum((item->>'price')::numeric * (item->>'qty')::numeric)
        FROM jsonb_array_elements(order_data->'items') AS item
    )
) AS order_json
FROM orders;

-- Aggregate relational data to nested JSON
SELECT 
    customer_id,
    jsonb_agg(
        jsonb_build_object(
            'order_id', order_id,
            'status', order_data->>'status',
            'items', order_data->'items'
        )
    ) AS orders
FROM orders
GROUP BY customer_id;

-- JSONB object aggregation
SELECT 
    'orders' AS collection,
    jsonb_object_agg(order_id, order_data) AS data
FROM orders;

-- Recursive JSONB operations (PostgreSQL 14+)
SELECT 
    order_id,
    jsonb_path_query(order_data, '$.items[*] ? (@.qty >= 2)') AS large_order_items
FROM orders;

-- Nested JSON transformation
SELECT 
    customer_id,
    jsonb_build_object(
        'customer_orders', 
        jsonb_agg(
            jsonb_build_object(
                'order_id', order_id,
                'line_items', 
                (SELECT jsonb_agg(
                    jsonb_build_object(
                        'product_id', item->>'product_id',
                        'quantity', item->>'qty',
                        'subtotal', (item->>'price')::numeric * (item->>'qty')::numeric
                    )
                )
                FROM jsonb_array_elements(order_data->'items') AS item)
            )
        )
    ) AS customer_data
FROM orders
GROUP BY customer_id;
```

## Array Operations

PostgreSQL's array type enables efficient storage and querying of multi-valued attributes without the overhead of separate tables. This is particularly useful for tags, categories, and other scenarios where items naturally belong to multiple values.

### Array Basics and Queries

```sql
-- Create table with array columns
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    tags TEXT[],
    category_ids INTEGER[],
    price NUMERIC(10, 2),
    colors VARCHAR(50)[] DEFAULT '{}'
);

-- Insert array data
INSERT INTO products (name, tags, category_ids, colors) VALUES
('Laptop Pro', ARRAY['electronics', 'computer', 'premium'], 
 ARRAY[1, 5, 10], ARRAY['silver', 'black']),
('Wireless Mouse', ARRAY['electronics', 'accessories'], 
 ARRAY[1, 3], ARRAY['black', 'blue', 'white']);

-- Query array contents (contains)
SELECT *
FROM products
WHERE tags @> ARRAY['electronics', 'computer'];

-- Query with ANY (matches any element)
SELECT *
FROM products
WHERE 'premium' = ANY(tags);

-- Query with ALL (matches all elements)
SELECT *
FROM products
WHERE ARRAY['electronics', 'computer'] <@ tags;

-- Array overlap (any common elements)
SELECT *
FROM products
WHERE tags && ARRAY['sale', 'clearance'];

-- Unnest array to rows
SELECT product_id, unnest(tags) AS tag
FROM products;

-- Array length and dimensions
SELECT 
    name,
    array_length(tags, 1) AS tag_count,
    array_length(colors, 1) AS color_count
FROM products;

-- Array slice (1-based indexing)
SELECT tags[1:2] AS first_two_tags FROM products;
```

### Array Indexing and Performance

```sql
-- Create GiST index for array containment queries
CREATE INDEX idx_products_tags ON products USING GIN(tags);

-- Create B-tree index for array ordering
CREATE INDEX idx_products_tags_btree ON products USING btree(tags);

-- GIN index also supports array overlap
CREATE INDEX idx_products_category_ids ON products USING GIN(category_ids);

-- Query plan analysis for array operations
EXPLAIN ANALYZE
SELECT *
FROM products
WHERE tags @> ARRAY['electronics'];

-- Using array positions in queries
SELECT *
FROM products
WHERE category_ids[1] = 1;  -- First category is electronics
```

### Array Functions and Operations

```sql
-- Array concatenation
SELECT ARRAY[1, 2] || ARRAY[3, 4];  -- {1, 2, 3, 4}

-- Array append/prepend
SELECT array_append(ARRAY[1, 2], 3);  -- {1, 2, 3}
SELECT array_prepend(1, ARRAY[2, 3]);  -- {1, 2, 3}

-- Array removal
SELECT array_remove(ARRAY[1, 2, 3, 2], 2);  -- {1, 3}

-- Array replacement
SELECT array_replace(ARRAY[1, 2, 3, 2], 2, 99);  -- {1, 99, 3, 99}

-- Array to string
SELECT array_to_string(ARRAY['a', 'b', 'c'], ', ');  -- 'a, b, c'

-- String to array
SELECT string_to_array('a,b,c', ',');  -- {a,b,c}

-- Array aggregation
SELECT array_agg(product_id) FROM products WHERE price > 100;

-- Distinct array elements
SELECT array_distinct(ARRAY[1, 2, 2, 3]);  -- {1, 2, 3}

-- Array position
SELECT array_position(ARRAY['a', 'b', 'c'], 'b');  -- 2

-- Advanced: Generate arrays
SELECT generate_series(1, 10);  -- 1, 2, 3, ..., 10
SELECT ARRAY(SELECT generate_series(1, 10));  -- {1, 2, 3, ..., 10}
```

## Time-Series Queries

Time-series data requires specialized query patterns for efficient aggregation, downsampling, and gap filling. Whether using purpose-built time-series databases or relational tables, these patterns enable effective temporal analysis.

### Time-Based Aggregation Patterns

```sql
-- Create time-series table
CREATE TABLE sensor_readings (
    reading_id BIGSERIAL PRIMARY KEY,
    sensor_id INT,
    value NUMERIC(10, 4),
    recorded_at TIMESTAMPTZ NOT NULL
);

-- Create time-series optimized index
CREATE INDEX idx_sensor_recorded 
ON sensor_readings (sensor_id, recorded_at);

-- Bucket by hour with aggregation
SELECT 
    sensor_id,
    date_trunc('hour', recorded_at) AS hour,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    COUNT(*) AS reading_count
FROM sensor_reads
WHERE recorded_at >= NOW() - INTERVAL '7 days'
GROUP BY sensor_id, date_trunc('hour', recorded_at)
ORDER BY sensor_id, hour;

-- Multiple time buckets in one query
SELECT 
    sensor_id,
    date_trunc('day', recorded_at) AS day,
    date_trunc('hour', recorded_at) AS hour,
    AVG(value) AS avg_value
FROM sensor_readings
WHERE recorded_at >= NOW() - INTERVAL '24 hours'
GROUP BY ROLLUP(sensor_id, date_trunc('day', recorded_at), date_trunc('hour', recorded_at))
ORDER BY sensor_id, day, hour;

-- Time-weighted average (important for irregular intervals)
SELECT 
    sensor_id,
    date_trunc('hour', recorded_at) AS hour,
    SUM((value + LEAD(value) OVER w) / 2 * 
        EXTRACT(EPOCH FROM (LEAD(recorded_at) OVER w - recorded_at))) /
    NULLIF(SUM(EXTRACT(EPOCH FROM (LEAD(recorded_at) OVER w - recorded_at))), 0) AS twa
FROM sensor_readings
WHERE recorded_at >= NOW() - INTERVAL '24 hours'
WINDOW w AS (PARTITION BY sensor_id ORDER BY recorded_at)
ORDER BY sensor_id, hour;

-- Moving window calculations
SELECT 
    sensor_id,
    recorded_at,
    value,
    AVG(value) OVER (
        PARTITION BY sensor_id 
        ORDER BY recorded_at 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS moving_avg_5,
    AVG(value) OVER (
        PARTITION BY sensor_id 
        ORDER BY recorded_at 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS moving_avg_30
FROM sensor_readings
WHERE sensor_id = 1
ORDER BY recorded_at;
```

### Gap Filling and Interpolation

Real-world time-series data often has gaps that need handling for analysis and visualization.

```sql
-- Generate continuous time series using generate_series
WITH time_bounds AS (
    SELECT 
        DATE_TRUNC('hour', MIN(recorded_at)) AS start_time,
        DATE_TRUNC('hour', MAX(recorded_at)) AS end_time
    FROM sensor_readings
    WHERE sensor_id = 1
),
time_series AS (
    SELECT generate_series(
        start_time,
        end_time,
        INTERVAL '1 hour'
    ) AS hour
    FROM time_bounds
)
SELECT 
    ts.hour,
    COALESCE(sr.avg_value, 0) AS value,
    INTERPOLATE(sr.avg_value) AS interpolated_value
FROM time_series ts
LEFT JOIN (
    SELECT 
        date_trunc('hour', recorded_at) AS hour,
        AVG(value) AS avg_value
    FROM sensor_readings
    WHERE sensor_id = 1
    GROUP BY date_trunc('hour', recorded_at)
) sr ON ts.hour = sr.hour;

-- Forward fill (last observation carried forward)
WITH RECURSIVE filled_data AS (
    SELECT 
        recorded_at,
        value,
        LAG(value) OVER (ORDER BY recorded_at) AS prev_value,
        ROW_NUMBER() OVER (ORDER BY recorded_at) AS rn
    FROM sensor_readings
    WHERE sensor_id = 1
)
SELECT 
    recorded_at,
    COALESCE(value, 
        (SELECT value FROM filled_data f2 
         WHERE f2.rn < f1.rn AND f2.value IS NOT NULL 
         ORDER BY f2.rn DESC LIMIT 1)
    ) AS filled_value
FROM filled_data f1;

-- Linear interpolation for gaps
WITH grouped AS (
    SELECT 
        recorded_at,
        value,
        LEAD(recorded_at) OVER (ORDER BY recorded_at) AS next_ts,
        LEAD(value) OVER (ORDER BY recorded_at) AS next_val,
        LAG(recorded_at) OVER (ORDER BY recorded_at) AS prev_ts,
        LAG(value) OVER (ORDER BY recorded_at) AS prev_val,
        COUNT(*) OVER () AS total_count
    FROM sensor_readings
    WHERE sensor_id = 1
)
SELECT 
    recorded_at,
    value,
    CASE 
        WHEN value IS NULL AND next_val IS NOT NULL AND prev_val IS NOT NULL THEN
            prev_val + (next_val - prev_val) * 
            EXTRACT(EPOCH FROM (recorded_at - prev_ts)) /
            NULLIF(EXTRACT(EPOCH FROM (next_ts - prev_ts)), 0)
        ELSE value
    END AS interpolated
FROM grouped
ORDER BY recorded_at;
```

### Time-Series Window Functions

```sql
-- Cumulative sum over time
SELECT 
    sensor_id,
    recorded_at,
    value,
    SUM(value) OVER (
        PARTITION BY sensor_id 
        ORDER BY recorded_at
    ) AS cumulative_value
FROM sensor_readings
WHERE sensor_id = 1
ORDER BY recorded_at;

-- Rate of change
SELECT 
    sensor_id,
    recorded_at,
    value,
    LAG(value) OVER w AS prev_value,
    value - LAG(value) OVER w AS change,
    (value - LAG(value) OVER w) / NULLIF(LAG(value) OVER w, 0) * 100 AS pct_change
FROM sensor_readings
WINDOW w AS (PARTITION BY sensor_id ORDER BY recorded_at);

-- Time-based joins for event correlation
SELECT 
    e.event_id,
    e.event_time,
    s.sensor_id,
    s.value AS sensor_value,
    s.recorded_at AS closest_reading
FROM events e
CROSS JOIN LATERAL (
    SELECT sensor_id, value, recorded_at
    FROM sensor_readings
    WHERE sensor_id = e.sensor_id
    ORDER BY ABS(EXTRACT(EPOCH FROM (recorded_at - e.event_time)))
    LIMIT 1
) s;

-- Detect anomalies using standard deviation
WITH stats AS (
    SELECT 
        AVG(value) AS mean,
        STDDEV(value) AS stddev
    FROM sensor_readings
    WHERE sensor_id = 1
    AND recorded_at >= NOW() - INTERVAL '7 days'
)
SELECT 
    sr.recorded_at,
    sr.value,
    s.mean,
    s.stddev,
    CASE 
        WHEN ABS(sr.value - s.mean) > 3 * s.stddev THEN 'ANOMALY'
        ELSE 'NORMAL'
    END AS status
FROM sensor_readings sr, stats s
WHERE sr.sensor_id = 1
AND sr.recorded_at >= NOW() - INTERVAL '7 days'
ORDER BY sr.recorded_at;
```

### Retention Policies and Data Management

```sql
-- Create automatic partition for time-series data
CREATE TABLE sensor_readings_2024_01 (
    CHECK (recorded_at >= '2024-01-01' AND recorded_at < '2024-02-01')
) INHERITS (sensor_readings);

-- Create trigger for automatic partition routing
CREATE OR REPLACE FUNCTION sensor_readings_insert_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.recorded_at >= '2024-01-01' AND NEW.recorded_at < '2024-02-01' THEN
        INSERT INTO sensor_readings_2024_01 VALUES (NEW.*);
    -- Add more partitions as needed
    ELSE
        RAISE EXCEPTION 'Date out of range for partitioning';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER insert_sensor_reading_trigger
    BEFORE INSERT ON sensor_readings
    FOR EACH ROW EXECUTE FUNCTION sensor_readings_insert_trigger();

-- Delete old data with retention policy
DELETE FROM sensor_readings
WHERE recorded_at < NOW() - INTERVAL '1 year';

-- Vacuum to reclaim space after deletion
VACUUM sensor_readings;

-- Archive to cheaper storage
CREATE TABLE sensor_readings_archive (LIKE sensor_readings);

-- Move old data to archive
INSERT INTO sensor_readings_archive
SELECT * FROM sensor_readings
WHERE recorded_at < NOW() - INTERVAL '2 years';

DELETE FROM sensor_readings
WHERE recorded_at < NOW() - INTERVAL '2 years';
```

This comprehensive guide provides practical patterns for specialized database querying. Each technique addresses specific data types and use cases commonly encountered in modern applications. Combine these patterns strategically based on your application's requirements and data characteristics.
