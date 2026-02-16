# Database API Design Patterns

This document provides comprehensive guidance on designing robust, scalable, and maintainable APIs for database operations. It covers REST API design, GraphQL integration, real-time subscriptions, batch processing, Change Data Capture patterns, and API versioning strategies essential for building modern data-driven applications.

## Table of Contents

1. [REST API Design for Database Operations](#1-rest-api-design-for-database-operations)
2. [GraphQL with Databases](#2-graphql-with-databases)
3. [Real-Time Database Subscriptions](#3-real-time-database-subscriptions)
4. [Batch Data Processing APIs](#4-batch-data-processing-apis)
5. [Change Data Capture (CDC) Patterns](#5-change-data-capture-cdc-patterns)
6. [API Versioning Strategies](#6-api-versioning-strategies)

---

## 1. REST API Design for Database Operations

### 1.1 Resource Modeling Fundamentals

Effective REST API design begins with proper resource modeling. Resources should represent nouns (entities) rather than verbs (actions), and should be organized in a logical hierarchy that reflects the underlying domain model. For database operations, the primary resources typically include collections (tables), individual items (rows), and relationships between entities. The resource URL structure should be intuitive and follow consistent conventions across the entire API surface.

When designing database-facing REST APIs, consider the relationship between database tables and API resources. A single database table might map to multiple API resources depending on the use case, while multiple tables might be combined into a single resource for efficiency. The key principle is to design resources around business capabilities rather than technical implementation details. This approach ensures that the API remains stable even as the underlying database schema evolves.

```
API Resource Structure:
/api/v1/users              -> User collection (GET, POST)
/api/v1/users/{id}         -> Individual user (GET, PUT, PATCH, DELETE)
/api/v1/users/{id}/orders  -> User's orders (GET, POST)
/api/v1/orders/{id}/items  -> Order items (GET)
```

### 1.2 HTTP Method Mapping

Proper HTTP method mapping is essential for creating intuitive and predictable APIs. Each HTTP method has specific semantics that should be carefully followed to ensure clients can correctly interact with the API. The standard mapping includes GET for retrieval, POST for creation, PUT for full updates, PATCH for partial updates, and DELETE for removal. Understanding these semantics and applying them consistently is crucial for building APIs that developers can easily understand and consume.

For database operations specifically, GET requests should be idempotent and never modify data. POST requests create new resources and return the created resource with its assigned identifier. PUT requests replace entire resources and should be idempotent, meaning multiple identical requests produce the same result. PATCH requests partially update resources and are not necessarily idempotent. DELETE operations remove resources and should be idempotent, with subsequent deletions of the same resource returning a successful response or indicating the resource no longer exists.

### 1.3 OpenAPI Specification for Database Operations

The following OpenAPI specification demonstrates a complete database API design for a typical user management system with order tracking capabilities. This specification includes comprehensive definitions for request bodies, response schemas, error handling, and authentication requirements.

```yaml
openapi: 3.0.3
info:
  title: Database Operations API
  description: REST API for managing users and orders with comprehensive database operations
  version: 1.0.0
  contact:
    name: API Support
    email: api-support@example.com

servers:
  - url: https://api.example.com/v1
    description: Production server
  - url: https://staging-api.example.com/v1
    description: Staging server

security:
  - BearerAuth: []

paths:
  /users:
    get:
      summary: List all users with pagination and filtering
      description: |
        Retrieves a paginated list of users. Supports filtering by status, 
        role, and date range. Results are sorted by creation date by default.
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
            minimum: 1
          description: Page number for pagination
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
            minimum: 1
          description: Number of items per page
        - name: status
          in: query
          schema:
            type: string
            enum: [active, inactive, suspended]
          description: Filter by user status
        - name: role
          in: query
          schema:
            type: string
            enum: [admin, user, guest]
          description: Filter by user role
        - name: sort
          in: query
          schema:
            type: string
            default: created_at
          description: Sort field (prefix with - for descending)
        - name: created_after
          in: query
          schema:
            type: string
            format: date-time
          description: Filter users created after this timestamp
      responses:
        '200':
          description: Successful response with paginated users
          headers:
            X-Total-Count:
              schema:
                type: integer
              description: Total number of matching records
            X-Total-Pages:
              schema:
                type: integer
              description: Total number of pages
            X-Current-Page:
              schema:
                type: integer
              description: Current page number
            Link:
              schema:
                type: string
              description: Pagination links (RFC 5988)
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  pagination:
                    $ref: '#/components/schemas/Pagination'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '429':
          $ref: '#/components/responses/TooManyRequests'

    post:
      summary: Create a new user
      description: Creates a new user record in the database
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
            examples:
              standard_user:
                summary: Standard user creation
                value:
                  email: john.doe@example.com
                  username: johndoe
                  first_name: John
                  last_name: Doe
                  password: SecurePassword123!
                  role: user
      responses:
        '201':
          description: User created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          $ref: '#/components/responses/BadRequest'
        '409':
          $ref: '#/components/responses/Conflict'
        '422':
          $ref: '#/components/responses/UnprocessableEntity'

  /users/{id}:
    parameters:
      - name: id
        in: path
        required: true
        schema:
          type: string
          format: uuid
        description: Unique identifier of the user

    get:
      summary: Get user by ID
      description: Retrieves detailed information about a specific user
      responses:
        '200':
          description: User found and returned
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserDetail'
        '404':
          $ref: '#/components/responses/NotFound'

    put:
      summary: Full user update
      description: Replaces all user fields with provided values
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateUserRequest'
      responses:
        '200':
          description: User updated successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          $ref: '#/components/responses/NotFound'
        '409':
          $ref: '#/components/responses/Conflict'

    patch:
      summary: Partial user update
      description: Updates only the provided fields
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PatchUserRequest'
      responses:
        '200':
          description: User patched successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          $ref: '#/components/responses/NotFound'

    delete:
      summary: Delete user
      description: Permanently removes user from database
      responses:
        '204':
          description: User deleted successfully
        '404':
          $ref: '#/components/responses/NotFound'
        '409':
          description: Cannot delete user with associated records
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /users/{id}/orders:
    parameters:
      - name: id
        in: path
        required: true
        schema:
          type: string
          format: uuid

    get:
      summary: Get user's orders
      description: Retrieves all orders associated with a specific user
      responses:
        '200':
          description: Orders retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Order'

    post:
      summary: Create order for user
      description: Creates a new order associated with the specified user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateOrderRequest'
      responses:
        '201':
          description: Order created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Order'

  /batch/users:
    post:
      summary: Batch create users
      description: Creates multiple users in a single request
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                users:
                  type: array
                  maxItems: 1000
                  items:
                    $ref: '#/components/schemas/CreateUserRequest'
              required: [users]
      responses:
        '201':
          description: Users created successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  created:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  errors:
                    type: array
                    items:
                      type: object
                      properties:
                        index:
                          type: integer
                        error:
                          type: string

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: JWT token authentication

  schemas:
    User:
      type: object
      properties:
        id:
          type: string
          format: uuid
          description: Unique identifier
        email:
          type: string
          format: email
        username:
          type: string
          minLength: 3
          maxLength: 50
        first_name:
          type: string
        last_name:
          type: string
        role:
          type: string
          enum: [admin, user, guest]
        status:
          type: string
          enum: [active, inactive, suspended]
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time
      required:
        - id
        - email
        - username
        - created_at
        - updated_at

    UserDetail:
      allOf:
        - $ref: '#/components/schemas/User'
        - type: object
          properties:
            last_login_at:
              type: string
              format: date-time
            order_count:
              type: integer
            total_spent:
              type: number

    CreateUserRequest:
      type: object
      properties:
        email:
          type: string
          format: email
        username:
          type: string
        first_name:
          type: string
        last_name:
          type: string
        password:
          type: string
          minLength: 8
          format: password
        role:
          type: string
          enum: [admin, user, guest]
          default: user
      required:
        - email
        - username
        - password

    UpdateUserRequest:
      allOf:
        - $ref: '#/components/schemas/CreateUserRequest'
        - type: object
          properties:
            id:
              type: string
              format: uuid

    PatchUserRequest:
      type: object
      properties:
        email:
          type: string
          format: email
        first_name:
          type: string
        last_name:
          type: string
        status:
          type: string
          enum: [active, inactive, suspended]

    Order:
      type: object
      properties:
        id:
          type: string
          format: uuid
        user_id:
          type: string
          format: uuid
        status:
          type: string
          enum: [pending, processing, shipped, delivered, cancelled]
        total:
          type: number
        currency:
          type: string
          default: USD
        items:
          type: array
          items:
            $ref: '#/components/schemas/OrderItem'
        created_at:
          type: string
          format: date-time

    OrderItem:
      type: object
      properties:
        product_id:
          type: string
          format: uuid
        quantity:
          type: integer
          minimum: 1
        unit_price:
          type: number

    CreateOrderRequest:
      type: object
      properties:
        items:
          type: array
          items:
            $ref: '#/components/schemas/OrderItem'
        shipping_address:
          type: string
      required:
        - items

    Pagination:
      type: object
      properties:
        page:
          type: integer
        limit:
          type: integer
        total:
          type: integer
        total_pages:
          type: integer
        has_next:
          type: boolean
        has_prev:
          type: boolean

    Error:
      type: object
      properties:
        code:
          type: string
        message:
          type: string
        details:
          type: array
          items:
            type: object
            properties:
              field:
                type: string
              message:
                type: string
        request_id:
          type: string

  responses:
    BadRequest:
      description: Invalid request parameters
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: BAD_REQUEST
            message: Invalid request parameters
            details:
              - field: email
                message: Invalid email format

    Unauthorized:
      description: Authentication required
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

    Conflict:
      description: Resource conflict
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

    UnprocessableEntity:
      description: Validation failed
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

    TooManyRequests:
      description: Rate limit exceeded
      headers:
        Retry-After:
          schema:
            type: integer
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
```

### 1.4 Query Parameter Design

Effective query parameter design enables flexible data retrieval without requiring multiple endpoints. Standard query parameters should include pagination controls (page, limit), sorting options (sort, order), and field selection (fields) to optimize bandwidth. Filtering parameters should be clearly documented and consistently named across the API. The filter syntax should support common operators such as equals, greater than, less than, contains, and in to provide maximum flexibility for clients.

For complex queries, consider implementing a filter expression language that allows clients to specify sophisticated conditions. This approach balances flexibility with API complexity. Additionally, include metadata parameters like include to support related resource embedding and expand to control the depth of included relationships. Response field filtering using the fields parameter allows clients to request only the specific fields they need, reducing payload size significantly.

### 1.5 Response Envelope Design

Response envelope design significantly impacts client experience and API usability. A well-designed response envelope provides consistent structure across all endpoints, includes helpful metadata, and maintains backward compatibility as the API evolves. The envelope should include the primary data payload, metadata about the request (such as processing time), pagination information when applicable, and optionally, linked resources or related data.

```
Standard Response Envelope:
{
  "data": { ... },
  "meta": {
    "request_id": "req_abc123",
    "timestamp": "2026-02-16T10:30:00Z",
    "processing_time_ms": 45
  },
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "total_pages": 8
  },
  "links": {
    "self": "/api/v1/users?page=1",
    "next": "/api/v1/users?page=2",
    "first": "/api/v1/users?page=1",
    "last": "/api/v1/users?page=8"
  },
  "included": {
    "orders": [...]
  }
}
```

---

## 2. GraphQL with Databases

### 2.1 Schema Design for Database Mappings

GraphQL provides a flexible type system that maps naturally to database schemas while offering significant advantages over REST for complex data requirements. The schema should be designed to mirror the domain model rather than the database structure directly, providing an abstraction layer that allows the underlying database to evolve without breaking clients. Each database table typically corresponds to a GraphQL type, with relationships defined through field connections rather than foreign key IDs.

When mapping database schemas to GraphQL types, consider the access patterns rather than simply exposing all database fields. The schema should enforce data loading patterns that prevent N+1 query problems through proper use of DataLoader implementations. Additionally, the schema should support both querying existing data and mutations for creating, updating, and deleting records, with input types that mirror the structure of the corresponding types.

```graphql
# GraphQL Schema Definition for Database Operations

scalar DateTime
scalar UUID
scalar JSON

type Query {
  # User queries
  user(id: UUID!): User
  users(
    filter: UserFilterInput
    sort: UserSortInput
    pagination: PaginationInput
  ): UserConnection!
  
  # Order queries
  order(id: UUID!): Order
  orders(
    filter: OrderFilterInput
    sort: OrderSortInput
    pagination: PaginationInput
  ): OrderConnection!
  userOrders(userId: UUID!, pagination: PaginationInput): OrderConnection!
  
  # Product queries with full-text search
  products(
    search: String
    filter: ProductFilterInput
    sort: ProductSortInput
    pagination: PaginationInput
  ): ProductConnection!
  
  # Analytics aggregation queries
  userOrderStats(userId: UUID!): OrderStats!
  dashboardMetrics(dateRange: DateRangeInput!): DashboardMetrics!
}

type Mutation {
  # User mutations
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(id: UUID!, input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(id: UUID!): DeleteUserPayload!
  bulkCreateUsers(input: BulkCreateUsersInput!): BulkCreateUsersPayload!
  
  # Order mutations
  createOrder(input: CreateOrderInput!): CreateOrderPayload!
  updateOrderStatus(id: UUID!, status: OrderStatus!): UpdateOrderPayload!
  cancelOrder(id: UUID!): CancelOrderPayload!
  
  # Transactional batch operations
  processOrderBatch(input: OrderBatchInput!): OrderBatchPayload!
}

# Node interface for relay cursor pagination
interface Node {
  id: UUID!
}

interface Edge {
  cursor: String!
  node: Node!
}

type User implements Node {
  id: UUID!
  email: String!
  username: String!
  firstName: String!
  lastName: String!
  role: UserRole!
  status: UserStatus!
  createdAt: DateTime!
  updatedAt: DateTime!
  
  # Relationships
  orders(
    filter: OrderFilterInput
    sort: OrderSortInput
    pagination: PaginationInput
  ): OrderConnection!
  orderCount: Int!
  totalSpent: Float!
  
  # Computed fields
  fullName: String!
  isActive: Boolean!
}

type Order implements Node {
  id: UUID!
  userId: UUID!
  user: User!
  status: OrderStatus!
  total: Float!
  currency: String!
  items: [OrderItem!]!
  shippingAddress: String
  createdAt: DateTime!
  updatedAt: DateTime!
  
  # Computed
  itemCount: Int!
  subtotal: Float!
  tax: Float!
}

type OrderItem {
  id: UUID!
  productId: UUID!
  product: Product!
  quantity: Int!
  unitPrice: Float!
  total: Float!
}

type Product implements Node {
  id: UUID!
  name: String!
  description: String
  price: Float!
  currency: String!
  category: Category!
  inventory: Int!
  isAvailable: Boolean!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type UserEdge {
  cursor: String!
  node: User!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

type OrderConnection {
  edges: [OrderEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
  aggregate: OrderAggregate!
}

type OrderAggregate {
  count: Int!
  sum: Float
  avg: Float
  min: Float
  max: Float
}

# Input types for filtering and sorting
input UserFilterInput {
  and: UserFilterInput
  or: [UserFilterInput!]
  id: UUIDFilter
  email: StringFilter
  username: StringFilter
  role: UserRoleFilter
  status: UserStatusFilter
  createdAt: DateTimeFilter
}

input UUIDFilter {
  eq: UUID
  in: [UUID!]
  ne: UUID
  nin: [UUID!]
}

input StringFilter {
  eq: String
  contains: String
  startsWith: String
  endsWith: String
  in: [String!]
}

input DateTimeFilter {
  eq: DateTime
  gt: DateTime
  gte: DateTime
  lt: DateTime
  lte: DateTime
  between: DateTimeRange
}

input UserSortInput {
  field: UserSortField!
  order: SortOrder
}

enum UserSortField {
  CREATED_AT
  UPDATED_AT
  EMAIL
  USERNAME
}

enum SortOrder {
  ASC
  DESC
}

input PaginationInput {
  first: Int
  after: String
  last: Int
  before: String
}

# Mutation input types
input CreateUserInput {
  email: String!
  username: String!
  firstName: String!
  lastName: String!
  password: String!
  role: UserRole = USER
}

input UpdateUserInput {
  email: String
  firstName: String
  lastName: String
  status: UserStatus
}

input CreateOrderInput {
  userId: UUID!
  items: [OrderItemInput!]!
  shippingAddress: String
}

input OrderItemInput {
  productId: UUID!
  quantity: Int!
}

# Mutation payload types
type CreateUserPayload {
  user: User!
  errors: [ValidationError!]
}

type UpdateUserPayload {
  user: User
  errors: [ValidationError!]
}

type DeleteUserPayload {
  success: Boolean!
  deletedId: UUID
}

type BulkCreateUsersPayload {
  users: [User!]!
  errors: [BulkOperationError!]
}

type ValidationError {
  field: String!
  message: String!
  code: String!
}

type BulkOperationError {
  index: Int!
  errors: [ValidationError!]!
}

# Enums
enum UserRole {
  ADMIN
  USER
  GUEST
}

enum UserStatus {
  ACTIVE
  INACTIVE
  SUSPENDED
}

enum OrderStatus {
  PENDING
  PROCESSING
  SHIPPED
  DELIVERED
  CANCELLED
}
```

### 2.2 DataLoader Implementation for Database Queries

DataLoader is essential for preventing N+1 query problems in GraphQL implementations. By batching and caching database queries, DataLoader ensures that multiple field resolutions requesting the same data execute a single database query rather than multiple individual queries. This pattern is particularly important when resolving nested relationships or when the same entity is requested multiple times in a single query.

The implementation should include loaders for each entity type with appropriate batch loading functions. Each loader should accept a list of identifiers and return a map of results keyed by identifier. The cache should be configured based on the query patterns, and the clear method should be used when mutations modify data to prevent stale cache issues. For complex queries, consider implementing query result caching at the database level to improve performance.

```javascript
// DataLoader implementation for database queries

const DataLoader = require('dataloader');
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

// Batch loader for users by ID
const createUserLoader = () => {
  return new DataLoader(async (userIds) => {
    const uniqueIds = [...new Set(userIds)];
    
    const result = await pool.query(
      `SELECT id, email, username, first_name, last_name, role, status, created_at, updated_at
       FROM users WHERE id = ANY($1)`,
      [uniqueIds]
    );
    
    const userMap = new Map();
    result.rows.forEach((user) => {
      userMap.set(user.id, user);
    });
    
    // Maintain order from userIds
    return userIds.map((id) => userMap.get(id) || null);
  });
};

// Batch loader for orders by user ID
const createOrdersByUserLoader = () => {
  return new DataLoader(async (userIds) => {
    const uniqueIds = [...new Set(userIds)];
    
    const result = await pool.query(
      `SELECT o.id, o.user_id, o.status, o.total, o.currency, o.created_at, o.updated_at,
              json_agg(
                json_build_object(
                  'id', oi.id,
                  'product_id', oi.product_id,
                  'quantity', oi.quantity,
                  'unit_price', oi.unit_price
                )
              ) as items
       FROM orders o
       LEFT JOIN order_items oi ON o.id = oi.order_id
       WHERE o.user_id = ANY($1)
       GROUP BY o.id`,
      [uniqueIds]
    );
    
    const ordersByUser = new Map();
    result.rows.forEach((order) => {
      const userOrders = ordersByUser.get(order.user_id) || [];
      userOrders.push(order);
      ordersByUser.set(order.user_id, userOrders);
    });
    
    return userIds.map((id) => ordersByUser.get(id) || []);
  });
};

// Batch loader for products by ID
const createProductLoader = () => {
  return new DataLoader(async (productIds) => {
    const uniqueIds = [...new Set(productIds)];
    
    const result = await pool.query(
      `SELECT p.id, p.name, p.description, p.price, p.currency, p.category_id,
              p.inventory, p.is_available, p.created_at, p.updated_at,
              c.name as category_name
       FROM products p
       LEFT JOIN categories c ON p.category_id = c.id
       WHERE p.id = ANY($1)`,
      [uniqueIds]
    );
    
    const productMap = new Map();
    result.rows.forEach((product) => {
      productMap.set(product.id, {
        ...product,
        category: product.category_name ? {
          id: product.category_id,
          name: product.category_name
        } : null
      });
    });
    
    return productIds.map((id) => productMap.get(id) || null);
  });
};

// Aggregate loader for user statistics
const createUserStatsLoader = () => {
  return new DataLoader(async (userIds) => {
    const uniqueIds = [...new Set(userIds)];
    
    const result = await pool.query(
      `SELECT user_id, 
              COUNT(*) as order_count, 
              COALESCE(SUM(total), 0) as total_spent
       FROM orders
       WHERE user_id = ANY($1)
       GROUP BY user_id`,
      [uniqueIds]
    );
    
    const statsMap = new Map();
    result.rows.forEach((stat) => {
      statsMap.set(stat.user_id, {
        orderCount: parseInt(stat.order_count),
        totalSpent: parseFloat(stat.total_spent)
      });
    });
    
    return userIds.map((id) => statsMap.get(id) || { orderCount: 0, totalSpent: 0 });
  });
};

// Context creation with loaders
const createContext = () => {
  return {
    userLoader: createUserLoader(),
    ordersByUserLoader: createOrdersByUserLoader(),
    productLoader: createProductLoader(),
    userStatsLoader: createUserStatsLoader(),
  };
};

module.exports = { createContext };
```

### 2.3 GraphQL Resolver Patterns for Database Operations

Resolver implementation requires careful attention to database interaction patterns, error handling, and data transformation. Each resolver should follow a consistent pattern that includes input validation, database query execution, result transformation, and error handling. For mutations, the pattern should include transaction management to ensure data consistency across multiple operations.

The resolver implementation should leverage the DataLoader instances from the context and should never execute database queries directly within field resolvers. Instead, use the DataLoader to batch and cache queries. For complex mutations that involve multiple database operations, implement proper transaction handling with rollback capability. Additionally, include proper logging and monitoring to track query performance and identify bottlenecks.

```javascript
// GraphQL resolver implementations

const { AuthenticationError, ForbiddenError, ValidationError } = require('graphql-errors');
const { v4: uuidv4 } = require('uuid');
const { pool } = require('../database');
const { createContext } = require('../dataloaders');

const resolvers = {
  Query: {
    // User queries
    user: async (parent, { id }, context) => {
      const users = await context.userLoader.load(id);
      if (!users) {
        throw new ValidationError('User not found');
      }
      return users;
    },
    
    users: async (parent, { filter, sort, pagination }, context) => {
      const { rows, total } = await queryUsersWithFilter(filter, sort, pagination);
      
      return {
        edges: rows.map((user) => ({
          cursor: Buffer.from(user.id).toString('base64'),
          node: user
        })),
        pageInfo: {
          hasNextPage: (pagination.page || 1) * pagination.limit < total,
          hasPreviousPage: (pagination.page || 1) > 1,
          startCursor: rows[0] ? Buffer.from(rows[0].id).toString('base64') : null,
          endCursor: rows[rows.length - 1] ? Buffer.from(rows[rows.length - 1].id).toString('base64') : null
        },
        totalCount: total
      };
    },
    
    // Order queries
    orders: async (parent, { filter, sort, pagination }, context) => {
      const { rows, total, aggregate } = await queryOrdersWithFilter(filter, sort, pagination);
      
      return {
        edges: rows.map((order) => ({
          cursor: Buffer.from(order.id).toString('base64'),
          node: order
        })),
        pageInfo: {
          hasNextPage: (pagination.page || 1) * pagination.limit < total,
          hasPreviousPage: (pagination.page || 1) > 1
        },
        totalCount: total,
        aggregate: {
          count: parseInt(aggregate.count),
          sum: parseFloat(aggregate.sum || 0),
          avg: parseFloat(aggregate.avg || 0)
        }
      };
    },
    
    userOrders: async (parent, { userId, pagination }, context) => {
      const filter = { userId: { eq: userId } };
      const { rows, total } = await queryOrdersWithFilter(filter, null, pagination);
      
      return {
        edges: rows.map((order) => ({
          cursor: Buffer.from(order.id).toString('base64'),
          node: order
        })),
        pageInfo: {
          hasNextPage: (pagination.page || 1) * pagination.limit < total,
          hasPreviousPage: (pagination.page || 1) > 1
        },
        totalCount: total
      };
    },
    
    userOrderStats: async (parent, { userId }, context) => {
      const stats = await context.userStatsLoader.load(userId);
      return stats;
    }
  },
  
  Mutation: {
    createUser: async (parent, { input }, context) => {
      // Validate input
      const errors = validateUserInput(input);
      if (errors.length > 0) {
        return { errors };
      }
      
      // Check for existing user
      const existingUser = await pool.query(
        'SELECT id FROM users WHERE email = $1 OR username = $2',
        [input.email, input.username]
      );
      
      if (existingUser.rows.length > 0) {
        return {
          errors: [{
            field: 'email/username',
            message: 'User with this email or username already exists',
            code: 'DUPLICATE_ENTRY'
          }]
        };
      }
      
      // Hash password
      const hashedPassword = await hashPassword(input.password);
      
      // Insert user
      const result = await pool.query(
        `INSERT INTO users (id, email, username, first_name, last_name, password_hash, role, status, created_at, updated_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())
         RETURNING *`,
        [
          uuidv4(),
          input.email,
          input.username,
          input.firstName,
          input.lastName,
          hashedPassword,
          input.role || 'USER',
          'ACTIVE'
        ]
      );
      
      return { user: result.rows[0], errors: [] };
    },
    
    updateUser: async (parent, { id, input }, context) => {
      // Verify user exists
      const existingUser = await context.userLoader.load(id);
      if (!existingUser) {
        return {
          errors: [{
            field: 'id',
            message: 'User not found',
            code: 'NOT_FOUND'
          }]
        };
      }
      
      // Build dynamic update query
      const updates = [];
      const values = [];
      let paramIndex = 1;
      
      if (input.email !== undefined) {
        updates.push(`email = $${paramIndex++}`);
        values.push(input.email);
      }
      if (input.firstName !== undefined) {
        updates.push(`first_name = $${paramIndex++}`);
        values.push(input.firstName);
      }
      if (input.lastName !== undefined) {
        updates.push(`last_name = $${paramIndex++}`);
        values.push(input.lastName);
      }
      if (input.status !== undefined) {
        updates.push(`status = $${paramIndex++}`);
        values.push(input.status);
      }
      
      if (updates.length === 0) {
        return { user: existingUser, errors: [] };
      }
      
      updates.push(`updated_at = NOW()`);
      values.push(id);
      
      const result = await pool.query(
        `UPDATE users SET ${updates.join(', ')} WHERE id = $${paramIndex} RETURNING *`,
        values
      );
      
      // Clear loader cache for this user
      context.userLoader.clear(id);
      
      return { user: result.rows[0], errors: [] };
    },
    
    deleteUser: async (parent, { id }, context) => {
      const result = await pool.query(
        'DELETE FROM users WHERE id = $1 RETURNING id',
        [id]
      );
      
      if (result.rows.length === 0) {
        return { success: false, deletedId: null };
      }
      
      // Clear loader cache
      context.userLoader.clear(id);
      
      return { success: true, deletedId: id };
    },
    
    bulkCreateUsers: async (parent, { input }, context) => {
      const { users } = input;
      const createdUsers = [];
      const errors = [];
      
      const client = await pool.connect();
      
      try {
        await client.query('BEGIN');
        
        for (let i = 0; i < users.length; i++) {
          const userInput = users[i];
          
          try {
            const validationErrors = validateUserInput(userInput);
            if (validationErrors.length > 0) {
              errors.push({ index: i, errors: validationErrors });
              continue;
            }
            
            const hashedPassword = await hashPassword(userInput.password);
            
            const result = await client.query(
              `INSERT INTO users (id, email, username, first_name, last_name, password_hash, role, status, created_at, updated_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())
               RETURNING *`,
              [
                uuidv4(),
                userInput.email,
                userInput.username,
                userInput.firstName,
                userInput.lastName,
                hashedPassword,
                userInput.role || 'USER',
                'ACTIVE'
              ]
            );
            
            createdUsers.push(result.rows[0]);
          } catch (error) {
            errors.push({
              index: i,
              errors: [{
                field: 'database',
                message: error.message,
                code: 'INSERT_FAILED'
              }]
            });
          }
        }
        
        await client.query('COMMIT');
      } catch (error) {
        await client.query('ROLLBACK');
        throw error;
      } finally {
        client.release();
      }
      
      return { users: createdUsers, errors };
    },
    
    createOrder: async (parent, { input }, context) => {
      const { userId, items, shippingAddress } = input;
      
      // Verify user exists
      const user = await context.userLoader.load(userId);
      if (!user) {
        return {
          errors: [{
            field: 'userId',
            message: 'User not found',
            code: 'NOT_FOUND'
          }]
        };
      }
      
      const client = await pool.connect();
      
      try {
        await client.query('BEGIN');
        
        // Verify products and calculate totals
        const productIds = items.map(item => item.productId);
        const products = await context.productLoader.loadMany(productIds);
        
        let total = 0;
        const orderItems = [];
        
        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          const product = products[i];
          
          if (!product) {
            throw new ValidationError(`Product not found: ${item.productId}`);
          }
          
          if (product.inventory < item.quantity) {
            throw new ValidationError(`Insufficient inventory for product: ${product.name}`);
          }
          
          const itemTotal = product.price * item.quantity;
          total += itemTotal;
          
          orderItems.push({
            productId: item.productId,
            quantity: item.quantity,
            unitPrice: product.price
          });
        }
        
        // Create order
        const orderId = uuidv4();
        
        await client.query(
          `INSERT INTO orders (id, user_id, status, total, currency, shipping_address, created_at, updated_at)
           VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())`,
          [orderId, userId, 'PENDING', total, 'USD', shippingAddress]
        );
        
        // Create order items
        for (const item of orderItems) {
          await client.query(
            `INSERT INTO order_items (id, order_id, product_id, quantity, unit_price, created_at)
             VALUES ($1, $2, $3, $4, $5, NOW())`,
            [uuidv4(), orderId, item.productId, item.quantity, item.unitPrice]
          );
          
          // Update inventory
          await client.query(
            'UPDATE products SET inventory = inventory - $1 WHERE id = $2',
            [item.quantity, item.productId]
          );
        }
        
        await client.query('COMMIT');
        
        // Clear caches
        context.ordersByUserLoader.clear(userId);
        context.userStatsLoader.clear(userId);
        
        // Fetch created order
        const orderResult = await pool.query('SELECT * FROM orders WHERE id = $1', [orderId]);
        
        return { order: { ...orderResult.rows[0], items: orderItems }, errors: [] };
      } catch (error) {
        await client.query('ROLLBACK');
        return {
          errors: [{
            field: 'order',
            message: error.message,
            code: 'ORDER_FAILED'
          }]
        };
      } finally {
        client.release();
      }
    }
  },
  
  // Field resolvers
  User: {
    orders: async (user, { filter, sort, pagination }, context) => {
      const userFilter = { ...filter, userId: { eq: user.id } };
      const { rows, total } = await queryOrdersWithFilter(userFilter, sort, pagination);
      
      return {
        edges: rows.map((order) => ({
          cursor: Buffer.from(order.id).toString('base64'),
          node: order
        })),
        pageInfo: {
          hasNextPage: (pagination.page || 1) * pagination.limit < total,
          hasPreviousPage: (pagination.page || 1) > 1
        },
        totalCount: total
      };
    },
    
    orderCount: async (user, args, context) => {
      const stats = await context.userStatsLoader.load(user.id);
      return stats.orderCount;
    },
    
    totalSpent: async (user, args, context) => {
      const stats = await context.userStatsLoader.load(user.id);
      return stats.totalSpent;
    },
    
    fullName: (user) => {
      return `${user.first_name} ${user.last_name}`;
    },
    
    isActive: (user) => {
      return user.status === 'ACTIVE';
    }
  },
  
  Order: {
    user: async (order, args, context) => {
      return context.userLoader.load(order.user_id);
    },
    
    items: async (order, args, context) => {
      const result = await pool.query(
        'SELECT * FROM order_items WHERE order_id = $1',
        [order.id]
      );
      return result.rows;
    },
    
    itemCount: async (order, args, context) => {
      const result = await pool.query(
        'SELECT COUNT(*) as count FROM order_items WHERE order_id = $1',
        [order.id]
      );
      return parseInt(result.rows[0].count);
    },
    
    subtotal: (order) => order.total,
    tax: (order) => order.total * 0.1
  },
  
  OrderItem: {
    product: async (item, args, context) => {
      return context.productLoader.load(item.product_id);
    },
    
    total: (item) => item.quantity * item.unit_price
  }
};

module.exports = resolvers;
```

---

## 3. Real-Time Database Subscriptions

### 3.1 WebSocket API Design

Real-time database subscriptions require WebSocket connections to push data changes to clients as they occur. The WebSocket API design should include connection establishment, authentication, subscription management, message format specifications, and graceful disconnection handling. Unlike HTTP APIs where clients initiate all requests, WebSocket subscriptions require the server to push updates, necessitating a different design approach for authentication and authorization.

The message protocol should support subscription requests, unsubscription requests, and data push messages. Each message should include a type identifier, a correlation ID for tracking request-response pairs, and the payload. The protocol should also support heartbeats to maintain connection health and detect stale connections. For database subscriptions, the server must track active subscriptions and efficiently forward relevant database changes to subscribed clients.

```
WebSocket Message Protocol:
Client -> Server:
{
  "type": "subscribe",
  "id": "msg_001",
  "payload": {
    "channel": "users",
    "filter": {
      "status": "active"
    },
    "fields": ["id", "email", "status"]
  }
}

Server -> Client:
{
  "type": "data",
  "id": "msg_001",
  "payload": {
    "event": "create",
    "data": {
      "id": "uuid-123",
      "email": "user@example.com",
      "status": "active"
    },
    "timestamp": "2026-02-16T10:30:00Z"
  }
}

Server -> Client (subscription confirmation):
{
  "type": "ack",
  "id": "msg_001",
  "payload": {
    "subscriptionId": "sub_abc123",
    "status": "active"
  }
}
```

### 3.2 Database Event Subscription Implementation

Implementing database event subscriptions requires capturing database changes and efficiently routing them to interested clients. This typically involves monitoring the database transaction log or using database-specific notification mechanisms. The implementation should handle connection failures, message buffering during disconnection, and proper cleanup of subscriptions when clients disconnect.

The following implementation demonstrates a complete WebSocket-based subscription system using PostgreSQL LISTEN/NOTIFY mechanism for real-time database change notifications. This pattern provides near-instantaneous delivery of database changes with minimal overhead.

```javascript
// WebSocket Server with Database Subscriptions

const WebSocket = require('ws');
const { Pool } = require('pg');
const EventEmitter = require('events');
const { v4: uuidv4 } = require('uuid');

class DatabaseSubscriptionServer extends EventEmitter {
  constructor(options = {}) {
    super();
    this.port = options.port || 8080;
    this.pool = new Pool({
      connectionString: options.databaseUrl,
      max: 20,
      idleTimeoutMillis: 30000
    });
    
    this.subscriptions = new Map(); // subscriptionId -> { client, channel, filter }
    this.clients = new Map(); // ws -> { id, subscriptions, auth }
    
    this.setupDatabaseListeners();
  }
  
  async start() {
    this.wss = new WebSocket.Server({ port: this.port });
    
    this.wss.on('connection', (ws, req) => {
      this.handleConnection(ws, req);
    });
    
    console.log(`Database subscription server running on port ${this.port}`);
  }
  
  setupDatabaseListeners() {
    // Listen for database notifications
    this.pool.on('notification', (msg) => {
      this.handleDatabaseNotification(msg);
    });
    
    // Subscribe to database change events
    this.setupDatabaseNotificationChannels();
  }
  
  async setupDatabaseNotificationChannels() {
    const client = await this.pool.connect();
    
    try {
      // Listen for user changes
      await client.query('LISTEN users_changes');
      await client.query('LISTEN orders_changes');
      await client.query('LISTEN products_changes');
    } finally {
      client.release();
    }
  }
  
  handleDatabaseNotification(msg) {
    // Parse the notification payload
    const payload = JSON.parse(msg.payload);
    const { table, operation, data, oldData, timestamp } = payload;
    
    // Determine need this update
    const event = {
      type: operation, which subscriptions // insert, update, delete
      table: table,
      data: operation === 'delete' ? oldData : data,
      timestamp: timestamp,
      sequence: payload.sequence
    };
    
    // Broadcast to matching subscriptions
    this.broadcastToSubscriptions(table, event);
  }
  
  broadcastToSubscriptions(channel, event) {
    for (const [subscriptionId, subscription] of this.subscriptions) {
      if (subscription.channel !== channel) continue;
      
      // Apply filter if specified
      if (subscription.filter && !this.matchesFilter(event.data, subscription.filter)) {
        continue;
      }
      
      // Send to client
      const message = {
        type: 'data',
        subscriptionId: subscriptionId,
        payload: {
          event: event.type,
          data: event.data,
          timestamp: event.timestamp
        }
      };
      
      this.sendToClient(subscription.client, message);
    }
  }
  
  matchesFilter(data, filter) {
    for (const [key, value] of Object.entries(filter)) {
      if (data[key] !== value) {
        return false;
      }
    }
    return true;
  }
  
  handleConnection(ws, req) {
    const clientId = uuidv4();
    this.clients.set(ws, { id: clientId, subscriptions: new Set(), auth: null });
    
    console.log(`Client connected: ${clientId}`);
    
    ws.on('message', (message) => {
      this.handleMessage(ws, message);
    });
    
    ws.on('close', () => {
      this.handleDisconnect(ws);
    });
    
    ws.on('error', (error) => {
      console.error(`WebSocket error for client ${clientId}:`, error);
    });
    
    // Send connection acknowledgment
    this.sendToClient(ws, {
      type: 'connected',
      clientId: clientId,
      timestamp: new Date().toISOString()
    });
  }
  
  async handleMessage(ws, rawMessage) {
    let message;
    try {
      message = JSON.parse(rawMessage);
    } catch (error) {
      this.sendToClient(ws, {
        type: 'error',
        error: 'Invalid message format'
      });
      return;
    }
    
    const client = this.clients.get(ws);
    
    switch (message.type) {
      case 'auth':
        await this.handleAuth(ws, message);
        break;
        
      case 'subscribe':
        this.handleSubscribe(ws, message);
        break;
        
      case 'unsubscribe':
        this.handleUnsubscribe(ws, message);
        break;
        
      case 'ping':
        this.sendToClient(ws, { type: 'pong', timestamp: Date.now() });
        break;
        
      default:
        this.sendToClient(ws, {
          type: 'error',
          error: `Unknown message type: ${message.type}`
        });
    }
  }
  
  async handleAuth(ws, message) {
    const { token } = message.payload || {};
    
    // Verify token (simplified - implement proper JWT verification)
    const isValid = await this.verifyToken(token);
    
    const client = this.clients.get(ws);
    client.auth = isValid ? { valid: true, token } : null;
    
    this.sendToClient(ws, {
      type: 'auth',
      payload: {
        success: isValid,
        message: isValid ? 'Authentication successful' : 'Invalid token'
      }
    });
  }
  
  async verifyToken(token) {
    // Implement actual token verification
    if (!token) return false;
    
    try {
      const result = await this.pool.query(
        'SELECT id FROM api_keys WHERE key_hash = $1 AND active = true',
        [this.hashToken(token)]
      );
      return result.rows.length > 0;
    } catch (error) {
      console.error('Token verification error:', error);
      return false;
    }
  }
  
  hashToken(token) {
    const crypto = require('crypto');
    return crypto.createHash('sha256').update(token).digest('hex');
  }
  
  handleSubscribe(ws, message) {
    const { channel, filter, fields } = message.payload;
    const client = this.clients.get(ws);
    
    // Check authorization
    if (!client.auth?.valid) {
      this.sendToClient(ws, {
        type: 'error',
        id: message.id,
        error: 'Authentication required'
      });
      return;
    }
    
    // Validate channel
    const validChannels = ['users', 'orders', 'products'];
    if (!validChannels.includes(channel)) {
      this.sendToClient(ws, {
        type: 'error',
        id: message.id,
        error: `Invalid channel: ${channel}`
      });
      return;
    }
    
    // Create subscription
    const subscriptionId = uuidv4();
    const subscription = {
      id: subscriptionId,
      client: ws,
      channel: channel,
      filter: filter || {},
      fields: fields || null,
      createdAt: new Date()
    };
    
    this.subscriptions.set(subscriptionId, subscription);
    client.subscriptions.add(subscriptionId);
    
    // Send acknowledgment
    this.sendToClient(ws, {
      type: 'ack',
      id: message.id,
      payload: {
        subscriptionId: subscriptionId,
        channel: channel,
        status: 'active'
      }
    });
    
    console.log(`Client ${client.id} subscribed to ${channel}`);
  }
  
  handleUnsubscribe(ws, message) {
    const { subscriptionId } = message.payload;
    const client = this.clients.get(ws);
    
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription || subscription.client !== ws) {
      this.sendToClient(ws, {
        type: 'error',
        id: message.id,
        error: 'Subscription not found'
      });
      return;
    }
    
    this.subscriptions.delete(subscriptionId);
    client.subscriptions.delete(subscriptionId);
    
    this.sendToClient(ws, {
      type: 'ack',
      id: message.id,
      payload: {
        subscriptionId: subscriptionId,
        status: 'unsubscribed'
      }
    });
  }
  
  handleDisconnect(ws) {
    const client = this.clients.get(ws);
    if (!client) return;
    
    // Clean up subscriptions
    for (const subscriptionId of client.subscriptions) {
      this.subscriptions.delete(subscriptionId);
    }
    
    this.clients.delete(ws);
    console.log(`Client disconnected: ${client.id}`);
  }
  
  sendToClient(ws, message) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }
  
  async shutdown() {
    // Close all client connections
    for (const ws of this.clients.keys()) {
      ws.close();
    }
    
    // Close database pool
    await this.pool.end();
    
    // Close WebSocket server
    this.wss.close();
  }
}

module.exports = DatabaseSubscriptionServer;
```

### 3.3 Database Trigger Configuration

Database triggers are essential for generating notifications when data changes occur. The following PostgreSQL trigger configuration demonstrates how to capture insert, update, and delete operations on database tables and emit notifications through the LISTEN/NOTIFY mechanism. This approach ensures that all database changes are captured at the source and can be propagated to subscribers.

```sql
-- Database triggers for change notification

-- Create notification function for users table
CREATE OR REPLACE FUNCTION notify_user_change()
RETURNS TRIGGER AS $$
BEGIN
  PERFORM pg_notify(
    'users_changes',
    json_build_object(
      'table', 'users',
      'operation', TG_OP,
      'data', CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE NEW END,
      'oldData', CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NULL END,
      'timestamp', now()::text,
      'sequence', nextval('change_sequence')::text
    )::text
  );
  
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create sequence for ordering changes
CREATE SEQUENCE change_sequence;

-- Create triggers for INSERT, UPDATE, DELETE
CREATE TRIGGER user_change_insert
AFTER INSERT ON users
FOR EACH ROW
EXECUTE FUNCTION notify_user_change();

CREATE TRIGGER user_change_update
AFTER UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION notify_user_change();

CREATE TRIGGER user_change_delete
AFTER DELETE ON users
FOR EACH ROW
EXECUTE FUNCTION notify_user_change();

-- Similar triggers for orders table
CREATE OR REPLACE FUNCTION notify_order_change()
RETURNS TRIGGER AS $$
BEGIN
  PERFORM pg_notify(
    'orders_changes',
    json_build_object(
      'table', 'orders',
      'operation', TG_OP,
      'data', CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE NEW END,
      'oldData', CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NULL END,
      'timestamp', now()::text,
      'sequence', nextval('change_sequence')::text
    )::text
  );
  
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER order_change_insert
AFTER INSERT ON orders
FOR EACH ROW
EXECUTE FUNCTION notify_order_change();

CREATE TRIGGER order_change_update
AFTER UPDATE ON orders
FOR EACH ROW
EXECUTE FUNCTION notify_order_change();

CREATE TRIGGER order_change_delete
AFTER DELETE ON orders
FOR EACH ROW
EXECUTE FUNCTION notify_order_change();

-- Products triggers
CREATE OR REPLACE FUNCTION notify_product_change()
RETURNS TRIGGER AS $$
BEGIN
  PERFORM pg_notify(
    'products_changes',
    json_build_object(
      'table', 'products',
      'operation', TG_OP,
      'data', CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE NEW END,
      'oldData', CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NULL END,
      'timestamp', now()::text,
      'sequence', nextval('change_sequence')::text
    )::text
  );
  
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER product_change_insert
AFTER INSERT ON products
FOR EACH ROW
EXECUTE FUNCTION notify_product_change();

CREATE TRIGGER product_change_update
AFTER UPDATE ON products
FOR EACH ROW
EXECUTE FUNCTION notify_product_change();

CREATE TRIGGER product_change_delete
AFTER DELETE ON products
FOR EACH ROW
EXECUTE FUNCTION notify_product_change();
```

---

## 4. Batch Data Processing APIs

### 4.1 Batch Operation Design Patterns

Batch data processing APIs enable efficient handling of large volumes of data operations that would be impractical or inefficient to process one at a time. These APIs must handle various failure scenarios, provide progress tracking, support partial success, and maintain idempotency. The design should account for different operation types including bulk creation, bulk update, bulk deletion, and complex transformations.

When designing batch APIs, consider the trade-offs between synchronous and asynchronous processing. Small batches might be processed synchronously for simplicity, while larger batches should be queued for asynchronous processing. The API should support both approaches and provide clear mechanisms for clients to track progress and retrieve results. Additionally, implement proper concurrency controls to prevent database overload and ensure data consistency.

### 4.2 Asynchronous Batch Processing Implementation

This implementation demonstrates a complete asynchronous batch processing system with job queuing, progress tracking, error handling, and result retrieval. The system uses a job queue pattern where batch requests are queued for background processing, and clients can poll for status or receive callbacks upon completion.

```javascript
// Asynchronous Batch Processing API

const express = require('express');
const { Pool } = require('pg');
const Queue = require('bull');
const { v4: uuidv4 } = require('uuid');

const app = express();
app.use(express.json());

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

// Create job queues
const userBatchQueue = new Queue('user-batch-processing', {
  redis: { host: 'localhost', port: 6379 }
});

const orderBatchQueue = new Queue('order-batch-processing', {
  redis: { host: 'localhost', port: 6379 }
});

// Job status tracking database
const createJobTables = async () => {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS batch_jobs (
      id UUID PRIMARY KEY,
      type VARCHAR(50) NOT NULL,
      status VARCHAR(20) NOT NULL,
      total_count INTEGER NOT NULL DEFAULT 0,
      processed_count INTEGER NOT NULL DEFAULT 0,
      success_count INTEGER NOT NULL DEFAULT 0,
      failure_count INTEGER NOT NULL DEFAULT 0,
      created_by VARCHAR(100),
      created_at TIMESTAMP DEFAULT NOW(),
      started_at TIMESTAMP,
      completed_at TIMESTAMP,
      error_message TEXT,
      result_url TEXT,
      metadata JSONB
    );
    
    CREATE TABLE IF NOT EXISTS batch_job_items (
      id UUID PRIMARY KEY,
      job_id UUID REFERENCES batch_jobs(id),
      item_index INTEGER NOT NULL,
      status VARCHAR(20) NOT NULL,
      input_data JSONB,
      output_data JSONB,
      error_message TEXT,
      created_at TIMESTAMP DEFAULT NOW(),
      processed_at TIMESTAMP
    );
    
    CREATE INDEX idx_batch_jobs_status ON batch_jobs(status);
    CREATE INDEX idx_batch_job_items_job_id ON batch_job_items(job_id);
  `);
};

// API Endpoints

// Submit batch user creation
app.post('/api/v1/batch/users', async (req, res) => {
  const { users, callbackUrl } = req.body;
  
  if (!users || !Array.isArray(users) || users.length === 0) {
    return res.status(400).json({
      error: 'Invalid request: users array required'
    });
  }
  
  if (users.length > 10000) {
    return res.status(400).json({
      error: 'Batch size exceeds maximum of 10,000 items'
    });
  }
  
  const jobId = uuidv4();
  
  try {
    // Create job record
    await pool.query(
      `INSERT INTO batch_jobs (id, type, status, total_count, created_by, metadata)
       VALUES ($1, 'USER_CREATE', 'PENDING', $2, $3, $4)`,
      [jobId, users.length, req.user?.id, JSON.stringify({ callbackUrl })]
    );
    
    // Create job items
    const itemValues = users.map((user, index) => ({
      id: uuidv4(),
      jobId,
      index,
      inputData: user
    }));
    
    for (const item of itemValues) {
      await pool.query(
        `INSERT INTO batch_job_items (id, job_id, item_index, input_data, status)
         VALUES ($1, $2, $3, $4, 'PENDING')`,
        [item.id, item.jobId, item.index, JSON.stringify(item.inputData)]
      );
    }
    
    // Add to processing queue
    await userBatchQueue.add({
      jobId,
      userIds: itemValues.map(i => i.id)
    });
    
    res.status(202).json({
      jobId,
      status: 'PENDING',
      totalCount: users.length,
      message: 'Batch job queued for processing',
      statusUrl: `/api/v1/batch/jobs/${jobId}`
    });
  } catch (error) {
    console.error('Batch job creation error:', error);
    res.status(500).json({ error: 'Failed to create batch job' });
  }
});

// Get batch job status
app.get('/api/v1/batch/jobs/:jobId', async (req, res) => {
  const { jobId } = req.params;
  
  try {
    const jobResult = await pool.query(
      'SELECT * FROM batch_jobs WHERE id = $1',
      [jobId]
    );
    
    if (jobResult.rows.length === 0) {
      return res.status(404).json({ error: 'Job not found' });
    }
    
    const job = jobResult.rows[0];
    
    // Include error details if failed
    const response = {
      jobId: job.id,
      type: job.type,
      status: job.status,
      totalCount: job.total_count,
      processedCount: job.processed_count,
      successCount: job.success_count,
      failureCount: job.failure_count,
      progress: job.total_count > 0 
        ? Math.round((job.processed_count / job.total_count) * 100) 
        : 0,
      createdAt: job.created_at,
      startedAt: job.started_at,
      completedAt: job.completed_at
    };
    
    if (job.status === 'FAILED') {
      response.errorMessage = job.error_message;
    }
    
    if (job.result_url) {
      response.resultUrl = job.result_url;
    }
    
    // Include first few errors for quick debugging
    if (job.failure_count > 0) {
      const errorsResult = await pool.query(
        `SELECT item_index, error_message, input_data 
         FROM batch_job_items 
         WHERE job_id = $1 AND status = 'FAILED'
         LIMIT 5`,
        [jobId]
      );
      response.sampleErrors = errorsResult.rows;
    }
    
    res.json(response);
  } catch (error) {
    console.error('Job status error:', error);
    res.status(500).json({ error: 'Failed to retrieve job status' });
  }
});

// Get batch job results (paginated)
app.get('/api/v1/batch/jobs/:jobId/results', async (req, res) => {
  const { jobId } = req.params;
  const { status, page = 1, limit = 50 } = req.query;
  
  try {
    const jobResult = await pool.query(
      'SELECT status, result_url FROM batch_jobs WHERE id = $1',
      [jobId]
    );
    
    if (jobResult.rows.length === 0) {
      return res.status(404).json({ error: 'Job not found' });
    }
    
    const job = jobResult.rows[0];
    
    if (job.status !== 'COMPLETED') {
      return res.status(400).json({
        error: 'Job not yet completed',
        status: job.status
      });
    }
    
    let query = 'SELECT * FROM batch_job_items WHERE job_id = $1';
    const params = [jobId];
    
    if (status) {
      query += ' AND status = $2';
      params.push(status.toUpperCase());
    }
    
    query += ' ORDER BY item_index LIMIT $' + (params.length + 1) + ' OFFSET $' + (params.length + 2);
    params.push(parseInt(limit), (parseInt(page) - 1) * parseInt(limit));
    
    const results = await pool.query(query, params);
    
    const countQuery = status 
      ? 'SELECT COUNT(*) FROM batch_job_items WHERE job_id = $1 AND status = $2'
      : 'SELECT COUNT(*) FROM batch_job_items WHERE job_id = $1';
    const countParams = status ? [jobId, status.toUpperCase()] : [jobId];
    const countResult = await pool.query(countQuery, countParams);
    
    res.json({
      data: results.rows.map(row => ({
        index: row.item_index,
        status: row.status,
        outputData: row.output_data,
        errorMessage: row.error_message
      })),
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total: parseInt(countResult.rows[0].count)
      }
    });
  } catch (error) {
    console.error('Job results error:', error);
    res.status(500).json({ error: 'Failed to retrieve job results' });
  }
});

// Download results as CSV
app.get('/api/v1/batch/jobs/:jobId/download', async (req, res) => {
  const { jobId } = req.params;
  
  try {
    const jobResult = await pool.query(
      'SELECT status, type FROM batch_jobs WHERE id = $1',
      [jobId]
    );
    
    if (jobResult.rows.length === 0) {
      return res.status(404).json({ error: 'Job not found' });
    }
    
    if (jobResult.rows[0].status !== 'COMPLETED') {
      return res.status(400).json({ error: 'Job not yet completed' });
    }
    
    // Stream results to CSV
    const stream = await pool.query(
      `SELECT item_index, status, output_data, error_message 
       FROM batch_job_items 
       WHERE job_id = $1 
       ORDER BY item_index`,
      [jobId]
    );
    
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="batch-results-${jobId}.csv"`);
    
    res.write('index,status,output,error\n');
    
    for (const row of stream.rows) {
      const output = row.output_data ? JSON.stringify(row.output_data).replace(/"/g, '""') : '';
      const error = row.error_message ? row.error_message.replace(/"/g, '""') : '';
      res.write(`${row.item_index},${row.status},"${output}","${error}"\n`);
    }
    
    res.end();
  } catch (error) {
    console.error('Download error:', error);
    res.status(500).json({ error: 'Failed to generate download' });
  }
});

// Batch job processor
userBatchQueue.process(async (job) => {
  const { jobId, userIds } = job.data;
  
  // Update job status to processing
  await pool.query(
    `UPDATE batch_jobs SET status = 'PROCESSING', started_at = NOW() WHERE id = $1`,
    [jobId]
  );
  
  const client = await pool.connect();
  
  try {
    await client.query('BEGIN');
    
    const usersResult = await client.query(
      'SELECT * FROM batch_job_items WHERE job_id = $1 AND status = $2 ORDER BY item_index',
      [jobId, 'PENDING']
    );
    
    let successCount = 0;
    let failureCount = 0;
    
    for (const item of usersResult.rows) {
      try {
        const userData = item.input_data;
        
        // Validate user data
        if (!userData.email || !userData.username) {
          throw new Error('Missing required fields: email or username');
        }
        
        // Check for duplicates
        const existingResult = await client.query(
          'SELECT id FROM users WHERE email = $1 OR username = $2',
          [userData.email, userData.username]
        );
        
        if (existingResult.rows.length > 0) {
          throw new Error('User with this email or username already exists');
        }
        
        // Create user
        const newUserResult = await client.query(
          `INSERT INTO users (id, email, username, first_name, last_name, created_at, updated_at)
           VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
           RETURNING id, email, username`,
          [uuidv4(), userData.email, userData.username, userData.first_name, userData.last_name]
        );
        
        // Update item status
        await client.query(
          `UPDATE batch_job_items 
           SET status = 'COMPLETED', output_data = $1, processed_at = NOW() 
           WHERE id = $2`,
          [JSON.stringify(newUserResult.rows[0]), item.id]
        );
        
        successCount++;
      } catch (error) {
        // Update item status to failed
        await client.query(
          `UPDATE batch_job_items 
           SET status = 'FAILED', error_message = $1, processed_at = NOW() 
           WHERE id = $2`,
          [error.message, item.id]
        );
        
        failureCount++;
      }
      
      // Update job progress
      await pool.query(
        `UPDATE batch_jobs 
         SET processed_count = processed_count + 1,
             success_count = $1,
             failure_count = $2
         WHERE id = $3`,
        [successCount, failureCount, jobId]
      );
    }
    
    await client.query('COMMIT');
    
    // Update final job status
    const finalStatus = failureCount === 0 ? 'COMPLETED' : 'COMPLETED_WITH_ERRORS';
    await pool.query(
      `UPDATE batch_jobs 
       SET status = $1, completed_at = NOW(), success_count = $2, failure_count = $3
       WHERE id = $4`,
      [finalStatus, successCount, failureCount, jobId]
    );
    
    return { successCount, failureCount };
  } catch (error) {
    await client.query('ROLLBACK');
    
    await pool.query(
      `UPDATE batch_jobs SET status = 'FAILED', error_message = $1 WHERE id = $2`,
      [error.message, jobId]
    );
    
    throw error;
  } finally {
    client.release();
  }
});

// Queue event handlers for monitoring
userBatchQueue.on('completed', (job, result) => {
  console.log(`Batch job ${job.data.jobId} completed:`, result);
});

userBatchQueue.on('failed', (job, err) => {
  console.error(`Batch job ${job.data.jobId} failed:`, err.message);
});

app.listen(3000, () => {
  console.log('Batch processing API running on port 3000');
});
```

---

## 5. Change Data Capture (CDC) Patterns

### 5.1 CDC Architecture Overview

Change Data Capture (CDC) is a design pattern that identifies and captures changes made to data in a database and delivers these changes in real-time to downstream consumers. CDC is essential for building event-driven architectures, maintaining data replicas, synchronizing caches, and enabling streaming analytics. The architecture consists of several components working together to capture, transform, and deliver database changes reliably.

The core CDC architecture includes the database transaction log as the source of truth, a CDC connector that reads the log, a message broker for reliable delivery, and consumer applications that process the change events. This architecture provides several key advantages over polling-based approaches: near real-time latency, no impact on source database performance, capture of all changes including deletes, and preservation of change ordering.

```
CDC Architecture Diagram:

─┐    ┌─────────────────┐┌────────────────    ┌─────────────────┐
│   PostgreSQL    │    │   Debezium      │    │    Kafka        │
│   Database      │───▶│   Connector     │───▶│    Cluster      │
│                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │Write-Ahead│  │    │  │Transaction│  │    │  │  Topics   │  │
│  │  Log      │  │    │  │  Log      │  │    │  │           │  │
│  │( WAL )    │  │    │  │ Reader   │  │    │  │ db.server │  │
│  └───────────┘  │    │  └───────────┘  │    │  │  .users   │  │
│                 │    │                 │    │  │ db.server │  │
│                 │    │  ┌───────────┐  │    │  │  .orders  │  │
│                 │    │  │Event      │  │    │  └───────────┘  │
│                 │    │  │Transform  │  │    │        ▲        │
│                 │    │  └───────────┘  │    │        │        │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │    Consumers        │
                                            │                     │
                                            │ ┌───────────────┐   │
                                            │ │Data Warehouse │   │
                                            │ │    Loader     │   │
                                            │ └───────────────┘   │
                                            │ ┌───────────────┐   │
                                            │ │Search Index   │   │
                                            │ │   Update      │   │
                                            │ └───────────────┘   │
                                            │ ┌───────────────┐   │
                                            │ │   Cache       │   │
                                            │ │  Invalidation │   │
                                            │ └───────────────┘   │
                                            └─────────────────────┘
```

### 5.2 CDC Event Schema Design

The CDC event schema must capture comprehensive information about each database change, including the operation type, before and after state, metadata about the change, and sequencing information. A well-designed schema enables consumers to properly handle each event type and reconstruct the current state from the event stream. The schema should be versioned to allow for evolution while maintaining backward compatibility.

```json
// CDC Event Schema - Insert Operation
{
  "schema": {
    "type": "struct",
    "fields": [
      {
        "type": "string",
        "optional": false,
        "field": "version"
      },
      {
        "type": "string",
        "optional": false,
        "field": "connector"
      },
      {
        "type": "string",
        "optional": false,
        "field": "name"
      },
      {
        "type": "int64",
        "optional": false,
        "field": "ts_ms"
      }
    ],
    "optional": false,
    "name": "io.debezium.connector.postgresql.Source"
  },
  "payload": {
    "version": "2.4.0.Final",
    "connector": "postgresql",
    "name": "orders-db",
    "ts_ms": 1708066200000,
    "snapshot": false,
    "db": "orders",
    "table": "users",
    "txId": 1234,
    "lsn": 12345678,
    "xmin": null
  }
}

// CDC Event Payload - Insert
{
  "before": null,
  "after": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "john.doe@example.com",
    "username": "johndoe",
    "first_name": "John",
    "last_name": "Doe",
    "status": "active",
    "created_at": 1708066200000,
    "updated_at": 1708066200000
  },
  "op": "c",  // c=create, u=update, d=delete, r=read (snapshot)
  "ts_ms": 1708066200123,
  "ts_us": 1708066200123456,
  "ts_ns": 1708066200123456789
}

// CDC Event Payload - Update
{
  "before": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "john.doe@example.com",
    "username": "johndoe",
    "first_name": "John",
    "last_name": "Doe",
    "status": "active",
    "created_at": 1708066200000,
    "updated_at": 1708066200000
  },
  "after": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "john.doe@example.com",
    "username": "johndoe",
    "first_name": "John",
    "last_name": "Doe",
    "status": "inactive",
    "created_at": 1708066200000,
    "updated_at": 1708070000000
  },
  "op": "u",
  "ts_ms": 1708070000123
}

// CDC Event Payload - Delete
{
  "before": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "john.doe@example.com",
    "username": "johndoe",
    "first_name": "John",
    "last_name": "Doe",
    "status": "active",
    "created_at": 1708066200000,
    "updated_at": 1708066200000
  },
  "after": null,
  "op": "d",
  "ts_ms": 1708071000123
}
```

---

## 6. API Versioning Strategies

### 6.1 Versioning Strategy Comparison

API versioning is essential for evolving APIs while maintaining backward compatibility for existing clients. Several versioning strategies exist, each with distinct advantages and trade-offs. URL path versioning is the most common approach, providing clear visibility into which version is being used. Header versioning keeps URLs clean but requires additional client configuration. Media type versioning follows HATEOAS principles but is less visible to developers.

When selecting a versioning strategy, consider the expected frequency of breaking changes, the client ecosystem, and team preferences. URL path versioning typically provides the best developer experience for public APIs, while header versioning might be preferred for internal APIs where URL cleanliness is valued. Regardless of the strategy chosen, maintain clear deprecation policies and provide ample migration time for clients.

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| URL Path (/v1/users) | Visible, easy to test, cacheable | URL pollution, duplicate endpoints | Public APIs, frequent changes |
| Header (Accept: v1) | Clean URLs, flexible | Less visible, requires headers | Internal APIs, stable interfaces |
| Query Param (?version=1) | Bookmarkable, simple | Caching issues, extra parsing | Simple APIs, few versions |
| Media Type (application/vnd.api.v1+json) | Standards-compliant | Complex tooling support | Hypermedia APIs |

### 6.2 Versioned API Implementation

The following implementation demonstrates a comprehensive versioning strategy using URL path versioning with support for multiple API versions, graceful version negotiation, deprecation handling, and comprehensive documentation of version changes.

```javascript
// Versioned API Implementation with Express

const express = require('express');
const { Pool } = require('pg');
const swaggerUi = require('swagger-ui-express');
const YAML = require('yamljs');

const app = express();
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

// API Version configuration
const API_VERSIONS = {
  v1: {
    status: 'deprecated',
    deprecatedAt: '2025-06-01',
    sunsetAt: '2026-06-01',
    documentation: '/api-docs/v1',
    features: ['basic_crud', 'pagination', 'filtering'],
    breakingChanges: []
  },
  v2: {
    status: 'active',
    documentation: '/api-docs/v2',
    features: ['basic_crud', 'pagination', 'filtering', 'batch_operations', 'graphql'],
    breakingChanges: [
      'Removed response envelope wrapper',
      'Changed date format to ISO 8601',
      'Updated error response structure'
    ]
  },
  v3: {
    status: 'beta',
    documentation: '/api-docs/v3',
    features: ['basic_crud', 'pagination', 'filtering', 'batch_operations', 'graphql', 'subscriptions'],
    breakingChanges: [
      'All v2 breaking changes plus:',
      'Required authentication header',
      'Changed pagination to cursor-based'
    ]
  }
};

// Version middleware
const versionMiddleware = (req, res, next) => {
  // Extract version from URL path
  const pathMatch = req.path.match(/^\/api\/v(\d+)/);
  const requestedVersion = pathMatch ? `v${pathMatch[1]}` : 'v1';
  
  // Check if version exists
  if (!API_VERSIONS[requestedVersion]) {
    return res.status(404).json({
      error: 'API version not found',
      availableVersions: Object.keys(API_VERSIONS)
    });
  }
  
  const versionConfig = API_VERSIONS[requestedVersion];
  
  // Add version info to request
  req.apiVersion = requestedVersion;
  req.versionConfig = versionConfig;
  
  // Add deprecation headers for deprecated versions
  if (versionConfig.status === 'deprecated') {
    res.setHeader('Deprecation', `date="${versionConfig.deprecatedAt}"`);
    res.setHeader('Sunset', versionConfig.sunsetAt);
    res.setHeader('Link', `<${versionConfig.documentation}>; rel="deprecation"`);
  }
  
  next();
};

// Apply version middleware
app.use('/api/v', versionMiddleware, express.json());

// Health check endpoint (unversioned)
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// API version discovery endpoint
app.get('/api/versions', (req, res) => {
  const versions = Object.entries(API_VERSIONS).map(([version, config]) => ({
    version,
    status: config.status,
    url: `/api/${version}`,
    documentation: config.documentation,
    features: config.features,
    deprecatedAt: config.deprecatedAt,
    sunsetAt: config.sunsetAt
  }));
  
  res.json({ versions });
});

// V1 API Routes (Deprecated)
const v1Router = express.Router();

v1Router.get('/users', async (req, res) => {
  // V1 implementation - wrapped response
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 20;
  const offset = (page - 1) * limit;
  
  try {
    const countResult = await pool.query('SELECT COUNT(*) FROM users');
    const result = await pool.query(
      'SELECT * FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2',
      [limit, offset]
    );
    
    // V1 response envelope
    res.json({
      response: {
        users: result.rows,
        pagination: {
          page,
          limit,
          total: parseInt(countResult.rows[0].count),
          pages: Math.ceil(countResult.rows[0].count / limit)
        }
      }
    });
  } catch (error) {
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: error.message
      }
    });
  }
});

v1Router.get('/users/:id', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM users WHERE id = $1', [req.params.id]);
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        error: {
          code: 'NOT_FOUND',
          message: 'User not found'
        }
      });
    }
    
    res.json({ response: { user: result.rows[0] } });
  } catch (error) {
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: error.message
      }
    });
  }
});

v1Router.post('/users', async (req, res) => {
  const { email, username, first_name, last_name } = req.body;
  
  try {
    const result = await pool.query(
      `INSERT INTO users (email, username, first_name, last_name, created_at, updated_at)
       VALUES ($1, $2, $3, $4, NOW(), NOW()) RETURNING *`,
      [email, username, first_name, last_name]
    );
    
    res.status(201).json({ response: { user: result.rows[0] } });
  } catch (error) {
    if (error.code === '23505') { // Unique violation
      return res.status(409).json({
        error: {
          code: 'DUPLICATE_ENTRY',
          message: 'User already exists'
        }
      });
    }
    
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: error.message
      }
    });
  }
});

// V2 API Routes (Active)
const v2Router = express.Router();

v2Router.get('/users', async (req, res) => {
  // V2 implementation - flat response
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 20;
  const offset = (page - 1) * limit;
  
  try {
    const countResult = await pool.query('SELECT COUNT(*) FROM users');
    const result = await pool.query(
      'SELECT * FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2',
      [limit, offset]
    );
    
    // V2 flat response
    res.json({
      data: result.rows,
      meta: {
        requestId: req.headers['x-request-id'],
        timestamp: new Date().toISOString()
      },
      pagination: {
        page,
        limit,
        total: parseInt(countResult.rows[0].count),
        pages: Math.ceil(countResult.rows[0].count / limit)
      }
    });
  } catch (error) {
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: error.message,
        requestId: req.headers['x-request-id']
      }
    });
  }
});

v2Router.get('/users/:id', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM users WHERE id = $1', [req.params.id]);
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        error: {
          code: 'NOT_FOUND',
          message: 'User not found',
          requestId: req.headers['x-request-id']
        }
      });
    }
    
    res.json({
      data: result.rows[0],
      meta: {
        requestId: req.headers['x-request-id'],
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: error.message,
        requestId: req.headers['x-request-id']
      }
    });
  }
});

v2Router.post('/users', async (req, res) => {
  const { email, username, first_name, last_name } = req.body;
  
  try {
    const result = await pool.query(
      `INSERT INTO users (email, username, first_name, last_name, created_at, updated_at)
       VALUES ($1, $2, $3, $4, NOW(), NOW()) RETURNING *`,
      [email, username, first_name, last_name]
    );
    
    res.status(201).json({
      data: result.rows[0],
      meta: {
        requestId: req.headers['x-request-id'],
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    if (error.code === '23505') {
      return res.status(409).json({
        error: {
          code: 'DUPLICATE_ENTRY',
          message: 'User already exists',
          requestId: req.headers['x-request-id']
        }
      });
    }
    
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: error.message,
        requestId: req.headers['x-request-id']
      }
    });
  }
});

// V3 API Routes (Beta - Cursor-based pagination)
const v3Router = express.Router();

v3Router.get('/users', async (req, res) => {
  // V3 implementation - cursor-based pagination
  const limit = Math.min(parseInt(req.query.limit) || 20, 100);
  const cursor = req.query.cursor;
  
  try {
    let query = 'SELECT * FROM users';
    let params = [limit + 1]; // Get one extra to determine hasMore
    
    if (cursor) {
      const decodedCursor = JSON.parse(Buffer.from(cursor, 'base64').toString());
      query += ' WHERE created_at < $2 OR (created_at = $2 AND id < $3)';
      params = [limit + 1, decodedCursor.timestamp, decodedCursor.id];
    }
    
    query += ' ORDER BY created_at DESC, id DESC LIMIT $1';
    
    const result = await pool.query(query, params);
    
    const hasMore = result.rows.length > limit;
    if (hasMore) {
      result.rows.pop(); // Remove the extra record
    }
    
    const nextCursor = hasMore ? Buffer.from(JSON.stringify({
      timestamp: result.rows[result.rows.length - 1].created_at,
      id: result.rows[result.rows.length - 1].id
    })).toString('base64') : null;
    
    res.json({
      data: result.rows,
      meta: {
        requestId: req.headers['x-request-id'],
        timestamp: new Date().toISOString()
      },
      pagination: {
        limit,
        cursor: nextCursor,
        hasMore
      }
    });
  } catch (error) {
    res.status(500).json({
      error: {
        code: 'INTERNAL_ERROR',
        message: error.message,
        requestId: req.headers['x-request-id']
      }
    });
  }
});

// Mount versioned routers
app.use('/api/v1', v1Router);
app.use('/api/v2', v2Router);
app.use('/api/v3', v3Router);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    error: {
      code: 'INTERNAL_ERROR',
      message: 'An unexpected error occurred'
    }
  });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`API server running on port ${PORT}`);
});
```

### 6.3 Deprecation Strategy Implementation

A comprehensive deprecation strategy ensures smooth transitions between API versions while providing adequate notice to consumers. This implementation includes automated deprecation warnings, sunset headers, alternative version recommendations, and usage tracking to identify clients still using deprecated versions.

```javascript
// Deprecation tracking and notification system

const { Pool } = require('pg');
const axios = require('axios');

class DeprecationManager {
  constructor(pool) {
    this.pool = pool;
    this.setupDatabase();
  }
  
  async setupDatabase() {
    await this.pool.query(`
      CREATE TABLE IF NOT EXISTS api_usage_tracking (
        id SERIAL PRIMARY KEY,
        api_version VARCHAR(10) NOT NULL,
        endpoint VARCHAR(100) NOT NULL,
        client_id VARCHAR(100),
        user_agent VARCHAR(500),
        request_timestamp TIMESTAMP DEFAULT NOW(),
        response_time_ms INTEGER
      );
      
      CREATE INDEX idx_usage_version ON api_usage_tracking(api_version, request_timestamp);
    `);
  }
  
  // Track API usage
  async trackUsage(req, res, responseTime) {
    try {
      await this.pool.query(
        `INSERT INTO api_usage_tracking (api_version, endpoint, client_id, user_agent, response_time_ms)
         VALUES ($1, $2, $3, $4, $5)`,
        [
          req.apiVersion,
          req.path,
          req.headers['x-client-id'],
          req.headers['user-agent'],
          responseTime
        ]
      );
    } catch (error) {
      console.error('Usage tracking error:', error);
    }
  }
  
  // Get usage statistics for a version
  async getVersionUsage(version, days = 30) {
    const result = await this.pool.query(
      `SELECT 
         DATE(request_timestamp) as date,
         COUNT(*) as requests,
         COUNT(DISTINCT client_id) as unique_clients,
         AVG(response_time_ms) as avg_response_time
       FROM api_usage_tracking
       WHERE api_version = $1 
         AND request_timestamp > NOW() - INTERVAL '${days} days'
       GROUP BY DATE(request_timestamp)
       ORDER BY date DESC`,
      [version]
    );
    
    return result.rows;
  }
  
  // Identify clients using deprecated versions
  async getDeprecatedVersionClients(version) {
    const result = await this.pool.query(
      `SELECT 
         client_id,
         COUNT(*) as request_count,
         MAX(request_timestamp) as last_request,
         MIN(request_timestamp) as first_request
       FROM api_usage_tracking
       WHERE api_version = $1
         AND client_id IS NOT NULL
       GROUP BY client_id
       ORDER BY request_count DESC`,
      [version]
    );
    
    return result.rows;
  }
  
  // Generate deprecation notification
  async sendDeprecationNotification(clientId, version, sunsetDate) {
    const message = {
      to: clientId,
      subject: `API Deprecation Notice - Version ${version}`,
      body: `
        Dear API Consumer,
        
        This is a notification that API version ${version} will be deprecated.
        
        Key Dates:
        - Deprecation Date: ${new Date().toISOString()}
        - Sunset Date: ${sunsetDate}
        
        Action Required:
        Please migrate to a supported API version before the sunset date.
        
        Recommended Version: v2 (active)
        
        For migration assistance, please visit our documentation at /api-docs/v2/migration
        
        If you have any questions, please contact api-support@example.com.
        
        Thank you for your cooperation.
      `
    };
    
    // Send notification (implementation depends on notification system)
    console.log('Deprecation notification:', message);
  }
}

// Deprecation monitoring job
async function runDeprecationMonitoring() {
  const manager = new DeprecationManager(pool);
  
  // Check for deprecated versions
  const deprecatedVersions = ['v1'];
  
  for (const version of deprecatedVersions) {
    const clients = await manager.getDeprecatedVersionClients(version);
    
    for (const client of clients) {
      // Calculate client usage percentage
      const totalUsageResult = await manager.pool.query(
        'SELECT COUNT(*) FROM api_usage_tracking WHERE client_id = $1',
        [client.client_id]
      );
      
      const versionUsageResult = await manager.pool.query(
        'SELECT COUNT(*) FROM api_usage_tracking WHERE client_id = $1 AND api_version = $2',
        [client.client_id, version]
      );
      
      const percentage = (versionUsageResult.rows[0].count / totalUsageResult.rows[0].count) * 100;
      
      // Notify clients with significant usage (>10% of requests)
      if (percentage > 10) {
        await manager.sendDeprecationNotification(
          client.client_id,
          version,
          '2026-06-01'
        );
      }
    }
  }
}

// Run monitoring daily
setInterval(runDeprecationMonitoring, 24 * 60 * 60 * 1000);
```

---

## Conclusion

This documentation has provided comprehensive coverage of database API design patterns essential for building modern, scalable applications. The patterns and implementations presented address the most common challenges in database-facing API development, from RESTful design principles to real-time event streaming and version management.

The key takeaways include the importance of proper resource modeling and consistent HTTP method usage in REST APIs, the power of GraphQL's type system for flexible data retrieval while preventing query performance issues, the necessity of WebSocket-based subscriptions for real-time applications, the efficiency of batch processing for bulk operations, the value of Change Data Capture for building event-driven architectures, and the critical nature of versioning strategies for maintaining API stability during evolution.

These patterns form the foundation for building robust, maintainable APIs that can scale with growing requirements while providing excellent developer experiences for consumers.
