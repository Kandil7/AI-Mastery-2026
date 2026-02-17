# Learning Management System Technical Architecture

## Table of Contents

1. [Frontend Architecture](#1-frontend-architecture)
2. [Backend Architecture](#2-backend-architecture)
3. [Database Design Patterns](#3-database-design-patterns)
4. [Authentication and Authorization](#4-authentication-and-authorization)
5. [Multi-Tenancy Patterns](#5-multi-tenancy-patterns)
6. [API Design](#6-api-design)
7. [Integration Patterns](#7-integration-patterns)

---

## 1. Frontend Architecture

### 1.1 Technology Stack Options

Modern LMS platforms leverage various frontend frameworks to deliver responsive, engaging user experiences. The three dominant frameworks are **React**, **Vue.js**, and **Angular**.

#### React LMS Architecture

React's component-based architecture makes it ideal for building complex LMS interfaces with reusable learning components.

**Key Advantages:**
- **Virtual DOM**: Efficient rendering of dynamic content like progress bars and real-time updates
- **Component Reusability**: Pre-built components for courses, assessments, and dashboards
- **Ecosystem**: Extensive libraries for state management (Redux, Zustand) and routing
- **Mobile Support**: React Native enables native mobile applications

**Recommended Libraries:**
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-router-dom": "^6.x",
    "zustand": "^4.x",
    "@tanstack/react-query": "^5.x",
    "recharts": "^2.x",
    "react-i18next": "^14.x"
  }
}
```

#### Vue.js LMS Architecture

Vue offers a gentler learning curve while providing powerful features:

**Key Advantages:**
- **Progressive Framework**: Can be adopted incrementally
- **Reactive Data Binding**: Automatic UI updates when learner data changes
- **Single File Components**: Clean organization of template, script, and styles
- **Transition System**: Built-in animations for engaging learner experiences

#### Angular LMS Architecture

Angular provides a comprehensive framework for enterprise-grade LMS:

**Key Advantages:**
- **TypeScript Foundation**: Built-in type safety for large-scale applications
- **Dependency Injection**: Better testability and maintainability
- **RxJS Integration**: Powerful handling of asynchronous learning data streams
- **Enterprise Ready**: Strong support for large development teams

### 1.2 Headless LMS Architecture

A emerging trend is the **headless LMS** architecture that separates backend functionality from frontend presentation:

```
┌─────────────────────────────────────────────────────────────┐
│                    Headless LMS Architecture                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │  Web App    │  │ Mobile App  │  │ Custom UI   │        │
│   │  (React)    │  │(React Native)│ │ (Any)       │        │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│          │                │                │                │
│          └────────────────┼────────────────┘                │
│                           │                                  │
│                    ┌──────▼──────┐                          │
│                    │  GraphQL/REST│                          │
│                    │    Gateway   │                          │
│                    └──────┬──────┘                          │
│                           │                                  │
│          ┌────────────────┼────────────────┐                │
│          │                │                │                │
│   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐        │
│   │ Auth Service │  │Course Service│  │User Service │        │
│   └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- **API-First Design**: All learning logic exposed through RESTful or GraphQL APIs
- **Multi-Platform Delivery**: Single backend serving web, mobile, and custom interfaces
- **Unlimited Customization**: Frontend teams can build any user experience
- **Integration Flexibility**: Easy connection with third-party systems

Research indicates that headless LMS architecture reduces frontend development time by **38% in multi-platform environments** and cuts integration effort by **up to 35%**.

### 1.3 Component Architecture

```
LMS Component Hierarchy
├── App Shell
│   ├── Header
│   │   ├── Logo
│   │   ├── Search
│   │   ├── Notifications
│   │   └── User Menu
│   ├── Sidebar
│   │   ├── Navigation
│   │   └── Quick Links
│   └── Main Content Area
│
├── Course Components
│   ├── CourseCard
│   ├── CoursePlayer
│   ├── VideoPlayer
│   ├── DocumentViewer
│   └── InteractiveContent
│
├── Assessment Components
│   ├── QuizBuilder
│   ├── QuestionTypes
│   ├── GradingInterface
│   └── ProctoringView
│
├── Progress Components
│   ├── ProgressBar
│   ├── CompletionTracker
│   ├── AchievementBadge
│   └── CertificateDisplay
│
└── Analytics Components
    ├── Dashboard
    ├── ChartLibrary
    └── ReportBuilder
```

### 1.4 State Management

For complex LMS applications, proper state management is crucial:

```javascript
// Example: Zustand store for learner progress
import { create } from 'zustand'

const useLearnerProgress = create((set, get) => ({
  currentCourse: null,
  modules: [],
  progress: {},
  
  fetchProgress: async (courseId) => {
    const response = await fetch(`/api/courses/${courseId}/progress`)
    const data = await response.json()
    set({ 
      currentCourse: data.course,
      modules: data.modules,
      progress: data.progress 
    })
  },
  
  updateModuleProgress: async (moduleId, progressData) => {
    await fetch(`/api/modules/${moduleId}/progress`, {
      method: 'PUT',
      body: JSON.stringify(progressData)
    })
    set((state) => ({
      progress: { ...state.progress, [moduleId]: progressData }
    }))
  },
  
  completeModule: async (moduleId) => {
    const { updateModuleProgress } = get()
    await updateModuleProgress(moduleId, { 
      status: 'completed', 
      completedAt: new Date().toISOString() 
    })
  }
}))
```

---

## 2. Backend Architecture

### 2.1 Service-Oriented Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LMS Backend Services                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      API Gateway                         │   │
│  │         (Rate Limiting, Authentication, Routing)         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐           │
│         │                    │                    │             │
│  ┌──────▼──────┐    ┌───────▼───────┐    ┌──────▼──────┐     │
│  │ Auth Service │    │Course Service  │    │User Service  │     │
│  │              │    │                │    │              │     │
│  │ - Login      │    │ - CRUD         │    │ - Profiles   │     │
│  │ - SSO/SAML   │    │ - Publishing   │    │ - Roles      │     │
│  │ - Tokens     │    │ - Versioning   │    │ - Groups     │     │
│  │ - Sessions   │    │ - Search       │    │ - Imports    │     │
│  └──────┬───────┘    └───────┬────────┘    └──────┬───────┘     │
│         │                    │                    │             │
│  ┌──────▼──────┐    ┌───────▼───────┐    ┌──────▼──────┐       │
│  │Assessment   │    │Progress       │    │Notification │       │
│  │Service      │    │Service        │    │Service      │       │
│  │              │    │                │    │              │       │
│  │ - Quizzes    │    │ - Tracking     │    │ - Email      │       │
│  │ - Grading    │    │ - Completion   │    │ - Push       │       │
│  │ - Attempts   │    │ - Analytics    │    │ - In-app     │       │
│  │ - Rubrics    │    │ - Time spent   │    │ - Webhooks   │       │
│  └─────────────┘    └────────────────┘    └──────────────┘     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Supporting Services                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │ File Service │  │Search Service│  │Analytics    │      │  │
│  │  │             │  │             │  │Service      │      │  │
│  │  │ - Upload    │  │ - Indexing  │  │ - Events    │      │  │
│  │  │ - Storage   │  │ - Queries   │  │ - Reports   │      │  │
│  │  │ - CDN      │  │ - Filters   │  │ - Dashboards│      │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 REST API Architecture

Standard REST endpoints for LMS:

```
LMS Backend REST API:

Authentication:
POST   /api/auth/login              # User login
POST   /api/auth/logout             # User logout
POST   /api/auth/refresh            # Refresh token
GET    /api/auth/me                 # Current user info
POST   /api/auth/saml               # SAML callback

Courses:
GET    /api/courses                 # List courses
POST   /api/courses                 # Create course
GET    /api/courses/:id             # Get course details
PUT    /api/courses/:id             # Update course
DELETE /api/courses/:id             # Delete course
GET    /api/courses/:id/content     # Get course content
POST   /api/courses/:id/publish     # Publish course

Modules:
GET    /api/courses/:courseId/modules
POST   /api/courses/:courseId/modules
GET    /api/modules/:id
PUT    /api/modules/:id
DELETE /api/modules/:id

Enrollments:
GET    /api/enrollments             # List enrollments
POST   /api/enrollments             # Enroll user
GET    /api/enrollments/:id         # Get enrollment
PUT    /api/enrollments/:id         # Update enrollment
DELETE /api/enrollments/:id         # Unenroll

Progress:
GET    /api/progress/:enrollmentId   # Get progress
PUT    /api/progress/:enrollmentId  # Update progress
POST   /api/progress/:enrollmentId/complete  # Mark complete

Assessments:
GET    /api/assessments             # List assessments
POST   /api/assessments             # Create assessment
GET    /api/assessments/:id         # Get assessment
POST   /api/assessments/:id/submit # Submit attempt
GET    /api/assessments/:id/results # Get results

Users:
GET    /api/users                   # List users
POST   /api/users                   # Create user
GET    /api/users/:id               # Get user
PUT    /api/users/:id               # Update user
GET    /api/users/:id/certificates  # Get user certs

Analytics:
GET    /api/analytics/dashboard     # Dashboard data
GET    /api/analytics/courses/:id    # Course analytics
GET    /api/analytics/users/:id      # User analytics
GET    /api/reports                  # Custom reports
```

### 2.3 GraphQL Implementation

GraphQL offers advantages for complex LMS data requirements:

```graphql
type Query {
  # Course queries
  course(id: ID!): Course
  courses(filter: CourseFilter, pagination: Pagination): CourseConnection!
  
  # User queries  
  user(id: ID!): User
  currentUser: User
  
  # Enrollment queries
  enrollments(userId: ID, courseId: ID): [Enrollment!]!
  
  # Progress queries
  progress(enrollmentId: ID!): Progress
  
  # Analytics queries
  dashboardMetrics: DashboardMetrics!
  courseAnalytics(courseId: ID!): CourseAnalytics!
}

type Mutation {
  # Course mutations
  createCourse(input: CreateCourseInput!): Course!
  updateCourse(id: ID!, input: UpdateCourseInput!): Course!
  publishCourse(id: ID!): Course!
  
  # Enrollment mutations
  enrollUser(input: EnrollUserInput!): Enrollment!
  
  # Assessment mutations
  submitAssessment(input: SubmitAssessmentInput!): AssessmentResult!
  
  # Progress mutations
  updateProgress(input: UpdateProgressInput!): Progress!
}

type Course {
  id: ID!
  title: String!
  description: String
  instructor: User!
  modules: [Module!]!
  enrollments: [Enrollment!]!
  progress(currentUser: ID): Progress
  analytics: CourseAnalytics
}
```

### 2.4 Microservices Architecture

For large-scale LMS implementations, microservices architecture provides better scalability and maintainability:

| Service Domain | Responsibilities |
|----------------|------------------|
| **Auth Service** | Authentication, authorization, token management |
| **Course Service** | Content management, SCORM handling |
| **Enrollment Service** | User-course assignments, tracking |
| **Assessment Service** | Quiz processing, grading |
| **Analytics Service** | Data aggregation, reporting |
| **Notification Service** | Emails, push notifications |
| **File Service** | Media upload, storage, CDN |

---

## 3. Database Design Patterns

### 3.1 Core Database Schema

#### Users Table

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    role ENUM('admin', 'instructor', 'learner') NOT NULL DEFAULT 'learner',
    department_id UUID REFERENCES departments(id),
    avatar_url VARCHAR(500),
    locale VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'UTC',
    is_active BOOLEAN DEFAULT true,
    email_verified_at TIMESTAMP,
    last_login_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_tenant ON users(tenant_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
```

#### Courses Table

```sql
CREATE TABLE courses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    title VARCHAR(500) NOT NULL,
    slug VARCHAR(500) UNIQUE,
    description TEXT,
    thumbnail_url VARCHAR(500),
    instructor_id UUID REFERENCES users(id),
    status ENUM('draft', 'published', 'archived') DEFAULT 'draft',
    duration_hours DECIMAL(5,2),
    difficulty_level ENUM('beginner', 'intermediate', 'advanced'),
    category_id UUID REFERENCES categories(id),
    tags VARCHAR(255)[],
    language VARCHAR(10) DEFAULT 'en',
    certificate_enabled BOOLEAN DEFAULT false,
    passing_score INTEGER DEFAULT 70,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_courses_tenant ON courses(tenant_id);
CREATE INDEX idx_courses_status ON courses(status);
CREATE INDEX idx_courses_instructor ON courses(instructor_id);
```

#### Enrollments Table

```sql
CREATE TABLE enrollments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    course_id UUID REFERENCES courses(id),
    status ENUM('enrolled', 'in_progress', 'completed', 'dropped') DEFAULT 'enrolled',
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    progress_percentage DECIMAL(5,2) DEFAULT 0,
    final_grade DECIMAL(5,2),
    certificate_id UUID REFERENCES certificates(id),
    assigned_by UUID REFERENCES users(id),
    due_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, course_id)
);

CREATE INDEX idx_enrollments_user ON enrollments(user_id);
CREATE INDEX idx_enrollments_course ON enrollments(course_id);
CREATE INDEX idx_enrollments_status ON enrollments(status);
```

#### Progress Tracking Table

```sql
CREATE TABLE module_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    enrollment_id UUID REFERENCES enrollments(id),
    module_id UUID REFERENCES modules(id),
    status ENUM('not_started', 'in_progress', 'completed') DEFAULT 'not_started',
    time_spent_minutes INTEGER DEFAULT 0,
    score DECIMAL(5,2),
    attempts_count INTEGER DEFAULT 0,
    last_attempt_at TIMESTAMP,
    completed_at TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    UNIQUE(enrollment_id, module_id)
);

CREATE INDEX idx_module_progress_enrollment ON module_progress(enrollment_id);
CREATE INDEX idx_module_progress_module ON module_progress(module_id);
```

#### Assessments Table

```sql
CREATE TABLE assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module_id UUID REFERENCES modules(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    type ENUM('quiz', 'exam', 'assignment', 'project') DEFAULT 'quiz',
    passing_score INTEGER DEFAULT 70,
    time_limit_minutes INTEGER,
    randomize_questions BOOLEAN DEFAULT false,
    randomize_answers BOOLEAN DEFAULT false,
    show_results ENUM('immediate', 'after_deadline', 'never') DEFAULT 'immediate',
    max_attempts INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID REFERENCES assessments(id),
    type ENUM('multiple_choice', 'true_false', 'fill_blank', 'essay', 'matching') NOT NULL,
    question_text TEXT NOT NULL,
    points INTEGER DEFAULT 1,
    options JSONB,
    correct_answer JSONB,
    explanation TEXT,
    order_index INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE assessment_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID REFERENCES assessments(id),
    user_id UUID REFERENCES users(id),
    enrollment_id UUID REFERENCES enrollments(id),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP,
    score DECIMAL(5,2),
    answers JSONB,
    graded_at TIMESTAMP,
    graded_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3.2 Multi-Tenant Database Strategies

| Strategy | Description | Use Case | Pros | Cons |
|----------|-------------|----------|------|------|
| **Database per Tenant** | Complete isolation, separate databases | Enterprise, regulatory compliance | Maximum isolation, custom configs | Higher cost, complex management |
| **Schema per Tenant** | Shared database, separate schemas | Mid-market, cost-effective | Better isolation than shared | Moderate complexity |
| **Shared Database** | All data in single schema with tenant_id | SaaS, high scale | Cost-efficient, easy operations | Requires careful queries |

**Recommended Implementation:**

```sql
-- Shared database with tenant_id (most common for SaaS)
-- All tables include tenant_id column

CREATE TABLE users (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    -- ... other columns
);

-- Row-Level Security Policy
CREATE POLICY tenant_isolation_policy ON users
    USING (tenant_id = current_setting('app.tenant_id')::uuid);
```

---

## 4. Authentication and Authorization

### 4.1 OAuth 2.0 Implementation

OAuth 2.0 provides secure authentication for modern LMS platforms:

```
OAuth 2.0 Flow:
1. User clicks "Login with SSO"
2. Redirect to Identity Provider (IdP)
3. User authenticates with IdP
4. IdP redirects with authorization code
5. LMS exchanges code for access token
6. LMS retrieves user profile from IdP
7. Session created for user
```

**Implementation Example:**

```javascript
// OAuth2 Configuration
const oauthConfig = {
  authorizationURL: 'https://idp.example.com/oauth/authorize',
  tokenURL: 'https://idp.example.com/oauth/token',
  clientID: process.env.OAUTH_CLIENT_ID,
  clientSecret: process.env.OAUTH_CLIENT_SECRET,
  callbackURL: 'https://lms.example.com/auth/callback',
  scope: 'openid profile email'
};

// Authorization URL generation
const authUrl = `${oauthConfig.authorizationURL}?` +
  `client_id=${oauthConfig.clientID}&` +
  `redirect_uri=${encodeURIComponent(oauthConfig.callbackURL)}&` +
  `response_type=code&` +
  `scope=${encodeURIComponent(oauthConfig.scope)}&` +
  `state=${generateStateToken()}`;
```

### 4.2 SAML Integration

SAML remains prevalent in enterprise environments:

- **Single Sign-On (SSO)**: One login for multiple applications
- **Enterprise Integration**: Works with Okta, Azure AD, OneLogin
- **Security Assertions**: XML-based authentication tokens
- **Attribute Sharing**: User data passed from IdP to LMS

```xml
<!-- SAML AuthnRequest -->
<samlp:AuthnRequest 
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="_unique_id"
    Version="2.0"
    IssueInstant="2026-01-01T00:00:00Z"
    AssertionConsumerServiceURL="https://lms.example.com/saml/acs">
    <saml:Issuer>https://lms.example.com</saml:Issuer>
</samlp:AuthnRequest>
```

### 4.3 Role-Based Access Control (RBAC)

```
LMS Role Hierarchy:
├── System Administrator
│   ├── Full platform access
│   └── Tenant management
├── Tenant Administrator
│   ├── Tenant-specific settings
│   └── User management
├── Course Instructor
│   ├── Course creation/editing
│   └── Grade management
├── Content Creator
│   └── Content development
└── Learner
    ├── Course access
    └── Self-profile management
```

**Permission Matrix:**

| Permission | Admin | Instructor | Content Creator | Learner |
|------------|-------|------------|------------------|---------|
| Manage Users | ✓ | - | - | - |
| Create Courses | ✓ | ✓ | ✓ | - |
| Edit Own Courses | ✓ | ✓ | ✓ | - |
| Edit All Courses | ✓ | ✓ | - | - |
| View Analytics | ✓ | Own | Own | - |
| Take Courses | ✓ | ✓ | ✓ | ✓ |
| Create Assessments | ✓ | ✓ | ✓ | - |

---

## 5. Multi-Tenancy Patterns

### 5.1 Implementation Approaches

#### Shared Everything Model
- Single application instance serves all tenants
- Tenant identification via URL subdomain or header
- Data isolation through tenant_id columns
- Cost-efficient but requires careful resource management

```javascript
// Tenant identification middleware
app.use(async (req, res, next) => {
  // Extract tenant from subdomain
  const host = req.headers.host;
  const subdomain = host.split('.')[0];
  
  // Or from header (for API clients)
  const tenantId = req.headers['x-tenant-id'];
  
  req.tenantId = tenantId || await getTenantIdBySubdomain(subdomain);
  next();
});
```

#### Tenant Context in Queries

```javascript
// Automatically add tenant_id to all queries
const addTenantFilter = (query) => {
  return query.where('tenant_id', req.tenantId);
};

// Usage in service layer
const getCourses = async (req) => {
  return db('courses')
    .where('status', 'published')
    .andWhere('tenant_id', req.tenantId)
    .orderBy('created_at', 'desc');
};
```

### 5.2 Tenant Configuration

```sql
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    domain VARCHAR(255),
    settings JSONB DEFAULT '{}',
    features JSONB DEFAULT '{}',
    branding JSONB,
    storage_limit_gb INTEGER DEFAULT 10,
    max_users INTEGER,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 6. API Design

### 6.1 RESTful API Conventions

```
Base URL: https://api.lms.example.com/v1

Headers:
  Authorization: Bearer {access_token}
  Content-Type: application/json
  Accept: application/json
  X-Tenant-ID: {tenant_id}

Response Format:
{
  "data": { ... },
  "meta": {
    "page": 1,
    "per_page": 20,
    "total": 100
  }
}

Error Format:
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid email format",
    "details": [
      { "field": "email", "message": "Must be a valid email" }
    ]
  }
}
```

### 6.2 API Versioning

```javascript
// URL-based versioning
app.use('/v1', v1Routes);
app.use('/v2', v2Routes);

// Deprecation headers
app.use((req, res, next) => {
  res.set('Sunset', 'Sat, 01 Jan 2027 00:00:00 GMT');
  res.set('Link', '<https://api.lms.example.com/v2>; rel="successor-version"');
  next();
});
```

---

## 7. Integration Patterns

### 7.1 Webhook System

```javascript
// Webhook configuration
const webhookService = {
  async trigger(event, payload) {
    const subscriptions = await db('webhooks')
      .where('events', 'contains', event)
      .where('is_active', true);
    
    for (const subscription of subscriptions) {
      await this.deliver(subscription, event, payload);
    }
  },
  
  async deliver(subscription, event, payload) {
    const signature = crypto
      .createHmac('sha256', subscription.secret)
      .update(JSON.stringify(payload))
      .digest('hex');
    
    await fetch(subscription.url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Webhook-Signature': signature,
        'X-Webhook-Event': event
      },
      body: JSON.stringify(payload)
    });
  }
};
```

### 7.2 Event Types

| Event | Description | Payload |
|-------|-------------|---------|
| `user.created` | New user registered | User object |
| `course.enrolled` | User enrolled in course | Enrollment object |
| `course.completed` | Course completed | Progress object |
| `certificate.issued` | Certificate awarded | Certificate object |
| `assessment.submitted` | Assessment submitted | Attempt object |

---

## Quick Reference

### Common Integrations

| Integration Type | Common Tools | Purpose |
|-----------------|--------------|---------|
| **Video Conferencing** | Zoom, Microsoft Teams, Google Meet | Virtual classrooms |
| **HRIS** | Workday, BambooHR, ADP | User synchronization |
| **CRM** | Salesforce, HubSpot | Sales training |
| **Productivity** | Google Workspace, Microsoft 365 | Document handling |
| **Content** | YouTube, Vimeo, Go1 | Content libraries |
| **Assessment** | Pearsons, CertNexus | Certification |
| **Communication** | Slack, Microsoft Teams | Notifications |

### Technology Recommendations

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Frontend | React 18+ | Largest ecosystem, component reusability |
| Mobile | React Native | Code sharing with web |
| API Gateway | Kong/AWS API Gateway | Rate limiting, auth |
| Backend | Node.js/TypeScript | JavaScript everywhere |
| Database | PostgreSQL | ACID, JSON support |
| Cache | Redis | Session, real-time |
| Queue | RabbitMQ/Redis | Async processing |
| Search | Elasticsearch | Full-text search |
| Storage | S3/MinIO | Media storage |

---

## Next Steps

Continue with:

1. **[Implementation Guide](./03_implementation/)** - Planning and executing LMS projects
2. **[Production Readiness](./04_production/)** - Security, scalability, and operations
3. **[Platform Comparison](./05_platforms/)** - Evaluating different LMS solutions
