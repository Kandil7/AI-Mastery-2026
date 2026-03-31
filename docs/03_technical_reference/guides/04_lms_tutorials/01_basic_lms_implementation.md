---
title: "Building a Basic Learning Management System: Step-by-Step Tutorial"
category: "tutorials"
subcategory: "lms_tutorials"
tags: ["lms", "tutorial", "beginner", "implementation"]
related: ["01_lms_fundamentals.md", "02_lms_architecture.md"]
difficulty: "beginner"
estimated_reading_time: 45
---

# Building a Basic Learning Management System: Step-by-Step Tutorial

This tutorial provides a hands-on guide to building a basic Learning Management System (LMS) from scratch. You'll implement core LMS functionality including user management, course creation, enrollment, and basic assessments.

## Prerequisites

### Technology Stack
- **Backend**: Node.js 18+ with Express.js
- **Database**: PostgreSQL 14+
- **Frontend**: React 18+ with Vite
- **Authentication**: JWT (JSON Web Tokens)
- **Testing**: Jest for unit tests, Cypress for E2E tests

### Development Environment
```bash
# Install Node.js and npm
node --version # Should be v18.x or higher
npm --version # Should be v9.x or higher

# Install PostgreSQL
psql --version # Should be 14.x or higher

# Install required tools
npm install -g typescript
npm install -g @types/node
```

## Project Setup

### Initialize the Project
```bash
mkdir lms-tutorial
cd lms-tutorial

# Create backend directory
mkdir backend
cd backend
npm init -y
npm install express pg dotenv cors bcryptjs jsonwebtoken uuid

# Create frontend directory
cd ..
mkdir frontend
cd frontend
npm create vite@latest . -- --template react-ts
cd frontend
npm install
npm install axios react-router-dom
```

### Database Setup
```sql
-- Create database
CREATE DATABASE lms_tutorial;

-- Connect to database
\c lms_tutorial;

-- Create users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'student',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create courses table
CREATE TABLE courses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    instructor_id UUID NOT NULL REFERENCES users(id),
    created_by UUID NOT NULL REFERENCES users(id),
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'published', 'archived')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create enrollments table
CREATE TABLE enrollments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    course_id UUID NOT NULL REFERENCES courses(id),
    progress NUMERIC(5,2) DEFAULT 0.0 CHECK (progress BETWEEN 0 AND 100),
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'enrolled' CHECK (status IN ('enrolled', 'completed', 'dropped')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Backend Implementation

### Server Configuration
```typescript
// backend/src/server.ts
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { Pool } from 'pg';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'healthy' });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### User Service Implementation
```typescript
// backend/src/services/userService.ts
import { Pool } from 'pg';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { v4 as uuidv4 } from 'uuid';

export class UserService {
  private pool: Pool;

  constructor(pool: Pool) {
    this.pool = pool;
  }

  async createUser(email: string, name: string, password: string, role: string = 'student') {
    const hashedPassword = await bcrypt.hash(password, 12);
    
    const query = `
      INSERT INTO users (email, name, password, role)
      VALUES ($1, $2, $3, $4)
      RETURNING id, email, name, role, created_at
    `;
    
    const result = await this.pool.query(query, [email, name, hashedPassword, role]);
    return result.rows[0];
  }

  async getUserByEmail(email: string) {
    const query = `
      SELECT id, email, name, password, role, created_at
      FROM users
      WHERE email = $1
    `;
    
    const result = await this.pool.query(query, [email]);
    return result.rows[0];
  }

  async validatePassword(password: string, hashedPassword: string) {
    return await bcrypt.compare(password, hashedPassword);
  }

  generateToken(user: any) {
    return jwt.sign(
      { id: user.id, email: user.email, role: user.role },
      process.env.JWT_SECRET || 'secret-key',
      { expiresIn: '24h' }
    );
  }
}
```

### Authentication Middleware
```typescript
// backend/src/middleware/auth.ts
import jwt from 'jsonwebtoken';
import { Request, Response, NextFunction } from 'express';

export const authenticate = (req: Request, res: Response, next: NextFunction) => {
  const token = req.headers.authorization?.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ message: 'Authentication required' });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'secret-key');
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(401).json({ message: 'Invalid token' });
  }
};

export const authorize = (roles: string[]) => {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!req.user || !roles.includes(req.user.role)) {
      return res.status(403).json({ message: 'Permission denied' });
    }
    next();
  };
};
```

### Course Service Implementation
```typescript
// backend/src/services/courseService.ts
import { Pool } from 'pg';

export class CourseService {
  private pool: Pool;

  constructor(pool: Pool) {
    this.pool = pool;
  }

  async createCourse(title: string, description: string, instructorId: string, createdBy: string) {
    const query = `
      INSERT INTO courses (title, description, instructor_id, created_by)
      VALUES ($1, $2, $3, $4)
      RETURNING id, title, description, instructor_id, created_by, status, created_at
    `;
    
    const result = await this.pool.query(query, [title, description, instructorId, createdBy]);
    return result.rows[0];
  }

  async getCourses(status?: string) {
    let query = `
      SELECT c.id, c.title, c.description, c.status, c.created_at,
             u.name as instructor_name
      FROM courses c
      JOIN users u ON c.instructor_id = u.id
    `;
    
    const params: any[] = [];
    
    if (status) {
      query += ' WHERE c.status = $1';
      params.push(status);
    }
    
    query += ' ORDER BY c.created_at DESC';
    
    const result = await this.pool.query(query, params);
    return result.rows;
  }

  async getCourseById(id: string) {
    const query = `
      SELECT c.id, c.title, c.description, c.status, c.created_at,
             u.name as instructor_name
      FROM courses c
      JOIN users u ON c.instructor_id = u.id
      WHERE c.id = $1
    `;
    
    const result = await this.pool.query(query, [id]);
    return result.rows[0];
  }
}
```

### Enrollment Service Implementation
```typescript
// backend/src/services/enrollmentService.ts
import { Pool } from 'pg';

export class EnrollmentService {
  private pool: Pool;

  constructor(pool: Pool) {
    this.pool = pool;
  }

  async enrollUser(userId: string, courseId: string) {
    // Check if already enrolled
    const checkQuery = `
      SELECT id FROM enrollments 
      WHERE user_id = $1 AND course_id = $2
    `;
    
    const checkResult = await this.pool.query(checkQuery, [userId, courseId]);
    
    if (checkResult.rows.length > 0) {
      return checkResult.rows[0];
    }

    const query = `
      INSERT INTO enrollments (user_id, course_id)
      VALUES ($1, $2)
      RETURNING id, user_id, course_id, progress, status, created_at
    `;
    
    const result = await this.pool.query(query, [userId, courseId]);
    return result.rows[0];
  }

  async getEnrollmentsByUser(userId: string) {
    const query = `
      SELECT e.id, e.course_id, e.progress, e.status, e.created_at,
             c.title as course_title, c.description as course_description
      FROM enrollments e
      JOIN courses c ON e.course_id = c.id
      WHERE e.user_id = $1
      ORDER BY e.created_at DESC
    `;
    
    const result = await this.pool.query(query, [userId]);
    return result.rows;
  }

  async updateProgress(enrollmentId: string, progress: number) {
    const query = `
      UPDATE enrollments
      SET progress = $1, updated_at = NOW()
      WHERE id = $2
      RETURNING id, user_id, course_id, progress, status, updated_at
    `;
    
    const result = await this.pool.query(query, [progress, enrollmentId]);
    return result.rows[0];
  }
}
```

### API Routes
```typescript
// backend/src/routes/index.ts
import express from 'express';
import { Router } from 'express';
import { UserService } from './services/userService';
import { CourseService } from './services/courseService';
import { EnrollmentService } from './services/enrollmentService';
import { authenticate, authorize } from './middleware/auth';

const router = Router();

// Initialize services
const userService = new UserService(pool);
const courseService = new CourseService(pool);
const enrollmentService = new EnrollmentService(pool);

// Auth routes
router.post('/auth/register', async (req, res) => {
  try {
    const { email, name, password } = req.body;
    const user = await userService.createUser(email, name, password);
    const token = userService.generateToken(user);
    
    res.status(201).json({ 
      user: { id: user.id, email: user.email, name: user.name, role: user.role },
      token
    });
  } catch (error) {
    res.status(500).json({ message: 'Registration failed', error: error.message });
  }
});

router.post('/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = await userService.getUserByEmail(email);
    
    if (!user) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }
    
    const isValid = await userService.validatePassword(password, user.password);
    
    if (!isValid) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }
    
    const token = userService.generateToken(user);
    res.json({ 
      user: { id: user.id, email: user.email, name: user.name, role: user.role },
      token
    });
  } catch (error) {
    res.status(500).json({ message: 'Login failed', error: error.message });
  }
});

// Course routes
router.get('/courses', authenticate, async (req, res) => {
  try {
    const courses = await courseService.getCourses('published');
    res.json(courses);
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch courses', error: error.message });
  }
});

router.post('/courses', authenticate, authorize(['instructor', 'admin']), async (req, res) => {
  try {
    const { title, description } = req.body;
    const user = req.user as any;
    
    const course = await courseService.createCourse(
      title, 
      description, 
      user.id, 
      user.id
    );
    
    res.status(201).json(course);
  } catch (error) {
    res.status(500).json({ message: 'Failed to create course', error: error.message });
  }
});

// Enrollment routes
router.post('/enrollments', authenticate, async (req, res) => {
  try {
    const { courseId } = req.body;
    const user = req.user as any;
    
    const enrollment = await enrollmentService.enrollUser(user.id, courseId);
    res.status(201).json(enrollment);
  } catch (error) {
    res.status(500).json({ message: 'Failed to enroll', error: error.message });
  }
});

router.get('/enrollments', authenticate, async (req, res) => {
  try {
    const user = req.user as any;
    const enrollments = await enrollmentService.getEnrollmentsByUser(user.id);
    res.json(enrollments);
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch enrollments', error: error.message });
  }
});

router.put('/enrollments/:id/progress', authenticate, async (req, res) => {
  try {
    const { id } = req.params;
    const { progress } = req.body;
    const user = req.user as any;
    
    // Verify user owns the enrollment
    const enrollmentQuery = `
      SELECT id FROM enrollments 
      WHERE id = $1 AND user_id = $2
    `;
    
    const enrollmentResult = await pool.query(enrollmentQuery, [id, user.id]);
    
    if (enrollmentResult.rows.length === 0) {
      return res.status(404).json({ message: 'Enrollment not found' });
    }
    
    const updatedEnrollment = await enrollmentService.updateProgress(id, progress);
    res.json(updatedEnrollment);
  } catch (error) {
    res.status(500).json({ message: 'Failed to update progress', error: error.message });
  }
});

export default router;
```

## Frontend Implementation

### React Component Structure
```typescript
// frontend/src/components/App.tsx
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Courses from './pages/Courses';
import CourseDetail from './pages/CourseDetail';
import Enrollments from './pages/Enrollments';

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            
            <Route 
              path="/dashboard" 
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/courses" 
              element={
                <ProtectedRoute>
                  <Courses />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/courses/:id" 
              element={
                <ProtectedRoute>
                  <CourseDetail />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/enrollments" 
              element={
                <ProtectedRoute>
                  <Enrollments />
                </ProtectedRoute>
              } 
            />
            
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;
```

### Authentication Context
```typescript
// frontend/src/contexts/AuthContext.tsx
import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

interface AuthContextType {
  user: any;
  login: (token: string) => void;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const decoded = JSON.parse(atob(token.split('.')[1]));
        setUser(decoded);
      } catch (e) {
        localStorage.removeItem('token');
      }
    }
  }, []);

  const login = (token: string) => {
    localStorage.setItem('token', token);
    const decoded = JSON.parse(atob(token.split('.')[1]));
    setUser(decoded);
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  const isAuthenticated = !!user;

  return (
    <AuthContext.Provider value={{ user, login, logout, isAuthenticated }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
```

### Course List Component
```typescript
// frontend/src/components/CourseList.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

interface Course {
  id: string;
  title: string;
  description: string;
  instructor_name: string;
  created_at: string;
  status: string;
}

interface CourseListProps {
  courses: Course[];
  onEnroll?: (courseId: string) => void;
}

const CourseList: React.FC<CourseListProps> = ({ courses, onEnroll }) => {
  const { user } = useAuth();

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {courses.map((course) => (
        <div key={course.id} className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow">
          <div className="p-6">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-xl font-bold text-gray-900">{course.title}</h3>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                course.status === 'published' ? 'bg-green-100 text-green-800' :
                course.status === 'draft' ? 'bg-yellow-100 text-yellow-800' :
                'bg-gray-100 text-gray-800'
              }`}>
                {course.status}
              </span>
            </div>
            
            <p className="text-gray-600 mb-4">{course.description}</p>
            
            <div className="flex items-center text-sm text-gray-500 mb-4">
              <span>By {course.instructor_name}</span>
              <span className="mx-2">•</span>
              <span>{new Date(course.created_at).toLocaleDateString()}</span>
            </2>
            
            <div className="flex space-x-3">
              <Link 
                to={`/courses/${course.id}`}
                className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors text-center"
              >
                View Details
              </Link>
              
              {user?.role === 'student' && onEnroll && (
                <button
                  onClick={() => onEnroll(course.id)}
                  className="bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 transition-colors"
                >
                  Enroll
                </button>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default CourseList;
```

### Enrollment Component
```typescript
// frontend/src/components/EnrollmentCard.tsx
import React from 'react';
import { Link } from 'react-router-dom';

interface Enrollment {
  id: string;
  course_id: string;
  progress: number;
  status: string;
  created_at: string;
  course_title: string;
  course_description: string;
}

interface EnrollmentCardProps {
  enrollment: Enrollment;
}

const EnrollmentCard: React.FC<EnrollmentCardProps> = ({ enrollment }) => {
  const getStatusColor = () => {
    switch (enrollment.status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'enrolled': return 'bg-blue-100 text-blue-800';
      case 'dropped': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-xl font-bold text-gray-900">{enrollment.course_title}</h3>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor()}`}>
          {enrollment.status}
        </span>
      </div>
      
      <p className="text-gray-600 mb-4">{enrollment.course_description}</p>
      
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span>Progress</span>
          <span>{Math.round(enrollment.progress)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-600 h-2 rounded-full" 
            style={{ width: `${enrollment.progress}%` }}
          ></div>
        </div>
      </div>
      
      <div className="flex space-x-3">
        <Link 
          to={`/courses/${enrollment.course_id}`}
          className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors text-center"
        >
          Continue Learning
        </Link>
        
        <button className="bg-gray-200 text-gray-800 py-2 px-4 rounded-md hover:bg-gray-300 transition-colors">
          Details
        </button>
      </div>
    </div>
  );
};

export default EnrollmentCard;
```

## Testing Implementation

### Unit Tests
```typescript
// backend/src/__tests__/userService.test.ts
import { Pool } from 'pg';
import { UserService } from '../services/userService';

describe('UserService', () => {
  let pool: Pool;
  let userService: UserService;

  beforeAll(async () => {
    pool = new Pool({
      connectionString: process.env.TEST_DATABASE_URL,
      ssl: false
    });
    userService = new UserService(pool);
  });

  afterAll(async () => {
    await pool.end();
  });

  describe('createUser', () => {
    it('should create a new user', async () => {
      const user = await userService.createUser(
        'test@example.com',
        'Test User',
        'password123'
      );
      
      expect(user).toBeDefined();
      expect(user.email).toBe('test@example.com');
      expect(user.name).toBe('Test User');
      expect(user.role).toBe('student');
    });
  });

  describe('getUserByEmail', () => {
    it('should find existing user by email', async () => {
      const user = await userService.getUserByEmail('test@example.com');
      
      expect(user).toBeDefined();
      expect(user.email).toBe('test@example.com');
    });
  });

  describe('validatePassword', () => {
    it('should validate correct password', async () => {
      const user = await userService.getUserByEmail('test@example.com');
      const isValid = await userService.validatePassword('password123', user.password);
      
      expect(isValid).toBe(true);
    });
  });
});
```

### E2E Tests with Cypress
```javascript
// frontend/cypress/e2e/auth.spec.js
describe('Authentication', () => {
  beforeEach(() => {
    cy.visit('http://localhost:5173/login');
  });

  it('should register a new user', () => {
    cy.get('input[name="email"]').type('test@example.com');
    cy.get('input[name="name"]').type('Test User');
    cy.get('input[name="password"]').type('password123');
    cy.get('button[type="submit"]').click();
    
    cy.url().should('include', '/dashboard');
    cy.contains('Welcome, Test User').should('be.visible');
  });

  it('should login with existing user', () => {
    cy.get('input[name="email"]').type('test@example.com');
    cy.get('input[name="password"]').type('password123');
    cy.get('button[type="submit"]').click();
    
    cy.url().should('include', '/dashboard');
    cy.contains('Welcome, Test User').should('be.visible');
  });
});
```

## Running the Application

### Development Setup
```bash
# In backend directory
cd backend
npm install
npm run dev

# In frontend directory
cd frontend
npm install
npm run dev
```

### Environment Variables
Create `.env` files in both directories:

**backend/.env**:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/lms_tutorial
JWT_SECRET=your-secret-key-here
NODE_ENV=development
```

**frontend/.env**:
```env
VITE_API_BASE_URL=http://localhost:3001
```

## Next Steps and Extensions

### Add Basic Assessment Functionality
1. **Create questions table** in PostgreSQL
2. **Implement question service** for CRUD operations
3. **Add assessment endpoints** for creating and taking quizzes
4. **Build frontend components** for quiz interface

### Implement Progress Tracking
1. **Add activity logging** for user interactions
2. **Create analytics dashboard** with charts
3. **Implement completion certificates**
4. **Add notifications system**

### Scale to Production
1. **Containerize with Docker**
2. **Deploy to Kubernetes**
3. **Add Redis for caching**
4. **Implement CI/CD pipeline**

This tutorial provides a solid foundation for building a Learning Management System. The architecture follows modern best practices with separation of concerns, proper authentication, and scalable design patterns. You can extend this basic implementation with additional features like content delivery, advanced assessments, and AI-powered personalization.

## Related Resources

- [LMS Fundamentals] - Core concepts and architecture
- [API Implementation Guide] - Advanced API design patterns
- [Database Design Best Practices] - PostgreSQL optimization techniques
- [React Frontend Patterns] - Modern React development practices

Happy coding! 🎓