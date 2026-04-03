"""
LMS Backend - Learning Management System API
=============================================
FastAPI backend for AI-Mastery-2026 educational platform.

Handles:
- User authentication
- Course enrollment
- Progress tracking
- Quiz submissions
- Certificate generation
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import hashlib
import uuid
import json

# Initialize FastAPI app
app = FastAPI(
    title="AI-Mastery-2026 LMS API",
    description="Learning Management System API for AI-Mastery-2026",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# ============================================================================
# Database Models (In-memory for demo, use PostgreSQL in production)
# ============================================================================

class UserDB:
    """In-memory user database."""
    
    def __init__(self):
        self.users: Dict[str, dict] = {}
        self.sessions: Dict[str, dict] = {}
    
    def create_user(self, email: str, password: str, name: str) -> dict:
        """Create new user."""
        user_id = str(uuid.uuid4())
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        user = {
            "id": user_id,
            "email": email,
            "password": hashed_password,
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "enrolled_courses": [],
            "progress": {},
            "certificates": [],
            "badges": []
        }
        
        self.users[user_id] = user
        return {k: v for k, v in user.items() if k != "password"}
    
    def authenticate(self, email: str, password: str) -> Optional[dict]:
        """Authenticate user."""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        for user in self.users.values():
            if user["email"] == email and user["password"] == hashed_password:
                return {k: v for k, v in user.items() if k != "password"}
        
        return None
    
    def create_session(self, user_id: str) -> str:
        """Create session token."""
        token = str(uuid.uuid4())
        self.sessions[token] = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
        }
        return token
    
    def get_user_by_token(self, token: str) -> Optional[dict]:
        """Get user by session token."""
        session = self.sessions.get(token)
        if not session:
            return None
        
        # Check expiration
        if datetime.fromisoformat(session["expires_at"]) < datetime.utcnow():
            del self.sessions[token]
            return None
        
        user = self.users.get(session["user_id"])
        if user:
            return {k: v for k, v in user.items() if k != "password"}
        
        return None


# Initialize database
db = UserDB()

# ============================================================================
# Request/Response Models
# ============================================================================

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str = Field(..., min_length=2)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class CourseEnrollment(BaseModel):
    course_id: str
    tier: int

class ProgressUpdate(BaseModel):
    course_id: str
    module_id: str
    status: str = Field(..., pattern="^(not_started|in_progress|completed)$")
    score: Optional[float] = None

class QuizSubmission(BaseModel):
    quiz_id: str
    answers: Dict[str, str]

class QuizResult(BaseModel):
    quiz_id: str
    score: float
    passed: bool
    total_questions: int
    correct_answers: int

class CertificateResponse(BaseModel):
    certificate_id: str
    certificate_name: str
    issue_date: str
    verification_url: str

# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/api/v1/auth/register", response_model=dict)
async def register(user_data: UserRegister):
    """Register new user."""
    # Check if user exists
    for user in db.users.values():
        if user["email"] == user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Create user
    user = db.create_user(
        email=user_data.email,
        password=user_data.password,
        name=user_data.name
    )
    
    return {
        "message": "User registered successfully",
        "user": user
    }

@app.post("/api/v1/auth/login", response_model=AuthResponse)
async def login(credentials: UserLogin):
    """Login user."""
    user = db.authenticate(credentials.email, credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create session
    token = db.create_session(user["id"])
    
    return AuthResponse(
        access_token=token,
        user=user
    )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    user = db.get_user_by_token(credentials.credentials)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    return user

# ============================================================================
# Course Endpoints
# ============================================================================

@app.get("/api/v1/courses")
async def list_courses():
    """List all available courses."""
    courses = [
        {
            "id": "tier-0",
            "name": "Absolute Beginner",
            "tier": 0,
            "modules": 6,
            "hours": 45,
            "description": "Python basics, math foundations, AI intro"
        },
        {
            "id": "tier-1",
            "name": "Fundamentals",
            "tier": 1,
            "modules": 7,
            "hours": 102,
            "description": "Linear algebra, calculus, probability, statistics"
        },
        {
            "id": "tier-2",
            "name": "ML Practitioner",
            "tier": 2,
            "modules": 8,
            "hours": 109,
            "description": "Classical ML, deep learning, CNNs, RNNs"
        },
        {
            "id": "tier-3",
            "name": "LLM Engineer",
            "tier": 3,
            "modules": 10,
            "hours": 195,
            "description": "Transformers, RAG, fine-tuning, agents"
        },
        {
            "id": "tier-4",
            "name": "Production Expert",
            "tier": 4,
            "modules": 8,
            "hours": 130,
            "description": "MLOps, scaling, security, monitoring"
        },
        {
            "id": "tier-5",
            "name": "Capstone",
            "tier": 5,
            "modules": 6,
            "hours": 240,
            "description": "Real-world portfolio projects"
        }
    ]
    
    return {"courses": courses}

@app.post("/api/v1/courses/enroll")
async def enroll_course(
    enrollment: CourseEnrollment,
    current_user: dict = Depends(get_current_user)
):
    """Enroll in a course."""
    user_id = current_user["id"]
    course_id = enrollment.course_id
    
    # Check if already enrolled
    if course_id in current_user["enrolled_courses"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already enrolled in this course"
        )
    
    # Enroll user
    db.users[user_id]["enrolled_courses"].append(course_id)
    db.users[user_id]["progress"][course_id] = {
        "enrolled_at": datetime.utcnow().isoformat(),
        "modules_completed": 0,
        "total_modules": enrollment.tier + 6,  # Approximate
        "status": "active"
    }
    
    return {
        "message": f"Successfully enrolled in {course_id}",
        "course_id": course_id
    }

@app.get("/api/v1/progress")
async def get_progress(current_user: dict = Depends(get_current_user)):
    """Get user's course progress."""
    return {
        "user_id": current_user["id"],
        "enrolled_courses": current_user["enrolled_courses"],
        "progress": current_user["progress"],
        "certificates": current_user["certificates"],
        "badges": current_user["badges"]
    }

@app.post("/api/v1/progress/update")
async def update_progress(
    progress_update: ProgressUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update course progress."""
    user_id = current_user["id"]
    course_id = progress_update.course_id
    
    # Check enrollment
    if course_id not in current_user["enrolled_courses"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not enrolled in this course"
        )
    
    # Update progress
    if course_id not in db.users[user_id]["progress"]:
        db.users[user_id]["progress"][course_id] = {
            "modules": {},
            "quizzes": {},
            "status": "active"
        }
    
    # Update module status
    db.users[user_id]["progress"][course_id]["modules"][progress_update.module_id] = {
        "status": progress_update.status,
        "completed_at": datetime.utcnow().isoformat() if progress_update.status == "completed" else None,
        "score": progress_update.score
    }
    
    # Calculate completion percentage
    modules = db.users[user_id]["progress"][course_id]["modules"]
    completed = sum(1 for m in modules.values() if m["status"] == "completed")
    total = len(modules)
    completion_pct = (completed / total * 100) if total > 0 else 0
    
    return {
        "message": "Progress updated",
        "course_id": course_id,
        "module_id": progress_update.module_id,
        "completion_percentage": round(completion_pct, 2)
    }

# ============================================================================
# Quiz Endpoints
# ============================================================================

@app.post("/api/v1/quizzes/submit", response_model=QuizResult)
async def submit_quiz(
    submission: QuizSubmission,
    current_user: dict = Depends(get_current_user)
):
    """Submit quiz answers."""
    # Mock quiz data (in production, load from database)
    quiz_data = {
        "quiz_1_1": {
            "name": "Linear Algebra Quiz",
            "passing_score": 80.0,
            "correct_answers": {
                "q1": "A",
                "q2": "B",
                "q3": "C",
                "q4": "B",
                "q5": "A"
            }
        }
    }
    
    quiz = quiz_data.get(submission.quiz_id)
    if not quiz:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quiz not found"
        )
    
    # Grade quiz
    correct = sum(
        1 for q_id, answer in submission.answers.items()
        if quiz["correct_answers"].get(q_id) == answer
    )
    
    total = len(quiz["correct_answers"])
    score = (correct / total * 100) if total > 0 else 0
    passed = score >= quiz["passing_score"]
    
    # Save result
    user_id = current_user["id"]
    if "quiz_results" not in db.users[user_id]:
        db.users[user_id]["quiz_results"] = {}
    
    db.users[user_id]["quiz_results"][submission.quiz_id] = {
        "score": score,
        "passed": passed,
        "submitted_at": datetime.utcnow().isoformat()
    }
    
    # Award badge if passed
    if passed:
        badge = f"quiz_{submission.quiz_id}_passed"
        if badge not in db.users[user_id]["badges"]:
            db.users[user_id]["badges"].append(badge)
    
    return QuizResult(
        quiz_id=submission.quiz_id,
        score=round(score, 2),
        passed=passed,
        total_questions=total,
        correct_answers=correct
    )

# ============================================================================
# Certificate Endpoints
# ============================================================================

@app.post("/api/v1/certificates/generate", response_model=CertificateResponse)
async def generate_certificate(
    course_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Generate certificate for completed course."""
    user_id = current_user["id"]
    
    # Check if course completed
    progress = db.users[user_id]["progress"].get(course_id)
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Course not started"
        )
    
    # Verify completion (simplified check)
    modules = progress.get("modules", {})
    if not modules:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No modules completed"
        )
    
    completed = sum(1 for m in modules.values() if m["status"] == "completed")
    total = len(modules)
    
    if completed < total * 0.8:  # 80% completion required
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Need 80% completion. Currently at {completed/total*100:.1f}%"
        )
    
    # Generate certificate
    cert_id = f"CERT-{datetime.utcnow().strftime('%Y')}-{uuid.uuid4().hex[:8].upper()}"
    cert_name = f"{course_id.replace('-', ' ').title()} Certificate"
    
    certificate = {
        "certificate_id": cert_id,
        "certificate_name": cert_name,
        "user_id": user_id,
        "user_name": current_user["name"],
        "course_id": course_id,
        "issue_date": datetime.utcnow().isoformat(),
        "verification_url": f"https://verify.ai-mastery-2026.dev/{cert_id}",
        "blockchain_hash": hashlib.sha256(f"{cert_id}{user_id}".encode()).hexdigest()
    }
    
    # Save certificate
    db.users[user_id]["certificates"].append(certificate)
    
    return CertificateResponse(
        certificate_id=cert_id,
        certificate_name=cert_name,
        issue_date=certificate["issue_date"],
        verification_url=certificate["verification_url"]
    )

@app.get("/api/v1/certificates")
async def list_certificates(current_user: dict = Depends(get_current_user)):
    """List user's certificates."""
    return {
        "certificates": current_user["certificates"],
        "total": len(current_user["certificates"])
    }

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AI-Mastery-2026 LMS API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI-Mastery-2026 LMS API...")
    print("📚 API Docs: http://localhost:8000/docs")
    print("💓 Health Check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)
