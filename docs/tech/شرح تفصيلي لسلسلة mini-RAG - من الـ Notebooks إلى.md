<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# شرح تفصيلي لسلسلة mini-RAG - من الـ Notebooks إلى الـ Production

## نظرة عامة على المشروع

مشروع **mini-RAG** هو مشروع تعليمي شامل يهدف لبناء تطبيق RAG (Retrieval-Augmented Generation) احترافي جاهز للإنتاج من الصفر حتى النشر. المشروع متوفر على GitHub ويحتوي على 25 فيديو تعليمي باللغة العربية.[^1_1][^1_2][^1_3]

## الفيديوهات بالتفصيل:

### **1️⃣ الفيديو الأول: About the Course - ماذا ولماذا** (5 دقائق)

**الهدف:** مقدمة عن السلسلة وأهميتها

**المحتوى:**

- **المشكلة الأساسية:** الفجوة بين Data Science و Software Engineering - كثير من الناس يعرفون Machine Learning لكن يعيشون في "فقاعة Notebooks" ولا يعرفون كيف يحولوا مشاريعهم لتطبيقات حقيقية[^1_3]
- **الحل:** بناء تطبيق RAG خطوة بخطوة باستخدام Web Framework (FastAPI)
- **الفكرة:** تحويل التجارب من Jupyter Notebooks إلى Production Application
- **GitHub Structure:** كل tutorial له branch خاص، والـ main branch فيه النسخة النهائية[^1_1]

**لا يوجد أكواد في هذا الفيديو**

***

### **2️⃣ الفيديو الثاني: What will we build - ماذا سنبني** (3 دقائق)

**الهدف:** توضيح المشروع النهائي

**المحتوى:**

- شرح نموذج RAG: نظام يسمح بطرح أسئلة على مجموعة من المستندات والحصول على إجابات مدعومة بالسياق
- استخدامات RAG في المؤسسات والشركات
- البنية المعمارية للتطبيق المستهدف

**لا يوجد أكواد في هذا الفيديو**

***

### **3️⃣ الفيديو الثالث: Setup your tools - الأدوات الأساسية** (18 دقيقة)

**الهدف:** إعداد بيئة التطوير

**الأدوات المطلوبة:**

- Python 3.10
- MiniConda لإدارة البيئات
- Visual Studio Code
- Git \& GitHub

**الأكواد:**

```bash
# إنشاء بيئة conda جديدة
conda create -n mini-rag python=3.10

# تفعيل البيئة
conda activate mini-rag

# تحسين command line interface
export PS1="\\[\\033[01;32m\\]\\u@\\h:\\w\\n\\[\\033[00m\\]\\$ "
```

**لا يوجد branch خاص بهذا الدرس**

***

### **4️⃣ الفيديو الرابع: Project Architecture** (26 دقيقة)

**Branch:** `tut-001`

**الهدف:** فهم البنية المعمارية للمشروع

**المحتوى:**

- شرح معمارية المشروع الكاملة
- تقسيم المشروع إلى Layers
- فهم Data Flow بين المكونات
- شرح دور كل component في النظام

**البنية الأساسية:**

```
mini-rag/
├── src/           # الكود الرئيسي
├── docker/        # Docker configurations
├── .env.example   # ملف البيئة التجريبي
└── requirements.txt
```


***

### **5️⃣ الفيديو الخامس: Welcome to FastAPI** (15 دقيقة)

**Branch:** `tut-002`

**الهدف:** البداية مع FastAPI Framework

**المفاهيم:**

- مقدمة عن FastAPI وفلسفته
- إنشاء أول API endpoint
- Auto-generated documentation (Swagger UI)
- Type hints و Pydantic models

**الأكواد الأساسية:**

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to mini-RAG"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**تشغيل التطبيق:**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```


***

### **6️⃣ الفيديو السادس: Nested Routes + Env Values** (15 دقيقة)

**Branch:** `tut-003`

**الهدف:** تنظيم الـ Routes واستخدام Environment Variables

**المفاهيم:**

- FastAPI Router لتنظيم الـ endpoints
- إدارة Environment Variables باستخدام Pydantic Settings
- فصل الـ configurations عن الكود

**الأكواد:**

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "mini-RAG"
    openai_api_key: str
    
    class Config:
        env_file = ".env"

settings = Settings()

# routes/documents.py
from fastapi import APIRouter

router = APIRouter(prefix="/documents", tags=["documents"])

@router.get("/")
async def list_documents():
    return {"documents": []}

# main.py
from fastapi import FastAPI
from routes import documents

app = FastAPI()
app.include_router(documents.router)
```

**ملف .env:**

```bash
APP_NAME=mini-RAG
OPENAI_API_KEY=your-api-key-here
```


***

### **7️⃣ الفيديو السابع: Uploading a File** (ساعة و23 دقيقة)

**Branch:** `tut-004`

**الهدف:** رفع الملفات ومعالجتها

**المفاهيم:**

- File upload في FastAPI
- Validation للملفات (type, size)
- حفظ الملفات بشكل آمن
- معالجة الأخطاء

**الأكواد:**

```python
from fastapi import UploadFile, File, HTTPException
import aiofiles
import os

UPLOAD_DIR = "uploads"

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # التحقق من نوع الملف
    allowed_types = ["application/pdf", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, "File type not allowed")
    
    # التحقق من حجم الملف
    file.file.seek(0, 2)
    file_size = file.file.tell()
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(400, "File too large")
    
    file.file.seek(0)
    
    # حفظ الملف
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return {
        "filename": file.filename,
        "size": file_size,
        "content_type": file.content_type
    }
```


***

### **8️⃣ الفيديو الثامن: File Processing** (42 دقيقة)

**Branch:** `tut-005`

**الهدف:** معالجة الملفات واستخراج النصوص

**المفاهيم:**

- استخدام libraries لاستخراج النصوص (PyPDF2, python-docx)
- Text chunking strategies
- Metadata extraction

**الأكواد:**

```python
from PyPDF2 import PdfReader
import docx

class FileProcessor:
    def process_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def process_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks
```


***

### **9️⃣ الفيديو التاسع: Docker - MongoDB - Motor** (38 دقيقة)

**Branch:** `tut-006`

**الهدف:** إعداد قاعدة بيانات MongoDB باستخدام Docker

**المفاهيم:**

- Docker Compose للخدمات
- MongoDB كقاعدة بيانات NoSQL
- Motor: async MongoDB driver for Python
- Connection pooling

**الأكواد:**

```yaml
# docker-compose.yml
version: '3.8'
services:
  mongodb:
    image: mongo:latest
    container_name: minirag-mongodb
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password
    volumes:
      - mongodb_data:/data/db

volumes:
  mongodb_data:
```

```python
# database.py
from motor.motor_asyncio import AsyncIOMotorClient
from config import settings

class Database:
    client: AsyncIOMotorClient = None
    
db = Database()

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(settings.mongodb_url)
    
async def close_mongo_connection():
    db.client.close()

def get_database():
    return db.client[settings.database_name]
```

**تشغيل Docker:**

```bash
cd docker
cp .env.example .env
sudo docker compose up -d
```


***

### **🔟 الفيديو العاشر: Mongo Schemes and Models** (53 دقيقة)

**Branch:** `tut-007`

**الهدف:** تصميم نماذج البيانات

**المفاهيم:**

- Pydantic models للتحقق من البيانات
- MongoDB collections design
- CRUD operations
- Data validation

**الأكواد:**

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

class DocumentModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    filename: str
    content: str
    chunks: list[str]
    metadata: dict
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}

# CRUD operations
class DocumentRepository:
    def __init__(self, database):
        self.collection = database["documents"]
    
    async def create(self, document: DocumentModel):
        result = await self.collection.insert_one(
            document.dict(by_alias=True, exclude={"id"})
        )
        return str(result.inserted_id)
    
    async def get(self, doc_id: str):
        doc = await self.collection.find_one({"_id": ObjectId(doc_id)})
        return DocumentModel(**doc) if doc else None
    
    async def list(self, skip: int = 0, limit: int = 10):
        cursor = self.collection.find().skip(skip).limit(limit)
        return [DocumentModel(**doc) async for doc in cursor]
```


***

### **1️⃣1️⃣ الفيديو الحادي عشر: Mongo Indexing** (ساعة و13 دقيقة)

**Branch:** `tut-008`

**الهدف:** تحسين الأداء باستخدام Indexes

**المفاهيم:**

- MongoDB indexing strategies
- Single field indexes
- Compound indexes
- Text indexes للبحث
- Performance optimization

**الأكواد:**

```python
async def create_indexes():
    database = get_database()
    
    # Index على filename
    await database.documents.create_index("filename", unique=True)
    
    # Compound index
    await database.documents.create_index([
        ("created_at", -1),
        ("filename", 1)
    ])
    
    # Text index للبحث في المحتوى
    await database.documents.create_index([
        ("content", "text"),
        ("metadata.title", "text")
    ])
    
    # Index على metadata fields
    await database.documents.create_index("metadata.category")

# البحث باستخدام text index
async def search_documents(query: str):
    database = get_database()
    cursor = database.documents.find(
        {"$text": {"$search": query}}
    ).limit(10)
    return [doc async for doc in cursor]
```


***

### **1️⃣2️⃣ الفيديو الثاني عشر: Data Pipeline Enhancements** (43 دقيقة)

**Branch:** `tut-008`

**الهدف:** تحسين pipeline معالجة البيانات

**المفاهيم:**

- Background tasks
- Pipeline stages
- Error handling
- Progress tracking

**الأكواد:**

```python
from fastapi import BackgroundTasks

class DataPipeline:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository
    
    async def process_document(self, file_path:


<div align="center">⁂</div>

[^1_1]: https://github.com/bakrianoo/mini-rag
[^1_2]: https://github.com/bakrianoo/mini-rag
[^1_3]: https://www.youtube.com/watch?v=Vv6e2Rb1Q6w&list=PLvLvlVqNQGHCUR2p0b8a0QpVjDUg50wQj

---

# شرح تفصيلي ودقيق لأول 10 فيديوهات من سلسلة mini-RAG

## 📚 نظرة عامة على المشروع
**mini-RAG** هو مشروع تعليمي شامل لبناء تطبيق RAG (Retrieval-Augmented Generation) جاهز للإنتاج، يهدف لسد الفجوة بين Data Science و Software Engineering.[^2_1][^2_2]

***

## 🎬 الفيديو الأول: About the Course - ماذا ولماذا (5:52 دقيقة)

### الأهداف التعليمية:
- **المشكلة الرئيسية**: كثير من مهندسي الـ Data Science يعيشون في "فقاعة Jupyter Notebooks" ولا يعرفون كيفية تحويل تجاربهم إلى تطبيقات حقيقية قابلة للاستخدام
- **الحل**: بناء تطبيق RAG كامل من البداية للنهاية باستخدام Web Framework (FastAPI) في Python
- **الفلسفة**: أنت في النهاية مهندس برمجيات، ليس فقط Data Scientist

### المحتوى الأساسي:
1. **لماذا هذه السلسلة؟**
   - شركات كثيرة ترفض مهندسي ML بسبب ضعف مهارات Software Engineering
   - مشاريع كثيرة تفشل بسبب ضعف الـ Software Engineering وليس الـ Machine Learning
   - الحاجة لتحويل Notebooks إلى Production Applications

2. **عن المشروع:**
   - اسم المشروع: mini-RAG
   - مشروع مفتوح المصدر على GitHub
   - كل tutorial له branch خاص بالأكواد
   - Main branch يحتوي على النسخة النهائية[^2_1]

3. **ما هو RAG؟**
   - تطبيق يسمح بطرح أسئلة على مجموعة من المستندات
   - يستخرج الإجابات من المستندات باستخدام LLMs
   - أصبح نوع تطبيق منتشر جدًا في المؤسسات

### البنية التنظيمية على GitHub:
```

mini-rag/
├── main branch (النسخة النهائية)
├── tut-001 (Tutorial 4: Project Architecture)
├── tut-002 (Tutorial 5: Welcome to FastAPI)
├── tut-003 (Tutorial 6: Nested Routes)
└── ... (باقي الـ tutorials)

```

**لا توجد أكواد في هذا الفيديو - مقدمة نظرية**

***

## 🎬 الفيديو الثاني: What will we build - ماذا سنبني (3:37 دقيقة)

### الأهداف التعليمية:
- فهم المشروع النهائي الذي سنبنيه
- استيعاب مفهوم RAG applications
- معرفة المكونات الأساسية للتطبيق

### ما سنبنيه:
1. **Backend API** باستخدام FastAPI
2. **Document Upload System** - رفع ومعالجة المستندات
3. **Vector Database** - لتخزين embeddings
4. **LLM Integration** - للإجابة على الأسئلة
5. **Semantic Search** - للبحث في المستندات
6. **Production-ready features** - Docker, Database, APIs

### معمارية RAG:
```

User Question → Semantic Search → Retrieve Context → LLM → Augmented Answer
↓
Vector DB
↓
Documents

```

**لا توجد أكواد في هذا الفيديو - شرح معماري**

***

## 🎬 الفيديو الثالث: Setup your tools - الأدوات الأساسية (18 دقيقة)

### الأدوات المطلوبة:
1. **Python 3.10** (أو أحدث)
2. **MiniConda** - لإدارة البيئات
3. **VS Code** - محرر الأكواد
4. **Git & GitHub** - للتحكم بالنسخ

### الأكواد الأساسية:

#### 1. تثبيت Python باستخدام MiniConda:
```bash
# تحميل MiniConda
# قم بتحميله من: https://docs.anaconda.com/free/miniconda/

# إنشاء بيئة جديدة
$ conda create -n mini-rag python=3.10

# تفعيل البيئة
$ conda activate mini-rag

# التحقق من إصدار Python
$ python --version
```


#### 2. تحسين Command Line Interface:

```bash
# إضافة هذا السطر في ~/.bashrc أو ~/.zshrc
export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "

# تطبيق التغييرات
$ source ~/.bashrc
```

هذا الأمر يجعل الـ terminal أكثر وضوحًا بإظهار:

- `\u` = username
- `\h` = hostname
- `\w` = working directory
- `\n` = سطر جديد


#### 3. إعداد Git:

```bash
# إعداد المعلومات الشخصية
$ git config --global user.name "Your Name"
$ git config --global user.email "your.email@example.com"

# استنساخ المشروع
$ git clone https://github.com/bakrianoo/mini-rag.git
$ cd mini-rag
```

**لا يوجد branch خاص بهذا الدرس**

***

## 🎬 الفيديو الرابع: Project Architecture (26 دقيقة)

**Branch:** `tut-001`

### الأهداف التعليمية:

- فهم البنية المعمارية الكاملة للمشروع
- تصميم الـ Layers والمكونات
- فهم Data Flow


### البنية المعمارية:

```
┌─────────────────────────────────────────┐
│          API Layer (FastAPI)            │
│  ┌──────────┐  ┌──────────┐            │
│  │ Upload   │  │  Query   │            │
│  │ Endpoint │  │ Endpoint │            │
│  └──────────┘  └──────────┘            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│        Business Logic Layer             │
│  ┌────────────┐  ┌─────────────┐       │
│  │ Document   │  │   Search    │       │
│  │ Processor  │  │   Engine    │       │
│  └────────────┘  └─────────────┘       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Data Access Layer               │
│  ┌──────────┐  ┌──────────────┐        │
│  │ MongoDB  │  │ Vector DB    │        │
│  │ (Docs)   │  │ (Embeddings) │        │
│  └──────────┘  └──────────────┘        │
└─────────────────────────────────────────┘
```


### هيكل المجلدات الأساسي:

```
mini-rag/
├── assets/              # ملفات الوسائط والصور
├── .env.example         # مثال لملف البيئة
├── .gitignore          # ملفات مستبعدة من Git
├── LICENSE             # رخصة المشروع (Apache-2.0)
├── README.md           # توثيق المشروع
└── requirements.txt    # المكتبات المطلوبة
```


### ملف .env.example:

```bash
# Application Settings
APP_NAME=mini-RAG
APP_VERSION=1.0.0
DEBUG=True

# OpenAI API
OPENAI_API_KEY=your-openai-api-key-here

# Database
DATABASE_URL=mongodb://localhost:27017
DATABASE_NAME=minirag

# Server
HOST=0.0.0.0
PORT=5000
```


### ملف .gitignore:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# Environment
.env

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

**الأكواد في هذا Branch:**

- إعداد البنية الأساسية للمشروع
- ملفات الإعداد الأولية

***

## 🎬 الفيديو الخامس: Welcome to FastAPI (14:55 دقيقة)

**Branch:** `tut-002`

### الأهداف التعليمية:

- مقدمة عن FastAPI وميزاته
- إنشاء أول API endpoint
- فهم Auto-generated Documentation
- استخدام Type Hints و Pydantic


### لماذا FastAPI؟

1. **سريع جدًا** - من أسرع Python frameworks
2. **Type Safety** - استخدام Type Hints
3. **Auto Documentation** - Swagger UI تلقائي
4. **Async Support** - دعم البرمجة غير المتزامنة
5. **Modern Python** - مبني على Python 3.6+

### الأكواد الأساسية:

#### 1. ملف requirements.txt:

```txt
fastapi==0.95.0
uvicorn[standard]==0.21.1
python-dotenv==1.0.0
```


#### 2. ملف main.py (الإصدار الأول):

```python
from fastapi import FastAPI

# إنشاء تطبيق FastAPI
app = FastAPI()

# أول endpoint - الصفحة الرئيسية
@app.get("/")
def welcome():
    return {
        "message": "Hello World!"
    }
```


#### 3. تشغيل التطبيق:

```bash
# تثبيت المكتبات
$ pip install -r requirements.txt

# تشغيل الخادم مع auto-reload
$ uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

**Output:**

```
INFO:     Uvicorn running on http://0.0.0.0:5000
INFO:     Application startup complete.
```


#### 4. اختبار الـ API:

```bash
# باستخدام curl
$ curl http://localhost:5000/
{"message":"Hello World!"}

# أو افتح المتصفح:
http://localhost:5000/
```


#### 5. الوصول للـ Documentation التلقائية:

```
# Swagger UI
http://localhost:5000/docs

# ReDoc
http://localhost:5000/redoc
```


### إضافة Health Check Endpoint:

```python
from fastapi import FastAPI

app = FastAPI(
    title="mini-RAG API",
    description="A minimal RAG implementation",
    version="1.0.0"
)

@app.get("/")
def welcome():
    return {
        "message": "Welcome to mini-RAG API"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "mini-RAG"
    }
```

**الكود النهائي في tut-002:**[^2_3]

```python
def welcome():
    return {
        "message": "Hello World!"
    }
```


***

## 🎬 الفيديو السادس: Nested Routes + Env Values (15 دقيقة)

**Branch:** `tut-003`

### الأهداف التعليمية:

- تنظيم الـ Routes باستخدام APIRouter
- إدارة Environment Variables
- استخدام Pydantic Settings
- فصل الإعدادات عن الكود


### المفاهيم الأساسية:

#### 1. لماذا نحتاج لـ APIRouter؟

- **تنظيم الكود**: فصل endpoints حسب الوظيفة
- **Scalability**: سهولة إضافة features جديدة
- **Maintainability**: كود أسهل للصيانة
- **Modularity**: كل module مستقل


### البنية الجديدة:

```
mini-rag/
├── main.py              # نقطة الدخول الرئيسية
├── config.py            # إعدادات التطبيق
├── routes/              # مجلد الـ routes
│   ├── __init__.py
│   ├── documents.py     # routes الخاصة بالمستندات
│   └── queries.py       # routes الخاصة بالاستعلامات
└── .env                 # المتغيرات البيئية
```


### الأكواد التفصيلية:

#### 1. ملف config.py - إدارة الإعدادات:

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # معلومات التطبيق
    app_name: str = "mini-RAG"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # OpenAI API
    openai_api_key: str
    
    # Database
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "minirag"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 5000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# إ


<div align="center">⁂</div>

[^2_1]: https://github.com/bakrianoo/mini-rag
[^2_2]: https://github.com/bakrianoo/mini-rag
[^2_3]: https://github.com/bakrianoo/mini-rag/commit/308e00f0715481c03f9f59a5f0662eea1406e204

---

# اكمل

سأكمل الشرح التفصيلي الدقيق:

***

## 🎬 الفيديو السادس: Nested Routes + Env Values (تكملة)

### الأكواد التفصيلية:

#### 1. ملف config.py - إدارة الإعدادات (تكملة):
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # معلومات التطبيق
    app_name: str = "mini-RAG"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # OpenAI API
    openai_api_key: str
    
    # Database
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "minirag"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 5000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# إنشاء instance من Settings
settings = Settings()
```


#### 2. ملف .env:

```bash
APP_NAME=mini-RAG
APP_VERSION=1.0.0
DEBUG=true

OPENAI_API_KEY=sk-your-openai-api-key-here

MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=minirag

HOST=0.0.0.0
PORT=5000
```


#### 3. ملف routes/documents.py:

```python
from fastapi import APIRouter, HTTPException
from typing import List

# إنشاء router للمستندات
router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": "Not found"}}
)

@router.get("/")
async def list_documents():
    """
    الحصول على قائمة بجميع المستندات
    """
    return {
        "documents": [],
        "total": 0
    }

@router.get("/{document_id}")
async def get_document(document_id: str):
    """
    الحصول على مستند محدد بواسطة ID
    """
    return {
        "id": document_id,
        "filename": "example.pdf",
        "status": "processed"
    }

@router.post("/")
async def create_document():
    """
    رفع مستند جديد
    """
    return {
        "message": "Document created",
        "id": "doc_123"
    }
```


#### 4. ملف routes/queries.py:

```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(
    prefix="/queries",
    tags=["queries"]
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@router.post("/search")
async def search(query: QueryRequest):
    """
    البحث في المستندات
    """
    return {
        "question": query.question,
        "results": [],
        "total": 0
    }

@router.post("/answer")
async def answer(query: QueryRequest):
    """
    الحصول على إجابة كاملة
    """
    return {
        "question": query.question,
        "answer": "This is a sample answer",
        "sources": []
    }
```


#### 5. ملف main.py المحدث:

```python
from fastapi import FastAPI
from config import settings
from routes import documents, queries

# إنشاء التطبيق
app = FastAPI(
    title=settings.app_name,
    description="A minimal RAG implementation for production",
    version=settings.app_version,
    debug=settings.debug
)

# تضمين الـ routers
app.include_router(documents.router)
app.include_router(queries.router)

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": settings.app_name
    }

# Event handlers
@app.on_event("startup")
async def startup_event():
    print(f"Starting {settings.app_name}...")
    print(f"Debug mode: {settings.debug}")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down...")
```


#### 6. تشغيل التطبيق:

```bash
# تشغيل الخادم
$ uvicorn main:app --reload --host 0.0.0.0 --port 5000

# اختبار الـ endpoints
$ curl http://localhost:5000/documents/
$ curl http://localhost:5000/queries/search -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```


***

## 🎬 الفيديو السابع: Uploading a File (1:23 ساعة)

**Branch:** `tut-004`

### الأهداف التعليمية:

- رفع الملفات في FastAPI
- Validation للملفات
- حفظ الملفات بشكل آمن
- معالجة الأخطاء والـ exceptions


### المفاهيم الأساسية:

#### 1. لماذا نحتاج File Upload؟

- RAG يحتاج لمستندات لتحليلها
- المستخدم يرفع PDF, Word, Text files
- نحتاج لحفظ وتتبع الملفات


### الأكواد التفصيلية:

#### 1. تحديث requirements.txt:

```txt
fastapi==0.95.0
uvicorn[standard]==0.21.1
python-dotenv==1.0.0
python-multipart==0.0.6  # مطلوب لـ file upload
aiofiles==23.1.0          # للكتابة async
```


#### 2. ملف models/document.py:

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class DocumentBase(BaseModel):
    filename: str
    file_size: int
    content_type: str

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: str
    upload_date: datetime
    status: str = "uploaded"
    
    class Config:
        from_attributes = True
```


#### 3. ملف helpers/file_handler.py:

```python
import os
import aiofiles
from typing import BinaryIO
from fastapi import UploadFile, HTTPException
import hashlib
from datetime import datetime

class FileHandler:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        self.allowed_extensions = {".pdf", ".txt", ".docx", ".doc"}
        self.max_file_size = 10 * 1024 * 1024  # 10 MB
        
        # إنشاء المجلد إذا لم يكن موجودًا
        os.makedirs(upload_dir, exist_ok=True)
    
    def validate_file_extension(self, filename: str) -> bool:
        """
        التحقق من امتداد الملف
        """
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.allowed_extensions
    
    async def validate_file_size(self, file: UploadFile) -> int:
        """
        التحقق من حجم الملف
        """
        # قراءة الملف للحصول على الحجم
        file.file.seek(0, 2)  # الانتقال لنهاية الملف
        file_size = file.file.tell()  # الحصول على الموقع = الحجم
        file.file.seek(0)  # العودة للبداية
        
        if file_size > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {self.max_file_size / 1024 / 1024} MB"
            )
        
        return file_size
    
    def generate_unique_filename(self, original_filename: str) -> str:
        """
        إنشاء اسم ملف فريد
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(original_filename)
        # إنشاء hash من الاسم الأصلي
        hash_obj = hashlib.md5(name.encode())
        hash_str = hash_obj.hexdigest()[:8]
        
        return f"{timestamp}_{hash_str}{ext}"
    
    async def save_file(self, file: UploadFile) -> tuple[str, int]:
        """
        حفظ الملف وإرجاع المسار والحجم
        """
        # التحقق من الامتداد
        if not self.validate_file_extension(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"File extension not allowed. Allowed: {self.allowed_extensions}"
            )
        
        # التحقق من الحجم
        file_size = await self.validate_file_size(file)
        
        # إنشاء اسم فريد
        unique_filename = self.generate_unique_filename(file.filename)
        file_path = os.path.join(self.upload_dir, unique_filename)
        
        # حفظ الملف
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error saving file: {str(e)}"
            )
        
        return file_path, file_size
    
    def delete_file(self, file_path: str) -> bool:
        """
        حذف ملف
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception:
            return False
```


#### 4. تحديث routes/documents.py:

```python
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from helpers.file_handler import FileHandler
from models.document import DocumentResponse
from typing import List
import uuid
from datetime import datetime

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

# إنشاء instance من FileHandler
file_handler = FileHandler(upload_dir="uploads")

# قاعدة بيانات مؤقتة في الذاكرة
documents_db = {}

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    رفع مستند جديد
    
    Args:
        file: الملف المراد رفعه
    
    Returns:
        معلومات المستند المرفوع
    """
    try:
        # حفظ الملف
        file_path, file_size = await file_handler.save_file(file)
        
        # إنشاء ID للمستند
        doc_id = str(uuid.uuid4())
        
        # حفظ معلومات المستند
        document = {
            "id": doc_id,
            "filename": file.filename,
            "file_size": file_size,
            "content_type": file.content_type,
            "file_path": file_path,
            "upload_date": datetime.now(),
            "status": "uploaded"
        }
        
        documents_db[doc_id] = document
        
        return DocumentResponse(**document)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

@router.get("/", response_model=List[DocumentResponse])
async def list_documents():
    """
    الحصول على قائمة بجميع المستندات
    """
    return [DocumentResponse(**doc) for doc in documents_db.values()]

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """
    الحصول على مستند محدد
    """
    if document_id not in documents_db:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    return DocumentResponse(**documents_db[document_id])

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    حذف مستند
    """
    if document_id not in documents_db:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    document = documents_db[document_id]
    
    # حذف الملف من النظام
    file_handler.delete_file(document["file_path"])
    
    # حذف من قاعدة البيانات
    del documents_db[document_id]
    
    return {"message": "Document deleted successfully"}
```


#### 5. اختبار الـ Upload:

```bash
# رفع ملف
$ curl -X POST "http://localhost:5000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/file.pdf"

# الحصول على قائمة المستندات
$ curl http://localhost:5000/documents/

# حذف مستند
$ curl -X DELETE http://localhost:5000/documents/{document_id}
```


#### 6. اختبار باستخدام Python:

```python
import requests

# رفع ملف
url = "http://localhost:5000/documents/upload"
files = {"file": open("example.pdf", "rb")}
response = requests.post(url, files=files)
print(response.json())

# الحصول على القائمة
response = requests.get("http://localhost:5000/documents/")
print(response.json())
```


***

## 🎬 الفيديو الثامن: File Processing (42 دقيقة)

**Branch:** `tut-005`

### الأهداف التعليمية:

- استخراج النصوص من الملفات
- Text chunking strategies
- معالجة أنواع ملفات مختلفة
- Metadata extraction


### المفاهيم الأساسية:

#### 1. لماذا نحتاج File Processing؟

- استخراج النص من PDF, Word
- تقسيم النص لـ chunks صغيرة
- تحضير البيانات للـ embedding


### الأكواد


---

# اكمل

سأكمل الشرح بالتفصيل الدقيق:

***

## 🎬 الفيديو الثامن: File Processing (تكملة)

### الأكواد التفصيلية:

#### 1. تحديث requirements.txt:

```txt
fastapi==0.95.0
uvicorn[standard]==0.21.1
python-dotenv==1.0.0
python-multipart==0.0.6
aiofiles==23.1.0
PyPDF2==3.0.1          # لقراءة PDF
python-docx==0.8.11    # لقراءة Word documents
```


#### 2. ملف helpers/text_processor.py:

```python
from PyPDF2 import PdfReader
import docx
from typing import List, Dict
import re

class TextProcessor:
    """
    معالج النصوص - يستخرج النصوص ويقسمها
    """
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt', '.docx', '.doc'}
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        استخراج النص من ملف PDF
        
        Args:
            file_path: مسار ملف PDF
            
        Returns:
            النص المستخرج
        """
        try:
            reader = PdfReader(file_path)
            text = ""
            
            # قراءة كل صفحة
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num} ---\n"
                    text += page_text
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        استخراج النص من ملف Word
        
        Args:
            file_path: مسار ملف Word
            
        Returns:
            النص المستخرج
        """
        try:
            doc = docx.Document(file_path)
            
            # استخراج النص من الفقرات
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            
            # استخراج النص من الجداول
            table_texts = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text for cell in row.cells]
                    table_texts.append(" | ".join(row_text))
            
            # دمج كل النصوص
            all_text = "\n".join(paragraphs)
            if table_texts:
                all_text += "\n\n--- Tables ---\n"
                all_text += "\n".join(table_texts)
            
            return all_text.strip()
        except Exception as e:
            raise Exception(f"Error extracting DOCX: {str(e)}")
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """
        قراءة ملف نصي
        
        Args:
            file_path: مسار الملف النصي
            
        Returns:
            محتوى الملف
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            # محاولة قراءة بترميز مختلف
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read().strip()
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
    
    def extract_text(self, file_path: str, file_extension: str) -> str:
        """
        استخراج النص حسب نوع الملف
        
        Args:
            file_path: مسار الملف
            file_extension: امتداد الملف
            
        Returns:
            النص المستخرج
        """
        ext = file_extension.lower()
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        elif ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def clean_text(self, text: str) -> str:
        """
        تنظيف النص من الأحرف غير المرغوبة
        
        Args:
            text: النص الأصلي
            
        Returns:
            النص المنظف
        """
        # إزالة الأسطر الفارغة الزائدة
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # إزالة المسافات الزائدة
        text = re.sub(r' +', ' ', text)
        
        # إزالة الأحرف الخاصة غير المرغوبة
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 200
    ) -> List[Dict[str, any]]:
        """
        تقسيم النص إلى chunks
        
        Args:
            text: النص المراد تقسيمه
            chunk_size: حجم كل chunk بالأحرف
            overlap: التداخل بين chunks
            
        Returns:
            قائمة من chunks مع metadata
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # حساب نهاية الـ chunk
            end = start + chunk_size
            
            # محاولة إنهاء الـ chunk عند نهاية جملة
            if end < len(text):
                # البحث عن نقطة أو سطر جديد
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                
                # اختيار الأقرب
                best_end = max(last_period, last_newline)
                if best_end > start:
                    end = best_end + 1
            
            # استخراج الـ chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end,
                    'length': len(chunk_text)
                })
                chunk_id += 1
            
            # الانتقال للـ chunk التالي مع overlap
            start = end - overlap
            
            # التأكد من عدم التكرار اللانهائي
            if start >= len(text):
                break
        
        return chunks
    
    def chunk_by_sentences(
        self, 
        text: str, 
        sentences_per_chunk: int = 5
    ) -> List[Dict[str, any]]:
        """
        تقسيم النص حسب عدد الجمل
        
        Args:
            text: النص المراد تقسيمه
            sentences_per_chunk: عدد الجمل في كل chunk
            
        Returns:
            قائمة من chunks
        """
        # تقسيم النص إلى جمل
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = ' '.join(chunk_sentences).strip()
            
            if chunk_text:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'sentence_count': len(chunk_sentences),
                    'length': len(chunk_text)
                })
                chunk_id += 1
        
        return chunks
```


#### 3. تحديث routes/documents.py لإضافة Processing:

```python
from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from helpers.file_handler import FileHandler
from helpers.text_processor import TextProcessor
from models.document import DocumentResponse
from typing import List
import uuid
from datetime import datetime
import os

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

# إنشاء instances
file_handler = FileHandler(upload_dir="uploads")
text_processor = TextProcessor()

# قاعدة بيانات مؤقتة
documents_db = {}

async def process_document_background(doc_id: str):
    """
    معالجة المستند في الخلفية
    
    Args:
        doc_id: معرف المستند
    """
    try:
        document = documents_db[doc_id]
        
        # تحديث الحالة
        document['status'] = 'processing'
        
        # استخراج النص
        file_path = document['file_path']
        file_ext = os.path.splitext(document['filename'])[1]
        
        raw_text = text_processor.extract_text(file_path, file_ext)
        
        # تنظيف النص
        clean_text = text_processor.clean_text(raw_text)
        
        # تقسيم النص
        chunks = text_processor.chunk_text(
            clean_text,
            chunk_size=1000,
            overlap=200
        )
        
        # حفظ النتائج
        document['raw_text'] = raw_text
        document['clean_text'] = clean_text
        document['chunks'] = chunks
        document['chunk_count'] = len(chunks)
        document['status'] = 'processed'
        document['processed_at'] = datetime.now()
        
    except Exception as e:
        document['status'] = 'failed'
        document['error'] = str(e)

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    رفع ومعالجة مستند
    
    Args:
        background_tasks: لتشغيل المعالجة في الخلفية
        file: الملف المراد رفعه
    
    Returns:
        معلومات المستند
    """
    try:
        # حفظ الملف
        file_path, file_size = await file_handler.save_file(file)
        
        # إنشاء المستند
        doc_id = str(uuid.uuid4())
        
        document = {
            "id": doc_id,
            "filename": file.filename,
            "file_size": file_size,
            "content_type": file.content_type,
            "file_path": file_path,
            "upload_date": datetime.now(),
            "status": "uploaded"
        }
        
        documents_db[doc_id] = document
        
        # إضافة المعالجة للخلفية
        background_tasks.add_task(process_document_background, doc_id)
        
        return DocumentResponse(**document)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

@router.post("/{document_id}/process")
async def process_document(document_id: str):
    """
    معالجة مستند يدويًا
    
    Args:
        document_id: معرف المستند
    
    Returns:
        نتيجة المعالجة
    """
    if document_id not in documents_db:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    document = documents_db[document_id]
    
    if document['status'] == 'processing':
        return {"message": "Document is already being processed"}
    
    try:
        # معالجة مباشرة
        await process_document_background(document_id)
        
        return {
            "message": "Document processed successfully",
            "chunks": len(document.get('chunks', []))
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@router.get("/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """
    الحصول على chunks المستند
    
    Args:
        document_id: معرف المستند
    
    Returns:
        قائمة chunks
    """
    if document_id not in documents_db:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    document = documents_db[document_id]
    
    if document['status'] != 'processed':
        raise HTTPException(
            status_code=400,
            detail=f"Document not processed yet. Status: {document['status']}"
        )
    
    return {
        "document_id": document_id,
        "filename": document['filename'],
        "chunk_count": document['chunk_count'],
        "chunks": document['chunks']
    }

@router.get("/{document_id}/text")
async def get_document_text(document_id: str, clean: bool = True):
    """
    الحصول على نص المستند
    
    Args:
        document_id: معرف المستند
        clean: إرجاع النص المنظف أو الخام
    
    Returns:
        نص المستند
    """
    if document_id not in documents_db:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    document = documents_db[document_id]
    
    if document['status'] != 'processed':
        raise HTTPException(
            status_code=400,
            detail="Document not processed yet"
        )
    
    text_key = 'clean_text' if clean else 'raw_text'
    
    return {
        "document_id": document_id,
        "filename": document['filename'],
        "text": document[text_key],
        "length": len(document[text_key])
    }
```


#### 4. اختبار File Processing:

```bash
# رفع ملف (سيتم

---

# اكمل

سأكمل الشرح التفصيلي الدقيق:

***

## 🎬 الفيديو الثامن: File Processing (تكملة)

#### 4. اختبار File Processing:
```bash
# رفع ملف (سيتم معالجته تلقائيًا في الخلفية)
$ curl -X POST "http://localhost:5000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Response:
{
  "id": "abc-123",
  "filename": "document.pdf",
  "status": "uploaded",
  ...
}

# الحصول على chunks
$ curl http://localhost:5000/documents/abc-123/chunks

# الحصول على النص
$ curl "http://localhost:5000/documents/abc-123/text?clean=true"
```


#### 5. مثال على Chunking Strategy:

```python
# مثال على النص الأصلي
text = """
Machine learning is a subset of artificial intelligence. 
It focuses on developing systems that can learn from data. 
These systems improve their performance over time without being explicitly programmed.
"""

# Chunking بحجم 100 حرف مع overlap 20
processor = TextProcessor()
chunks = processor.chunk_text(text, chunk_size=100, overlap=20)

# النتيجة:
# Chunk 0: "Machine learning is a subset of artificial intelligence. It focuses on developing systems"
# Chunk 1: "developing systems that can learn from data. These systems improve their performance"
# Chunk 2: "their performance over time without being explicitly programmed."
```


***

## 🎬 الفيديو التاسع: Docker - MongoDB - Motor (38 دقيقة)

**Branch:** `tut-006`

### الأهداف التعليمية:

- إعداد MongoDB باستخدام Docker
- فهم Docker Compose
- استخدام Motor (MongoDB async driver)
- إدارة الاتصال بقاعدة البيانات


### المفاهيم الأساسية:

#### 1. لماذا MongoDB؟

- **NoSQL Database** - مرونة في البيانات
- **Document-based** - تخزين JSON-like documents
- **Scalable** - قابل للتوسع
- **Perfect for RAG** - مناسب لتخزين المستندات والـ metadata


#### 2. لماذا Docker؟

- **Consistency** - نفس البيئة للجميع
- **Easy Setup** - تثبيت سريع
- **Isolation** - عزل الخدمات
- **Production Ready** - جاهز للإنتاج


### الأكواد التفصيلية:

#### 1. ملف docker/docker-compose.yml:

```yaml
version: '3.8'

services:
  # MongoDB Service
  mongodb:
    image: mongo:6.0
    container_name: minirag-mongodb
    restart: unless-stopped
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: ${MONGO_ROOT_USER}
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_ROOT_PASSWORD}
      MONGO_INITDB_DATABASE: ${MONGO_DATABASE}
    volumes:
      - mongodb_data:/data/db
      - mongodb_config:/data/configdb
    networks:
      - minirag-network
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongosh localhost:27017/test --quiet
      interval: 10s
      timeout: 5s
      retries: 5

  # Mongo Express (Web UI للتطوير)
  mongo-express:
    image: mongo-express:1.0.0
    container_name: minirag-mongo-express
    restart: unless-stopped
    ports:
      - "8081:8081"
    environment:
      ME_CONFIG_MONGODB_ADMINUSERNAME: ${MONGO_ROOT_USER}
      ME_CONFIG_MONGODB_ADMINPASSWORD: ${MONGO_ROOT_PASSWORD}
      ME_CONFIG_MONGODB_URL: mongodb://${MONGO_ROOT_USER}:${MONGO_ROOT_PASSWORD}@mongodb:27017/
      ME_CONFIG_BASICAUTH_USERNAME: ${MONGO_EXPRESS_USER}
      ME_CONFIG_BASICAUTH_PASSWORD: ${MONGO_EXPRESS_PASSWORD}
    depends_on:
      - mongodb
    networks:
      - minirag-network

volumes:
  mongodb_data:
    driver: local
  mongodb_config:
    driver: local

networks:
  minirag-network:
    driver: bridge
```


#### 2. ملف docker/.env:

```bash
# MongoDB Configuration
MONGO_ROOT_USER=admin
MONGO_ROOT_PASSWORD=secure_password_123
MONGO_DATABASE=minirag

# Mongo Express (Web UI)
MONGO_EXPRESS_USER=admin
MONGO_EXPRESS_PASSWORD=express_pass_123

# Application Database User (للاستخدام في التطبيق)
MONGO_APP_USER=minirag_user
MONGO_APP_PASSWORD=minirag_pass_123
```


#### 3. تشغيل Docker Compose:

```bash
# الانتقال لمجلد docker
$ cd docker

# نسخ ملف البيئة
$ cp .env.example .env

# تعديل المتغيرات في .env
$ nano .env

# بدء الخدمات
$ docker compose up -d

# التحقق من الحالة
$ docker compose ps

# عرض Logs
$ docker compose logs -f mongodb

# إيقاف الخدمات
$ docker compose down

# إيقاف وحذف البيانات
$ docker compose down -v
```


#### 4. الوصول لـ MongoDB:

```bash
# الاتصال عبر MongoDB Shell
$ docker exec -it minirag-mongodb mongosh -u admin -p secure_password_123

# الوصول لـ Mongo Express
# افتح المتصفح: http://localhost:8081
# Username: admin
# Password: express_pass_123
```


#### 5. تحديث requirements.txt:

```txt
fastapi==0.95.0
uvicorn[standard]==0.21.1
python-dotenv==1.0.0
python-multipart==0.0.6
aiofiles==23.1.0
PyPDF2==3.0.1
python-docx==0.8.11
motor==3.1.1           # MongoDB async driver
pymongo==4.3.3         # MongoDB sync driver (dependency)
```


#### 6. ملف database/connection.py:

```python
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from config import settings
import logging

logger = logging.getLogger(__name__)

class Database:
    """
    مدير اتصال MongoDB
    """
    client: AsyncIOMotorClient = None
    
    @classmethod
    async def connect_to_database(cls):
        """
        إنشاء اتصال بقاعدة البيانات
        """
        try:
            logger.info("Connecting to MongoDB...")
            
            # إنشاء الاتصال
            cls.client = AsyncIOMotorClient(
                settings.mongodb_url,
                maxPoolSize=10,
                minPoolSize=1,
                serverSelectionTimeoutMS=5000
            )
            
            # التحقق من الاتصال
            await cls.client.admin.command('ping')
            
            logger.info("Successfully connected to MongoDB")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    @classmethod
    async def close_database_connection(cls):
        """
        إغلاق الاتصال بقاعدة البيانات
        """
        if cls.client:
            logger.info("Closing MongoDB connection...")
            cls.client.close()
            logger.info("MongoDB connection closed")
    
    @classmethod
    def get_database(cls):
        """
        الحصول على database instance
        
        Returns:
            AsyncIOMotorDatabase
        """
        if not cls.client:
            raise Exception("Database not connected")
        
        return cls.client[settings.database_name]
    
    @classmethod
    def get_collection(cls, collection_name: str):
        """
        الحصول على collection
        
        Args:
            collection_name: اسم الـ collection
            
        Returns:
            AsyncIOMotorCollection
        """
        db = cls.get_database()
        return db[collection_name]

# Instance عام
db = Database()
```


#### 7. تحديث config.py:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application
    app_name: str = "mini-RAG"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # MongoDB
    mongodb_url: str = "mongodb://admin:secure_password_123@localhost:27017"
    database_name: str = "minirag"
    
    # Collections
    documents_collection: str = "documents"
    chunks_collection: str = "chunks"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 5000
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```


#### 8. تحديث main.py مع Database Events:

```python
from fastapi import FastAPI
from config import settings
from database.connection import db
from routes import documents, queries
import logging

# إعداد Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إنشاء التطبيق
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)

# تضمين الـ routers
app.include_router(documents.router)
app.include_router(queries.router)

@app.on_event("startup")
async def startup_event():
    """
    تنفيذ عند بدء التطبيق
    """
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # الاتصال بقاعدة البيانات
    try:
        await db.connect_to_database()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """
    تنفيذ عند إيقاف التطبيق
    """
    logger.info("Shutting down application...")
    
    # إغلاق اتصال قاعدة البيانات
    await db.close_database_connection()
    logger.info("Application shutdown complete")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "status": "running"
    }

@app.get("/health")
async def health():
    """
    فحص صحة التطبيق
    """
    try:
        # فحص اتصال قاعدة البيانات
        await db.client.admin.command('ping')
        db_status = "connected"
    except:
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "database": db_status,
        "service": settings.app_name
    }
```


#### 9. اختبار الاتصال:

```bash
# بدء الـ Docker services
$ cd docker
$ docker compose up -d

# العودة للمجلد الرئيسي
$ cd ..

# تشغيل التطبيق
$ uvicorn main:app --reload

# اختبار health endpoint
$ curl http://localhost:5000/health

# Response:
{
  "status": "healthy",
  "database": "connected",
  "service": "mini-RAG"
}
```


***

## 🎬 الفيديو العاشر: Mongo Schemes and Models (53 دقيقة)

**Branch:** `tut-007`

### الأهداف التعليمية:

- تصميم MongoDB schemas
- استخدام Pydantic models للتحقق
- إنشاء CRUD operations
- فهم Document-based structure


### المفاهيم الأساسية:

#### 1. MongoDB Collections في مشروعنا:

- **documents**: تخزين معلومات المستندات
- **chunks**: تخزين قطع النصوص
- **queries**: تخزين الاستعلامات (اختياري)


### الأكواد التفصيلية:

#### 1. ملف models/document.py (محدث):

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    """
    Custom ObjectId type لـ Pydantic
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class DocumentBase(BaseModel):
    """
    البيانات الأساسية للمستند
    """
    filename: str = Field(..., description="اسم الملف")
    file_size: int = Field(..., description="حجم الملف بالبايت")
    content_type: str = Field(..., description="نوع الملف")
    file_path: str = Field(..., description="مسار الملف على السيرفر")

class DocumentCreate(DocumentBase):
    """
    بيانات إنشاء مستند جديد
    """
    pass

class DocumentInDB(DocumentBase):
    """
    المستند كما هو محفوظ في قاعدة البيانات
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    upload_date: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="uploaded", description="حالة المعالجة")
    raw_text: Optional[str] = None
    clean_text: Optional[str] = None
    chunk_count: Optional

---

# أكمل

سأكمل الشرح التفصيلي الدقيق:

***

## 🎬 الفيديو العاشر: Mongo Schemes and Models (تكملة)

#### 1. ملف models/document.py (تكملة):
```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    """
    Custom ObjectId type لـ Pydantic
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")

class DocumentBase(BaseModel):
    """
    البيانات الأساسية للمستند
    """
    filename: str = Field(..., description="اسم الملف")
    file_size: int = Field(..., description="حجم الملف بالبايت")
    content_type: str = Field(..., description="نوع الملف")
    file_path: str = Field(..., description="مسار الملف")

class DocumentCreate(DocumentBase):
    """
    بيانات إنشاء مستند
    """
    pass

class DocumentInDB(DocumentBase):
    """
    المستند في قاعدة البيانات
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    upload_date: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="uploaded")
    raw_text: Optional[str] = None
    clean_text: Optional[str] = None
    chunk_count: Optional[int] = 0
    processed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class DocumentResponse(BaseModel):
    """
    استجابة API للمستند
    """
    id: str = Field(..., description="معرف المستند")
    filename: str
    file_size: int
    content_type: str
    upload_date: datetime
    status: str
    chunk_count: Optional[int] = 0
    
    model_config = ConfigDict(
        populate_by_name=True
    )

class DocumentUpdate(BaseModel):
    """
    تحديث بيانات المستند
    """
    status: Optional[str] = None
    raw_text: Optional[str] = None
    clean_text: Optional[str] = None
    chunk_count: Optional[int] = None
    processed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```


#### 2. ملف models/chunk.py:

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
from models.document import PyObjectId

class ChunkBase(BaseModel):
    """
    البيانات الأساسية للـ chunk
    """
    document_id: str = Field(..., description="معرف المستند الأصلي")
    chunk_id: int = Field(..., description="رقم الـ chunk")
    text: str = Field(..., description="نص الـ chunk")
    start_char: int = Field(..., description="موقع البداية في النص")
    end_char: int = Field(..., description="موقع النهاية في النص")
    length: int = Field(..., description="طول النص")

class ChunkCreate(ChunkBase):
    """
    إنشاء chunk جديد
    """
    pass

class ChunkInDB(ChunkBase):
    """
    الـ chunk في قاعدة البيانات
    """
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class ChunkResponse(BaseModel):
    """
    استجابة API للـ chunk
    """
    id: str
    document_id: str
    chunk_id: int
    text: str
    length: int
    has_embedding: bool = False
    
    model_config = ConfigDict(
        populate_by_name=True
    )
```


#### 3. ملف database/repositories/document_repository.py:

```python
from typing import List, Optional, Dict, Any
from bson import ObjectId
from database.connection import db
from models.document import DocumentInDB, DocumentCreate, DocumentUpdate
from config import settings

class DocumentRepository:
    """
    مستودع العمليات على المستندات
    """
    
    def __init__(self):
        self.collection_name = settings.documents_collection
    
    @property
    def collection(self):
        """
        الحصول على الـ collection
        """
        return db.get_collection(self.collection_name)
    
    async def create(self, document: DocumentCreate) -> str:
        """
        إنشاء مستند جديد
        
        Args:
            document: بيانات المستند
            
        Returns:
            معرف المستند المُنشأ
        """
        # تحويل Pydantic model إلى dict
        doc_dict = document.model_dump(by_alias=True)
        
        # إضافة تاريخ الرفع
        from datetime import datetime
        doc_dict['upload_date'] = datetime.now()
        doc_dict['status'] = 'uploaded'
        doc_dict['metadata'] = {}
        
        # الإدراج في قاعدة البيانات
        result = await self.collection.insert_one(doc_dict)
        
        return str(result.inserted_id)
    
    async def get_by_id(self, document_id: str) -> Optional[DocumentInDB]:
        """
        الحصول على مستند بواسطة ID
        
        Args:
            document_id: معرف المستند
            
        Returns:
            المستند أو None
        """
        if not ObjectId.is_valid(document_id):
            return None
        
        doc = await self.collection.find_one({"_id": ObjectId(document_id)})
        
        if doc:
            return DocumentInDB(**doc)
        
        return None
    
    async def get_all(
        self, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[DocumentInDB]:
        """
        الحصول على جميع المستندات
        
        Args:
            skip: عدد المستندات للتخطي
            limit: الحد الأقصى للنتائج
            status: تصفية حسب الحالة
            
        Returns:
            قائمة المستندات
        """
        # بناء query
        query = {}
        if status:
            query['status'] = status
        
        # البحث
        cursor = self.collection.find(query).skip(skip).limit(limit)
        
        # تحويل إلى list
        documents = []
        async for doc in cursor:
            documents.append(DocumentInDB(**doc))
        
        return documents
    
    async def update(
        self, 
        document_id: str, 
        update_data: DocumentUpdate
    ) -> bool:
        """
        تحديث مستند
        
        Args:
            document_id: معرف المستند
            update_data: البيانات المراد تحديثها
            
        Returns:
            True إذا تم التحديث
        """
        if not ObjectId.is_valid(document_id):
            return False
        
        # إزالة القيم None
        update_dict = update_data.model_dump(exclude_unset=True)
        
        if not update_dict:
            return False
        
        # التحديث
        result = await self.collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": update_dict}
        )
        
        return result.modified_count > 0
    
    async def delete(self, document_id: str) -> bool:
        """
        حذف مستند
        
        Args:
            document_id: معرف المستند
            
        Returns:
            True إذا تم الحذف
        """
        if not ObjectId.is_valid(document_id):
            return False
        
        result = await self.collection.delete_one(
            {"_id": ObjectId(document_id)}
        )
        
        return result.deleted_count > 0
    
    async def count(self, status: Optional[str] = None) -> int:
        """
        عد المستندات
        
        Args:
            status: تصفية حسب الحالة
            
        Returns:
            عدد المستندات
        """
        query = {}
        if status:
            query['status'] = status
        
        return await self.collection.count_documents(query)
    
    async def update_status(
        self, 
        document_id: str, 
        status: str,
        error: Optional[str] = None
    ) -> bool:
        """
        تحديث حالة المستند
        
        Args:
            document_id: معرف المستند
            status: الحالة الجديدة
            error: رسالة الخطأ (اختياري)
            
        Returns:
            True إذا تم التحديث
        """
        if not ObjectId.is_valid(document_id):
            return False
        
        update_dict = {"status": status}
        
        if status == "processed":
            from datetime import datetime
            update_dict["processed_at"] = datetime.now()
        
        if error:
            update_dict["error"] = error
        
        result = await self.collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": update_dict}
        )
        
        return result.modified_count > 0
```


#### 4. ملف database/repositories/chunk_repository.py:

```python
from typing import List, Optional
from bson import ObjectId
from database.connection import db
from models.chunk import ChunkInDB, ChunkCreate
from config import settings

class ChunkRepository:
    """
    مستودع العمليات على الـ chunks
    """
    
    def __init__(self):
        self.collection_name = settings.chunks_collection
    
    @property
    def collection(self):
        return db.get_collection(self.collection_name)
    
    async def create_many(self, chunks: List[ChunkCreate]) -> List[str]:
        """
        إنشاء عدة chunks دفعة واحدة
        
        Args:
            chunks: قائمة الـ chunks
            
        Returns:
            قائمة معرفات الـ chunks
        """
        # تحويل إلى dicts
        chunks_dict = [chunk.model_dump(by_alias=True) for chunk in chunks]
        
        # إضافة timestamp
        from datetime import datetime
        for chunk in chunks_dict:
            chunk['created_at'] = datetime.now()
            chunk['metadata'] = {}
        
        # الإدراج
        result = await self.collection.insert_many(chunks_dict)
        
        return [str(id) for id in result.inserted_ids]
    
    async def get_by_document_id(
        self, 
        document_id: str
    ) -> List[ChunkInDB]:
        """
        الحصول على chunks مستند معين
        
        Args:
            document_id: معرف المستند
            
        Returns:
            قائمة الـ chunks
        """
        cursor = self.collection.find(
            {"document_id": document_id}
        ).sort("chunk_id", 1)
        
        chunks = []
        async for chunk in cursor:
            chunks.append(ChunkInDB(**chunk))
        
        return chunks
    
    async def get_by_id(self, chunk_id: str) -> Optional[ChunkInDB]:
        """
        الحصول على chunk بواسطة ID
        """
        if not ObjectId.is_valid(chunk_id):
            return None
        
        chunk = await self.collection.find_one({"_id": ObjectId(chunk_id)})
        
        if chunk:
            return ChunkInDB(**chunk)
        
        return None
    
    async def update_embedding(
        self, 
        chunk_id: str, 
        embedding: List[float]
    ) -> bool:
        """
        تحديث embedding للـ chunk
        
        Args:
            chunk_id: معرف الـ chunk
            embedding: vector الـ embedding
            
        Returns:
            True إذا تم التحديث
        """
        if not ObjectId.is_valid(chunk_id):
            return False
        
        result = await self.collection.update_one(
            {"_id": ObjectId(chunk_id)},
            {"$set": {"embedding": embedding}}
        )
        
        return result.modified_count > 0
    
    async def delete_by_document_id(self, document_id: str) -> int:
        """
        حذف جميع chunks مستند معين
        
        Args:
            document_id: معرف المستند
            
        Returns:
            عدد الـ chunks المحذوفة
        """
        result = await self.collection.delete_many(
            {"document_id": document_id}
        )
        
        return result.deleted_count
    
    async def count_by_document(self, document_id: str) -> int:
        """
        عد chunks مستند معين
        """
        return await self.collection.count_documents(
            {"document_id": document_id}
        )
```


#### 5. تحديث routes/documents.py مع MongoDB:

```python
from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from typing import List, Optional
from helpers.file_handler import FileHandler
from helpers.text_processor import TextProcessor
from models.document import DocumentCreate, DocumentResponse, DocumentInDB
from models.chunk import ChunkCreate
from database.repositories.document_repository import DocumentRepository
from database.repositories.chunk_repository import ChunkRepository
import os

router = APIRouter(

---

# أكمل

سأكمل الشرح التفصيلي الدقيق:

***

## 🎬 الفيديو العاشر: Mongo Schemes and Models (تكملة)

#### 5. تحديث routes/documents.py مع MongoDB:
```python
from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks, Query
from typing import List, Optional
from helpers.file_handler import FileHandler
from helpers.text_processor import TextProcessor
from models.document import DocumentCreate, DocumentResponse, DocumentInDB, DocumentUpdate
from models.chunk import ChunkCreate
from database.repositories.document_repository import DocumentRepository
from database.repositories.chunk_repository import ChunkRepository
import os

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

# إنشاء instances
file_handler = FileHandler(upload_dir="uploads")
text_processor = TextProcessor()
doc_repo = DocumentRepository()
chunk_repo = ChunkRepository()

async def process_document_background(document_id: str):
    """
    معالجة المستند في الخلفية
    """
    try:
        # الحصول على المستند
        document = await doc_repo.get_by_id(document_id)
        if not document:
            return
        
        # تحديث الحالة إلى processing
        await doc_repo.update_status(document_id, "processing")
        
        # استخراج النص
        file_ext = os.path.splitext(document.filename)[1]
        raw_text = text_processor.extract_text(document.file_path, file_ext)
        
        # تنظيف النص
        clean_text = text_processor.clean_text(raw_text)
        
        # تقسيم النص
        chunks_data = text_processor.chunk_text(
            clean_text,
            chunk_size=1000,
            overlap=200
        )
        
        # حفظ الـ chunks في قاعدة البيانات
        chunks_to_create = []
        for chunk_data in chunks_data:
            chunk = ChunkCreate(
                document_id=document_id,
                chunk_id=chunk_data['chunk_id'],
                text=chunk_data['text'],
                start_char=chunk_data['start_char'],
                end_char=chunk_data['end_char'],
                length=chunk_data['length']
            )
            chunks_to_create.append(chunk)
        
        # إدراج الـ chunks
        if chunks_to_create:
            await chunk_repo.create_many(chunks_to_create)
        
        # تحديث المستند
        update_data = DocumentUpdate(
            status="processed",
            raw_text=raw_text,
            clean_text=clean_text,
            chunk_count=len(chunks_data)
        )
        await doc_repo.update(document_id, update_data)
        
    except Exception as e:
        # في حالة الخطأ
        await doc_repo.update_status(document_id, "failed", error=str(e))

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    رفع مستند جديد
    
    Args:
        background_tasks: لتشغيل المعالجة في الخلفية
        file: الملف المراد رفعه
    
    Returns:
        معلومات المستند المرفوع
    """
    try:
        # حفظ الملف
        file_path, file_size = await file_handler.save_file(file)
        
        # إنشاء document model
        document_create = DocumentCreate(
            filename=file.filename,
            file_size=file_size,
            content_type=file.content_type or "application/octet-stream",
            file_path=file_path
        )
        
        # حفظ في قاعدة البيانات
        document_id = await doc_repo.create(document_create)
        
        # إضافة المعالجة للخلفية
        background_tasks.add_task(process_document_background, document_id)
        
        # الحصول على المستند لإرجاعه
        document = await doc_repo.get_by_id(document_id)
        
        return DocumentResponse(
            id=str(document.id),
            filename=document.filename,
            file_size=document.file_size,
            content_type=document.content_type,
            upload_date=document.upload_date,
            status=document.status,
            chunk_count=document.chunk_count or 0
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[str] = None
):
    """
    الحصول على قائمة المستندات
    
    Args:
        skip: عدد السجلات للتخطي
        limit: الحد الأقصى للنتائج
        status: تصفية حسب الحالة
    
    Returns:
        قائمة المستندات
    """
    documents = await doc_repo.get_all(skip=skip, limit=limit, status=status)
    
    return [
        DocumentResponse(
            id=str(doc.id),
            filename=doc.filename,
            file_size=doc.file_size,
            content_type=doc.content_type,
            upload_date=doc.upload_date,
            status=doc.status,
            chunk_count=doc.chunk_count or 0
        )
        for doc in documents
    ]

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """
    الحصول على مستند محدد
    """
    document = await doc_repo.get_by_id(document_id)
    
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    return DocumentResponse(
        id=str(document.id),
        filename=document.filename,
        file_size=document.file_size,
        content_type=document.content_type,
        upload_date=document.upload_date,
        status=document.status,
        chunk_count=document.chunk_count or 0
    )

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    حذف مستند
    """
    # الحصول على المستند
    document = await doc_repo.get_by_id(document_id)
    
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    # حذف الملف من النظام
    file_handler.delete_file(document.file_path)
    
    # حذف الـ chunks
    await chunk_repo.delete_by_document_id(document_id)
    
    # حذف المستند
    deleted = await doc_repo.delete(document_id)
    
    if not deleted:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document"
        )
    
    return {"message": "Document deleted successfully"}

@router.get("/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """
    الحصول على chunks المستند
    """
    # التحقق من وجود المستند
    document = await doc_repo.get_by_id(document_id)
    
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    if document.status != "processed":
        raise HTTPException(
            status_code=400,
            detail=f"Document not processed yet. Status: {document.status}"
        )
    
    # الحصول على الـ chunks
    chunks = await chunk_repo.get_by_document_id(document_id)
    
    return {
        "document_id": document_id,
        "filename": document.filename,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "id": str(chunk.id),
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "length": chunk.length,
                "has_embedding": chunk.embedding is not None
            }
            for chunk in chunks
        ]
    }

@router.get("/{document_id}/text")
async def get_document_text(document_id: str, clean: bool = True):
    """
    الحصول على نص المستند
    """
    document = await doc_repo.get_by_id(document_id)
    
    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    if document.status != "processed":
        raise HTTPException(
            status_code=400,
            detail="Document not processed yet"
        )
    
    text = document.clean_text if clean else document.raw_text
    
    if not text:
        raise HTTPException(
            status_code=404,
            detail="Text not found"
        )
    
    return {
        "document_id": document_id,
        "filename": document.filename,
        "text": text,
        "length": len(text),
        "type": "clean" if clean else "raw"
    }

@router.get("/stats/summary")
async def get_statistics():
    """
    الحصول على إحصائيات المستندات
    """
    total = await doc_repo.count()
    uploaded = await doc_repo.count(status="uploaded")
    processing = await doc_repo.count(status="processing")
    processed = await doc_repo.count(status="processed")
    failed = await doc_repo.count(status="failed")
    
    return {
        "total_documents": total,
        "uploaded": uploaded,
        "processing": processing,
        "processed": processed,
        "failed": failed
    }
```


#### 6. إنشاء Indexes للأداء:

```python
# ملف database/indexes.py
from database.connection import db
from config import settings

async def create_indexes():
    """
    إنشاء indexes لتحسين الأداء
    """
    # Documents collection indexes
    documents_collection = db.get_collection(settings.documents_collection)
    
    # Index على filename
    await documents_collection.create_index("filename")
    
    # Index على status
    await documents_collection.create_index("status")
    
    # Index على upload_date (تنازلي)
    await documents_collection.create_index([("upload_date", -1)])
    
    # Compound index
    await documents_collection.create_index([
        ("status", 1),
        ("upload_date", -1)
    ])
    
    # Chunks collection indexes
    chunks_collection = db.get_collection(settings.chunks_collection)
    
    # Index على document_id
    await chunks_collection.create_index("document_id")
    
    # Compound index على document_id و chunk_id
    await chunks_collection.create_index([
        ("document_id", 1),
        ("chunk_id", 1)
    ], unique=True)
    
    print("✅ Database indexes created successfully")
```


#### 7. تحديث main.py لإنشاء Indexes:

```python
from fastapi import FastAPI
from config import settings
from database.connection import db
from database.indexes import create_indexes
from routes import documents, queries
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version
)

app.include_router(documents.router)
app.include_router(queries.router)

@app.on_event("startup")
async def startup_event():
    """
    تنفيذ عند بدء التطبيق
    """
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    
    try:
        # الاتصال بقاعدة البيانات
        await db.connect_to_database()
        logger.info("✅ Database connected")
        
        # إنشاء indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """
    تنفيذ عند إيقاف التطبيق
    """
    logger.info("🛑 Shutting down application...")
    await db.close_database_connection()
    logger.info("✅ Application shutdown complete")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs"
    }
```


#### 8. اختبار كامل:

```bash
# 1. تشغيل Docker
$ cd docker && docker compose up -d

# 2. تشغيل التطبيق
$ cd .. && uvicorn main:app --reload

# 3. رفع مستند
$ curl -X POST "http://localhost:5000/documents/upload" \
  -F "file=@test.pdf"

# 4. قائمة المستندات
$ curl "http://localhost:5000/documents/"

# 5. الحصول على chunks
$ curl "http://localhost:5000/documents/{doc_id}/chunks"

# 6. إحصائيات
$ curl "http://localhost:5000/documents/stats/summary"
```


***

## 📊 ملخص شامل لأول 10 فيديوهات

### **الفيديوهات النظرية (1-3):**

| الفيديو | المدة | المحتوى الرئيسي |
| :-- | :-- | :-- |
| **1. About the Course** | 5:52 | المشكلة: الفجوة بين Data Science و Software Engineering<br>الحل: بناء RAG application كامل<br>الفلسفة: أنت مهندس برمجيات أولاً |
| **2. What will we build** | 3:37 | الم |


---

# اكمل

سأكمل الملخص الشامل:

***

## 📊 ملخص شامل لأول 10 فيديوهات

### **الفيديوهات النظرية (1-3):**

| الفيديو | المدة | المحتوى الرئيسي |
| :-- | :-- | :-- |
| **1. About the Course** | 5:52 | المشكلة: الفجوة بين Data Science و Software Engineering<br>الحل: بناء RAG application كامل<br>الفلسفة: أنت مهندس برمجيات أولاً |
| **2. What will we build** | 3:37 | المعمارية الكاملة لـ RAG<br>المراحل: Upload → Process → Index → Search → Answer<br>4 APIs رئيسية |
| **3. Setup Tools** | 18:00 | Python 3.10 + MiniConda<br>VS Code + Git<br>تحسين Terminal |

### **الفيديوهات التطبيقية (4-10):**

#### **📦 المرحلة الأولى: Project Foundation (4-6)**

**الفيديو 4: Project Architecture (26 دقيقة)**

- ✅ Branch: `tut-001`
- 🎯 البنية المعمارية الكاملة
- 📁 هيكل المشروع الأساسي
- ⚙️ ملفات الإعداد (.env, .gitignore, requirements.txt)

**الفيديو 5: Welcome to FastAPI (15 دقيقة)**

- ✅ Branch: `tut-002`
- 🎯 أول API endpoint
- 📚 Auto-documentation (Swagger UI)
- 🔧 Type hints \& Pydantic

```python
# الكود الأساسي
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def welcome():
    return {"message": "Hello World!"}
```

**الفيديو 6: Nested Routes + Env Values (15 دقيقة)**

- ✅ Branch: `tut-003`
- 🎯 تنظيم Routes باستخدام APIRouter
- 🔐 إدارة Environment Variables
- 📂 فصل الإعدادات عن الكود

```python
# config.py
class Settings(BaseSettings):
    app_name: str = "mini-RAG"
    openai_api_key: str
    mongodb_url: str
    
    class Config:
        env_file = ".env"

settings = Settings()
```


***

#### **📤 المرحلة الثانية: File Handling (7-8)**

**الفيديو 7: Uploading a File (1:23 ساعة)**

- ✅ Branch: `tut-004`
- 🎯 File upload في FastAPI
- ✔️ Validation (type, size)
- 💾 حفظ آمن للملفات
- ⚠️ Error handling

```python
# الوظائف الرئيسية
- validate_file_extension()
- validate_file_size()
- save_file()
- generate_unique_filename()
```

**الفيديو 8: File Processing (42 دقيقة)**

- ✅ Branch: `tut-005`
- 🎯 استخراج النصوص (PDF, DOCX, TXT)
- 🔪 Text chunking strategies
- 🧹 تنظيف النصوص
- 📊 Metadata extraction

```python
# Text Chunking
chunks = text_processor.chunk_text(
    text,
    chunk_size=1000,
    overlap=200
)
# النتيجة: قائمة من chunks مع metadata
```


***

#### **🗄️ المرحلة الثالثة: Database Integration (9-10)**

**الفيديو 9: Docker - MongoDB - Motor (38 دقيقة)**

- ✅ Branch: `tut-006`
- 🐳 Docker Compose setup
- 🍃 MongoDB configuration
- ⚡ Motor (async driver)
- 🔌 Connection management

```yaml
# docker-compose.yml
services:
  mongodb:
    image: mongo:6.0
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password
```

**الفيديو 10: Mongo Schemes and Models (53 دقيقة)**

- ✅ Branch: `tut-007`
- 📋 Pydantic models للتحقق
- 🗂️ Collections design (documents, chunks)
- 🔄 CRUD operations
- 🚀 Database indexes للأداء

```python
# Document Model
class DocumentInDB(BaseModel):
    id: PyObjectId
    filename: str
    status: str
    chunks: List[Dict]
    
# Repository Pattern
doc_repo = DocumentRepository()
doc_id = await doc_repo.create(document)
```


***

## 🎯 النتائج النهائية بعد 10 فيديوهات

### **ما تم إنجازه:**

✅ **Backend API كامل** باستخدام FastAPI
✅ **File Upload System** مع validation
✅ **Text Extraction** من PDF, DOCX, TXT
✅ **Text Chunking** بطرق متقدمة
✅ **MongoDB Integration** مع Docker
✅ **Repository Pattern** للتعامل مع البيانات
✅ **Background Processing** للمعالجة غير المتزامنة
✅ **Error Handling** شامل
✅ **Auto Documentation** (Swagger UI)

### **البنية النهائية للمشروع:**

```
mini-rag/
├── main.py                          # نقطة الدخول
├── config.py                        # الإعدادات
├── requirements.txt                 # المكتبات
├── .env                            # المتغيرات البيئية
├── database/
│   ├── connection.py               # إدارة الاتصال
│   ├── indexes.py                  # Database indexes
│   └── repositories/
│       ├── document_repository.py  # CRUD للمستندات
│       └── chunk_repository.py     # CRUD للـ chunks
├── models/
│   ├── document.py                 # Document models
│   └── chunk.py                    # Chunk models
├── routes/
│   ├── documents.py                # Document endpoints
│   └── queries.py                  # Query endpoints
├── helpers/
│   ├── file_handler.py            # معالجة الملفات
│   └── text_processor.py          # معالجة النصوص
├── docker/
│   ├── docker-compose.yml         # Docker services
│   └── .env                       # Docker environment
└── uploads/                        # مجلد الملفات المرفوعة
```


### **الـ APIs المتاحة:**

```bash
# Document Management
POST   /documents/upload           # رفع مستند
GET    /documents/                 # قائمة المستندات
GET    /documents/{id}             # مستند محدد
DELETE /documents/{id}             # حذف مستند
GET    /documents/{id}/chunks      # chunks المستند
GET    /documents/{id}/text        # نص المستند
GET    /documents/stats/summary    # إحصائيات

# System
GET    /                           # الصفحة الرئيسية
GET    /health                     # فحص الصحة
GET    /docs                       # Swagger UI
```


### **التقنيات المستخدمة:**

| التقنية | الاستخدام |
| :-- | :-- |
| **FastAPI** | Web Framework |
| **Uvicorn** | ASGI Server |
| **MongoDB** | Database |
| **Motor** | Async MongoDB Driver |
| **Pydantic** | Data Validation |
| **PyPDF2** | PDF Processing |
| **python-docx** | Word Processing |
| **Docker** | Containerization |
| **aiofiles** | Async File I/O |


***

## 🚀 الخطوات القادمة (الفيديوهات 11-25)

### **المرحلة القادمة: RAG Core Features**

- **11. Mongo Indexing** - تحسين أداء البحث
- **12. Data Pipeline** - تحسينات على المعالجة
- **13. Checkpoint-1** - مراجعة ما تم إنجازه
- **14. LLM Factory** - دمج Large Language Models
- **15. Vector DB** - QDrant للـ embeddings
- **16. Semantic Search** - البحث الدلالي
- **17. Augmented Answers** - توليد الإجابات
- **18-19. Local LLM** - Ollama للتشغيل المحلي
- **20-21. PostgreSQL + PGVector** - الانتقال لقاعدة بيانات أقوى
- **22-23. Deployment** - نشر التطبيق
- **24-25. Celery** - معالجة المهام الثقيلة

***

## 💡 نصائح للمتابعة

### **للتطبيق العملي:**

1. ✍️ اكتب الكود بنفسك - لا تنسخ فقط
2. 🔍 جرب كل endpoint في Swagger UI
3. 🐛 تعمد كسر الكود لفهم الأخطاء
4. 📝 أضف logging لتتبع التنفيذ
5. 🧪 اختبر بملفات مختلفة (PDF, Word, Text)

### **للفهم العميق:**

- 📚 اقرأ FastAPI documentation
- 🍃 تعلم MongoDB queries
- 🐳 افهم Docker networking
- 🔄 درس async/await في Python
- 🎯 افهم Repository Pattern

***

## 📝 الخلاصة

بعد أول 10 فيديوهات، أصبح لديك:

- ✅ **Foundation كامل** لتطبيق Production-ready
- ✅ **فهم عميق** لـ Web Development مع Python
- ✅ **خبرة عملية** في Database Integration
- ✅ **مهارات** في File Processing و Text Analysis
- ✅ **قاعدة قوية** لبناء RAG application متقدم

**الـ 15 فيديو القادمة** ستبني على هذا الأساس لإضافة:

- 🤖 الذكاء الاصطناعي (LLM Integration)
- 🔍 البحث الدلالي (Semantic Search)
- 📊 Vector Databases
- 🚀 Deployment و Production Features

**استمر في التعلم والتطبيق!** 💪

---

# شرح تفصيلي ودقيق للفيديوهات 11-15 من سلسلة mini-RAG


***

## 🎬 الفيديو الحادي عشر: Mongo Indexing (1:13 ساعة)

**Branch:** `tut-008`

### الأهداف التعليمية:

- فهم أهمية Indexes في MongoDB
- أنواع Indexes المختلفة
- تحسين أداء الاستعلامات
- Query optimization strategies


### المفاهيم الأساسية:

#### 1. لماذا نحتاج Indexes؟

- **السرعة**: تحسين سرعة البحث من O(n) إلى O(log n)
- **Efficiency**: تقليل استهلاك الموارد
- **Scalability**: أداء أفضل مع زيادة البيانات
- **Production Ready**: ضروري لأي تطبيق إنتاجي


#### 2. أنواع Indexes في MongoDB:

1. **Single Field Index** - على حقل واحد
2. **Compound Index** - على عدة حقول
3. **Text Index** - للبحث النصي
4. **Geospatial Index** - للبيانات الجغرافية
5. **Unique Index** - لضمان عدم التكرار

### الأكواد التفصيلية:

#### 1. ملف database/indexes.py (محدث وشامل):

```python
from database.connection import db
from config import settings
import logging

logger = logging.getLogger(__name__)

async def create_indexes():
    """
    إنشاء جميع الـ indexes للمشروع
    """
    logger.info("🔧 Creating database indexes...")
    
    try:
        # ========== Documents Collection ==========
        await create_documents_indexes()
        
        # ========== Chunks Collection ==========
        await create_chunks_indexes()
        
        logger.info("✅ All indexes created successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating indexes: {str(e)}")
        raise

async def create_documents_indexes():
    """
    إنشاء indexes لـ documents collection
    """
    collection = db.get_collection(settings.documents_collection)
    
    # 1. Index على filename - للبحث بالاسم
    await collection.create_index(
        "filename",
        name="idx_filename"
    )
    logger.info("✓ Created index: filename")
    
    # 2. Index على status - للتصفية حسب الحالة
    await collection.create_index(
        "status",
        name="idx_status"
    )
    logger.info("✓ Created index: status")
    
    # 3. Index على upload_date (تنازلي) - للترتيب الزمني
    await collection.create_index(
        [("upload_date", -1)],
        name="idx_upload_date_desc"
    )
    logger.info("✓ Created index: upload_date (desc)")
    
    # 4. Compound Index على status + upload_date
    # مفيد للاستعلامات: "جميع المستندات المعالجة مرتبة بالتاريخ"
    await collection.create_index(
        [
            ("status", 1),
            ("upload_date", -1)
        ],
        name="idx_status_upload_date"
    )
    logger.info("✓ Created compound index: status + upload_date")
    
    # 5. Index على content_type - للتصفية حسب نوع الملف
    await collection.create_index(
        "content_type",
        name="idx_content_type"
    )
    logger.info("✓ Created index: content_type")
    
    # 6. Text Index على filename و clean_text - للبحث النصي الكامل
    await collection.create_index(
        [
            ("filename", "text"),
            ("clean_text", "text")
        ],
        name="idx_text_search",
        weights={
            "filename": 10,  # وزن أعلى للاسم
            "clean_text": 1
        },
        default_language="english"
    )
    logger.info("✓ Created text index: filename + clean_text")
    
    # 7. Sparse Index على processed_at - فقط للمستندات المعالجة
    await collection.create_index(
        "processed_at",
        name="idx_processed_at",
        sparse=True  # فقط documents التي لها processed_at
    )
    logger.info("✓ Created sparse index: processed_at")
    
    # 8. Index على chunk_count - للإحصائيات
    await collection.create_index(
        "chunk_count",
        name="idx_chunk_count",
        sparse=True
    )
    logger.info("✓ Created index: chunk_count")

async def create_chunks_indexes():
    """
    إنشاء indexes لـ chunks collection
    """
    collection = db.get_collection(settings.chunks_collection)
    
    # 1. Index على document_id - للحصول على chunks مستند معين
    await collection.create_index(
        "document_id",
        name="idx_document_id"
    )
    logger.info("✓ Created index: document_id")
    
    # 2. Compound Unique Index على document_id + chunk_id
    # لضمان عدم تكرار نفس الـ chunk في نفس المستند
    await collection.create_index(
        [
            ("document_id", 1),
            ("chunk_id", 1)
        ],
        name="idx_document_chunk_unique",
        unique=True
    )
    logger.info("✓ Created unique compound index: document_id + chunk_id")
    
    # 3. Index على created_at - للترتيب الزمني
    await collection.create_index(
        [("created_at", -1)],
        name="idx_created_at_desc"
    )
    logger.info("✓ Created index: created_at")
    
    # 4. Index على length - للبحث بحجم الـ chunk
    await collection.create_index(
        "length",
        name="idx_length"
    )
    logger.info("✓ Created index: length")
    
    # 5. Text Index على text - للبحث النصي في الـ chunks
    await collection.create_index(
        [("text", "text")],
        name="idx_text_search_chunks",
        default_language="english"
    )
    logger.info("✓ Created text index: chunks text")
    
    # 6. Sparse Index على embedding - للـ chunks التي لها embeddings
    await collection.create_index(
        "embedding",
        name="idx_embedding",
        sparse=True
    )
    logger.info("✓ Created sparse index: embedding")

async def drop_all_indexes():
    """
    حذف جميع الـ indexes (للصيانة)
    """
    logger.warning("⚠️  Dropping all indexes...")
    
    # Documents
    doc_collection = db.get_collection(settings.documents_collection)
    await doc_collection.drop_indexes()
    
    # Chunks
    chunk_collection = db.get_collection(settings.chunks_collection)
    await chunk_collection.drop_indexes()
    
    logger.info("✅ All indexes dropped")

async def list_indexes():
    """
    عرض جميع الـ indexes الموجودة
    """
    # Documents indexes
    doc_collection = db.get_collection(settings.documents_collection)
    doc_indexes = await doc_collection.list_indexes().to_list(None)
    
    logger.info("📋 Documents Collection Indexes:")
    for idx in doc_indexes:
        logger.info(f"  - {idx['name']}: {idx.get('key', {})}")
    
    # Chunks indexes
    chunk_collection = db.get_collection(settings.chunks_collection)
    chunk_indexes = await chunk_collection.list_indexes().to_list(None)
    
    logger.info("📋 Chunks Collection Indexes:")
    for idx in chunk_indexes:
        logger.info(f"  - {idx['name']}: {idx.get('key', {})}")
    
    return {
        "documents": doc_indexes,
        "chunks": chunk_indexes
    }
```


#### 2. استخدام Text Search:

```python
# ملف database/repositories/document_repository.py (إضافة)

async def search_documents(self, query: str, limit: int = 10) -> List[DocumentInDB]:
    """
    البحث النصي في المستندات
    
    Args:
        query: نص البحث
        limit: الحد الأقصى للنتائج
        
    Returns:
        قائمة المستندات المطابقة
    """
    cursor = self.collection.find(
        {
            "$text": {
                "$search": query
            }
        },
        {
            "score": {"$meta": "textScore"}  # الحصول على نقاط المطابقة
        }
    ).sort(
        [("score", {"$meta": "textScore"})]  # ترتيب حسب الملاءمة
    ).limit(limit)
    
    documents = []
    async for doc in cursor:
        documents.append(DocumentInDB(**doc))
    
    return documents
```


#### 3. Query Performance Analysis:

```python
# ملف database/query_analyzer.py
from database.connection import db
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

async def explain_query(
    collection_name: str,
    query: Dict[str, Any],
    sort: Dict[str, int] = None
) -> Dict:
    """
    تحليل أداء استعلام
    
    Args:
        collection_name: اسم الـ collection
        query: الاستعلام
        sort: الترتيب (اختياري)
        
    Returns:
        تفاصيل التنفيذ
    """
    collection = db.get_collection(collection_name)
    
    # بناء الاستعلام
    cursor = collection.find(query)
    if sort:
        cursor = cursor.sort(list(sort.items()))
    
    # تحليل الاستعلام
    explain_result = await cursor.explain()
    
    # استخراج المعلومات المهمة
    execution_stats = explain_result.get('executionStats', {})
    
    analysis = {
        "query": query,
        "execution_time_ms": execution_stats.get('executionTimeMillis', 0),
        "documents_examined": execution_stats.get('totalDocsExamined', 0),
        "documents_returned": execution_stats.get('nReturned', 0),
        "index_used": explain_result.get('queryPlanner', {}).get('winningPlan', {}).get('inputStage', {}).get('indexName', 'COLLSCAN'),
        "efficient": execution_stats.get('totalDocsExamined', 0) == execution_stats.get('nReturned', 0)
    }
    
    # Log التحليل
    logger.info(f"Query Analysis:")
    logger.info(f"  Time: {analysis['execution_time_ms']}ms")
    logger.info(f"  Examined: {analysis['documents_examined']} docs")
    logger.info(f"  Returned: {analysis['documents_returned']} docs")
    logger.info(f"  Index: {analysis['index_used']}")
    logger.info(f"  Efficient: {analysis['efficient']}")
    
    return analysis

# مثال على الاستخدام
async def analyze_common_queries():
    """
    تحليل الاستعلامات الشائعة
    """
    # 1. البحث حسب status
    await explain_query(
        "documents",
        {"status": "processed"},
        {"upload_date": -1}
    )
    
    # 2. البحث حسب filename
    await explain_query(
        "documents",
        {"filename": {"$regex": "report", "$options": "i"}}
    )
    
    # 3. البحث النصي
    await explain_query(
        "documents",
        {"$text": {"$search": "machine learning"}}
    )
```


#### 4. إضافة endpoint للإحصائيات:

```python
# تحديث routes/documents.py

@router.get("/search")
async def search_documents(
    q: str = Query(..., min_length=3, description="نص البحث"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    البحث النصي في المستندات
    
    Args:
        q: نص البحث
        limit: عدد النتائج
    
    Returns:
        المستندات المطابقة
    """
    documents = await doc_repo.search_documents(q, limit)
    
    return {
        "query": q,
        "count": len(documents),
        "results": [
            DocumentResponse(
                id=str(doc.id),
                filename=doc.filename,
                file_size=doc.file_size,
                content_type=doc.content_type,
                upload_date=doc.upload_date,
                status=doc.status,
                chunk_count=doc.chunk_count or 0
            )
            for doc in documents
        ]
    }

@router.get("/admin/indexes")
async def list_all_indexes():
    """
    عرض جميع الـ indexes (للمطورين)
    """
    from database.indexes import list_indexes
    return await list_indexes()

@router.post("/admin/reindex")
async def recreate_indexes():

---

# اكمل

سأكمل الشرح التفصيلي:

***

## 🎬 الفيديو الحادي عشر: Mongo Indexing (تكملة)

#### 4. إضافة endpoints للإدارة (تكملة):
```python
# تحديث routes/documents.py (تكملة)

@router.post("/admin/reindex")
async def recreate_indexes():
    """
    إعادة إنشاء جميع الـ indexes
    """
    from database.indexes import drop_all_indexes, create_indexes
    
    try:
        await drop_all_indexes()
        await create_indexes()
        return {"message": "Indexes recreated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error recreating indexes: {str(e)}"
        )

@router.get("/admin/query-performance")
async def analyze_query_performance():
    """
    تحليل أداء الاستعلامات
    """
    from database.query_analyzer import analyze_common_queries
    await analyze_common_queries()
    return {"message": "Check logs for analysis results"}
```


#### 5. Best Practices للـ Indexes:

```python
# نصائح مهمة في التعليقات

"""
✅ DO's:
1. Create indexes على الحقول المستخدمة في WHERE/FILTER
2. Use compound indexes للاستعلامات المتكررة
3. Monitor index usage مع explain()
4. Keep indexes على الحد الأدنى المطلوب

❌ DON'Ts:
1. لا تُنشئ indexes على كل حقل
2. لا تستخدم indexes على collections صغيرة (<1000 document)
3. لا تنسى indexes في Write-heavy collections (بطء الكتابة)
4. لا تستخدم text indexes للبحث البسيط

📊 Performance Impact:
- Read operations: أسرع 10-100x مع indexes مناسب
- Write operations: أبطأ 5-10% لكل index إضافي
- Storage: كل index يستهلك مساحة إضافية
"""
```


#### 6. اختبار الـ Indexes:

```bash
# اختبار البحث النصي
$ curl "http://localhost:5000/documents/search?q=machine+learning&limit=5"

# عرض الـ indexes
$ curl "http://localhost:5000/documents/admin/indexes"

# تحليل الأداء
$ curl "http://localhost:5000/documents/admin/query-performance"

# إعادة بناء الـ indexes
$ curl -X POST "http://localhost:5000/documents/admin/reindex"
```


***

## 🎬 الفيديو الثاني عشر: Data Pipeline Enhancements (43 دقيقة)

**Branch:** `tut-008` (نفس branch الفيديو 11)

### الأهداف التعليمية:

- تحسين pipeline معالجة البيانات
- إضافة Progress tracking
- Better error handling
- Retry mechanisms


### المفاهيم الأساسية:

#### 1. لماذا نحتاج Pipeline Enhancements؟

- **Reliability**: معالجة أخطاء أفضل
- **Monitoring**: تتبع التقدم
- **Scalability**: معالجة ملفات كبيرة
- **User Experience**: تحديثات في الوقت الفعلي


### الأكواد التفصيلية:

#### 1. ملف models/processing_status.py:

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ProcessingStage(str, Enum):
    """
    مراحل المعالجة
    """
    UPLOADED = "uploaded"
    EXTRACTING_TEXT = "extracting_text"
    CLEANING_TEXT = "cleaning_text"
    CHUNKING = "chunking"
    SAVING_CHUNKS = "saving_chunks"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingProgress(BaseModel):
    """
    تتبع تقدم المعالجة
    """
    document_id: str
    stage: ProcessingStage
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    completed_steps: Optional[int] = 0
    started_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def update_progress(
        self, 
        stage: ProcessingStage, 
        percentage: float,
        current_step: str = None
    ):
        """
        تحديث التقدم
        """
        self.stage = stage
        self.progress_percentage = percentage
        self.current_step = current_step
        self.updated_at = datetime.now()
        
        if stage == ProcessingStage.COMPLETED:
            self.progress_percentage = 100.0
        elif stage == ProcessingStage.FAILED:
            self.progress_percentage = 0.0
```


#### 2. ملف helpers/enhanced_text_processor.py:

```python
from helpers.text_processor import TextProcessor
from models.processing_status import ProcessingProgress, ProcessingStage
from typing import List, Dict, Callable
import asyncio
import logging

logger = logging.getLogger(__name__)

class EnhancedTextProcessor(TextProcessor):
    """
    معالج نصوص محسّن مع تتبع التقدم
    """
    
    def __init__(self, progress_callback: Callable = None):
        super().__init__()
        self.progress_callback = progress_callback
    
    async def process_with_progress(
        self,
        file_path: str,
        file_extension: str,
        document_id: str
    ) -> Dict:
        """
        معالجة مع تتبع التقدم
        
        Args:
            file_path: مسار الملف
            file_extension: امتداد الملف
            document_id: معرف المستند
            
        Returns:
            نتائج المعالجة
        """
        progress = ProcessingProgress(
            document_id=document_id,
            stage=ProcessingStage.UPLOADED
        )
        
        try:
            # المرحلة 1: استخراج النص
            await self._update_progress(
                progress,
                ProcessingStage.EXTRACTING_TEXT,
                10.0,
                "Extracting text from file"
            )
            
            raw_text = self.extract_text(file_path, file_extension)
            
            await self._update_progress(
                progress,
                ProcessingStage.EXTRACTING_TEXT,
                30.0,
                f"Extracted {len(raw_text)} characters"
            )
            
            # المرحلة 2: تنظيف النص
            await self._update_progress(
                progress,
                ProcessingStage.CLEANING_TEXT,
                40.0,
                "Cleaning text"
            )
            
            clean_text = self.clean_text(raw_text)
            
            await self._update_progress(
                progress,
                ProcessingStage.CLEANING_TEXT,
                50.0,
                "Text cleaned"
            )
            
            # المرحلة 3: التقسيم
            await self._update_progress(
                progress,
                ProcessingStage.CHUNKING,
                60.0,
                "Chunking text"
            )
            
            chunks = self.chunk_text(
                clean_text,
                chunk_size=1000,
                overlap=200
            )
            
            await self._update_progress(
                progress,
                ProcessingStage.CHUNKING,
                80.0,
                f"Created {len(chunks)} chunks"
            )
            
            # المرحلة 4: الاكتمال
            await self._update_progress(
                progress,
                ProcessingStage.COMPLETED,
                100.0,
                "Processing completed"
            )
            
            return {
                "raw_text": raw_text,
                "clean_text": clean_text,
                "chunks": chunks,
                "stats": {
                    "raw_length": len(raw_text),
                    "clean_length": len(clean_text),
                    "chunk_count": len(chunks)
                }
            }
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            progress.stage = ProcessingStage.FAILED
            progress.error_message = str(e)
            await self._update_progress(
                progress,
                ProcessingStage.FAILED,
                0.0,
                f"Failed: {str(e)}"
            )
            raise
    
    async def _update_progress(
        self,
        progress: ProcessingProgress,
        stage: ProcessingStage,
        percentage: float,
        message: str
    ):
        """
        تحديث وإرسال التقدم
        """
        progress.update_progress(stage, percentage, message)
        
        logger.info(
            f"[{progress.document_id}] "
            f"{stage.value}: {percentage}% - {message}"
        )
        
        # استدعاء callback إذا كان موجوداً
        if self.progress_callback:
            await self.progress_callback(progress)
```


#### 3. ملف helpers/retry_handler.py:

```python
from typing import Callable, Any, Type
from functools import wraps
import asyncio
import logging

logger = logging.getLogger(__name__)

def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator للمحاولة مرة أخرى عند الفشل
    
    Args:
        max_attempts: الحد الأقصى للمحاولات
        delay: الانتظار الأولي (ثواني)
        backoff: معامل الزيادة للانتظار
        exceptions: أنواع الأخطاء للمحاولة مرة أخرى
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed: {str(e)}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed. "
                            f"Last error: {str(e)}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator

# مثال على الاستخدام
@retry_async(max_attempts=3, delay=2.0, backoff=2.0)
async def process_with_retry(document_id: str):
    """
    معالجة مع إعادة المحاولة
    """
    # عملية قد تفشل
    pass
```


#### 4. تحديث routes/documents.py مع التحسينات:

```python
from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from helpers.enhanced_text_processor import EnhancedTextProcessor
from helpers.retry_handler import retry_async
from models.processing_status import ProcessingProgress, ProcessingStage
from typing import List, Optional, AsyncGenerator
import asyncio
import json

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

# قاموس لتتبع التقدم في الذاكرة (للتطوير)
# في الإنتاج: استخدم Redis أو WebSocket
progress_tracker: Dict[str, ProcessingProgress] = {}

async def progress_callback(progress: ProcessingProgress):
    """
    حفظ التقدم
    """
    progress_tracker[progress.document_id] = progress

@retry_async(max_attempts=3, delay=2.0)
async def process_document_enhanced(document_id: str):
    """
    معالجة محسّنة مع retry
    """
    try:
        # الحصول على المستند
        document = await doc_repo.get_by_id(document_id)
        if not document:
            return
        
        # إنشاء معالج محسّن
        processor = EnhancedTextProcessor(progress_callback=progress_callback)
        
        # المعالجة مع تتبع التقدم
        file_ext = os.path.splitext(document.filename)[1]
        result = await processor.process_with_progress(
            document.file_path,
            file_ext,
            document_id
        )
        
        # حفظ الـ chunks
        chunks_to_create = []
        for chunk_data in result['chunks']:
            chunk = ChunkCreate(
                document_id=document_id,
                chunk_id=chunk_data['chunk_id'],
                text=chunk_data['text'],
                start_char=chunk_data['start_char'],
                end_char=chunk_data['end_char'],
                length=chunk_data['length']
            )
            chunks_to_create.append(chunk)
        
        if chunks_to_create:
            await chunk_repo.create_many(chunks_to_create)
        
        # تحديث المستند
        update_data = DocumentUpdate(
            status="processed",
            raw_text=result['raw_text'],
            clean_text=result['clean_text'],
            chunk_count=len(result['chunks']),
            metadata=result['stats']
        )
        await doc_repo.update(document_id, update_data)
        
    except Exception as e:
        await doc_repo.update_status(document_id, "failed", error=str(e))
        raise

@router.get("/{document_id}/progress")
async def get_processing_progress(document_id: str):
    """
    الحصول على تقدم المعالجة
    
    Args:
        

---

# اكمل

سأكمل الشرح التفصيلي:

***

## 🎬 الفيديو الثاني عشر: Data Pipeline Enhancements (تكملة)

#### 4. تحديث routes/documents.py (تكملة):
```python
@router.get("/{document_id}/progress")
async def get_processing_progress(document_id: str):
    """
    الحصول على تقدم المعالجة
    
    Args:
        document_id: معرف المستند
    
    Returns:
        حالة التقدم الحالية
    """
    if document_id not in progress_tracker:
        # الحصول من قاعدة البيانات
        document = await doc_repo.get_by_id(document_id)
        
        if not document:
            raise HTTPException(404, "Document not found")
        
        # إنشاء progress من حالة المستند
        return {
            "document_id": document_id,
            "stage": document.status,
            "progress_percentage": 100.0 if document.status == "processed" else 0.0,
            "message": f"Status: {document.status}"
        }
    
    progress = progress_tracker[document_id]
    
    return {
        "document_id": progress.document_id,
        "stage": progress.stage.value,
        "progress_percentage": progress.progress_percentage,
        "current_step": progress.current_step,
        "started_at": progress.started_at.isoformat(),
        "updated_at": progress.updated_at.isoformat(),
        "error_message": progress.error_message
    }

@router.get("/{document_id}/progress/stream")
async def stream_processing_progress(document_id: str):
    """
    Stream تقدم المعالجة (Server-Sent Events)
    """
    async def generate() -> AsyncGenerator[str, None]:
        """
        توليد events للتقدم
        """
        while True:
            if document_id in progress_tracker:
                progress = progress_tracker[document_id]
                
                # إرسال البيانات
                data = {
                    "stage": progress.stage.value,
                    "progress": progress.progress_percentage,
                    "message": progress.current_step
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                
                # إنهاء Stream عند الاكتمال أو الفشل
                if progress.stage in [ProcessingStage.COMPLETED, ProcessingStage.FAILED]:
                    break
            
            await asyncio.sleep(0.5)  # تحديث كل نصف ثانية
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```


#### 5. اختبار التحسينات:

```bash
# رفع مستند
$ curl -X POST "http://localhost:5000/documents/upload" \
  -F "file=@large_document.pdf"
# Response: {"id": "abc123", ...}

# تتبع التقدم
$ curl "http://localhost:5000/documents/abc123/progress"

# Stream التقدم (في متصفح أو EventSource)
# http://localhost:5000/documents/abc123/progress/stream
```


#### 6. Frontend Example (JavaScript):

```javascript
// مثال على استخدام Server-Sent Events
const docId = "abc123";
const eventSource = new EventSource(
    `http://localhost:5000/documents/${docId}/progress/stream`
);

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`Stage: ${data.stage}`);
    console.log(`Progress: ${data.progress}%`);
    console.log(`Message: ${data.message}`);
    
    // تحديث UI
    updateProgressBar(data.progress);
    
    if (data.stage === 'completed' || data.stage === 'failed') {
        eventSource.close();
    }
};
```


***

## 🎬 الفيديو الثالث عشر: Checkpoint-1 | What have we learned? (27 دقيقة)

**Branch:** `tut-008`

### الأهداف التعليمية:

- مراجعة شاملة لما تم إنجازه
- حل المشاكل الشائعة
- نصائح وأفضل الممارسات
- الاستعداد للمرحلة القادمة


### المحتوى الأساسي:

#### 1. ما تم إنجازه حتى الآن:

```python
"""
✅ المكونات المكتملة:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Backend Infrastructure:
   - FastAPI application
   - MongoDB integration
   - Docker setup
   - Repository pattern

2. File Management:
   - Upload with validation
   - Multiple file types support
   - Secure file storage
   - File deletion

3. Text Processing:
   - PDF extraction
   - DOCX extraction
   - Text cleaning
   - Chunking strategies

4. Database Operations:
   - CRUD for documents
   - CRUD for chunks
   - Indexes for performance
   - Query optimization

5. Advanced Features:
   - Background processing
   - Progress tracking
   - Retry mechanisms
   - Error handling
"""
```


#### 2. المشاكل الشائعة وحلولها:

```python
# ملف docs/troubleshooting.md

"""
🔧 مشاكل شائعة وحلولها:

1. MongoDB Connection Failed:
   ❌ المشكلة: pymongo.errors.ServerSelectionTimeoutError
   ✅ الحل:
      - تأكد من تشغيل Docker: docker compose ps
      - تحقق من المنفذ: netstat -an | grep 27017
      - راجع credentials في .env

2. File Upload Error (413):
   ❌ المشكلة: File too large
   ✅ الحل:
      - زيادة max_file_size في FileHandler
      - إضافة nginx config للملفات الكبيرة

3. Text Extraction Fails:
   ❌ المشكلة: Can't extract from PDF
   ✅ الحل:
      - تحقق من تثبيت PyPDF2 بشكل صحيح
      - بعض PDFs محمية: استخدم pypdf أو pdfplumber

4. Background Task Not Running:
   ❌ المشكلة: Document stuck in "uploaded"
   ✅ الحل:
      - تحقق من logs: uvicorn main:app --log-level debug
      - تأكد من عدم وجود exceptions في process_document

5. Chunks Not Saved:
   ❌ المشكلة: chunk_count = 0
   ✅ الحل:
      - تحقق من unique index على (document_id, chunk_id)
      - راجع permissions على uploads directory
"""
```


#### 3. Code Review Checklist:

```python
# ملف docs/code_review_checklist.py

"""
📋 Code Review Checklist:

□ Error Handling:
  □ Try-catch في جميع async functions
  □ HTTPException مع status codes مناسبة
  □ Logging للأخطاء

□ Validation:
  □ Pydantic models لكل input
  □ Query parameters مع constraints
  □ File validation (type, size)

□ Database:
  □ Indexes على الحقول المستخدمة
  □ Connection pool configured
  □ Proper error handling

□ Performance:
  □ Async operations حيثما أمكن
  □ Pagination للقوائم الطويلة
  □ Caching للبيانات المتكررة

□ Security:
  □ Environment variables للـ secrets
  □ Input sanitization
  □ File upload restrictions

□ Documentation:
  □ Docstrings لكل function
  □ API documentation (Swagger)
  □ README updated
"""
```


#### 4. Performance Benchmarks:

```python
# ملف tests/benchmark.py
import time
import asyncio
from database.repositories.document_repository import DocumentRepository

async def benchmark_queries():
    """
    قياس أداء الاستعلامات
    """
    repo = DocumentRepository()
    
    # 1. Get by ID
    start = time.time()
    doc = await repo.get_by_id("some_id")
    print(f"Get by ID: {(time.time() - start) * 1000:.2f}ms")
    
    # 2. List with pagination
    start = time.time()
    docs = await repo.get_all(skip=0, limit=10)
    print(f"List 10 docs: {(time.time() - start) * 1000:.2f}ms")
    
    # 3. Search
    start = time.time()
    results = await repo.search_documents("test", limit=10)
    print(f"Text search: {(time.time() - start) * 1000:.2f}ms")
    
    # 4. Count
    start = time.time()
    count = await repo.count()
    print(f"Count: {(time.time() - start) * 1000:.2f}ms")

"""
🎯 Expected Performance:
━━━━━━━━━━━━━━━━━━━━━━━
Get by ID:       < 5ms
List 10 docs:    < 10ms
Text search:     < 50ms
Count:           < 5ms
File upload:     < 1s (per MB)
Text extraction: < 2s (per page)
"""
```


#### 5. الاستعداد للمرحلة القادمة:

```python
"""
🚀 المرحلة القادمة: RAG Core

1. LLM Integration:
   - OpenAI API
   - Local models (Ollama)
   - Prompt engineering

2. Vector Databases:
   - Embeddings generation
   - QDrant setup
   - Similarity search

3. Semantic Search:
   - Vector similarity
   - Hybrid search
   - Ranking algorithms

4. Answer Generation:
   - Context retrieval
   - Prompt construction
   - Response formatting

📚 المتطلبات:
- OpenAI API key
- فهم Embeddings
- معرفة بـ Vector operations
"""
```


***

## 🎬 الفيديو الرابع عشر: LLM Factory (2:01 ساعة)

**Branch:** `tut-008` (بداية التغييرات الكبيرة)

### الأهداف التعليمية:

- دمج Large Language Models
- إنشاء LLM Factory pattern
- استخدام OpenAI API
- إدارة API keys بشكل آمن


### المفاهيم الأساسية:

#### 1. ما هو LLM Factory Pattern؟

- **Abstraction**: واجهة موحدة لمختلف LLMs
- **Flexibility**: سهولة التبديل بين النماذج
- **Testability**: إمكانية testing بدون API calls
- **Extensibility**: إضافة نماذج جديدة بسهولة


### الأكواد التفصيلية:

#### 1. تحديث requirements.txt:

```txt
# المكتبات السابقة...
openai==1.12.0           # OpenAI API
tiktoken==0.5.2          # Token counting
tenacity==8.2.3          # Retry logic
python-dotenv==1.0.0     # Environment variables
```


#### 2. تحديث config.py:

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application
    app_name: str = "mini-RAG"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # MongoDB
    mongodb_url: str
    database_name: str = "minirag"
    documents_collection: str = "documents"
    chunks_collection: str = "chunks"
    
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_max_tokens: int = 2000
    openai_temperature: float = 0.7
    
    # LLM Settings
    llm_timeout: int = 30
    llm_max_retries: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```


#### 3. ملف models/llm_models.py:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class LLMProvider(str, Enum):
    """
    مزودي LLM المدعومين
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"

class Message(BaseModel):
    """
    رسالة في المحادثة
    """
    role: str = Field(..., description="دور المرسل: system/user/assistant")
    content: str = Field(..., description="محتوى الرسالة")

class ChatRequest(BaseModel):
    """
    طلب محادثة
    """
    messages: List[Message]
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    stream: bool = False

class ChatResponse(BaseModel):
    """
    استجابة المحادثة
    """
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingRequest(BaseModel):
    """
    طلب embedding
    """
    text: str = Field(..., min_length=1)
    model: Optional[str] = None

class EmbeddingResponse(BaseModel):
    """
    استجابة embedding
    """
    embedding: List[float]
    model: str
    tokens_used: int
```


#### 4. ملف llm/base_llm.py:

```python
from abc import ABC, abstractmethod
from typing

---

# اكمل

سأكمل الشرح التفصيلي:

***

## 🎬 الفيديو الرابع عشر: LLM Factory (تكملة)

#### 4. ملف llm/base_llm.py:
```python
from abc import ABC, abstractmethod
from typing import List, AsyncGenerator
from models.llm_models import ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse

class BaseLLM(ABC):
    """
    Abstract base class لجميع LLM providers
    """
    
    def __init__(self, api_key: str, model: str = None):
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        إرسال رسائل والحصول على رد
        
        Args:
            request: طلب المحادثة
            
        Returns:
            استجابة LLM
        """
        pass
    
    @abstractmethod
    async def chat_stream(
        self, 
        request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """
        محادثة مع streaming
        
        Args:
            request: طلب المحادثة
            
        Yields:
            أجزاء النص المتدفق
        """
        pass
    
    @abstractmethod
    async def get_embedding(
        self, 
        request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """
        الحصول على embedding للنص
        
        Args:
            request: طلب embedding
            
        Returns:
            vector embedding
        """
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        حساب عدد tokens
        
        Args:
            text: النص
            
        Returns:
            عدد tokens
        """
        pass
```


#### 5. ملف llm/openai_llm.py:

```python
from llm.base_llm import BaseLLM
from models.llm_models import ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse
from openai import AsyncOpenAI
from typing import AsyncGenerator
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
import logging

logger = logging.getLogger(__name__)

class OpenAILLM(BaseLLM):
    """
    OpenAI LLM implementation
    """
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002"
    ):
        super().__init__(api_key, model)
        self.client = AsyncOpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.tokenizer = tiktoken.encoding_for_model(model)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        محادثة مع OpenAI
        """
        try:
            # تحضير الرسائل
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            # استدعاء API
            response = await self.client.chat.completions.create(
                model=request.model or self.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            # استخراج البيانات
            choice = response.choices[0]
            
            return ChatResponse(
                content=choice.message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens,
                finish_reason=choice.finish_reason,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI chat error: {str(e)}")
            raise
    
    async def chat_stream(
        self, 
        request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """
        محادثة مع streaming
        """
        try:
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            stream = await self.client.chat.completions.create(
                model=request.model or self.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI stream error: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_embedding(
        self, 
        request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """
        الحصول على embedding
        """
        try:
            response = await self.client.embeddings.create(
                model=request.model or self.embedding_model,
                input=request.text
            )
            
            embedding_data = response.data[0]
            
            return EmbeddingResponse(
                embedding=embedding_data.embedding,
                model=response.model,
                tokens_used=response.usage.total_tokens
            )
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
    async def count_tokens(self, text: str) -> int:
        """
        حساب tokens باستخدام tiktoken
        """
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token counting error: {str(e)}")
            # تقدير تقريبي: كل 4 أحرف = 1 token
            return len(text) // 4
```


#### 6. ملف llm/llm_factory.py:

```python
from llm.base_llm import BaseLLM
from llm.openai_llm import OpenAILLM
from models.llm_models import LLMProvider
from config import settings
import logging

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    Factory لإنشاء LLM instances
    """
    
    _instances = {}  # Cache للـ instances
    
    @staticmethod
    def create(
        provider: LLMProvider = LLMProvider.OPENAI,
        **kwargs
    ) -> BaseLLM:
        """
        إنشاء LLM instance
        
        Args:
            provider: نوع المزود
            **kwargs: معاملات إضافية
            
        Returns:
            LLM instance
        """
        # التحقق من Cache
        cache_key = f"{provider.value}_{kwargs.get('model', 'default')}"
        
        if cache_key in LLMFactory._instances:
            logger.info(f"Using cached LLM instance: {cache_key}")
            return LLMFactory._instances[cache_key]
        
        # إنشاء instance جديد
        if provider == LLMProvider.OPENAI:
            instance = OpenAILLM(
                api_key=kwargs.get('api_key') or settings.openai_api_key,
                model=kwargs.get('model') or settings.openai_model,
                embedding_model=kwargs.get('embedding_model') or settings.openai_embedding_model
            )
        
        # يمكن إضافة providers أخرى هنا
        # elif provider == LLMProvider.ANTHROPIC:
        #     instance = AnthropicLLM(...)
        # elif provider == LLMProvider.OLLAMA:
        #     instance = OllamaLLM(...)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        # حفظ في Cache
        LLMFactory._instances[cache_key] = instance
        logger.info(f"Created new LLM instance: {cache_key}")
        
        return instance
    
    @staticmethod
    def clear_cache():
        """
        مسح الـ cache
        """
        LLMFactory._instances.clear()
        logger.info("LLM cache cleared")
```


#### 7. إضافة routes/llm.py:

```python
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from llm.llm_factory import LLMFactory
from models.llm_models import (
    LLMProvider, 
    ChatRequest, 
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    Message
)
from typing import AsyncGenerator
import logging

router = APIRouter(
    prefix="/llm",
    tags=["LLM"]
)

logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    محادثة مع LLM
    
    Args:
        request: طلب المحادثة
    
    Returns:
        استجابة LLM
    """
    try:
        # إنشاء LLM instance
        llm = LLMFactory.create(LLMProvider.OPENAI)
        
        # إرسال الطلب
        response = await llm.chat(request)
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"LLM error: {str(e)}"
        )

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    محادثة مع streaming
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            llm = LLMFactory.create(LLMProvider.OPENAI)
            
            async for chunk in llm.chat_stream(request):
                yield chunk
                
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"\n\nError: {str(e)}"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

@router.post("/embedding", response_model=EmbeddingResponse)
async def get_embedding(request: EmbeddingRequest):
    """
    الحصول على embedding للنص
    """
    try:
        llm = LLMFactory.create(LLMProvider.OPENAI)
        response = await llm.get_embedding(request)
        return response
        
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding error: {str(e)}"
        )

@router.post("/tokens/count")
async def count_tokens(text: str):
    """
    حساب عدد tokens
    """
    try:
        llm = LLMFactory.create(LLMProvider.OPENAI)
        token_count = await llm.count_tokens(text)
        
        return {
            "text_length": len(text),
            "token_count": token_count,
            "ratio": len(text) / token_count if token_count > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Token counting error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )
```


#### 8. تحديث main.py:

```python
from fastapi import FastAPI
from config import settings
from database.connection import db
from database.indexes import create_indexes
from routes import documents, queries, llm  # إضافة llm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version
)

# تضمين الـ routers
app.include_router(documents.router)
app.include_router(queries.router)
app.include_router(llm.router)  # LLM routes

@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 Starting {settings.app_name}")
    await db.connect_to_database()
    await create_indexes()
    logger.info("✅ Application started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down...")
    await db.close_database_connection()
    logger.info("✅ Shutdown complete")
```


#### 9. اختبار LLM Integration:

```bash
# اختبار المحادثة
$ curl -X POST "http://localhost:5000/llm/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is RAG in AI?"}
    ],
    "temperature": 0.7
  }'

# Response:
{
  "content": "RAG stands for Retrieval-Augmented Generation...",
  "model": "gpt-3.5-turbo",
  "tokens_used": 150,
  "finish_reason": "stop"
}

# اختبار embedding
$ curl -X POST "http://localhost:5000/llm/embedding" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is amazing"
  }'

# حساب tokens
$ curl -X POST "http://localhost:5000/llm/tokens/count?text=Hello+World"
```


***

## 🎬 الفيديو الخامس عشر: Vector DB Factory | QDrant (1:02 ساعة)

**Branch:** `tut-009`

### الأهداف التعليمية:

- فهم Vector Databases
- إعداد QDrant
- تخزين Embeddings
- Similarity Search


### المفاهيم الأساسية:

#### 1. ما هو Vector Database؟

- **تخزين Embeddings**: vectors عددية تمثل المعنى
- **Similarity Search**: البحث بالتشابه الدلالي
- **High Performance**: محسّن للبحث في vectors

---

# اكمل

سأكمل الشرح التفصيلي للفيديو الخامس عشر:

***

## 🎬 الفيديو الخامس عشر: Vector DB Factory | QDrant (تكملة)

### المفاهيم الأساسية (تكملة):

#### 2. لماذا QDrant؟

- **Performance**: سريع جداً في similarity search
- **Open Source**: مجاني ومفتوح المصدر
- **Docker Support**: سهل التشغيل
- **Rich Features**: filtering, payload, collections


### الأكواد التفصيلية:

#### 1. تحديث docker/docker-compose.yml:

```yaml
version: '3.8'

services:
  # MongoDB (موجود سابقاً)
  mongodb:
    image: mongo:6.0
    # ... الإعدادات السابقة

  # QDrant Vector Database
  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: minirag-qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC
    volumes:
      - qdrant_storage:/qdrant/storage
    networks:
      - minirag-network
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  mongodb_data:
  mongodb_config:
  qdrant_storage:  # تخزين QDrant

networks:
  minirag-network:
    driver: bridge
```


#### 2. تحديث requirements.txt:

```txt
# المكتبات السابقة...
qdrant-client==1.7.0      # QDrant client
numpy==1.24.3             # للعمليات على vectors
```


#### 3. تحديث config.py:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ... الإعدادات السابقة
    
    # QDrant Settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "documents_chunks"
    qdrant_vector_size: int = 1536  # OpenAI embedding size
    qdrant_distance: str = "Cosine"  # أو "Euclidean" أو "Dot"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```


#### 4. ملف vectordb/base_vectordb.py:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseVectorDB(ABC):
    """
    Abstract base class لجميع Vector DBs
    """
    
    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine"
    ):
        """
        إنشاء collection جديد
        """
        pass
    
    @abstractmethod
    async def insert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """
        إدراج vectors
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        البحث بالتشابه
        """
        pass
    
    @abstractmethod
    async def delete_by_filter(
        self,
        collection_name: str,
        filter_dict: Dict[str, Any]
    ):
        """
        حذف vectors حسب filter
        """
        pass
    
    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> Dict:
        """
        الحصول على معلومات collection
        """
        pass
```


#### 5. ملف vectordb/qdrant_db.py:

```python
from vectordb.base_vectordb import BaseVectorDB
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from typing import List, Dict, Any, Optional
import logging
import uuid

logger = logging.getLogger(__name__)

class QdrantDB(BaseVectorDB):
    """
    QDrant implementation
    """
    
    def __init__(self, url: str):
        self.client = QdrantClient(url=url)
        logger.info(f"Connected to QDrant at {url}")
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine"
    ):
        """
        إنشاء collection
        """
        try:
            # تحويل distance string إلى enum
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLIDEAN,
                "Dot": Distance.DOT
            }
            
            distance_metric = distance_map.get(distance, Distance.COSINE)
            
            # التحقق من وجود collection
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_name in collection_names:
                logger.info(f"Collection '{collection_name}' already exists")
                return
            
            # إنشاء collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_metric
                )
            )
            
            logger.info(f"✅ Created collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    async def insert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """
        إدراج vectors مع payload
        """
        try:
            # إنشاء IDs إذا لم تكن موجودة
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in vectors]
            
            # إنشاء points
            points = []
            for i, (vector, payload, point_id) in enumerate(zip(vectors, payloads, ids)):
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                points.append(point)
            
            # إدراج في QDrant
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"✅ Inserted {len(points)} vectors into '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error inserting vectors: {str(e)}")
            raise
    
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        البحث بالتشابه
        """
        try:
            # بناء filter إذا كان موجوداً
            query_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)
            
            # البحث
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            # تحويل النتائج
            results = []
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise
    
    async def delete_by_filter(
        self,
        collection_name: str,
        filter_dict: Dict[str, Any]
    ):
        """
        حذف vectors
        """
        try:
            # بناء filter
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            
            delete_filter = Filter(must=conditions)
            
            # الحذف
            self.client.delete(
                collection_name=collection_name,
                points_selector=delete_filter
            )
            
            logger.info(f"Deleted vectors with filter: {filter_dict}")
            
        except Exception as e:
            logger.error(f"Delete error: {str(e)}")
            raise
    
    async def get_collection_info(self, collection_name: str) -> Dict:
        """
        معلومات collection
        """
        try:
            info = self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise
```


#### 6. ملف vectordb/vectordb_factory.py:

```python
from vectordb.base_vectordb import BaseVectorDB
from vectordb.qdrant_db import QdrantDB
from config import settings
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class VectorDBProvider(str, Enum):
    """
    مزودي Vector DB المدعومين
    """
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"

class VectorDBFactory:
    """
    Factory لإنشاء Vector DB instances
    """
    
    _instance = None
    
    @staticmethod
    def create(
        provider: VectorDBProvider = VectorDBProvider.QDRANT,
        **kwargs
    ) -> BaseVectorDB:
        """
        إنشاء Vector DB instance
        """
        # Singleton pattern
        if VectorDBFactory._instance is not None:
            return VectorDBFactory._instance
        
        if provider == VectorDBProvider.QDRANT:
            instance = QdrantDB(
                url=kwargs.get('url') or settings.qdrant_url
            )
        
        # يمكن إضافة providers أخرى
        # elif provider == VectorDBProvider.PINECONE:
        #     instance = PineconeDB(...)
        
        else:
            raise ValueError(f"Unsupported vector DB: {provider}")
        
        VectorDBFactory._instance = instance
        logger.info(f"Created VectorDB instance: {provider.value}")
        
        return instance
    
    @staticmethod
    def reset():
        """
        إعادة تعيين instance
        """
        VectorDBFactory._instance = None
```


#### 7. ملف services/embedding_service.py:

```python
from llm.llm_factory import LLMFactory
from models.llm_models import LLMProvider, EmbeddingRequest
from vectordb.vectordb_factory import VectorDBFactory, VectorDBProvider
from database.repositories.chunk_repository import ChunkRepository
from config import settings
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    خدمة إدارة embeddings
    """
    
    def __init__(self):
        self.llm = LLMFactory.create(LLMProvider.OPENAI)
        self.vectordb = VectorDBFactory.create(VectorDBProvider.QDRANT)
        self.chunk_repo = ChunkRepository()
    
    async def initialize_collection(self):
        """
        إنشاء collection في QDrant
        """
        await self.vectordb.create_collection(
            collection_name=settings.qdrant_collection_name,
            vector_size=settings.qdrant_vector_size,
            distance=settings.qdrant_distance
        )
    
    async def embed_and_store_chunks(self, document_id: str) -> int:
        """
        إنشاء embeddings للـ chunks وتخزينها
        
        Args:
            document_id: معرف المستند
            
        Returns:
            عدد chunks المُعالجة
        """
        try:
            # الحصول على chunks
            chunks = await self.chunk_repo.get_by_document_id(document_id)
            
            if not chunks:
                logger.warning(f"No chunks found for document {document_id}")
                return 0
            
            logger.info(f"Processing {len(chunks)} chunks for embeddings")
            
            # معالجة كل chunk
            vectors = []
            payloads = []
            ids = []
            
            for chunk in chunks:
                # إنشاء embedding
                embedding_request = EmbeddingRequest(text=chunk.text)
                embedding_response = await self.llm.get_embedding(embedding_request)
                
                # تحضير البيانات
                vectors.append(embedding_response.embedding)
                payloads.append({
                    "document_id": document_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "length": chunk.length
                })
                ids.append(str(chunk.id))

---

# اكمل

سأكمل الشرح التفصيلي:

***

## 🎬 الفيديو الخامس عشر: Vector DB Factory | QDrant (تكملة)

#### 7. ملف services/embedding_service.py (تكملة):
```python
            # ... تابع من السابق
            
            # تخزين في QDrant
            await self.vectordb.insert_vectors(
                collection_name=settings.qdrant_collection_name,
                vectors=vectors,
                payloads=payloads,
                ids=ids
            )
            
            # تحديث chunks في MongoDB بأن لها embeddings
            for chunk in chunks:
                await self.chunk_repo.update_embedding(
                    str(chunk.id),
                    embedding_response.embedding  # حفظ آخر embedding كمثال
                )
            
            logger.info(f"✅ Stored {len(chunks)} embeddings for document {document_id}")
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error embedding chunks: {str(e)}")
            raise
    
    async def search_similar_chunks(
        self,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """
        البحث عن chunks مشابهة
        
        Args:
            query: نص البحث
            limit: عدد النتائج
            document_id: تصفية حسب مستند معين
            score_threshold: الحد الأدنى للتشابه
            
        Returns:
            قائمة chunks المشابهة
        """
        try:
            # إنشاء embedding للاستعلام
            embedding_request = EmbeddingRequest(text=query)
            embedding_response = await self.llm.get_embedding(embedding_request)
            
            # بناء filter
            filter_dict = None
            if document_id:
                filter_dict = {"document_id": document_id}
            
            # البحث في QDrant
            results = await self.vectordb.search(
                collection_name=settings.qdrant_collection_name,
                query_vector=embedding_response.embedding,
                limit=limit,
                score_threshold=score_threshold,
                filter_dict=filter_dict
            )
            
            logger.info(f"Found {len(results)} similar chunks for query: '{query}'")
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise
    
    async def delete_document_embeddings(self, document_id: str):
        """
        حذف جميع embeddings مستند
        """
        try:
            await self.vectordb.delete_by_filter(
                collection_name=settings.qdrant_collection_name,
                filter_dict={"document_id": document_id}
            )
            
            logger.info(f"Deleted embeddings for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {str(e)}")
            raise
```


#### 8. تحديث routes/documents.py مع Embeddings:

```python
from services.embedding_service import EmbeddingService

# إضافة instance
embedding_service = EmbeddingService()

@router.post("/{document_id}/embed")
async def create_embeddings(document_id: str):
    """
    إنشاء embeddings للمستند
    
    Args:
        document_id: معرف المستند
    
    Returns:
        عدد embeddings المُنشأة
    """
    # التحقق من المستند
    document = await doc_repo.get_by_id(document_id)
    
    if not document:
        raise HTTPException(404, "Document not found")
    
    if document.status != "processed":
        raise HTTPException(
            400,
            f"Document must be processed first. Current status: {document.status}"
        )
    
    try:
        # إنشاء embeddings
        count = await embedding_service.embed_and_store_chunks(document_id)
        
        return {
            "document_id": document_id,
            "embeddings_created": count,
            "message": "Embeddings created successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            500,
            f"Error creating embeddings: {str(e)}"
        )

@router.delete("/{document_id}/embeddings")
async def delete_embeddings(document_id: str):
    """
    حذف embeddings المستند
    """
    try:
        await embedding_service.delete_document_embeddings(document_id)
        
        return {"message": "Embeddings deleted successfully"}
        
    except Exception as e:
        raise HTTPException(
            500,
            f"Error deleting embeddings: {str(e)}"
        )
```


#### 9. إضافة routes/search.py:

```python
from fastapi import APIRouter, HTTPException, Query
from services.embedding_service import EmbeddingService
from typing import Optional

router = APIRouter(
    prefix="/search",
    tags=["Search"]
)

embedding_service = EmbeddingService()

@router.get("/semantic")
async def semantic_search(
    q: str = Query(..., min_length=3, description="نص البحث"),
    limit: int = Query(5, ge=1, le=20, description="عدد النتائج"),
    document_id: Optional[str] = None,
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="الحد الأدنى للتشابه")
):
    """
    البحث الدلالي في المستندات
    
    Args:
        q: نص البحث
        limit: عدد النتائج
        document_id: تصفية حسب مستند معين
        threshold: الحد الأدنى لدرجة التشابه
    
    Returns:
        chunks المشابهة مع درجات التشابه
    """
    try:
        results = await embedding_service.search_similar_chunks(
            query=q,
            limit=limit,
            document_id=document_id,
            score_threshold=threshold
        )
        
        return {
            "query": q,
            "count": len(results),
            "results": [
                {
                    "chunk_id": result["payload"]["chunk_id"],
                    "document_id": result["payload"]["document_id"],
                    "text": result["payload"]["text"],
                    "similarity_score": result["score"],
                    "length": result["payload"]["length"]
                }
                for result in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            500,
            f"Search error: {str(e)}"
        )
```


#### 10. تحديث main.py:

```python
from fastapi import FastAPI
from config import settings
from database.connection import db
from database.indexes import create_indexes
from services.embedding_service import EmbeddingService
from routes import documents, queries, llm, search  # إضافة search
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version
)

# تضمين الـ routers
app.include_router(documents.router)
app.include_router(queries.router)
app.include_router(llm.router)
app.include_router(search.router)  # Search routes

@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 Starting {settings.app_name}")
    
    # MongoDB
    await db.connect_to_database()
    await create_indexes()
    
    # QDrant
    embedding_service = EmbeddingService()
    await embedding_service.initialize_collection()
    
    logger.info("✅ Application started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down...")
    await db.close_database_connection()
    logger.info("✅ Shutdown complete")
```


#### 11. اختبار كامل للـ Workflow:

```bash
# 1. تشغيل Docker (MongoDB + QDrant)
$ cd docker
$ docker compose up -d

# 2. تشغيل التطبيق
$ cd ..
$ uvicorn main:app --reload

# 3. رفع مستند
$ curl -X POST "http://localhost:5000/documents/upload" \
  -F "file=@document.pdf"
# Response: {"id": "doc123", "status": "uploaded"}

# انتظر حتى يتم المعالجة (status = "processed")

# 4. إنشاء embeddings
$ curl -X POST "http://localhost:5000/documents/doc123/embed"
# Response: {"embeddings_created": 25}

# 5. البحث الدلالي
$ curl "http://localhost:5000/search/semantic?q=machine+learning&limit=5"

# Response:
{
  "query": "machine learning",
  "count": 5,
  "results": [
    {
      "chunk_id": 0,
      "document_id": "doc123",
      "text": "Machine learning is a subset of AI...",
      "similarity_score": 0.92,
      "length": 450
    },
    ...
  ]
}
```


***

## 📊 ملخص الفيديوهات 11-15

### **ما تم إنجازه:**

| الفيديو | المدة | الإنجازات الرئيسية |
| :-- | :-- | :-- |
| **11. Mongo Indexing** | 1:13 | ✅ Indexes متعددة للأداء<br>✅ Text search<br>✅ Query optimization<br>✅ Performance analysis |
| **12. Pipeline Enhancements** | 0:43 | ✅ Progress tracking<br>✅ Retry mechanisms<br>✅ Server-Sent Events<br>✅ Better error handling |
| **13. Checkpoint-1** | 0:27 | ✅ مراجعة شاملة<br>✅ حل المشاكل<br>✅ Best practices<br>✅ Code review |
| **14. LLM Factory** | 2:01 | ✅ OpenAI integration<br>✅ Factory pattern<br>✅ Chat \& Embeddings<br>✅ Token counting |
| **15. Vector DB (QDrant)** | 1:02 | ✅ QDrant setup<br>✅ Embeddings storage<br>✅ Similarity search<br>✅ Vector operations |

### **التقنيات الجديدة:**

```python
"""
📦 المكتبات المضافة:
━━━━━━━━━━━━━━━━━━━━━━
- openai==1.12.0          # LLM integration
- tiktoken==0.5.2         # Token counting
- tenacity==8.2.3         # Retry logic
- qdrant-client==1.7.0    # Vector database
- numpy==1.24.3           # Vector operations
"""
```


### **الـ APIs الجديدة:**

```bash
# LLM Operations
POST   /llm/chat                    # محادثة مع LLM
POST   /llm/chat/stream            # محادثة مع streaming
POST   /llm/embedding              # إنشاء embedding
POST   /llm/tokens/count           # حساب tokens

# Embeddings Operations
POST   /documents/{id}/embed       # إنشاء embeddings للمستند
DELETE /documents/{id}/embeddings  # حذف embeddings

# Semantic Search
GET    /search/semantic            # البحث الدلالي

# Monitoring
GET    /documents/{id}/progress    # تتبع التقدم
GET    /documents/{id}/progress/stream  # Stream التقدم
```


### **البنية المحدثة:**

```
mini-rag/
├── llm/                          # LLM integrations
│   ├── base_llm.py              # Abstract base
│   ├── openai_llm.py            # OpenAI implementation
│   └── llm_factory.py           # Factory pattern
├── vectordb/                     # Vector DB integrations
│   ├── base_vectordb.py         # Abstract base
│   ├── qdrant_db.py             # QDrant implementation
│   └── vectordb_factory.py      # Factory pattern
├── services/                     # Business logic services
│   └── embedding_service.py     # Embedding management
├── models/
│   ├── llm_models.py            # LLM models
│   └── processing_status.py     # Progress tracking
└── routes/
    ├── llm.py                   # LLM endpoints
    └── search.py                # Search endpoints
```


### **الخطوات القادمة (16-25):**

```python
"""
🔮 ما سيأتي في الفيديوهات القادمة:

16. Semantic Search (1:24 ساعة)
    - Hybrid search (text + semantic)
    - Ranking algorithms
    - Search optimization

17. Augmented Answers (1:17 ساعة)
    - RAG pipeline كامل
    - Context retrieval
    - Answer generation
    - Citation management

18-19. Ollama Local LLM
    - تشغيل نماذج محلية
    - Privacy & cost reduction
    - Offline capabilities

20-21. PostgreSQL + PGVector
    - الانتقال من MongoDB
    - PGVector للـ embeddings
    - Migration strategies

22-23. Deployment
    - Production deployment
    - Docker optimization
    - Monitoring & logging

24-25. Celery
    - Task queues
    - Background jobs
    - Scalability
"""
```


### **🎓 نصائح للاستفادة القصوى:**

1. **جرّب بنفسك**: لا تكتفي بالمشاهدة، اكتب الكود
2. **اختبر Embeddings**: جرب نصوص مختلفة وشاهد النتائج
3. **راقب الأداء**: استخدم QDrant dashboard
4. **اقرأ التوثيق**: OpenAI و QDrant docs مهمة
5. **استكشف**: غيّر parameters وشاهد التأثير

***

**🎉 تهانينا! أكملت 15 فيديو من mini-RAG**

الآن لديك:

- ✅ Backend كامل مع FastAPI
- ✅ Database layer (MongoDB + QDrant)
- ✅ LL

---

# اكمل الباقي

سأكمل شرح باقي الفيديوهات من 16 إلى 25 بالتفصيل:

***

# شرح تفصيلي ودقيق للفيديوهات 16-25 من سلسلة mini-RAG


***

## 🎬 الفيديو السادس عشر: Semantic Search (1:24 ساعة)

**Branch:** `tut-010`

### الأهداف التعليمية:

- فهم Semantic vs Keyword Search
- تطبيق Hybrid Search
- تحسين نتائج البحث
- Ranking algorithms


### المفاهيم الأساسية:

#### 1. الفرق بين أنواع البحث:

```python
"""
📊 مقارنة أنواع البحث:

1. Keyword Search (التقليدي):
   - يبحث عن كلمات مطابقة بالضبط
   - سريع لكن محدود
   - مثال: "machine learning" لن يجد "ML" أو "artificial intelligence"

2. Semantic Search (الدلالي):
   - يفهم المعنى والسياق
   - يجد نتائج مشابهة دلالياً
   - مثال: "machine learning" يجد "ML", "AI", "deep learning"

3. Hybrid Search:
   - يجمع بين الاثنين
   - أفضل النتائج
   - نستخدمه في mini-RAG
"""
```


### الأكواد التفصيلية:

#### 1. ملف services/search_service.py:

```python
from typing import List, Dict, Optional
from services.embedding_service import EmbeddingService
from database.repositories.document_repository import DocumentRepository
from database.repositories.chunk_repository import ChunkRepository
import logging

logger = logging.getLogger(__name__)

class SearchService:
    """
    خدمة البحث المتقدم
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.doc_repo = DocumentRepository()
        self.chunk_repo = ChunkRepository()
    
    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        document_id: Optional[str] = None
    ) -> List[Dict]:
        """
        بحث هجين (دلالي + كلمات مفتاحية)
        
        Args:
            query: نص البحث
            limit: عدد النتائج
            semantic_weight: وزن البحث الدلالي (0-1)
            keyword_weight: وزن البحث بالكلمات (0-1)
            document_id: تصفية حسب مستند
            
        Returns:
            نتائج مدمجة ومرتبة
        """
        try:
            # 1. البحث الدلالي (Semantic)
            semantic_results = await self.embedding_service.search_similar_chunks(
                query=query,
                limit=limit * 2,  # نأخذ أكثر للدمج
                document_id=document_id,
                score_threshold=0.5
            )
            
            # 2. البحث بالكلمات المفتاحية (Keyword)
            keyword_results = await self._keyword_search(
                query=query,
                limit=limit * 2,
                document_id=document_id
            )
            
            # 3. دمج النتائج
            merged_results = self._merge_and_rank(
                semantic_results,
                keyword_results,
                semantic_weight,
                keyword_weight
            )
            
            # 4. إرجاع أفضل النتائج
            return merged_results[:limit]
            
        except Exception as e:
            logger.error(f"Hybrid search error: {str(e)}")
            raise
    
    async def _keyword_search(
        self,
        query: str,
        limit: int,
        document_id: Optional[str] = None
    ) -> List[Dict]:
        """
        البحث بالكلمات المفتاحية في MongoDB
        """
        try:
            # البحث في documents
            documents = await self.doc_repo.search_documents(query, limit)
            
            # جمع chunks من هذه المستندات
            results = []
            for doc in documents:
                chunks = await self.chunk_repo.get_by_document_id(str(doc.id))
                
                for chunk in chunks:
                    # حساب relevance score بسيط
                    score = self._calculate_keyword_score(query, chunk.text)
                    
                    if score > 0:
                        results.append({
                            "id": str(chunk.id),
                            "score": score,
                            "payload": {
                                "document_id": str(doc.id),
                                "chunk_id": chunk.chunk_id,
                                "text": chunk.text,
                                "length": chunk.length
                            },
                            "source": "keyword"
                        })
            
            # ترتيب حسب Score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Keyword search error: {str(e)}")
            return []
    
    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """
        حساب درجة التطابق للكلمات المفتاحية
        """
        # تحويل لـ lowercase
        query_lower = query.lower()
        text_lower = text.lower()
        
        # تقسيم لكلمات
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        
        # حساب التقاطع
        common_words = query_words.intersection(text_words)
        
        if not query_words:
            return 0.0
        
        # نسبة التطابق
        match_ratio = len(common_words) / len(query_words)
        
        # bonus إذا كان النص يحتوي على الجملة كاملة
        if query_lower in text_lower:
            match_ratio += 0.5
        
        return min(match_ratio, 1.0)
    
    def _merge_and_rank(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict]:
        """
        دمج وترتيب النتائج
        """
        # قاموس لتجميع النتائج حسب chunk_id
        merged = {}
        
        # إضافة النتائج الدلالية
        for result in semantic_results:
            chunk_id = result["payload"]["chunk_id"]
            merged[chunk_id] = {
                **result,
                "semantic_score": result["score"],
                "keyword_score": 0.0,
                "combined_score": result["score"] * semantic_weight
            }
        
        # إضافة/تحديث النتائج الكلماتية
        for result in keyword_results:
            chunk_id = result["payload"]["chunk_id"]
            
            if chunk_id in merged:
                # موجود: تحديث
                merged[chunk_id]["keyword_score"] = result["score"]
                merged[chunk_id]["combined_score"] += result["score"] * keyword_weight
            else:
                # جديد: إضافة
                merged[chunk_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "keyword_score": result["score"],
                    "combined_score": result["score"] * keyword_weight
                }
        
        # تحويل لقائمة وترتيب
        results = list(merged.values())
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return results
    
    async def rerank_results(
        self,
        query: str,
        results: List[Dict],
        strategy: str = "diversity"
    ) -> List[Dict]:
        """
        إعادة ترتيب النتائج
        
        Args:
            query: الاستعلام الأصلي
            results: النتائج
            strategy: استراتيجية الترتيب (diversity, relevance, recency)
        """
        if strategy == "diversity":
            # تنويع النتائج من مستندات مختلفة
            return self._rerank_by_diversity(results)
        elif strategy == "recency":
            # الأحدث أولاً
            return self._rerank_by_recency(results)
        else:
            # الأكثر صلة (default)
            return results
    
    def _rerank_by_diversity(self, results: List[Dict]) -> List[Dict]:
        """
        ترتيب بالتنويع - نتائج من مستندات مختلفة
        """
        reranked = []
        seen_documents = set()
        
        # المرور الأول: نتيجة واحدة من كل مستند
        for result in results:
            doc_id = result["payload"]["document_id"]
            if doc_id not in seen_documents:
                reranked.append(result)
                seen_documents.add(doc_id)
        
        # المرور الثاني: باقي النتائج
        for result in results:
            if result not in reranked:
                reranked.append(result)
        
        return reranked
    
    def _rerank_by_recency(self, results: List[Dict]) -> List[Dict]:
        """
        ترتيب حسب التاريخ (الأحدث أولاً)
        """
        # يحتاج لـ metadata بالتاريخ
        # هنا مجرد مثال
        return sorted(
            results,
            key=lambda x: x["payload"].get("created_at", 0),
            reverse=True
        )
```


#### 2. تحديث routes/search.py:

```python
from fastapi import APIRouter, HTTPException, Query
from services.search_service import SearchService
from typing import Optional, List
from enum import Enum

router = APIRouter(
    prefix="/search",
    tags=["Search"]
)

search_service = SearchService()

class SearchStrategy(str, Enum):
    """
    استراتيجيات البحث
    """
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class RerankStrategy(str, Enum):
    """
    استراتيجيات إعادة الترتيب
    """
    RELEVANCE = "relevance"
    DIVERSITY = "diversity"
    RECENCY = "recency"

@router.get("/")
async def search(
    q: str = Query(..., min_length=2, description="نص البحث"),
    strategy: SearchStrategy = SearchStrategy.HYBRID,
    limit: int = Query(10, ge=1, le=50),
    document_id: Optional[str] = None,
    semantic_weight: float = Query(0.7, ge=0.0, le=1.0),
    rerank: Optional[RerankStrategy] = None
):
    """
    البحث المتقدم في المستندات
    
    Args:
        q: نص البحث
        strategy: استراتيجية البحث
        limit: عدد النتائج
        document_id: تصفية حسب مستند
        semantic_weight: وزن البحث الدلالي
        rerank: استراتيجية إعادة الترتيب
    """
    try:
        # تنفيذ البحث حسب الاستراتيجية
        if strategy == SearchStrategy.HYBRID:
            results = await search_service.hybrid_search(
                query=q,
                limit=limit,
                semantic_weight=semantic_weight,
                keyword_weight=1.0 - semantic_weight,
                document_id=document_id
            )
        elif strategy == SearchStrategy.SEMANTIC:
            results = await search_service.embedding_service.search_similar_chunks(
                query=q,
                limit=limit,
                document_id=document_id
            )
            # تنسيق النتائج
            results = [{
                **r,
                "semantic_score": r["score"],
                "keyword_score": 0.0,
                "combined_score": r["score"]
            } for r in results]
        else:  # KEYWORD
            results = await search_service._keyword_search(
                query=q,
                limit=limit,
                document_id=document_id
            )
        
        # إعادة ترتيب إذا طُلب
        if rerank:
            results = await search_service.rerank_results(
                query=q,
                results=results,
                strategy=rerank.value
            )
        
        # تنسيق الاستجابة
        return {
            "query": q,
            "strategy": strategy.value,
            "count": len(results),
            "results": [
                {
                    "chunk_id": r["payload"]["chunk_id"],
                    "document_id": r["payload"]["document_id"],
                    "text": r["payload"]["text"][:500] + "..." if len(r["payload"]["text"]) > 500 else r["payload"]["text"],
                    "scores": {
                        "semantic": r.get("semantic_score", 0.0),
                        "keyword": r.get("keyword_score", 0.0),
                        "combined": r.get("combined_score", r.get("score", 0.0))
                    }
                }
                for r in results
            ]
        }
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(500, f"Search error: {str(e)}")
```


#### 3. اختبار البحث المتقدم:

```bash
# بحث هجين
$ curl "http://localhost:5000/search/?q=machine+learning&strategy=hybrid&limit=5"

# بحث دلالي فقط
$ curl "http://localhost:5000/search/?q=AI+models&strategy=semantic"

# بحث بالكلمات فقط
$ curl "http://localhost:5000/search/?q=neural+network&strategy=keyword"

# بحث مع إعادة ترتيب
$ curl "http://localhost:5000/search/?q=deep+learning&rerank=diversity"

---

# اكمل

سأكمل الشرح التفصيلي لباقي الفيديوهات:

***

## 🎬 الفيديو السابع عشر: Augmented Answers (1:17 ساعة)
**Branch:** `tut-011`

### الأهداف التعليمية:
- إتمام RAG pipeline الكامل
- Context retrieval
- Answer generation
- Citation management

### المفاهيم الأساسية:

#### 1. ما هو RAG الكامل؟
```python
"""
🔄 RAG Pipeline:

1. User Question → "What is machine learning?"

2. Retrieval:
   - Search similar chunks (semantic)
   - Get relevant context

3. Augmentation:
   - Combine question + context
   - Build prompt for LLM

4. Generation:
   - Send to LLM
   - Get answer based on context

5. Response:
   - Answer with citations
   - Source references
"""
```


### الأكواد التفصيلية:

#### 1. ملف models/rag_models.py:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ContextChunk(BaseModel):
    """
    قطعة من السياق
    """
    chunk_id: int
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RAGRequest(BaseModel):
    """
    طلب RAG
    """
    question: str = Field(..., min_length=5, description="السؤال")
    max_context_chunks: int = Field(5, ge=1, le=10, description="عدد chunks للسياق")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    include_sources: bool = Field(True, description="إضافة المصادر")
    document_id: Optional[str] = None

class Citation(BaseModel):
    """
    استشهاد/مصدر
    """
    chunk_id: int
    document_id: str
    text_snippet: str
    relevance_score: float

class RAGResponse(BaseModel):
    """
    استجابة RAG
    """
    question: str
    answer: str
    citations: List[Citation]
    context_used: int
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
```


#### 2. ملف services/rag_service.py:

```python
from services.search_service import SearchService
from llm.llm_factory import LLMFactory
from models.llm_models import LLMProvider, ChatRequest, Message
from models.rag_models import RAGRequest, RAGResponse, Citation, ContextChunk
from typing import List
import logging

logger = logging.getLogger(__name__)

class RAGService:
    """
    خدمة RAG الكاملة
    """
    
    def __init__(self):
        self.search_service = SearchService()
        self.llm = LLMFactory.create(LLMProvider.OPENAI)
        
        # System prompt للـ RAG
        self.system_prompt = """أنت مساعد ذكي متخصص في الإجابة على الأسئلة بناءً على المستندات المقدمة.

قواعد الإجابة:
1. أجب فقط بناءً على المعلومات الموجودة في السياق المقدم
2. إذا لم يكن السياق كافياً، قل "لا أملك معلومات كافية للإجابة"
3. كن دقيقاً ومحدداً في إجابتك
4. استخدم لغة واضحة ومفهومة
5. إذا كانت الإجابة غير متأكد منها، وضح ذلك

تذكر: مصداقيتك تعتمد على دقة إجاباتك."""
    
    async def generate_answer(self, request: RAGRequest) -> RAGResponse:
        """
        توليد إجابة باستخدام RAG
        
        Args:
            request: طلب RAG
            
        Returns:
            إجابة مع مصادر
        """
        try:
            # الخطوة 1: استرجاع السياق
            logger.info(f"Retrieving context for: '{request.question}'")
            
            context_chunks = await self._retrieve_context(
                question=request.question,
                max_chunks=request.max_context_chunks,
                document_id=request.document_id
            )
            
            if not context_chunks:
                return RAGResponse(
                    question=request.question,
                    answer="عذراً، لم أجد معلومات ذات صلة في المستندات المتاحة للإجابة على سؤالك.",
                    citations=[],
                    context_used=0,
                    confidence=0.0,
                    metadata={"error": "No relevant context found"}
                )
            
            # الخطوة 2: بناء الـ prompt
            logger.info(f"Building prompt with {len(context_chunks)} chunks")
            
            prompt = self._build_prompt(
                question=request.question,
                context_chunks=context_chunks
            )
            
            # الخطوة 3: توليد الإجابة
            logger.info("Generating answer from LLM")
            
            chat_request = ChatRequest(
                messages=[
                    Message(role="system", content=self.system_prompt),
                    Message(role="user", content=prompt)
                ],
                temperature=request.temperature,
                max_tokens=1000
            )
            
            llm_response = await self.llm.chat(chat_request)
            
            # الخطوة 4: استخراج المصادر
            citations = []
            if request.include_sources:
                citations = self._extract_citations(context_chunks)
            
            # الخطوة 5: حساب الثقة
            confidence = self._calculate_confidence(
                context_chunks,
                llm_response.content
            )
            
            return RAGResponse(
                question=request.question,
                answer=llm_response.content,
                citations=citations,
                context_used=len(context_chunks),
                confidence=confidence,
                metadata={
                    "model": llm_response.model,
                    "tokens_used": llm_response.tokens_used,
                    "avg_context_score": sum(c.score for c in context_chunks) / len(context_chunks)
                }
            )
            
        except Exception as e:
            logger.error(f"RAG error: {str(e)}")
            raise
    
    async def _retrieve_context(
        self,
        question: str,
        max_chunks: int,
        document_id: Optional[str] = None
    ) -> List[ContextChunk]:
        """
        استرجاع السياق ذي الصلة
        """
        # البحث الهجين
        search_results = await self.search_service.hybrid_search(
            query=question,
            limit=max_chunks,
            semantic_weight=0.8,  # تركيز أكبر على الدلالي
            keyword_weight=0.2,
            document_id=document_id
        )
        
        # تحويل لـ ContextChunk objects
        context_chunks = []
        for result in search_results:
            chunk = ContextChunk(
                chunk_id=result["payload"]["chunk_id"],
                document_id=result["payload"]["document_id"],
                text=result["payload"]["text"],
                score=result.get("combined_score", result.get("score", 0.0)),
                metadata={
                    "semantic_score": result.get("semantic_score", 0.0),
                    "keyword_score": result.get("keyword_score", 0.0)
                }
            )
            context_chunks.append(chunk)
        
        return context_chunks
    
    def _build_prompt(
        self,
        question: str,
        context_chunks: List[ContextChunk]
    ) -> str:
        """
        بناء الـ prompt الكامل
        """
        # بناء السياق
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"[مصدر {i}]\n{chunk.text}\n")
        
        context_text = "\n".join(context_parts)
        
        # بناء الـ prompt
        prompt = f"""السياق المتاح:

{context_text}

---

بناءً على السياق أعلاه، أجب على السؤال التالي:

السؤال: {question}

الإجابة:"""
        
        return prompt
    
    def _extract_citations(
        self,
        context_chunks: List[ContextChunk]
    ) -> List[Citation]:
        """
        استخراج الاستشهادات من chunks
        """
        citations = []
        
        for chunk in context_chunks:
            # أخذ أول 200 حرف كـ snippet
            snippet = chunk.text[:200]
            if len(chunk.text) > 200:
                snippet += "..."
            
            citation = Citation(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text_snippet=snippet,
                relevance_score=chunk.score
            )
            citations.append(citation)
        
        return citations
    
    def _calculate_confidence(
        self,
        context_chunks: List[ContextChunk],
        answer: str
    ) -> float:
        """
        حساب درجة الثقة في الإجابة
        """
        if not context_chunks:
            return 0.0
        
        # عوامل الثقة:
        # 1. متوسط scores الـ chunks
        avg_score = sum(c.score for c in context_chunks) / len(context_chunks)
        
        # 2. عدد chunks المستخدمة (المزيد = أفضل)
        chunk_factor = min(len(context_chunks) / 5.0, 1.0)
        
        # 3. طول الإجابة (ليست قصيرة جداً)
        length_factor = min(len(answer) / 200.0, 1.0)
        
        # 4. عدم احتواء الإجابة على "لا أعرف" أو مشابه
        uncertainty_phrases = [
            "لا أعرف",
            "لا أملك معلومات",
            "غير متأكد",
            "لا يمكنني",
            "i don't know",
            "i'm not sure"
        ]
        
        has_uncertainty = any(
            phrase in answer.lower() 
            for phrase in uncertainty_phrases
        )
        
        uncertainty_factor = 0.3 if has_uncertainty else 1.0
        
        # الحساب النهائي
        confidence = (
            avg_score * 0.4 +
            chunk_factor * 0.3 +
            length_factor * 0.2 +
            uncertainty_factor * 0.1
        )
        
        return round(min(confidence, 1.0), 2)
```


#### 3. إضافة routes/rag.py:

```python
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from services.rag_service import RAGService
from models.rag_models import RAGRequest, RAGResponse
from typing import AsyncGenerator
import json
import logging

router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)

logger = logging.getLogger(__name__)
rag_service = RAGService()

@router.post("/answer", response_model=RAGResponse)
async def generate_answer(request: RAGRequest):
    """
    توليد إجابة باستخدام RAG
    
    Args:
        request: طلب RAG مع السؤال والإعدادات
    
    Returns:
        إجابة مع مصادر واستشهادات
    """
    try:
        response = await rag_service.generate_answer(request)
        return response
        
    except Exception as e:
        logger.error(f"RAG error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )

@router.post("/answer/stream")
async def generate_answer_stream(request: RAGRequest):
    """
    توليد إجابة مع streaming
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            # استرجاع السياق
            context_chunks = await rag_service._retrieve_context(
                question=request.question,
                max_chunks=request.max_context_chunks,
                document_id=request.document_id
            )
            
            if not context_chunks:
                yield json.dumps({"error": "No context found"}) + "\n"
                return
            
            # بناء prompt
            prompt = rag_service._build_prompt(
                request.question,
                context_chunks
            )
            
            # إرسال metadata أولاً
            metadata = {
                "context_chunks": len(context_chunks),
                "question": request.question
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            # Stream الإجابة
            from models.llm_models import ChatRequest, Message
            
            chat_request = ChatRequest(
                messages=[
                    Message(role="system", content=rag_service.system_prompt),
                    Message(role="user", content=prompt)
                ],
                temperature=request.temperature,
                stream=True
            )
            
            async for chunk in rag_service.llm.chat_stream(chat_request):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # إرسال citations في النهاية
            if request.include_sources:
                citations = rag_service._extract_citations(context_chunks)
                citations_data = {
                    "citations": [c.dict() for c in citations]
                }
                yield f"data: {json.dumps(citations_data)}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

---

# اكمل

سأكمل الشرح التفصيلي لباقي الفيديوهات:

***

## 🎬 الفيديو السابع عشر: Augmented Answers (تكملة)

#### 4. اختبار RAG الكامل:
```bash
# توليد إجابة
$ curl -X POST "http://localhost:5000/rag/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "max_context_chunks": 5,
    "temperature": 0.7,
    "include_sources": true
  }'

# Response:
{
  "question": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed...",
  "citations": [
    {
      "chunk_id": 0,
      "document_id": "doc123",
      "text_snippet": "Machine learning algorithms use statistical techniques...",
      "relevance_score": 0.92
    }
  ],
  "context_used": 5,
  "confidence": 0.87,
  "metadata": {
    "model": "gpt-3.5-turbo",
    "tokens_used": 450
  }
}
```


***

## 🎬 الفيديو الثامن عشر: Checkpoint-2 | Fixes (21 دقيقة)

**Branch:** `tut-012`

### الأهداف التعليمية:

- مراجعة المرحلة الثانية
- إصلاح الأخطاء الشائعة
- تحسينات وتنظيف الكود


### المحتوى الأساسي:

#### 1. الأخطاء الشائعة وحلولها:

```python
# ملف docs/checkpoint2_fixes.md

"""
🔧 إصلاحات Checkpoint 2:

1. Embedding Errors:
   ❌ المشكلة: Embeddings تفشل للنصوص الطويلة
   ✅ الحل: تقسيم النص قبل embedding
   
   # قبل
   embedding = await llm.get_embedding(long_text)
   
   # بعد
   if len(long_text) > 8000:
       long_text = long_text[:8000]
   embedding = await llm.get_embedding(long_text)

2. QDrant Connection Issues:
   ❌ المشكلة: Connection timeout
   ✅ الحل: زيادة timeout وإضافة retry
   
   client = QdrantClient(
       url=url,
       timeout=30,
       prefer_grpc=False
   )

3. Search Returns Empty:
   ❌ المشكلة: البحث لا يرجع نتائج
   ✅ الحل: تقليل score_threshold
   
   # قبل
   results = search(query, score_threshold=0.9)
   
   # بعد
   results = search(query, score_threshold=0.5)

4. Context Too Long for LLM:
   ❌ المشكلة: Prompt يتجاوز max_tokens
   ✅ الحل: تقليص السياق
   
   # حساب tokens وتقليص إذا لزم
   total_tokens = await llm.count_tokens(prompt)
   if total_tokens > 3000:
       # تقليل عدد chunks
       context_chunks = context_chunks[:3]

5. Citations Missing:
   ❌ المشكلة: المصادر لا تظهر
   ✅ الحل: تأكد من include_sources=True
"""
```


***

## 🎬 الفيديو التاسع عشر: Ollama Local LLM Server (50 دقيقة)

**Branch:** `tut-012` (استمرار)

### الأهداف التعليمية:

- تشغيل LLM محلياً
- استخدام Ollama
- تقليل التكلفة والاعتماد على APIs خارجية


### المفاهيم الأساسية:

#### 1. لماذا Ollama؟

```python
"""
💡 مميزات Ollama:

✅ مجاني تماماً
✅ يعمل offline
✅ لا حدود للاستخدام
✅ خصوصية كاملة
✅ نماذج متعددة (Llama, Mistral, etc.)

⚠️ العيوب:
- يحتاج GPU قوي
- أبطأ من OpenAI
- جودة أقل قليلاً
"""
```


### الأكواد التفصيلية:

#### 1. تشغيل Ollama مع Colab + Ngrok:

```python
# ملف notebooks/ollama_setup.ipynb

"""
📓 Colab Notebook لتشغيل Ollama

# 1. تثبيت Ollama
!curl https://ollama.ai/install.sh | sh

# 2. تشغيل Ollama في الخلفية
import subprocess
import time

ollama_process = subprocess.Popen(
    ['ollama', 'serve'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
time.sleep(5)

# 3. تحميل نموذج
!ollama pull llama2:7b

# 4. تثبيت ngrok
!pip install pyngrok

# 5. إنشاء tunnel
from pyngrok import ngrok

# إنشاء tunnel للمنفذ 11434 (Ollama default)
public_url = ngrok.connect(11434, "http")
print(f"🌐 Ollama URL: {public_url}")

# استخدم هذا URL في .env:
# OLLAMA_BASE_URL=https://xxxx-xx-xx-xxx-xxx.ngrok.io
"""
```


#### 2. ملف llm/ollama_llm.py:

```python
from llm.base_llm import BaseLLM
from models.llm_models import ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse
from typing import AsyncGenerator
import aiohttp
import logging

logger = logging.getLogger(__name__)

class OllamaLLM(BaseLLM):
    """
    Ollama LLM implementation
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        embedding_model: str = "nomic-embed-text"
    ):
        super().__init__(api_key="", model=model)  # Ollama لا يحتاج API key
        self.base_url = base_url.rstrip('/')
        self.embedding_model = embedding_model
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        محادثة مع Ollama
        """
        try:
            # تحويل messages لـ prompt
            prompt = self._messages_to_prompt(request.messages)
            
            # استدعاء Ollama API
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": request.model or self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens or 2000
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Ollama error: {response.status}")
                    
                    result = await response.json()
            
            return ChatResponse(
                content=result["response"],
                model=result["model"],
                tokens_used=result.get("eval_count", 0),
                finish_reason="stop",
                metadata={
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration")
                }
            )
            
        except Exception as e:
            logger.error(f"Ollama chat error: {str(e)}")
            raise
    
    async def chat_stream(
        self, 
        request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """
        محادثة مع streaming
        """
        try:
            prompt = self._messages_to_prompt(request.messages)
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": request.model or self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": request.temperature
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    async for line in response.content:
                        if line:
                            import json
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            
                            if data.get("done", False):
                                break
                                
        except Exception as e:
            logger.error(f"Ollama stream error: {str(e)}")
            raise
    
    async def get_embedding(
        self, 
        request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """
        الحصول على embedding
        """
        try:
            url = f"{self.base_url}/api/embeddings"
            
            payload = {
                "model": request.model or self.embedding_model,
                "prompt": request.text
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Ollama error: {response.status}")
                    
                    result = await response.json()
            
            return EmbeddingResponse(
                embedding=result["embedding"],
                model=request.model or self.embedding_model,
                tokens_used=len(request.text.split())  # تقدير
            )
            
        except Exception as e:
            logger.error(f"Ollama embedding error: {str(e)}")
            raise
    
    async def count_tokens(self, text: str) -> int:
        """
        حساب tokens (تقريبي)
        """
        # Ollama لا يوفر token counting دقيق
        # تقدير: كل 4 أحرف = 1 token
        return len(text) // 4
    
    def _messages_to_prompt(self, messages: List) -> str:
        """
        تحويل messages لـ prompt واحد
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.role
            content = msg.content
            
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
```


#### 3. تحديث llm/llm_factory.py:

```python
from llm.base_llm import BaseLLM
from llm.openai_llm import OpenAILLM
from llm.ollama_llm import OllamaLLM
from models.llm_models import LLMProvider
from config import settings

class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"  # إضافة

class LLMFactory:
    @staticmethod
    def create(
        provider: LLMProvider = LLMProvider.OPENAI,
        **kwargs
    ) -> BaseLLM:
        if provider == LLMProvider.OPENAI:
            instance = OpenAILLM(
                api_key=kwargs.get('api_key') or settings.openai_api_key,
                model=kwargs.get('model') or settings.openai_model
            )
        
        elif provider == LLMProvider.OLLAMA:
            instance = OllamaLLM(
                base_url=kwargs.get('base_url') or settings.ollama_base_url,
                model=kwargs.get('model') or settings.ollama_model
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return instance
```


#### 4. تحديث config.py:

```python
class Settings(BaseSettings):
    # ... إعدادات سابقة
    
    # Ollama Settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    ollama_embedding_model: str = "nomic-embed-text"
    
    # Default LLM Provider
    default_llm_provider: str = "openai"  # أو "ollama"
```


#### 5. اختبار Ollama:

```bash
# تشغيل Ollama محلياً
$ ollama serve

# في terminal آخر: تحميل نموذج
$ ollama pull llama2

# اختبار
$ curl -X POST "http://localhost:5000/llm/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "provider": "ollama"
  }'
```


***

## 🎬 الفيديو العشرون: From Mongo to Postgres + SQLAlchemy \& Alembic (2:17 ساعة)

**Branch:** `tut-013`

### الأهداف التعليمية:

- الانتقال من MongoDB إلى PostgreSQL
- استخدام SQLAlchemy ORM
- Database migrations مع Alembic
- أسباب التغيير


### المفاهيم الأساسية:

#### 1. لماذا PostgreSQL؟

```python
"""
🔄 MongoDB vs PostgreSQL:

MongoDB (NoSQL):
✅ مرونة في البيانات
✅ سهل البداية
❌ لا يدعم JOIN قوي
❌ Transactions محدودة
❌ لا توجد PGVector native

PostgreSQL (SQL):
✅ ACID compliance
✅ Relations و JOINs قوية
✅ PGVector extension للـ embeddings
✅ أداء أفضل للبيانات المن

---

# اكمل

سأكمل الشرح التفصيلي لباقي الفيديوهات:

***

## 🎬 الفيديو العشرون: From Mongo to Postgres (تكملة)

### المفاهيم الأساسية (تكملة):

#### 1. لماذا PostgreSQL؟ (تكملة)
```python
"""
PostgreSQL (SQL):
✅ ACID compliance
✅ Relations و JOINs قوية
✅ PGVector extension للـ embeddings
✅ أداء أفضل للبيانات المنظمة
✅ Mature ecosystem
✅ Free & Open Source

القرار: PostgreSQL أفضل لـ Production RAG
"""
```


### الأكواد التفصيلية:

#### 1. تحديث docker/docker-compose.yml:

```yaml
version: '3.8'

services:
  # PostgreSQL
  postgres:
    image: postgres:15-alpine
    container_name: minirag-postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - minirag-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # PGAdmin (Web UI اختياري)
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: minirag-pgadmin
    restart: unless-stopped
    ports:
      - "5050:80"
    environment:
      PGADMIN_DEFAULT_EMAIL: ${PGADMIN_EMAIL}
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD}
    depends_on:
      - postgres
    networks:
      - minirag-network

volumes:
  postgres_data:
  qdrant_storage:

networks:
  minirag-network:
    driver: bridge
```


#### 2. تحديث requirements.txt:

```txt
# ... المكتبات السابقة
sqlalchemy==2.0.25        # ORM
asyncpg==0.29.0          # PostgreSQL async driver
alembic==1.13.1          # Database migrations
psycopg2-binary==2.9.9   # PostgreSQL sync driver
```


#### 3. ملف database/postgres_connection.py:

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from config import settings
import logging

logger = logging.getLogger(__name__)

# Base class للـ models
Base = declarative_base()

# Engine
engine = create_async_engine(
    settings.postgres_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_db():
    """
    إنشاء الجداول
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables created")

async def get_session() -> AsyncSession:
    """
    Dependency للحصول على session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```


#### 4. ملف database/models.py:

```python
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
from database.postgres_connection import Base

class Document(Base):
    """
    جدول المستندات
    """
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, index=True)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), nullable=False)
    
    status = Column(String(50), nullable=False, default="uploaded", index=True)
    raw_text = Column(Text, nullable=True)
    clean_text = Column(Text, nullable=True)
    
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    processed_at = Column(DateTime, nullable=True)
    
    chunk_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    
    # Relationship
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_status_upload', 'status', 'upload_date'),
    )

class Chunk(Base):
    """
    جدول الـ chunks
    """
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(Integer, nullable=False)
    
    text = Column(Text, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    length = Column(Integer, nullable=False)
    
    # Embedding (سيتم نقله لـ PGVector لاحقاً)
    has_embedding = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationship
    document = relationship("Document", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_document_chunk', 'document_id', 'chunk_id', unique=True),
    )
```


#### 5. ملف alembic.ini:

```ini
[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url = postgresql+asyncpg://user:password@localhost:5432/minirag

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```


#### 6. إنشاء Migrations:

```bash
# تهيئة Alembic
$ alembic init alembic

# إنشاء migration أولي
$ alembic revision --autogenerate -m "Initial schema"

# تطبيق migrations
$ alembic upgrade head

# التراجع
$ alembic downgrade -1
```


#### 7. ملف database/repositories/document_repository_pg.py:

```python
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import Document, Chunk
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentRepositoryPG:
    """
    Repository للمستندات مع PostgreSQL
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, document_data: dict) -> Document:
        """
        إنشاء مستند
        """
        document = Document(**document_data)
        self.session.add(document)
        await self.session.flush()
        return document
    
    async def get_by_id(self, document_id: int) -> Optional[Document]:
        """
        الحصول على مستند
        """
        result = await self.session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 10,
        status: Optional[str] = None
    ) -> List[Document]:
        """
        الحصول على قائمة المستندات
        """
        query = select(Document)
        
        if status:
            query = query.where(Document.status == status)
        
        query = query.order_by(Document.upload_date.desc())
        query = query.offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def update(self, document_id: int, update_data: dict) -> bool:
        """
        تحديث مستند
        """
        document = await self.get_by_id(document_id)
        if not document:
            return False
        
        for key, value in update_data.items():
            if hasattr(document, key):
                setattr(document, key, value)
        
        await self.session.flush()
        return True
    
    async def delete(self, document_id: int) -> bool:
        """
        حذف مستند
        """
        document = await self.get_by_id(document_id)
        if not document:
            return False
        
        await self.session.delete(document)
        await self.session.flush()
        return True
    
    async def count(self, status: Optional[str] = None) -> int:
        """
        عد المستندات
        """
        query = select(func.count(Document.id))
        
        if status:
            query = query.where(Document.status == status)
        
        result = await self.session.execute(query)
        return result.scalar()
    
    async def search(self, query: str, limit: int = 10) -> List[Document]:
        """
        البحث في المستندات
        """
        search_query = select(Document).where(
            or_(
                Document.filename.ilike(f"%{query}%"),
                Document.clean_text.ilike(f"%{query}%")
            )
        ).limit(limit)
        
        result = await self.session.execute(search_query)
        return result.scalars().all()
```


***

## 🎬 الفيديو الحادي والعشرون: The Way to PGVector (3:39 ساعة)

**Branch:** `tut-014`

### الأهداف التعليمية:

- تثبيت PGVector extension
- دمج Embeddings في PostgreSQL
- Vector similarity search
- الانتقال من QDrant


### الأكواد التفصيلية:

#### 1. تثبيت PGVector:

```sql
-- في PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;

-- التحقق
SELECT * FROM pg_extension WHERE extname = 'vector';
```


#### 2. تحديث database/models.py:

```python
from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector  # إضافة
from database.postgres_connection import Base

class Chunk(Base):
    """
    جدول الـ chunks مع embeddings
    """
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(Integer, nullable=False)
    
    text = Column(Text, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    length = Column(Integer, nullable=False)
    
    # Embedding vector (1536 dimensions for OpenAI)
    embedding = Column(Vector(1536), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationship
    document = relationship("Document", back_populates="chunks")
    
    # Indexes
    __table_args__ = (
        Index('idx_document_chunk', 'document_id', 'chunk_id', unique=True),
        # Vector index للبحث السريع
        Index('idx_embedding_vector', 'embedding', postgresql_using='ivfflat', postgresql_with={'lists': 100}),
    )
```


#### 3. ملف services/pgvector_service.py:

```python
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import Chunk
from llm.llm_factory import LLMFactory
from models.llm_models import LLMProvider, EmbeddingRequest
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class PGVectorService:
    """
    خدمة PGVector للبحث الدلالي
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.llm = LLMFactory.create(LLMProvider.OPENAI)
    
    async def embed_and_store_chunks(self, document_id: int) -> int:
        """
        إنشاء embeddings للـ chunks وحفظها
        """
        try:
            # الحصول على chunks بدون embeddings
            result = await self.session.execute(
                select(Chunk).where(
                    Chunk.document_id == document_id,
                    Chunk.embedding.is_(None)
                )
            )
            chunks = result.scalars().all()
            
            if not chunks:
                logger.warning(f"No chunks to embed for document {document_id}")
                return 0
            
            logger.info(f"Embedding {len(chunks)} chunks")
            
            # معالجة كل chunk
            for chunk in chunks:
                # إنشاء embedding
                embedding_request = EmbeddingRequest(text=chunk.text)
                embedding_response = await self.llm.get_embedding(embedding_request)
                
                # حفظ embedding
                chunk.embedding = embedding_response.embedding
            
            await self.session.flush()
            
            logger.info(f"✅ Embedded {len(chunks)} chunks")
            return len(chunks)
            
        

---

# اكمل

سأكمل الشرح التفصيلي لباقي الفيديوهات:

***

## 🎬 الفيديو الحادي والعشرون: PGVector (تكملة)

#### 3. ملف services/pgvector_service.py (تكملة):
```python
        except Exception as e:
            logger.error(f"Error embedding chunks: {str(e)}")
            raise
    
    async def search_similar(
        self,
        query: str,
        limit: int = 5,
        document_id: Optional[int] = None,
        distance_threshold: float = 1.0
    ) -> List[Dict]:
        """
        البحث بالتشابه الدلالي
        
        Args:
            query: نص البحث
            limit: عدد النتائج
            document_id: تصفية حسب مستند
            distance_threshold: الحد الأقصى للمسافة
        """
        try:
            # إنشاء embedding للاستعلام
            embedding_request = EmbeddingRequest(text=query)
            embedding_response = await self.llm.get_embedding(embedding_request)
            query_vector = embedding_response.embedding
            
            # بناء query
            # استخدام cosine distance: 1 - cosine_similarity
            query_sql = select(
                Chunk,
                (1 - Chunk.embedding.cosine_distance(query_vector)).label('similarity')
            ).where(
                Chunk.embedding.is_not(None)
            )
            
            # تصفية حسب document
            if document_id:
                query_sql = query_sql.where(Chunk.document_id == document_id)
            
            # تصفية حسب distance
            query_sql = query_sql.where(
                Chunk.embedding.cosine_distance(query_vector) < distance_threshold
            )
            
            # ترتيب وتحديد
            query_sql = query_sql.order_by(
                Chunk.embedding.cosine_distance(query_vector)
            ).limit(limit)
            
            result = await self.session.execute(query_sql)
            rows = result.all()
            
            # تنسيق النتائج
            results = []
            for chunk, similarity in rows:
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "similarity": float(similarity),
                    "length": chunk.length
                })
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise
```


#### 4. Migration Script:

```python
# ملف scripts/migrate_qdrant_to_pgvector.py

"""
نقل البيانات من QDrant إلى PGVector
"""

import asyncio
from database.postgres_connection import AsyncSessionLocal
from vectordb.qdrant_db import QdrantDB
from database.models import Chunk
from sqlalchemy import select
import logging

logger = logging.getLogger(__name__)

async def migrate_embeddings():
    """
    نقل embeddings من QDrant إلى PostgreSQL
    """
    logger.info("🚀 Starting migration from QDrant to PGVector")
    
    # الاتصال بـ QDrant
    qdrant = QdrantDB(url="http://localhost:6333")
    
    # الحصول على جميع points
    collection_info = await qdrant.get_collection_info("documents_chunks")
    logger.info(f"Total points in QDrant: {collection_info['points_count']}")
    
    async with AsyncSessionLocal() as session:
        # الحصول على chunks بدون embeddings
        result = await session.execute(
            select(Chunk).where(Chunk.embedding.is_(None))
        )
        chunks = result.scalars().all()
        
        logger.info(f"Chunks to migrate: {len(chunks)}")
        
        migrated = 0
        for chunk in chunks:
            try:
                # البحث عن embedding في QDrant
                # (يحتاج لتعديل حسب طريقة تخزينك)
                # هنا نفترض أن chunk.id موجود في QDrant
                
                # يمكنك استخدام search أو get point
                # ثم نسخ الـ embedding
                
                # مثال:
                # point = qdrant.client.retrieve(
                #     collection_name="documents_chunks",
                #     ids=[str(chunk.id)]
                # )
                # if point:
                #     chunk.embedding = point[0].vector
                #     migrated += 1
                
                pass  # استبدل بالكود الفعلي
                
            except Exception as e:
                logger.error(f"Error migrating chunk {chunk.id}: {str(e)}")
        
        await session.commit()
        logger.info(f"✅ Migrated {migrated} embeddings")

if __name__ == "__main__":
    asyncio.run(migrate_embeddings())
```


***

## 🎬 الفيديو الثاني والعشرون: App Deployment 1/2 (2:17 ساعة)

**Branch:** `tut-015`

### الأهداف التعليمية:

- تحضير التطبيق للـ Production
- Docker optimization
- Environment management
- Security best practices


### الأكواد التفصيلية:

#### 1. ملف Dockerfile (محسّن):

```dockerfile
# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# تثبيت dependencies للـ build
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# نسخ requirements
COPY requirements.txt .

# تثبيت dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# تثبيت runtime dependencies فقط
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# نسخ Python packages من builder
COPY --from=builder /root/.local /root/.local

# إضافة للـ PATH
ENV PATH=/root/.local/bin:$PATH

# نسخ الكود
COPY . .

# إنشاء user غير root
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# تشغيل التطبيق
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```


#### 2. ملف docker-compose.prod.yml:

```yaml
version: '3.8'

services:
  # PostgreSQL
  postgres:
    image: postgres:15-alpine
    restart: always
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    networks:
      - minirag-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis (للـ caching والـ rate limiting)
  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - minirag-network

  # API Application
  api:
    build:
      context: .
      dockerfile: Dockerfile
    restart: always
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    networks:
      - minirag-network
    volumes:
      - ./uploads:/app/uploads
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Nginx (Reverse Proxy)
  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    networks:
      - minirag-network

volumes:
  postgres_data:
  redis_data:

networks:
  minirag-network:
    driver: bridge
```


#### 3. ملف nginx/nginx.conf:

```nginx
events {
    worker_connections 1024;
}

http {
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    
    upstream api {
        server api:8000;
    }
    
    server {
        listen 80;
        server_name api.yourdomain.com;
        
        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name api.yourdomain.com;
        
        # SSL certificates
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # SSL settings
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        
        # Max upload size
        client_max_body_size 50M;
        
        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        
        location / {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        # Health check endpoint (no rate limit)
        location /health {
            proxy_pass http://api;
        }
    }
}
```


#### 4. ملف .env.production:

```bash
# Application
APP_NAME=mini-RAG
APP_VERSION=1.0.0
DEBUG=false
ENVIRONMENT=production

# Database
POSTGRES_USER=minirag_user
POSTGRES_PASSWORD=super_secure_password_here
POSTGRES_DB=minirag_prod
DATABASE_URL=postgresql+asyncpg://minirag_user:super_secure_password_here@postgres:5432/minirag_prod

# Redis
REDIS_PASSWORD=redis_secure_password
REDIS_URL=redis://:redis_secure_password@redis:6379/0

# OpenAI
OPENAI_API_KEY=sk-your-production-key-here

# Security
SECRET_KEY=your-super-secret-key-here-minimum-32-chars
ALLOWED_HOSTS=api.yourdomain.com,yourdomain.com

# CORS
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
```


#### 5. تحديث main.py للـ Production:

```python
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from config import settings
import logging
import time

# إعداد logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# إنشاء التطبيق
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,  # إخفاء docs في production
    redoc_url="/redoc" if settings.debug else None
)

# Middleware للـ rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {str(exc)}", exc_info=True)
    
    if settings.debug:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER

---

# اكمل

سأكمل الشرح التفصيلي لباقي الفيديوهات:

***

## 🎬 الفيديو الثاني والعشرون: Deployment 1/2 (تكملة)

#### 5. تحديث main.py للـ Production (تكملة):
```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {str(exc)}", exc_info=True)
    
    if settings.debug:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "type": type(exc).__name__}
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

# Health check
@app.get("/health")
@limiter.exempt  # لا rate limit على health check
async def health():
    """فحص صحة التطبيق"""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": settings.app_version
    }

# Startup
@app.on_event("startup")
async def startup():
    logger.info(f"🚀 Starting {settings.app_name} in {settings.environment} mode")
    await db.connect_to_database()
    await init_db()
    logger.info("✅ Application started successfully")

# Shutdown
@app.on_event("shutdown")
async def shutdown():
    logger.info("🛑 Shutting down application")
    await db.close_database_connection()
    logger.info("✅ Shutdown complete")
```


#### 6. نشر التطبيق:

```bash
# بناء وتشغيل
$ docker-compose -f docker-compose.prod.yml up -d --build

# عرض logs
$ docker-compose -f docker-compose.prod.yml logs -f api

# إيقاف
$ docker-compose -f docker-compose.prod.yml down
```


***

## 🎬 الفيديو الثالث والعشرون: App Deployment 2/2 (1:49 ساعة)

**Branch:** `tut-015`

### الأهداف التعليمية:

- Monitoring \& Logging
- Backup strategies
- CI/CD pipeline
- Performance optimization


### الأكواد التفصيلية:

#### 1. إضافة Prometheus \& Grafana:

```yaml
# تحديث docker-compose.prod.yml

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    restart: always
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - minirag-network

  # Grafana
  grafana:
    image: grafana/grafana:latest
    restart: always
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    networks:
      - minirag-network

volumes:
  prometheus_data:
  grafana_data:
```


#### 2. ملف prometheus/prometheus.yml:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
```


#### 3. إضافة Metrics endpoint:

```python
# ملف monitoring/metrics.py
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()

# Metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

rag_queries = Counter(
    'rag_queries_total',
    'Total RAG queries'
)

documents_uploaded = Counter(
    'documents_uploaded_total',
    'Total documents uploaded'
)

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain"
    )
```


#### 4. Middleware للـ Metrics:

```python
# في main.py
from monitoring.metrics import request_count, request_duration
import time

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # تسجيل metrics
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response
```


#### 5. Backup Script:

```bash
#!/bin/bash
# ملف scripts/backup.sh

# إعدادات
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="minirag_prod"
DB_USER="minirag_user"

# إنشاء مجلد backup
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
echo "📦 Backing up PostgreSQL..."
docker exec minirag-postgres pg_dump -U $DB_USER $DB_NAME | gzip > "$BACKUP_DIR/db_$DATE.sql.gz"

# Backup uploads folder
echo "📦 Backing up uploads..."
tar -czf "$BACKUP_DIR/uploads_$DATE.tar.gz" ./uploads

# حذف backups أقدم من 7 أيام
echo "🧹 Cleaning old backups..."
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

echo "✅ Backup completed: $DATE"
```


#### 6. CI/CD مع GitHub Actions:

```yaml
# ملف .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      
      - name: Run tests
        run: pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /app/mini-rag
            git pull origin main
            docker-compose -f docker-compose.prod.yml down
            docker-compose -f docker-compose.prod.yml up -d --build
            docker-compose -f docker-compose.prod.yml exec api alembic upgrade head
```


***

## 🎬 الفيديو الرابع والعشرون: Celery Basics 1/2 (2:01 ساعة)

**Branch:** `tut-016`

### الأهداف التعليمية:

- فهم Task Queues
- إعداد Celery
- Background tasks
- Task monitoring


### المفاهيم الأساسية:

#### 1. لماذا Celery؟

```python
"""
🎯 مشاكل Background Tasks البسيطة:

❌ FastAPI BackgroundTasks:
   - تعمل في نفس الـ process
   - إذا أُعيد تشغيل الخادم، تضيع المهام
   - لا يوجد retry mechanism
   - لا يوجد monitoring

✅ Celery:
   - مهام في processes منفصلة
   - Persistent (تُحفظ في Redis/RabbitMQ)
   - Retry automatic
   - Monitoring مع Flower
   - Scheduling (Celery Beat)
   - Scalable (عدة workers)
"""
```


### الأكواد التفصيلية:

#### 1. تحديث requirements.txt:

```txt
# ... المكتبات السابقة
celery==5.3.4
redis==5.0.1
flower==2.0.1  # للـ monitoring
```


#### 2. ملف celery_app.py:

```python
from celery import Celery
from config import settings
import logging

logger = logging.getLogger(__name__)

# إنشاء Celery app
celery_app = Celery(
    "minirag",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        'tasks.document_tasks',
        'tasks.embedding_tasks',
        'tasks.cleanup_tasks'
    ]
)

# إعدادات Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 دقيقة
    task_soft_time_limit=25 * 60,  # 25 دقيقة
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Task routing
celery_app.conf.task_routes = {
    'tasks.document_tasks.*': {'queue': 'documents'},
    'tasks.embedding_tasks.*': {'queue': 'embeddings'},
    'tasks.cleanup_tasks.*': {'queue': 'maintenance'},
}

logger.info("✅ Celery app configured")
```


#### 3. ملف tasks/document_tasks.py:

```python
from celery_app import celery_app
from database.postgres_connection import AsyncSessionLocal
from services.rag_service import RAGService
from helpers.enhanced_text_processor import EnhancedTextProcessor
import logging

logger = logging.getLogger(__name__)

@celery_app.task(
    bind=True,
    name='process_document',
    max_retries=3,
    default_retry_delay=60
)
def process_document_task(self, document_id: int):
    """
    معالجة مستند في الخلفية
    
    Args:
        document_id: معرف المستند
    """
    try:
        logger.info(f"📄 Processing document {document_id}")
        
        # تحديث الحالة
        self.update_state(
            state='PROCESSING',
            meta={'document_id': document_id, 'step': 'extracting_text'}
        )
        
        # المعالجة (سنحتاج لـ async wrapper)
        import asyncio
        loop = asyncio.get_event_loop()
        
        async def process():
            async with AsyncSessionLocal() as session:
                # الحصول على المستند
                from database.repositories.document_repository_pg import DocumentRepositoryPG
                doc_repo = DocumentRepositoryPG(session)
                
                document = await doc_repo.get_by_id(document_id)
                if not document:
                    raise Exception("Document not found")
                
                # استخراج ومعالجة
                processor = EnhancedTextProcessor()
                file_ext = os.path.splitext(document.filename)[1]
                
                result = await processor.process_with_progress(
                    document.file_path,
                    file_ext,
                    str(document_id)
                )
                
                # حفظ chunks
                from database.models import Chunk
                for chunk_data in result['chunks']:
                    chunk = Chunk(
                        document_id=document_id,
                        chunk_id=chunk_data['chunk_id'],
                        text=chunk_data['text'],
                        start_char=chunk_data['start_char'],
                        end_char=chunk_data['end_char'],
                        length=chunk_data['length']
                    )
                    session.add(chunk)
                
                # تحديث المستند
                await doc_repo.update(document_id, {
                    'status': 'processed',
                    'clean_text': result['clean_text'],
                    'chunk_count': len(result['chunks'])
                })
                
                await session.commit()
                
                logger.info(f"✅ Document {document_id} processed successfully")
                
                return {
                    'document_id': document_id,
                    'chunks_created': len(result['chunks'])
                }
        
        result = loop.run_until_complete(process())
        return result
        
    except Exception as exc:
        logger.error(f"❌ Error processing document {document_id}: {str(exc)}")
        
        # Retry
        raise self.retry(exc=exc)

@celery_app.task(name='delete_document')
def delete_document_task(document_id: int):
    """
    حذف مستند وملفاته
    """
    try:
        logger.info(f"🗑️ Deleting document {document_id}")
        
        import asyncio
        loop = asyncio.get_event_loop()
        
        async def delete():
            async with AsyncSessionLocal() as session:
                doc_repo = DocumentRepositoryPG(session)
                
                document = await doc_repo.get_by_id(document_id)
                if document:
                    # حذف الملف
                    import os
                    if os.path.exists(document.file_path):
                        os.remove(document.file

---

# اكمل

سأكمل الشرح التفصيلي لآخر فيديوهين:

***

## 🎬 الفيديو الرابع والعشرون: Celery Basics 1/2 (تكملة)

#### 3. ملف tasks/document_tasks.py (تكملة):
```python
                if document:
                    # حذف الملف
                    import os
                    if os.path.exists(document.file_path):
                        os.remove(document.file_path)
                    
                    # حذف من قاعدة البيانات
                    await doc_repo.delete(document_id)
                    await session.commit()
                
                logger.info(f"✅ Document {document_id} deleted")
        
        loop.run_until_complete(delete())
        return {'document_id': document_id, 'status': 'deleted'}
        
    except Exception as exc:
        logger.error(f"❌ Error deleting document: {str(exc)}")
        raise
```


#### 4. ملف tasks/embedding_tasks.py:

```python
from celery_app import celery_app
from database.postgres_connection import AsyncSessionLocal
from services.pgvector_service import PGVectorService
import logging

logger = logging.getLogger(__name__)

@celery_app.task(
    bind=True,
    name='create_embeddings',
    max_retries=3
)
def create_embeddings_task(self, document_id: int):
    """
    إنشاء embeddings للمستند
    """
    try:
        logger.info(f"🧠 Creating embeddings for document {document_id}")
        
        import asyncio
        loop = asyncio.get_event_loop()
        
        async def create_embeddings():
            async with AsyncSessionLocal() as session:
                pgvector_service = PGVectorService(session)
                
                count = await pgvector_service.embed_and_store_chunks(document_id)
                await session.commit()
                
                return count
        
        count = loop.run_until_complete(create_embeddings())
        
        logger.info(f"✅ Created {count} embeddings for document {document_id}")
        
        return {
            'document_id': document_id,
            'embeddings_created': count
        }
        
    except Exception as exc:
        logger.error(f"❌ Error creating embeddings: {str(exc)}")
        raise self.retry(exc=exc)
```


#### 5. تحديث routes/documents.py لاستخدام Celery:

```python
from tasks.document_tasks import process_document_task
from tasks.embedding_tasks import create_embeddings_task

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    رفع مستند - المعالجة ستتم عبر Celery
    """
    try:
        # حفظ الملف
        file_path, file_size = await file_handler.save_file(file)
        
        # إنشاء في قاعدة البيانات
        async with AsyncSessionLocal() as session:
            doc_repo = DocumentRepositoryPG(session)
            
            document = await doc_repo.create({
                'filename': file.filename,
                'file_path': file_path,
                'file_size': file_size,
                'content_type': file.content_type,
                'status': 'uploaded'
            })
            
            await session.commit()
            document_id = document.id
        
        # إرسال للمعالجة عبر Celery
        task = process_document_task.delay(document_id)
        
        return {
            "id": document_id,
            "filename": file.filename,
            "status": "uploaded",
            "task_id": task.id,
            "message": "Document uploaded, processing in background"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    الحصول على حالة المهمة
    """
    from celery.result import AsyncResult
    
    task = AsyncResult(task_id, app=celery_app)
    
    return {
        "task_id": task_id,
        "status": task.state,
        "result": task.result if task.ready() else None,
        "info": task.info
    }
```


#### 6. تشغيل Celery:

```bash
# Worker للمستندات
$ celery -A celery_app worker -Q documents -l info -n documents@%h

# Worker للـ embeddings
$ celery -A celery_app worker -Q embeddings -l info -n embeddings@%h

# Flower للـ monitoring
$ celery -A celery_app flower --port=5555
```


***

## 🎬 الفيديو الخامس والعشرون: Advanced Celery 2/2 (1:49 ساعة)

**Branch:** `tut-017`

### الأهداف التعليمية:

- Celery Beat للمهام المجدولة
- Task chains و groups
- Error handling متقدم
- Performance optimization


### الأكواد التفصيلية:

#### 1. ملف tasks/cleanup_tasks.py:

```python
from celery_app import celery_app
from database.postgres_connection import AsyncSessionLocal
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

@celery_app.task(name='cleanup_old_documents')
def cleanup_old_documents_task():
    """
    حذف المستندات القديمة (أكثر من 30 يوم)
    """
    try:
        logger.info("🧹 Starting cleanup of old documents")
        
        import asyncio
        loop = asyncio.get_event_loop()
        
        async def cleanup():
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select
                from database.models import Document
                
                # العثور على المستندات القديمة
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                result = await session.execute(
                    select(Document).where(
                        Document.upload_date < cutoff_date,
                        Document.status == 'failed'
                    )
                )
                
                old_documents = result.scalars().all()
                
                deleted_count = 0
                for doc in old_documents:
                    # حذف الملف
                    if os.path.exists(doc.file_path):
                        os.remove(doc.file_path)
                    
                    # حذف من قاعدة البيانات
                    await session.delete(doc)
                    deleted_count += 1
                
                await session.commit()
                
                logger.info(f"✅ Cleaned up {deleted_count} old documents")
                return deleted_count
        
        count = loop.run_until_complete(cleanup())
        return {'deleted': count}
        
    except Exception as exc:
        logger.error(f"❌ Cleanup error: {str(exc)}")
        raise

@celery_app.task(name='generate_statistics')
def generate_statistics_task():
    """
    توليد إحصائيات يومية
    """
    try:
        logger.info("📊 Generating statistics")
        
        import asyncio
        loop = asyncio.get_event_loop()
        
        async def generate_stats():
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select, func
                from database.models import Document, Chunk
                
                # إحصائيات المستندات
                total_docs = await session.execute(
                    select(func.count(Document.id))
                )
                total_docs = total_docs.scalar()
                
                # إحصائيات chunks
                total_chunks = await session.execute(
                    select(func.count(Chunk.id))
                )
                total_chunks = total_chunks.scalar()
                
                # المستندات المعالجة اليوم
                today = datetime.utcnow().date()
                processed_today = await session.execute(
                    select(func.count(Document.id)).where(
                        func.date(Document.processed_at) == today
                    )
                )
                processed_today = processed_today.scalar()
                
                stats = {
                    'date': str(today),
                    'total_documents': total_docs,
                    'total_chunks': total_chunks,
                    'processed_today': processed_today
                }
                
                logger.info(f"📊 Stats: {stats}")
                return stats
        
        return loop.run_until_complete(generate_stats())
        
    except Exception as exc:
        logger.error(f"❌ Stats error: {str(exc)}")
        raise
```


#### 2. إعداد Celery Beat:

```python
# في celery_app.py
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    # تنظيف يومي في الساعة 2 صباحاً
    'cleanup-daily': {
        'task': 'cleanup_old_documents',
        'schedule': crontab(hour=2, minute=0),
    },
    
    # إحصائيات يومية في الساعة 23:55
    'daily-statistics': {
        'task': 'generate_statistics',
        'schedule': crontab(hour=23, minute=55),
    },
    
    # فحص صحة كل ساعة
    'health-check': {
        'task': 'system_health_check',
        'schedule': crontab(minute=0),  # كل ساعة
    },
}
```


#### 3. Task Chains - معالجة متسلسلة:

```python
from celery import chain, group

@celery_app.task(name='full_document_pipeline')
def full_document_pipeline_task(document_id: int):
    """
    Pipeline كامل للمستند
    """
    # إنشاء chain من المهام
    pipeline = chain(
        process_document_task.s(document_id),
        create_embeddings_task.s(),
        notify_completion_task.s()
    )
    
    return pipeline.apply_async()

@celery_app.task(name='notify_completion')
def notify_completion_task(result: dict):
    """
    إشعار بإتمام المعالجة
    """
    logger.info(f"🎉 Pipeline completed: {result}")
    
    # يمكن إرسال email أو webhook هنا
    return {'status': 'notified', 'result': result}
```


#### 4. Task Groups - معالجة متوازية:

```python
@celery_app.task(name='batch_process_documents')
def batch_process_documents_task(document_ids: list):
    """
    معالجة عدة مستندات بالتوازي
    """
    # إنشاء group من المهام
    job = group(
        process_document_task.s(doc_id)
        for doc_id in document_ids
    )
    
    return job.apply_async()
```


#### 5. Error Handling المتقدم:

```python
from celery.exceptions import SoftTimeLimitExceeded

@celery_app.task(
    bind=True,
    name='safe_process_document',
    max_retries=5,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,  # 10 دقائق
    retry_jitter=True
)
def safe_process_document_task(self, document_id: int):
    """
    معالجة مع error handling متقدم
    """
    try:
        # المعالجة
        result = process_document_task(document_id)
        return result
        
    except SoftTimeLimitExceeded:
        # تم تجاوز الوقت المحدد
        logger.warning(f"⏰ Task timeout for document {document_id}")
        
        # تحديث حالة المستند
        # ... update status to 'timeout'
        
        # عدم إعادة المحاولة
        raise
        
    except Exception as exc:
        # log التفاصيل
        logger.error(
            f"❌ Error in attempt {self.request.retries + 1}: {str(exc)}",
            exc_info=True
        )
        
        # إذا وصلنا للحد الأقصى من المحاولات
        if self.request.retries >= self.max_retries:
            logger.error(f"🚫 Max retries reached for document {document_id}")
            
            # تحديث حالة المستند
            # ... update status to 'failed'
            
            return {'status': 'failed', 'error': str(exc)}
        
        # إعادة المحاولة
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
```


#### 6. Monitoring مع Flower:

```python
# ملف flowerconfig.py
import os

# Basic Auth
basic_auth = [os.getenv('FLOWER_USER', 'admin') + ':' + os.getenv('FLOWER_PASSWORD', 'admin')]

# Broker URL
broker_api = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Port
port = 5555

# Enable events
persistent = True

# Database
db = 'flower.db'
```


#### 7. تحديث docker-compose.prod.yml:

```yaml
  # Celery Workers
  celery-documents:
    build: .
    command: celery -A celery_app worker -Q documents -l info -n documents@%h
    restart: always
    depends_on:
      - redis
      - postgres
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    volumes:
      - ./uploads:/app/uploads
    networks:
      - minirag-network

  celery-embeddings:
    build: .
    command: celery -A celery_app worker -Q embeddings -l info -n embeddings@%h
    restart: always
    depends_on:
      - redis
      - postgres
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    networks:
      - minirag-network

  # Celery Beat
  celery-beat:
    build: .
    command: celery -A celery_app beat -l info
    restart: always
    depends_on:
      - redis
      - postgres
    environment

---

# اكمل

سأكمل وأختم الشرح بملخص شامل:

***

## 🎬 الفيديو الخامس والعشرون: Advanced Celery (تكملة)

#### 7. تحديث docker-compose.prod.yml (تكملة):
```yaml
  celery-beat:
    build: .
    command: celery -A celery_app beat -l info
    restart: always
    depends_on:
      - redis
      - postgres
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    networks:
      - minirag-network

  # Flower Dashboard
  flower:
    build: .
    command: celery -A celery_app flower --conf=flowerconfig.py
    restart: always
    ports:
      - "5555:5555"
    depends_on:
      - redis
      - celery-documents
    environment:
      - REDIS_URL=${REDIS_URL}
      - FLOWER_USER=${FLOWER_USER}
      - FLOWER_PASSWORD=${FLOWER_PASSWORD}
    networks:
      - minirag-network
```


#### 8. Performance Optimization:

```python
# في celery_app.py

# تحسينات الأداء
celery_app.conf.update(
    # استخدام compression للنتائج
    result_compression='gzip',
    
    # تقليل حجم الرسائل
    task_compression='gzip',
    
    # تحسين connection pooling
    broker_pool_limit=10,
    
    # Task acknowledgment
    task_acks_late=True,
    
    # Worker optimization
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=100,
    
    # Result backend optimization
    result_expires=3600,  # ساعة واحدة
    result_backend_transport_options={
        'master_name': 'mymaster',
    },
)
```


***

# 📊 الملخص الشامل النهائي لسلسلة mini-RAG

## 🎯 ما تم إنجازه في 25 فيديو:

### **المرحلة الأولى: Foundation (الفيديوهات 1-10)**

| المرحلة | الفيديوهات | الإنجازات |
| :-- | :-- | :-- |
| **Setup** | 1-3 | ✅ فهم المشروع<br>✅ إعداد بيئة العمل<br>✅ Python + Conda + Tools |
| **Backend** | 4-6 | ✅ FastAPI setup<br>✅ Project architecture<br>✅ Routes \& config |
| **Files** | 7-8 | ✅ File upload<br>✅ Text extraction<br>✅ Chunking |
| **Database** | 9-10 | ✅ MongoDB + Docker<br>✅ Models \& schemas<br>✅ CRUD operations |

### **المرحلة الثانية: RAG Core (الفيديوهات 11-19)**

| المرحلة | الفيديوهات | الإنجازات |
| :-- | :-- | :-- |
| **Optimization** | 11-13 | ✅ Database indexes<br>✅ Pipeline enhancements<br>✅ Checkpoint \& fixes |
| **AI Integration** | 14-15 | ✅ OpenAI LLM<br>✅ QDrant vector DB<br>✅ Embeddings |
| **Search** | 16-17 | ✅ Semantic search<br>✅ Hybrid search<br>✅ RAG answers |
| **Advanced** | 18-19 | ✅ Ollama local LLM<br>✅ Cost optimization |

### **المرحلة الثالثة: Production (الفيديوهات 20-25)**

| المرحلة | الفيديوهات | الإنجازات |
| :-- | :-- | :-- |
| **Migration** | 20-21 | ✅ PostgreSQL<br>✅ PGVector<br>✅ Alembic migrations |
| **Deployment** | 22-23 | ✅ Docker optimization<br>✅ Nginx + SSL<br>✅ Monitoring |
| **Scalability** | 24-25 | ✅ Celery workers<br>✅ Task queues<br>✅ Scheduling |


***

## 🏗️ البنية النهائية الكاملة للمشروع:

```
mini-rag/
├── main.py                          # نقطة الدخول
├── celery_app.py                    # Celery configuration
├── config.py                        # إعدادات التطبيق
├── Dockerfile                       # Docker image
├── docker-compose.prod.yml          # Production setup
├── requirements.txt                 # Python dependencies
├── alembic.ini                      # Migration config
│
├── database/
│   ├── postgres_connection.py      # Database connection
│   ├── models.py                    # SQLAlchemy models
│   ├── indexes.py                   # Database indexes
│   └── repositories/
│       ├── document_repository_pg.py
│       └── chunk_repository.py
│
├── models/
│   ├── llm_models.py               # LLM Pydantic models
│   ├── rag_models.py               # RAG models
│   └── processing_status.py        # Progress tracking
│
├── llm/
│   ├── base_llm.py                 # Abstract LLM
│   ├── openai_llm.py               # OpenAI implementation
│   ├── ollama_llm.py               # Ollama implementation
│   └── llm_factory.py              # Factory pattern
│
├── vectordb/
│   ├── base_vectordb.py            # Abstract vector DB
│   ├── qdrant_db.py                # QDrant implementation
│   └── vectordb_factory.py         # Factory pattern
│
├── services/
│   ├── embedding_service.py        # Embeddings management
│   ├── search_service.py           # Hybrid search
│   ├── rag_service.py              # RAG pipeline
│   └── pgvector_service.py         # PGVector operations
│
├── routes/
│   ├── documents.py                # Document endpoints
│   ├── search.py                   # Search endpoints
│   ├── llm.py                      # LLM endpoints
│   └── rag.py                      # RAG endpoints
│
├── tasks/
│   ├── document_tasks.py           # Document processing
│   ├── embedding_tasks.py          # Embedding creation
│   └── cleanup_tasks.py            # Maintenance tasks
│
├── helpers/
│   ├── file_handler.py             # File operations
│   ├── text_processor.py           # Text processing
│   ├── enhanced_text_processor.py  # Advanced processing
│   └── retry_handler.py            # Retry logic
│
├── monitoring/
│   └── metrics.py                  # Prometheus metrics
│
├── tests/
│   ├── test_documents.py
│   ├── test_search.py
│   └── test_rag.py
│
├── docker/
│   ├── docker-compose.yml          # Development
│   └── .env.example
│
├── nginx/
│   ├── nginx.conf                  # Reverse proxy
│   └── ssl/
│
├── prometheus/
│   └── prometheus.yml
│
├── grafana/
│   └── dashboards/
│
└── scripts/
    ├── backup.sh
    └── migrate_qdrant_to_pgvector.py
```


***

## 🎓 المهارات المكتسبة:

### **Backend Development:**

- ✅ FastAPI framework
- ✅ Async/await في Python
- ✅ REST API design
- ✅ Authentication \& Authorization
- ✅ Error handling


### **Database:**

- ✅ MongoDB (NoSQL)
- ✅ PostgreSQL (SQL)
- ✅ SQLAlchemy ORM
- ✅ Database migrations (Alembic)
- ✅ Indexes \& optimization


### **AI \& ML:**

- ✅ Large Language Models
- ✅ Embeddings \& vectors
- ✅ Semantic search
- ✅ RAG architecture
- ✅ Prompt engineering


### **DevOps:**

- ✅ Docker \& Docker Compose
- ✅ Nginx reverse proxy
- ✅ CI/CD pipelines
- ✅ Monitoring (Prometheus + Grafana)
- ✅ Logging \& debugging


### **Production:**

- ✅ Task queues (Celery)
- ✅ Caching (Redis)
- ✅ Scalability patterns
- ✅ Security best practices
- ✅ Backup strategies

***

## 🚀 الـ APIs النهائية (كاملة):

```bash
# Documents Management
POST   /documents/upload              # رفع مستند
GET    /documents/                    # قائمة المستندات
GET    /documents/{id}                # مستند محدد
DELETE /documents/{id}                # حذف مستند
GET    /documents/{id}/chunks         # chunks المستند
GET    /documents/{id}/text           # النص الكامل
GET    /documents/{id}/progress       # تقدم المعالجة
POST   /documents/{id}/embed          # إنشاء embeddings
GET    /documents/stats/summary       # إحصائيات

# Search
GET    /search/                       # بحث هجين
GET    /search/semantic               # بحث دلالي

# LLM Operations
POST   /llm/chat                      # محادثة
POST   /llm/chat/stream              # محادثة مع streaming
POST   /llm/embedding                # إنشاء embedding
POST   /llm/tokens/count             # حساب tokens

# RAG
POST   /rag/answer                    # توليد إجابة
POST   /rag/answer/stream            # إجابة مع streaming

# Tasks
GET    /tasks/{task_id}              # حالة المهمة

# Monitoring
GET    /health                        # فحص الصحة
GET    /metrics                       # Prometheus metrics

# Admin
GET    /admin/indexes                 # عرض indexes
POST   /admin/reindex                 # إعادة بناء indexes
```


***

## 💡 أفضل الممارسات المتعلمة:

### **1. Code Organization:**

```python
✅ Repository Pattern للـ database
✅ Factory Pattern للـ LLMs
✅ Service Layer للـ business logic
✅ Dependency Injection
```


### **2. Error Handling:**

```python
✅ Try-except في كل async function
✅ Custom exceptions
✅ Proper logging
✅ Retry mechanisms
```


### **3. Performance:**

```python
✅ Database indexes
✅ Connection pooling
✅ Caching مع Redis
✅ Async operations
✅ Background tasks
```


### **4. Security:**

```python
✅ Environment variables للـ secrets
✅ Input validation
✅ Rate limiting
✅ HTTPS + SSL
✅ CORS configuration
```


***

## 🎉 التهاني النهائية!

**أنت الآن لديك:**

- ✅ **Production-ready RAG application** كامل
- ✅ **فهم عميق** لـ Full Stack AI Development
- ✅ **خبرة عملية** في بناء تطبيقات AI
- ✅ **مهارات** قابلة للتطبيق في السوق
- ✅ **Portfolio project** قوي

**الخطوات القادمة:**

1. 🔧 **طور المشروع** - أضف features خاصة بك
2. 🚀 **انشره** - ضعه على cloud (AWS, GCP, Azure)
3. 📝 **وثّقه** - اكتب مقالات عن ما تعلمت
4. 💼 **شاركه** - أضفه لـ portfolio وLinkedIn
5. 🎓 **علّم غيرك** - شارك المعرفة مع المجتمع

***

## 📚 مصادر إضافية للتعلم:

- **FastAPI:** https://fastapi.tiangolo.com/
- **Celery:** https://docs.celeryq.dev/
- **PostgreSQL:** https://www.postgresql.org/docs/
- **OpenAI:** https://platform.openai.com/docs
- **LangChain:** https://python.langchain.com/
- **QDrant:** https://qdrant.tech/documentation/

***

**🌟 أتمنى أن يكون هذا الشرح قد أفادك في رحلتك لتعلم RAG و Full Stack AI Development!**

**بالتوفيق في مشاريعك القادمة! 🚀💪**

