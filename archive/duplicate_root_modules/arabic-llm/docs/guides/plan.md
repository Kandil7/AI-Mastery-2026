تقدر تضيف كل الأدوار والمهارات اللي اقترحناها ما عدا أي حاجة متعلقة بـOCR، فهنشتغل على نسخة منقّحة من اللي فات.

## ✅ Implementation Status

**Status**: ✅ **COMPLETE** (March 26, 2026)

All roles and skills have been implemented in `arabic_llm/core/schema_enhanced.py`:
- ✅ 4 new roles added to `Role` enum
- ✅ 5 new skills added to `Skill` enum
- Total: 19 roles, 45+ skills

---

## الأدوار الجديدة (Roles) بدون OCR

### 1) DATAENGINEER_AR
- الوصف: مهندس بيانات عربي يحول النصوص العربية الخام لهيكل منظم (JSON / جداول) ويستخرج كيانات وعلاقات مفيدة للـRAG والأنظمة المعرفية.  
- أمثلة مهام:  
  - استخراج آيات/أحاديث/اقتباسات مع مراجعها.  
  - تلخيص كتاب عربي في شكل outline منظم.

### 2) RAG_ASSISTANT
- الوصف: مساعد مخصص للإجابة المعتمدة على مصادر (documents / آيات / أحاديث) مع إرجاع شواهد واضحة.  
- أمثلة مهام:  
  - سؤال وجواب مع citations.  
  - مقارنة بين قولين مع توثيق المصادر.

### 3) EDTECH_TUTOR
- الوصف: معلّم عربي متخصص في مناهج اللغة العربية الحديثة (مدارس/جامعات) يشرح بطريقة مبسطة وبتمارين.  
- أمثلة مهام:  
  - شرح درس نحو من كتاب مدرسي.  
  - وضع أسئلة اختيار من متعدد مع نموذج إجابة.

### 4) FATWA_ASSISTANT_SAFE
- الوصف: مساعد فتاوى حذر، يذكر أقوال المذاهب، يوضّح أنه ليس بديلاً عن المفتي، ويشير للمراجع الرسمية عند المسائل الحساسة.  
- أمثلة مهام:  
  - تلخيص أقوال المذاهب في مسألة.  
  - إرشاد المستخدم للتواصل مع دار الإفتاء.

(تقدر تضيف دول إلى `RoleEnum` بجانب الموجود في الوثيقة مثل tutor, faqih, muhaddith, إلخ.) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/57452145/35407437-d68b-4fd5-b357-6cc0f0c705c5/COMPLETE_DOCUMENTATION.md)

## المهارات الجديدة (Skills) بدون OCR

### 1) ERROR_ANALYSIS_AR
- الوصف: تحليل أخطاء الكتابة بالعربية (نحو، صرف، إملاء، أسلوب) مع شرح السبب والتصحيح.  
- استخدام: أدوات تصحيح عربي، feedback تعليمي.

### 2) RAG_GROUNDED_ANSWERING
- الوصف: توليد إجابات مبنية على مقاطع معطاة (context chunks)، مع إرجاع الشواهد كمراجع في آخر الجواب.  
- استخدام: أي RAG system عندك (إسلامي، تعليمي، قانوني).

### 3) CURRICULUM_ALIGNED_AR
- الوصف: ربط الشرح والتمارين بمناهج معينة (مثلاً “منهج اللغة العربية للصف الثالث الثانوي – مصر”).  
- استخدام: LMS، منصات تعليمية، مدرس خصوصي آلي.

### 4) DIALECT_HANDLING_EGY
- الوصف: فهم العامية المصرية وتحويلها لفصحى ثم المعالجة عليها، أو الرد بالعامية مع الحفاظ على دقة المحتوى.  
- استخدام: شات بوت مصري، دعم عملاء، تطبيقات شبابية.

### 5) LEGAL_ARABIC_DRAFTING
- الوصف: صياغة خطابات رسمية، شكاوى، وعقود مبسطة بالعربية بصياغة منضبطة.  
- استخدام: مساعد إداري/قانوني عربي.

(تضيف هذه إلى `SkillEnum` بجانب skills الحالية مثل nahw, balagha, fiqh, tafsir….) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/57452145/35407437-d68b-4fd5-b357-6cc0f0c705c5/COMPLETE_DOCUMENTATION.md)

## شكل التعريف في الكود (مقتطف جاهز)

### RoleEnum (إضافة للأصلي عندك)
```python
class RoleEnum(str, Enum):
    # موجودة بالفعل في المشروع
    TUTOR = "tutor"
    PROOFREADER = "proofreader"
    POET = "poet"
    MUHHAQIQ = "muhhaqiq"
    ASSISTANT_GENERAL = "assistant_general"
    FAQIH = "faqih"
    MUHADDITH = "muhaddith"
    MUFASSIR = "mufassir"
    AQEEDAH_SPECIALIST = "aqeedah_specialist"
    SUFI = "sufi"
    HISTORIAN = "historian"
    GENEALOGIST = "genealogist"
    GEOGRAPHER = "geographer"
    PHYSICIAN = "physician"
    LOGICIAN = "logician"
    ADAB_SPECIALIST = "adab_specialist"
    QURAN_RECITER = "quran_reciter"

    # جديدة
    DATAENGINEER_AR = "dataengineer_ar"
    RAG_ASSISTANT = "rag_assistant"
    EDTECH_TUTOR = "edtech_tutor"
    FATWA_ASSISTANT_SAFE = "fatwa_assistant_safe"
```

### SkillEnum (إضافة للأصلي عندك)
```python
class SkillEnum(str, Enum):
    # لغويات (من المشروع)
    NAHW = "nahw"
    SARF = "sarf"
    BALAGHA = "balagha"
    ORTHOGRAPHY = "orthography"
    PHONOLOGY = "phonology"
    SEMANTICS = "semantics"
    LEXICOGRAPHY = "lexicography"
    QIRAAT = "qiraat"

    # علوم شرعية (من المشروع)
    FIQH = "fiqh"
    USUL_FIQH = "usul_fiqh"
    HADITH = "hadith"
    HADITH_MUSTALAH = "hadith_mustalah"
    TAFSIR = "tafsir"
    AQEEDAH = "aqeedah"
    SECTS = "sects"
    TASAWWUF = "tasawwuf"
    ZAKAT = "zakat"
    INHERITANCE = "inheritance"
    FATWA = "fatwa"
    JUDICIAL = "judicial"
    # ... + باقي الـskills في الوثيقة

    # جديدة
    ERROR_ANALYSIS_AR = "error_analysis_ar"
    RAG_GROUNDED_ANSWERING = "rag_grounded_answering"
    CURRICULUM_ALIGNED_AR = "curriculum_aligned_ar"
    DIALECT_HANDLING_EGY = "dialect_handling_egy"
    LEGAL_ARABIC_DRAFTING = "legal_arabic_drafting"
```

