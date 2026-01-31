# Internationalization (i18n) Guide
# دليل التدويل

## Table of Contents / جدول المحتويات

1. [Introduction / مقدمة](#introduction)
2. [Why i18n? / لماذا التدويل؟](#why-i18n)
3. [Language Detection / كشف اللغة](#language-detection)
4. [Translation Management / إدارة الترجمات](#translation-management)
5. [RTL Support / دعم RTL](#rtl-support)
6. [Best Practices / أفضل الممارسات](#best-practices)
7. [Summary / الملخص](#summary)

---

## Introduction / مقدمة

### What is i18n? / ما هو التدويل؟

**Internationalization (i18n)** is the process of designing software applications so that they can be adapted to different languages and regions without requiring engineering changes.

**Key components:**
- **Translation**: Converting text from one language to another
- **Locale-aware formatting**: Dates, times, numbers, currency
- **Directionality**: LTR (left-to-right) vs RTL (right-to-left)
- **Character encoding**: Unicode support for multiple scripts

**التدويل** هو عملية تصميم تطبيقات البرمجيات بحيث يمكن تكييفها مع لغات ومناطق مختلفة دون الحاجة إلى تغييرات هندسية.

**المكونات الرئيسية:**
- **الترجمة**: تحويل النص من لغة إلى أخرى
- **التنسيق المحلي**: التواريخ والأوقات والأرقام والعملات
- **الاتجاهية**: LTR (من اليسار لليمين) مقابل RTL (من اليمين لليسار)
- **ترميز الأحرف**: دعم يونيكود للنصوص المتعددة

---

## Why i18n? / لماذا التدويل؟

### Benefits / الفوائد

**1. Global Reach / وصول عالمي**
- Support users in their native language
- Expand to new markets without code changes
- Improve user engagement and satisfaction

**2. Compliance / الامتثال**
- Legal requirements in some regions
- Accessibility standards (WCAG)
- Data privacy regulations

**3. User Experience / تجربة المستخدم**
- Reduced cognitive load (read in native language)
- Better cultural context
- Proper text direction handling

### الفوائد

**1. وصول عالمي**
**2. الامتثال**
**3. تجربة المستخدم**

---

## Language Detection / كشف اللغة

### Detection Priority / أولوية الكشف

```python
def detect_language(
    headers: Dict[str, str],
    query_params: Dict[str, str],
    user_preference: Optional[str],
) -> Language:
    """
    Detect language with priority-based strategy.
    كشف اللغة باستخدام استراتيجية قائمة على الأولوية.
    """
    # Priority 1: Query parameter (explicit user choice)
    if query_params and "lang" in query_params:
        return parse_language(query_params["lang"])

    # Priority 2: Accept-Language header (browser preference)
    if headers and "Accept-Language" in headers:
        return parse_accept_language(headers["Accept-Language"])

    # Priority 3: User preference (saved in database)
    if user_preference:
        return parse_language(user_preference)

    # Priority 4: Content analysis (detect from request content)
    return detect_from_content()
```

### Priority Levels / مستويات الأولوية

| Priority | Source | Example | Reliability |
|----------|--------|---------|-------------|
| 1 | Query parameter | `?lang=ar` | Highest (explicit) |
| 2 | Accept-Language header | `Accept-Language: ar-SA,en-US;q=0.9` | High (user preference) |
| 3 | User preference | Database field | High (saved preference) |
| 4 | Content analysis | Text detection algorithm | Medium (best effort) |

### مستويات الأولوية

---

## Translation Management / إدارة الترجمات

### Translation Key Structure / هيكلية مفاتح الترجمة

```python
# Organize translations by category
TRANSLATIONS = {
    "common": {
        "welcome": "Welcome",
        "loading": "Loading...",
        "success": "Success",
        "error": "Error",
    },
    "documents": {
        "uploaded": "Document uploaded",
        "indexing": "Indexing...",
        "deleted": "Document deleted",
    },
    "chat": {
        "new_session": "New chat session",
        "title_generated": "Title generated",
        "summarized": "Session summarized",
    },
}
```

### Interpolation / الاستيفال

```python
# Define templates with placeholders
template = "Hello, {name}! You have {count} messages."

# Interpolate with variables
message = template.format(name="John", count=5)
# Result: "Hello, John! You have 5 messages."
```

### هيكلية مفاتح الترجمة

### الاستيفال

---

## RTL Support / دعم RTL

### RTL Languages / اللغات RTL

Languages that use right-to-left text direction:

**Arabic / العربية**
- 22 countries
- 400+ million speakers
- Uses Arabic script

**Hebrew / العبرية**
- Israel
- 9 million speakers
- Uses Hebrew script

**Farsi (Persian) / الفارسية**
- Iran, Afghanistan, Tajikistan
- 110+ million speakers
- Uses Persian script

**Urdu / الأردية**
- Pakistan, India
- 100+ million speakers
- Uses Perso-Arabic script

### CSS Directionality / اتجاهية CSS

```css
/* LTR (default) */
body {
    direction: ltr;
    text-align: left;
}

/* RTL for Arabic, Hebrew, etc. */
[dir="rtl"] body {
    direction: rtl;
    text-align: right;
}

/* Layout adjustment */
[dir="rtl"] .sidebar {
    margin-left: auto;
    margin-right: 20px;
}
```

### Layout Considerations / اعتبارات التخطيط

**Mirroring / عكس:**
- Flip left/right margins
- Flip flex directions
- Mirror icons and arrows

**Iconography / الأيقونات:**
- Use SVG for easy flipping
- Provide RTL-specific icons if needed
- Avoid directional glyphs in text (→, ←)

**Spacing / المسافات:**
- Maintain consistent spacing
- Consider character width differences
- Test with actual text content

### اللغات RTL

### اتجاهية CSS

### اعتبارات التخطيط

---

## Best Practices / أفضل الممارسات

### 1. Use Translation Keys / استخدم مفاتح الترجمة

```python
# GOOD: Use keys for maintainability
message = i18n.get_translation(Language.ARABIC, "document_uploaded")

# BAD: Hardcode translations
message = "تم تحميل المستند بنجاح"
```

### 2. Externalize Strings / استخرج السلاسل النصية

**Avoid hardcoding UI text:**

```python
# Route with i18n integration
@router.post("/upload")
async def upload_document(
    file: UploadFile,
    lang: str = Query(default="en"),  # Language parameter
):
    # Detect language
    language = i18n.detect_language(lang=lang)
    
    # Get translated messages
    success_msg = i18n.get_translation(language, "document_uploaded")
    error_msg = i18n.get_translation(language, "upload_failed")
    
    try:
        # Upload logic
        return {"message": success_msg}
    except Exception:
        return {"error": error_msg}
```

### 3. Context-Aware Translations / ترجمات مدركة للسياق

```python
TRANSLATIONS = {
    "document_upload": {
        "success": "Document {filename} uploaded successfully",
        "error": "Failed to upload {filename}",
    },
}
```

### 4. Date and Number Formatting / تنسيق التواريخ والأرقام

```python
import locale

def format_date(date: datetime, language: str) -> str:
    """Format date according to locale."""
    locale.setlocale(locale.LC_TIME, f"{language}_UTF-8")
    return date.strftime("%B %d, %Y")  # e.g., "يناير 15, 2024"

def format_number(number: float, language: str) -> str:
    """Format number with locale-appropriate separators."""
    locale.setlocale(locale.LC_NUMERIC, f"{language}_UTF-8")
    return locale.format_string("%d", number)
```

### 1. استخدم مفاتح الترجمة

### 2. استخرج السلاسل النصية

### 3. ترجمات مدركة للسياق

### 4. تنسيق التواريخ والأرقام

---

## Summary / الملخص

### Key Takeaways / النقاط الرئيسية

1. **i18n is essential** for global products
2. **Language detection** should use priority-based strategy
3. **Translation organization** by categories improves maintainability
4. **RTL support** requires special CSS and layout handling
5. **Externalize strings** - avoid hardcoding
6. **Context-aware translations** with interpolation
7. **Locale formatting** for dates, numbers, currency

### النقاط الرئيسية

1. **التدويل ضروري** للمنتجات العالمية
2. **كشف اللغة** يجب استخدام استراتيجية قائمة على الأولوية
3. **تنظيم الترجمات** حسب الفئات يحسن الصيانة
4. **دعم RTL** يتطلب تعامل CSS خاص مع التخطيط
5. **استخراج السلاسل النصية** - تجنب التشفير الثابت
6. **الترجمات المدركة للسياق** مع الاستيفال
7. **التنسيق المحلي** للتواريخ والأرقام

---

## Further Reading / قراءة إضافية

- [Unicode Standard](https://unicode.org/)
- [MDN RTL Styling Guide](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Logical_Properties)
- [i18n Best Practices](https://www.w3.org/International/articles/)

---

## Arabic Summary / ملخص بالعربية

هذا الدليل يغطي التدويل في محرك RAG. تشمل المواضيع الرئيسية:

1. **ما هو التدويل**: تصميم البرمجيات لدعم لغات متعددة
2. **كشف اللغة**: استراتيجية قائمة على الأولوية
3. **إدارة الترجمات**: تنظيم حسب الفئات مع الاستيفال
4. **دعم RTL**: التعامل مع اللغات من اليمين لليسار
5. **أفضل الممارسات**: استخراج السلاسل، ترجمات مدركة للسياق، التنسيق المحلي

التدويل ضروري للوصول العالمي وتجربة مستخدم محسنة.
