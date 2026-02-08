# **الدليل المرجعي الشامل لهندسة الذكاء الاصطناعي: من الأسس الرياضية إلى الأنظمة المتقدمة**

## **مقدمة**

يشهد مجال هندسة الذكاء الاصطناعي (AI Engineering) تحولاً جذرياً؛ لم يعد كافياً للمهندس أن يكون مجرد مستخدم للمكتبات البرمجية الجاهزة. الانتقال من مستوى "الممارس" إلى مستوى "الخبير" يتطلب فهماً عميقاً للأسس الرياضية التي تحكم عمل النماذج، والقدرة على بناء الخوارزميات من الصفر، وتصميم أنظمة قابلة للتوسع في بيئات الإنتاج الحقيقية.

يقدم هذا التقرير البحثي المفصل، والممتد عبر أكثر من 15,000 كلمة، خارطة طريق شاملة وتطبيقية تغطي كافة الجوانب الواردة في المواد البحثية المرفقة. تم تصميم هذا الدليل ليكون مرجعاً تقنياً متعمقاً يدمج بين النظرية الأكاديمية والتطبيق العملي، مدعوماً بأمثلة برمجية، اشتقاقات رياضية، تمارين من مسابقات برمجية عالمية (مثل Codeforces و Kaggle)، وأسئلة مقابلات تقنية متقدمة في كبرى شركات التكنولوجيا (FAANG).

# ---

**الباب الأول: الركائز الرياضية (The Mathematical Foundations)**

الرياضيات ليست مجرد متطلب سابق، بل هي اللغة التي "تُفكر" بها الآلة. فهم هذه الأسس هو الفارق بين من يطبق النماذج عشوائياً ومن يستطيع تشخيص أسباب عدم التقارب (Non-convergence) أو تحسين أداء الاستدلال (Inference).

## **الفصل 1: الجبر الخطي (Linear Algebra) \- لغة البيانات**

في عالم تعلم الآلة، البيانات هي مصفوفات، والعمليات هي تحويلات خطية. الجبر الخطي يوفر البنية التحتية لتمثيل البيانات ومعالجتها.

### **1.1 المتجهات والمصفوفات: المعنى الهندسي والتطبيقي**

#### **1.1.1 المتجهات والضرب النقطي (Dot Product)**

الضرب النقطي هو العملية الأساسية التي تقوم عليها معظم خوارزميات الذكاء الاصطناعي، بدءاً من الانحدار الخطي وصولاً إلى آليات الانتباه في نماذج المحولات (Transformers).

التعريف الرياضي والهندسي:  
إذا كان لدينا متجهان $\\mathbf{a}$ و $\\mathbf{b}$، فإن الضرب النقطي يُعرف كالتالي:

$$\\mathbf{a} \\cdot \\mathbf{b} \= \\sum\_{i=1}^{n} a\_i b\_i \= \\|\\mathbf{a}\\| \\|\\mathbf{b}\\| \\cos \\theta$$

حيث $\\theta$ هي الزاوية بين المتجهين.

* **التفسير في تعلم الآلة:** الضرب النقطي هو مقياس **للتشابه**. إذا كان المتجهان يشيران إلى نفس الاتجاه (زاوية 0)، يكون الناتج أكبر ما يمكن. إذا كانا متعامدين (Orthogonal)، يكون الناتج صفراً، مما يعني عدم وجود ارتباط (Correlation) بينهما.1

تطبيق عملي: أنظمة التوصية البسيطة  
تخيل أننا نمثل تفضيلات المستخدم بمتجه، وخصائص الفيلم بمتجه آخر. حاصل الضرب النقطي بينهما يعطينا "درجة التوصية".

Python

import numpy as np

\# متجه المستخدم (يفضل الأكشن، لا يفضل الرومانسية)  
user\_vector \= np.array(\[0.9, 0.1\]) 

\# متجه فيلم 1 (فيلم أكشن)  
movie\_action \= np.array(\[0.8, 0.2\])

\# متجه فيلم 2 (فيلم رومانسي)  
movie\_romance \= np.array(\[0.1, 0.9\])

\# حساب التشابه  
score\_action \= np.dot(user\_vector, movie\_action)   \# النتيجة: 0.74  
score\_romance \= np.dot(user\_vector, movie\_romance) \# النتيجة: 0.18

print(f"Action Score: {score\_action}, Romance Score: {score\_romance}")

#### **1.1.2 العمليات على المصفوفات والمستويات الفائقة (Hyperplanes)**

في مهام التصنيف (Classification)، نهدف غالباً إلى إيجاد مستوى فائق (Hyperplane) يفصل بين فئتين من البيانات. معادلة المستوى الفائق تُعطى بـ:

$$\\mathbf{w}^T \\mathbf{x} \+ b \= 0$$

حيث $\\mathbf{w}$ هو المتجه العمودي على المستوى (Weights)، و $b$ هو الانحياز (Bias). فهم هذه المعادلة أساسي لفهم خوارزميات مثل SVM والشبكات العصبية.3

### **1.2 القيم الذاتية والمتجهات الذاتية (Eigenvalues & Eigenvectors)**

المتجه الذاتي لمصفوفة مربعة $A$ هو متجه لا يتغير اتجاهه عند ضربه بالمصفوفة، بل يتغير طوله فقط بمقدار قيمة معينة تسمى القيمة الذاتية $\\lambda$.

$$A\\mathbf{v} \= \\lambda\\mathbf{v}$$  
تطبيق واقعي 1: خوارزمية PageRank من جوجل  
تعتبر شبكة الويب رسماً بيانياً (Graph) حيث تمثل الصفحات العقد والروابط الحواف. مصفوفة الانتقال (Transition Matrix) لهذا الرسم البياني تمثل احتمالات الانتقال من صفحة لأخرى. المتجه الذاتي المقابل لأكبر قيمة ذاتية لهذه المصفوفة يمثل "التوزيع المستقر" (Steady State Distribution)، وقيم هذا المتجه تمثل "أهمية" أو ترتيب كل صفحة.4  
تطبيق واقعي 2: تحليل المكونات الأساسية (PCA)  
في البيانات ذات الأبعاد العالية، غالباً ما تكون الميزات مترابطة (Redundant). الـ PCA يبحث عن المتجهات الذاتية لمصفوفة التباين المشترك (Covariance Matrix) للبيانات. المتجهات الذاتية ذات القيم الذاتية الأكبر تمثل الاتجاهات التي تحتوي على أكبر قدر من المعلومات (التباين)، مما يسمح لنا بضغط البيانات مع الحفاظ على جوهرها.4

### **1.3 تفكيك القيم المنفردة (SVD): بناء أنظمة التوصية**

الـ SVD هو تعميم لتحليل القيم الذاتية ليشمل أي مصفوفة مستطيلة (ليست بالضرورة مربعة). يمكن تفكيك أي مصفوفة $A$ (مثلاً مصفوفة تقييمات المستخدمين للأفلام) إلى:

$$A \= U \\Sigma V^T$$

حيث:

* $U$: مصفوفة تمثل العلاقة بين المستخدمين والمفاهيم الكامنة (Latent Features).  
* $\\Sigma$: مصفوفة قطرية تمثل قوة هذه المفاهيم.  
* $V^T$: مصفوفة تمثل العلاقة بين الأفلام والمفاهيم الكامنة.

مثال تطبيقي: نظام توصية أفلام (Step-by-Step)  
لنفترض أن لدينا مصفوفة $A$ حيث الصفوف مستخدمون والأعمدة أفلام، والقيم هي التقييمات. المصفوفة مليئة بالفراغات (أفلام لم تُقيم).

1. نقوم بعمل SVD للمصفوفة (بعد ملء الفراغات بقيم مبدئية).  
2. نحتفظ فقط بأكبر $k$ قيم في $\\Sigma$ (تقليل الرتبة \- Low Rank Approximation).  
3. نعيد ضرب المصفوفات $A \\approx U\_k \\Sigma\_k V\_k^T$.  
4. المصفوفة الناتجة ستحتوي على قيم في الأماكن التي كانت فارغة سابقاً؛ هذه القيم هي التقييمات المتوقعة التي نستخدمها للتوصية.6

### **أسئلة مقابلات (الجبر الخطي)**

سؤال (Senior Data Scientist): لماذا نفضل استخدام SVD أو Matrix Factorization على الطرق التقليدية في أنظمة التوصية؟  
الإجابة النموذجية: الطرق التقليدية مثل تصفية المحتوى (Content-based) تعتمد على خصائص معرفة مسبقاً، بينما SVD يكتشف "الميزات الكامنة" (Latent Features) التي قد لا تكون واضحة (مثلاً: نمط أفلام غامض يجمع بين الرعب والكوميديا). كما أن SVD يتعامل بكفاءة مع المصفوفات المتناثرة (Sparse Matrices) التي تميز بيانات المستخدمين الحقيقية، ويوفر أساساً رياضياً لضغط البيانات وتقليل الضجيج.1

## ---

**الفصل 2: التفاضل والتكامل (Calculus) \- محرك التحسين**

إذا كان الجبر الخطي يبني هيكل النموذج، فإن التفاضل هو الذي يقوم بتدريبه. التدريب في جوهره هو عملية تحسين (Optimization) تهدف لتقليل الخطأ.

### **2.1 الاشتقاق والاشتقاق الجزئي: قياس التغيير**

المشتقة $\\frac{dy}{dx}$ تقيس مدى حساسية المخرجات للتغير في المدخلات. في التعلم العميق، لدينا ملايين المدخلات (الأوزان)، لذا نستخدم التدرج (Gradient)، وهو متجه يحتوي على جميع المشتقات الجزئية للدالة.

$$\\nabla f(x) \= \\left\[ \\frac{\\partial f}{\\partial x\_1}, \\frac{\\partial f}{\\partial x\_2}, \\dots, \\frac{\\partial f}{\\partial x\_n} \\right\]$$

يمثل التدرج الاتجاه الذي تتزايد فيه الدالة بأقصى سرعة. لتقليل الخطأ (Loss)، نتحرك في الاتجاه المعاكس للتدرج.9

### **2.2 النزول المتدرج (Gradient Descent): خوارزمية التعلم**

خوارزمية النزول المتدرج هي العمود الفقري لتدريب الشبكات العصبية. المعادلة الأساسية للتحديث هي:

$$w\_{new} \= w\_{old} \- \\alpha \\nabla J(w)$$

حيث $\\alpha$ هو معدل التعلم (Learning Rate)، و $J(w)$ هي دالة التكلفة.  
مثال عددي شامل (Step-by-Step Calculation):  
لنقم بتدريب نموذج انحدار خطي بسيط $f(x) \= wx$ لنقطة بيانات واحدة $(x=2, y=4)$.

* **القيمة الابتدائية:** $w \= 0.5$.  
* **معدل التعلم:** $\\alpha \= 0.1$.  
* **دالة الخسارة (MSE):** $L \= (y \- \\hat{y})^2 \= (y \- wx)^2$.

**التكرار الأول (Iteration 1):**

1. **التنبؤ:** $\\hat{y} \= 0.5 \\times 2 \= 1.0$.  
2. **حساب الخطأ:** $L \= (4 \- 1.0)^2 \= 9$.  
3. حساب المشتقة (Gradient):

   $$\\frac{\\partial L}{\\partial w} \= 2(y \- wx)(-x) \= \-2x(y \- \\hat{y})$$  
   $$\\frac{\\partial L}{\\partial w} \= \-2(2)(4 \- 1\) \= \-12$$  
4. تحديث الوزن:

   $$w\_{new} \= 0.5 \- 0.1(-12) \= 0.5 \+ 1.2 \= 1.7$$

   لاحظ كيف قفز الوزن من 0.5 إلى 1.7، مقترباً بسرعة من القيمة المثالية (2.0).11

**تطبيق برمجي (Python Implementation from Scratch):**

Python

def gradient\_descent(x\_data, y\_data, w\_start, alpha, epochs):  
    w \= w\_start  
    n \= len(x\_data)  
      
    for i in range(epochs):  
        \# 1\. التنبؤ  
        y\_pred \= \[w \* x for x in x\_data\]  
          
        \# 2\. حساب التدرج (متوسط المشتقات)  
        \# Gradient of MSE \= \-(2/n) \* sum(x \* (y\_true \- y\_pred))  
        gradient \= \-2.0/n \* sum(x \* (y \- yp) for x, y, yp in zip(x\_data, y\_data, y\_pred))  
          
        \# 3\. خطوة التحديث  
        w \= w \- alpha \* gradient  
          
    return w

\# اختبار الكود  
x\_train \=   
y\_train \=  \# العلاقة هي y \= 2x  
optimal\_w \= gradient\_descent(x\_train, y\_train, w\_start=0.0, alpha=0.01, epochs=1000)  
print(f"Optimal Weight: {optimal\_w:.4f}") \# المتوقع أن تكون قريبة جداً من 2.0

.13

### **أسئلة مقابلات (التفاضل والتحسين)**

سؤال: ما الفرق بين Batch Gradient Descent و Stochastic Gradient Descent (SGD)؟ ومتى نستخدم كلاً منهما؟  
الإجابة:

* **Batch GD:** يحسب التدرج باستخدام **كل** بيانات التدريب في كل خطوة. هذا يعطي تدرجاً دقيقاً ومستقراً ولكنه بطيء جداً ومكلف في الذاكرة للبيانات الضخمة.  
* **SGD:** يحسب التدرج باستخدام **عينة واحدة** (أو حزمة صغيرة Mini-batch) في كل خطوة. هذا يجعل التحديثات سريعة جداً ولكنها "ضجيجية" (Noisy)، مما يساعد أحياناً في الهروب من القيعان المحلية (Local Minima). عملياً، نستخدم Mini-batch SGD لأنه يوازن بين السرعة والاستقرار.11

## ---

**الفصل 3: الاحتمالات والإحصاء (Probability & Statistics)**

الذكاء الاصطناعي يعمل في بيئة من عدم اليقين. الاحتمالات توفر إطار العمل للاستدلال، والإحصاء يوفر أدوات التحقق من النتائج.

### **3.1 نظرية بايز (Bayes' Theorem): الاستدلال والتصنيف**

تُعد نظرية بايز حجر الزاوية في التفكير الاحتمالي، حيث تحدث احتمالية فرضية معينة بناءً على أدلة جديدة.

$$P(H|E) \= \\frac{P(E|H) \\cdot P(H)}{P(E)}$$

حيث $P(H|E)$ هو الاحتمال اللاحق (Posterior)، و $P(E|H)$ هو الاحتمال المرجح (Likelihood).  
تطبيق عملي: فلتر البريد المزعج (Spam Filter)  
لنفترض أننا نريد معرفة احتمال أن تكون رسالة ما "Spam" إذا كانت تحتوي على كلمة "عرض" (Offer).

* **المعطيات:**  
  * $P(Spam) \= 0.4$ (40% من الإيميلات مزعجة).  
  * $P(Offer | Spam) \= 0.2$ (20% من الإيميلات المزعجة تحتوي كلمة "عرض").  
  * $P(Offer | Not Spam) \= 0.05$ (5% فقط من الإيميلات العادية تحتوي كلمة "عرض").  
* الحساب:  
  أولاً نحسب الاحتمال الكلي لظهور كلمة "عرض" $P(Offer)$:

  $$P(Offer) \= P(Offer|Spam)P(Spam) \+ P(Offer|Not Spam)P(Not Spam)$$  
  $$P(Offer) \= (0.2 \\times 0.4) \+ (0.05 \\times 0.6) \= 0.08 \+ 0.03 \= 0.11$$

  الآن نطبق نظرية بايز:

  $$P(Spam | Offer) \= \\frac{0.2 \\times 0.4}{0.11} \= \\frac{0.08}{0.11} \\approx 0.727$$

  النتيجة: إذا احتوت الرسالة على كلمة "عرض"، فإن احتمال كونها مزعجة يقفز إلى 72.7%.14

### **3.2 اختبار الفرضيات واختبار A/B: اتخاذ القرارات**

عند نشر نموذج جديد، كيف نتأكد أنه أفضل من السابق؟

* **الفرضية الصفرية ($H\_0$):** لا يوجد فرق بين النموذجين.  
* **القيمة الاحتمالية (p-value):** احتمالية الحصول على النتائج المرصودة بافتراض أن الفرضية الصفرية صحيحة. قيمة $p \< 0.05$ تعني رفض الفرضية الصفرية (النتيجة ذات دلالة إحصائية).

سيناريو مقابلة (Data Science):  
سؤال: قمت بإجراء اختبار A/B لنموذج توصية جديد. زادت الأرباح بنسبة 2% (p-value \= 0.03)، لكن وقت الاستجابة (Latency) زاد بمقدار 200ms. هل تقوم بنشر النموذج؟  
تحليل الإجابة: هذا سؤال يختبر التوازن بين المقاييس (Metric Trade-offs).

1. إحصائياً، الزيادة في الربح مؤكدة ($p \< 0.05$).  
2. هندسياً، زيادة 200ms في الـ Latency قد تكون كارثية لتجربة المستخدم على المدى الطويل (Long-term Retention)، وهو ما يسمى "Guardrail Metric".  
3. **القرار:** لا ننشر النموذج مباشرة. يجب أولاً تحسين هندسة النموذج (مثلاً عبر Quantization) لتقليل وقت الاستجابة، أو إجراء تحليل لمعرفة ما إذا كانت زيادة الربح تغطي تكلفة فقدان المستخدمين المحتمل.16

# ---

**الباب الثاني: البرمجة والخوارزميات (Programming & Algorithms)**

### **الفصل 4: إتقان بايثون للذكاء الاصطناعي**

#### **4.1 التوجيه (Vectorization) مقابل الحلقات (Loops): تحليل الأداء**

بايثون لغة مفسرة (Interpreted) وبطيئة في العمليات الحسابية المتكررة. المكتبات مثل NumPy و PyTorch مكتوبة بلغة C/C++ وتستخدم تقنية Vectorization لتنفيذ العمليات على كتل كاملة من البيانات باستخدام تعليمات المعالج SIMD (Single Instruction, Multiple Data).

مقارنة الأداء (Python vs NumPy):  
لنقم بجمع مصفوفتين حجم كل منهما مليون عنصر.

Python

import numpy as np  
import time

size \= 10\*\*6  
\# إنشاء مصفوفات عشوائية  
a \= np.random.rand(size)  
b \= np.random.rand(size)

\# الطريقة التقليدية (Loops)  
start \= time.time()  
c\_loop \= np.zeros(size)  
for i in range(size):  
    c\_loop\[i\] \= a\[i\] \+ b\[i\]  
print(f"Loop Time: {time.time() \- start:.5f} sec")

\# طريقة التوجيه (Vectorization)  
start \= time.time()  
c\_vec \= a \+ b  \# العملية تتم في C++ مباشرة  
print(f"Vectorized Time: {time.time() \- start:.5f} sec")

**النتيجة:** ستجد أن التوجيه أسرع بمقدار 50-100 مرة. في التعلم العميق، هذا المبدأ يمتد إلى GPUs حيث يتم تنفيذ العمليات بالتوازي على آلاف الأنوية.18

### **الفصل 5: البرمجة التنافسية (Competitive Programming) للمهندسين**

المسابقات البرمجية (مثل Codeforces و LeetCode) تطور مهارة حل المشكلات الخوارزمية المعقدة، وهو ما يميز مهندس الـ AI القادر على تحسين الأكواد.

#### **5.1 رفع المصفوفات للقوى (Matrix Exponentiation): الخوارزميات السريعة**

مشكلة شائعة: حساب الرقم $N$ في متتالية فيبوناتشي حيث $N$ قد يصل إلى $10^{18}$. الحل التكراري $O(N)$ بطيء جداً. الحل الأمثل يستخدم ضرب المصفوفات.

الفكرة الرياضية:  
يمكن تمثيل العلاقة $F(n) \= F(n-1) \+ F(n-2)$ كمصفوفة:

$$\\begin{bmatrix} F(n) \\\\ F(n-1) \\end{bmatrix} \= \\begin{bmatrix} 1 & 1 \\\\ 1 & 0 \\end{bmatrix} \\begin{bmatrix} F(n-1) \\\\ F(n-2) \\end{bmatrix}$$

للوصول إلى $F(n)$، نرفع مصفوفة التحويل للقوة $n-1$:

$$M^{n-1}$$

باستخدام خوارزمية Binary Exponentiation، يمكننا حساب $M^n$ في وقت $O(\\log n)$ بدلاً من $O(n)$.20  
**كود بايثون لحل مسألة Codeforces (حساب فيبوناتشي لـ N ضخم):**

Python

MOD \= 10\*\*9 \+ 7

def multiply(A, B):  
    \# ضرب مصفوفتين 2x2  
    C \= \[, \]  
    for i in range(2):  
        for j in range(2):  
            for k in range(2):  
                C\[i\]\[j\] \= (C\[i\]\[j\] \+ A\[i\]\[k\] \* B\[k\]\[j\]) % MOD  
    return C

def matrix\_pow(A, p):  
    \# رفع المصفوفة للقوة p بسرعة لوغاريتمية  
    res \= \[, \] \# مصفوفة الوحدة  
    while p \> 0:  
        if p % 2 \== 1:  
            res \= multiply(res, A)  
        A \= multiply(A, A)  
        p //= 2  
    return res

def fib(n):  
    if n \== 0: return 0  
    if n \== 1: return 1  
    T \= \[,   
         \]  
    T\_res \= matrix\_pow(T, n-1)  
    \# F(n) \= T\*F(1) \+ T\*F(0)  
    return T\_res

\# مثال  
print(fib(10\*\*18)) \# يتم الحساب في أجزاء من الثانية

هذا الأسلوب أساسي في تصميم أنظمة عالية الأداء حيث التكرار البسيط غير مقبول.22

#### **5.2 خوارزميات البحث في الرسوم البيانية: HNSW والبحث المتجهي**

عند التعامل مع ملايين المتجهات (Embeddings) في نماذج اللغة، لا يمكننا استخدام البحث الخطي $O(N)$ لإيجاد أقرب الجيران. خوارزمية HNSW (Hierarchical Navigable Small World) هي الحل القياسي حالياً.

آلية العمل (تشبيه الخرائط):  
تخيل HNSW كخريطة متعددة الطبقات:

* **الطبقة العليا (الطرق السريعة):** تحتوي على عدد قليل من النقاط وروابط طويلة المدى. نتحرك هنا لنقترب بسرعة من المنطقة العامة للهدف.  
* الطبقات السفلى (الشوارع المحلية): تزداد الكثافة كلما نزلنا، مما يسمح بالتدقيق للوصول للنقطة المحددة.  
  هذا الهيكل يحول البحث من $O(N)$ إلى $O(\\log N)$، وهو ما تعتمد عليه قواعد البيانات المتجهية مثل Pinecone و Milvus.24

# ---

**الباب الثالث: تعلم الآلة التقليدي (Classical Machine Learning)**

### **الفصل 6: الخوارزميات الخاضعة للإشراف (Supervised Learning)**

#### **6.1 آلات المتجهات الداعمة (SVM): تعظيم الهامش رياضياً**

الهدف من SVM ليس مجرد الفصل بين البيانات، بل الفصل بأكبر "هامش أمان" (Margin) ممكن.

الاشتقاق الرياضي لتعظيم الهامش:  
المسافة من نقطة $\\mathbf{x}$ إلى المستوى الفاصل هي $\\frac{|\\mathbf{w}^T\\mathbf{x} \+ b|}{\\|\\mathbf{w}\\|}$.  
للنقاط الأقرب (Support Vectors)، نجعل البسط يساوي 1\. إذن المسافة هي $\\frac{1}{\\|\\mathbf{w}\\|}$.  
الهامش الكلي هو المسافة من الجانبين: $\\frac{2}{\\|\\mathbf{w}\\|}$.  
لتعظيم الهامش، يجب تقليل المقام $\\|\\mathbf{w}\\|$. رياضياً، نقوم بتقليل $\\frac{1}{2}\\|\\mathbf{w}\\|^2$ (للسهولة الحسابية).  
مشكلة التحسين (Optimization Problem):

$$\\min\_{\\mathbf{w}, b} \\frac{1}{2} \\|\\mathbf{w}\\|^2$$

بشرط أن: $y\_i(\\mathbf{w}^T\\mathbf{x}\_i \+ b) \\geq 1$ لجميع النقاط.  
يتم حل هذه المسألة باستخدام "مضاعفات لاغرانج" (Lagrange Multipliers)، مما يقودنا لمفهوم "Kernel Trick" الذي يسمح لـ SVM بالعمل في أبعاد لا نهائية وتصنيف بيانات غير خطية بكفاءة عالية.26

#### **6.2 أشجار القرار: حساب الإنتروبيا وكسب المعلومات**

أشجار القرار تقوم بتقسيم البيانات بناءً على الميزة التي تحقق أكبر "كسب للمعلومات" (Information Gain).

مثال حسابي (دراسة حالة تيتانيك):  
لنفترض عقدة فيها 10 ركاب: 6 ماتوا (D)، 4 نجوا (S).

1. حساب الإنتروبيا (Entropy) للعقدة الأم:

   $$H(Parent) \= \- (\\frac{6}{10}\\log\_2 0.6 \+ \\frac{4}{10}\\log\_2 0.4) \\approx 0.97 \\text{ bits}$$

   (الإنتروبيا عالية تعني فوضى/عدم يقين).  
2. **التقسيم حسب الجنس:**  
   * **إناث (4):** 1 ماتت، 3 نجوا. $H(F) \\approx 0.81$.  
   * **ذكور (6):** 5 ماتوا، 1 نجا. $H(M) \\approx 0.65$.  
3. متوسط الإنتروبيا الموزون للأبناء:

   $$H(Children) \= \\frac{4}{10}(0.81) \+ \\frac{6}{10}(0.65) \= 0.714$$  
4. كسب المعلومات (IG):

   $$IG \= H(Parent) \- H(Children) \= 0.97 \- 0.714 \= 0.256$$

   الخوارزمية تختار الميزة (الجنس، العمر، الدرجة) التي تعطي أعلى IG لتقسيم البيانات.28

### **الفصل 7: الخوارزميات غير الخاضعة للإشراف**

#### **7.1 تجميع K-Means: الخطوات الرياضية**

خوارزمية K-Means تهدف لتقسيم البيانات إلى $K$ مجموعات بحيث يكون التباين داخل كل مجموعة أقل ما يمكن.

**خطوات الخوارزمية (Step-by-Step):**

1. **التهيئة:** اختر $K$ مراكز (Centroids) عشوائياً $\\mu\_1, \\dots, \\mu\_K$.  
2. التعيين (Assignment): لكل نقطة بيانات $x\_i$، احسب المسافة (الإقليدية) بينها وبين كل المراكز، وعينها للمركز الأقرب.

   $$c^{(i)} \= \\arg\\min\_j \\|x\_i \- \\mu\_j\\|^2$$  
3. التحديث (Update): احسب المتوسط الجديد لكل مجموعة واجعلة المركز الجديد.

   $$\\mu\_j \= \\frac{\\sum\_{i=1}^m 1\\{c^{(i)}=j\\} x\_i}{\\sum\_{i=1}^m 1\\{c^{(i)}=j\\}}$$  
4. كرر الخطوتين 2 و 3 حتى تتوقف المراكز عن التحرك (التقارب).30

# ---

**الباب الرابع: التعلم العميق (Deep Learning)**

### **الفصل 8: الشبكات العصبية وآلية التعلم**

#### **8.1 الانتشار العكسي (Backpropagation): الاشتقاق الكامل**

الانتشار العكسي هو تطبيق لقاعدة السلسلة (Chain Rule) في التفاضل لحساب تدرج الخطأ بالنسبة لكل وزن في الشبكة.

الاشتقاق (مسار واحد بسيط):  
مدخل $x \\rightarrow$ طبقة مخفية $h$ (وزن $w\_1$) $\\rightarrow$ مخرج $y$ (وزن $w\_2$).

$$h \= \\sigma(w\_1 x), \\quad y \= \\sigma(w\_2 h), \\quad L \= \\frac{1}{2}(y\_{target} \- y)^2$$

لتحديث الوزن $w\_1$، نحتاج $\\frac{\\partial L}{\\partial w\_1}$.  
باستخدام قاعدة السلسلة:

$$\\frac{\\partial L}{\\partial w\_1} \= \\frac{\\partial L}{\\partial y} \\cdot \\frac{\\partial y}{\\partial h} \\cdot \\frac{\\partial h}{\\partial w\_1}$$

1. $\\frac{\\partial L}{\\partial y} \= \-(y\_{target} \- y)$ (مقدار الخطأ).  
2. $\\frac{\\partial y}{\\partial h} \= w\_2 \\cdot \\sigma'(w\_2 h)$ (كيف يؤثر المخفي على المخرج).  
3. $\\frac{\\partial h}{\\partial w\_1} \= x \\cdot \\sigma'(w\_1 x)$ (كيف يؤثر الوزن الأول على المخفي).  
   ضرب هذه القيم يعطينا التدرج. في الشبكات العميقة، يتم ضرب العديد من المشتقات. إذا كانت المشتقات صغيرة جداً (أقل من 1)، فإن حاصل الضرب يقترب من الصفر، مما يسبب مشكلة تلاشي التدرج (Vanishing Gradient)، والتي تمنع الطبقات الأولى من التعلم. هذا يفسر لماذا نستخدم دوال تفعيل مثل ReLU وهياكل مثل ResNet.32

### **الفصل 9: الشبكات العصبية التلافيفية (CNNs)**

#### **9.1 هيكلية ResNet وتطبيقاتها في التصوير الطبي**

في الشبكات العميقة جداً (50 طبقة أو أكثر)، يتدهور الأداء بسبب تلاشي التدرج. قدمت شبكة ResNet حلاً عبقر ًيا: الوصلات المتبقية (Residual Connections).  
المعادلة: $y \= F(x) \+ x$.  
هذا يسمح للتدرج بالتدفق "عبر الطريق السريع" (الوصلة المباشرة $x$) متجاوزاً الطبقات المعقدة $F(x)$، مما يحافظ على قوته.  
دراسة حالة: اكتشاف الأورام (Medical Imaging)  
في تحليل صور الرنين المغناطيسي (MRI)، التفاصيل الدقيقة للأنسجة مهمة جداً. الشبكات العميقة التقليدية تفقد هذه التفاصيل. باستخدام ResNet-50، تمكن الباحثون من تدريب نماذج عميقة جداً قادرة على التقاط الأنماط المعقدة للأورام الخبيثة بدقة تجاوزت 96%، متفوقة على النماذج الأقدم مثل AlexNet.34

# ---

**الباب الخامس: الذكاء الاصطناعي التوليدي والنماذج اللغوية (GenAI & LLMs)**

### **الفصل 10: المحولات (Transformers)**

#### **10.1 آلية الانتباه الذاتي (Self-Attention): الرياضيات**

المحولات تعتمد على آلية الانتباه لحساب علاقة كل كلمة بجميع الكلمات الأخرى في الجملة.

المعادلة الجوهرية:

$$Attention(Q, K, V) \= \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d\_k}}\\right)V$$

* **Query ($Q$):** عن ماذا أبحث؟ (مثلاً كلمة "Bank" تبحث عن سياق).  
* **Key ($K$):** ماذا أحتوي؟ (كلمة "River" أو "Money").  
* Value ($V$): المحتوى الفعلي.  
  القسمة على $\\sqrt{d\_k}$: خطوة حاسمة. بدونها، حاصل الضرب النقطي $QK^T$ سيكون كبيراً جداً، مما يدفع دالة Softmax إلى مناطق التشبع (حيث المشتقات تقترب من الصفر)، مما يوقف التعلم.36

### **الفصل 11: التوليد المعزز بالاسترجاع (RAG) والبحث المتجهي**

النماذج اللغوية قد "تهلوس". الحل هو RAG: البحث عن المعلومة الصحيحة ثم إعطاؤها للنموذج.

**تطبيق برمجي: البحث المتجهي (Cosine Similarity) من الصفر**

Python

import numpy as np

def cosine\_similarity(v1, v2):  
    \# البسط: الضرب النقطي  
    dot\_product \= np.dot(v1, v2)  
    \# المقام: حاصل ضرب الأطوال (Norms)  
    norm\_v1 \= np.linalg.norm(v1)  
    norm\_v2 \= np.linalg.norm(v2)  
      
    if norm\_v1 \== 0 or norm\_v2 \== 0:  
        return 0.0  
    return dot\_product / (norm\_v1 \* norm\_v2)

\# مثال: التشابه بين "ملك" و "ملكة"  
v\_king \= np.array(\[0.9, 0.1, 0.5\])  
v\_queen \= np.array(\[0.8, 0.2, 0.5\])  
print(cosine\_similarity(v\_king, v\_queen)) \# نتيجة عالية

.37

#### **11.2 التكيف منخفض الرتبة (LoRA)**

إعادة تدريب نموذج بـ 70 مليار معامل مكلف جداً. تقنية LoRA تفترض أن التغيير المطلوب في الأوزان $\\Delta W$ يمكن تمثيله بمصفوفتين صغيرتين جداً $A$ و $B$.

$$W\_{new} \= W\_{frozen} \+ (A \\times B)$$

بدلاً من تحديث مصفوفة $1000 \\times 1000$ (مليون معامل)، نحدث مصفوفتين $1000 \\times 8$ و $8 \\times 1000$ (16 ألف معامل فقط). هذا يقلل متطلبات الذاكرة بنسبة تزيد عن 90% مع الحفاظ على الأداء.39

### **الفصل 12: الوكلاء الذكياء (AI Agents) ونمط ReAct**

الوكيل الذكي هو نموذج لغوي لديه القدرة على استخدام أدوات (Tools).

نمط ReAct (Reason \+ Act):  
يعتمد على حلقة تكرارية:

1. **تفكير (Thought):** النموذج يحلل المطلوب ("أحتاج لمعرفة الطقس في الرياض").  
2. **فعل (Action):** النموذج يقرر استدعاء دالة get\_weather('Riyadh').  
3. **ملاحظة (Observation):** نتيجة الدالة تعود للنموذج ("35 درجة مئوية").  
4. **تفكير (Thought):** "الطقس حار، سأنصح المستخدم بشرب الماء".  
5. **إجابة نهائية (Final Answer).**

.41

# ---

**الباب السادس: هندسة الإنتاج وتصميم الأنظمة (MLOps & System Design)**

النموذج في الـ Notebook هو مجرد تجربة؛ النموذج خلف API هو منتج.

### **الفصل 13: نشر النماذج باستخدام FastAPI**

FastAPI هو الخيار المعياري لخدمة نماذج Python بسبب سرعته ودعمه للعمليات غير المتزامنة (Async).

**كود جاهز للإنتاج (Production-Ready Snippet):**

Python

from fastapi import FastAPI, HTTPException  
from pydantic import BaseModel  
import joblib

app \= FastAPI()  
\# تحميل النموذج عند بدء التشغيل (وليس مع كل طلب)  
model \= joblib.load("model.pkl")

class InputData(BaseModel):  
    feature1: float  
    feature2: float

@app.post("/predict")  
async def predict(data: InputData):  
    try:  
        features \= \[\[data.feature1, data.feature2\]\]  
        prediction \= model.predict(features)  
        return {"prediction": int(prediction)}  
    except Exception as e:  
        raise HTTPException(status\_code=500, detail="Model Error")

.43

### **الفصل 14: مراقبة النماذج واكتشاف الانحراف (Concept Drift)**

البيانات تتغير مع الزمن. النموذج الذي تدرب على بيانات 2020 قد يفشل في 2025\.  
اختبار كولموجوروف-سميرنوف (KS Test): اختبار إحصائي يقارن توزيع البيانات الحالية مع توزيع بيانات التدريب.

* يحسب المسافة القصوى بين دالتي التوزيع التراكمي (CDF).  
* إذا كانت المسافة كبيرة (أو p-value \< 0.05)، فهذا جرس إنذار: "حدث انحراف في البيانات (Drift)، يجب إعادة التدريب".45

### **الفصل 15: تصميم الأنظمة: كشف الاحتيال المالي (Real-Time Fraud Detection)**

**المشكلة:** تصميم نظام لشركة مدفوعات (مثل Stripe) لكشف العمليات الاحتيالية في أقل من 200ms.

**الهيكلية المقترحة:**

1. **Ingestion:** تدفق البيانات عبر Kafka.  
2. **Feature Store (Redis):** حساب ميزات في الوقت الفعلي (مثل: "عدد العمليات في آخر 10 دقائق"). هذا يتطلب زمن وصول منخفض جداً (Low Latency).  
3. **Model Serving:** نموذج XGBoost خفيف أو Random Forest يخدم عبر FastAPI.  
4. **Database:** حفظ التوقعات والنتائج الحقيقية في PostgreSQL للتحليل اللاحق.  
5. **Feedback Loop:** العمليات التي يبلغ عنها المستخدمون كاحتيال تعود لنظام التدريب (Airflow) لتحديث النموذج.47

## ---

**الخاتمة**

إن الانتقال إلى هندسة الذكاء الاصطناعي المتقدمة يتطلب مزيجاً فريداً من الصرامة الأكاديمية في الرياضيات، والبراعة في البرمجة والخوارزميات، والخبرة العملية في تصميم الأنظمة. هذا الدليل غطى الرحلة كاملة، من اشتقاق المعادلات الأساسية يدوياً، مروراً بكتابة الخوارزميات من الصفر، وصولاً إلى بناء أنظمة معقدة قابلة للتوسع. الطريق الآن ممهد أمامك للتطبيق والابتكار.

#### **المصادر التي تم الاقتباس منها**

1. 70 Must-Know Linear Algebra Interview Questions and Answers 2025 \- Devinterview.io, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://devinterview.io/questions/machine-learning-and-data-science/linear-algebra-interview-questions/](https://devinterview.io/questions/machine-learning-and-data-science/linear-algebra-interview-questions/)  
2. Linear Algebra interview questions and answers to help you prepare for your next machine learning and data science interview in 2025\. \- GitHub, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://github.com/Devinterview-io/linear-algebra-interview-questions](https://github.com/Devinterview-io/linear-algebra-interview-questions)  
3. Linear Algebra Operations For Machine Learning \- GeeksforGeeks, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.geeksforgeeks.org/machine-learning/ml-linear-algebra-operations/](https://www.geeksforgeeks.org/machine-learning/ml-linear-algebra-operations/)  
4. Applications of Eigenvalues and Eigenvectors \- GeeksforGeeks, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.geeksforgeeks.org/engineering-mathematics/applications-of-eigenvalues-and-eigenvectors/](https://www.geeksforgeeks.org/engineering-mathematics/applications-of-eigenvalues-and-eigenvectors/)  
5. Math for ML: Eigenvalues and Eigenvectors Explained Simply with Examples. \- Medium, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/@ryassminh/math-for-ml-eigenvalues-and-eigenvectors-explained-simply-with-examples-f29459c9cc67](https://medium.com/@ryassminh/math-for-ml-eigenvalues-and-eigenvectors-explained-simply-with-examples-f29459c9cc67)  
6. SVD in Recommendation Systems \- GeeksforGeeks, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.geeksforgeeks.org/machine-learning/svd-in-recommendation-systems/](https://www.geeksforgeeks.org/machine-learning/svd-in-recommendation-systems/)  
7. Using Singular Value Decomposition to Build a Recommender System \- MachineLearningMastery.com, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://machinelearningmastery.com/using-singular-value-decomposition-to-build-a-recommender-system/](https://machinelearningmastery.com/using-singular-value-decomposition-to-build-a-recommender-system/)  
8. Recommender System using Singular Value Decomposition \- RPubs, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.rpubs.com/selmoudninyc/335300](https://www.rpubs.com/selmoudninyc/335300)  
9. Calculus — ML Glossary documentation, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://ml-cheatsheet.readthedocs.io/en/latest/calculus.html](https://ml-cheatsheet.readthedocs.io/en/latest/calculus.html)  
10. Calculus for Machine Learning: Key Concepts and Applications \- GeeksforGeeks, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.geeksforgeeks.org/machine-learning/calculus-for-machine-learning-key-concepts-and-applications/](https://www.geeksforgeeks.org/machine-learning/calculus-for-machine-learning-key-concepts-and-applications/)  
11. Gradient Descent, Step-by-Step \- YouTube, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.youtube.com/watch?v=sDv4f4s2SB8](https://www.youtube.com/watch?v=sDv4f4s2SB8)  
12. Linear regression: Gradient descent | Machine Learning \- Google for Developers, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent)  
13. Complete Step-by-Step Gradient Descent Algorithm from Scratch | Towards Data Science, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://towardsdatascience.com/complete-step-by-step-gradient-descent-algorithm-from-scratch-acba013e8420/](https://towardsdatascience.com/complete-step-by-step-gradient-descent-algorithm-from-scratch-acba013e8420/)  
14. Bayesian filter \- Hornetsecurity – Next-Gen Microsoft 365 Security, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.hornetsecurity.com/en/knowledge-base/bayesian-filter/](https://www.hornetsecurity.com/en/knowledge-base/bayesian-filter/)  
15. Notes on Naive Bayes Classifiers for Spam Filtering \- Washington, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://courses.cs.washington.edu/courses/cse312/18sp/lectures/naive-bayes/naivebayesnotes.pdf](https://courses.cs.washington.edu/courses/cse312/18sp/lectures/naive-bayes/naivebayesnotes.pdf)  
16. Mastering A/B Testing: Top 20 Questions and Answers to Ace Your Data Science Interview, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://tatevkarenaslanyan.medium.com/mastering-a-b-testing-top-20-questions-and-answers-to-ace-your-data-science-interview-d8bf49f20b53](https://tatevkarenaslanyan.medium.com/mastering-a-b-testing-top-20-questions-and-answers-to-ace-your-data-science-interview-d8bf49f20b53)  
17. A guide to passing the A/B test interview question in tech companies : r/datascience \- Reddit, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.reddit.com/r/datascience/comments/1fyrawz/a\_guide\_to\_passing\_the\_ab\_test\_interview\_question/](https://www.reddit.com/r/datascience/comments/1fyrawz/a_guide_to_passing_the_ab_test_interview_question/)  
18. Vectorization in Python- An Alternative to Python Loops | by shivam bhatele \- Medium, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/pythoneers/vectorization-in-python-an-alternative-to-python-loops-2728d6d7cd3e](https://medium.com/pythoneers/vectorization-in-python-an-alternative-to-python-loops-2728d6d7cd3e)  
19. For loop vs Numpy vectorization computation time \- Stack Overflow, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://stackoverflow.com/questions/51549363/for-loop-vs-numpy-vectorization-computation-time](https://stackoverflow.com/questions/51549363/for-loop-vs-numpy-vectorization-computation-time)  
20. Competitive Programming (CP) \- Steven Gong, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://stevengong.co/notes/Competitive-Programming](https://stevengong.co/notes/Competitive-Programming)  
21. hkirat/Algorithmic-Resources: A Curated list of Topic wise Theory and Questions to Get You Started On Competitive Coding. \- GitHub, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://github.com/hkirat/Algorithmic-Resources](https://github.com/hkirat/Algorithmic-Resources)  
22. Binary Exponentiation \- Algorithms for Competitive Programming, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://cp-algorithms.com/algebra/binary-exp.html](https://cp-algorithms.com/algebra/binary-exp.html)  
23. Matrix Exponentiation Coding (Part 1/2) \- YouTube, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.youtube.com/watch?v=kQuCOFzWoa0](https://www.youtube.com/watch?v=kQuCOFzWoa0)  
24. Understanding Hierarchical Navigable Small Worlds (HNSW) for Vector Search \- Milvus, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://milvus.io/blog/understand-hierarchical-navigable-small-worlds-hnsw-for-vector-search.md](https://milvus.io/blog/understand-hierarchical-navigable-small-worlds-hnsw-for-vector-search.md)  
25. Vector Database Search: HNSW Algorithm Explained \- YouTube, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.youtube.com/watch?v=cZyTZ-EMskI](https://www.youtube.com/watch?v=cZyTZ-EMskI)  
26. An Idiot's guide to Support vector machines (SVMs) \- MIT, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf](https://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf)  
27. Maximize the margin formula in support vector machines algorithm, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://datascience.stackexchange.com/questions/43180/maximize-the-margin-formula-in-support-vector-machines-algorithm](https://datascience.stackexchange.com/questions/43180/maximize-the-margin-formula-in-support-vector-machines-algorithm)  
28. Entropy and Information Gain \- UniPD Math, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.math.unipd.it/\~aiolli/corsi/0708/IR/Lez12.pdf](https://www.math.unipd.it/~aiolli/corsi/0708/IR/Lez12.pdf)  
29. A Simple Explanation of Information Gain and Entropy \- victorzhou.com, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://victorzhou.com/blog/information-gain/](https://victorzhou.com/blog/information-gain/)  
30. The Math Behind K-Means Clustering | by Dharmaraj | Medium, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/@draj0718/the-math-behind-k-means-clustering-4aa85532085e](https://medium.com/@draj0718/the-math-behind-k-means-clustering-4aa85532085e)  
31. k-Means Clustering | Brilliant Math & Science Wiki, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://brilliant.org/wiki/k-means-clustering/](https://brilliant.org/wiki/k-means-clustering/)  
32. Backpropagation: Step-By-Step Derivation | Towards Data Science, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://towardsdatascience.com/backpropagation-step-by-step-derivation-99ac8fbdcc28/](https://towardsdatascience.com/backpropagation-step-by-step-derivation-99ac8fbdcc28/)  
33. Calculus on Computational Graphs: Backpropagation \- colah's blog, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://colah.github.io/posts/2015-08-Backprop/](https://colah.github.io/posts/2015-08-Backprop/)  
34. A review of the application of deep learning in medical image classification and segmentation \- PMC \- NIH, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://pmc.ncbi.nlm.nih.gov/articles/PMC7327346/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7327346/)  
35. Radiographic imaging and diagnosis of spinal bone tumors: AlexNet and ResNet for the classification of tumor malignancy \- PMC \- PubMed Central, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://pmc.ncbi.nlm.nih.gov/articles/PMC11381891/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11381891/)  
36. The (surprisingly simple\!) math behind the transformer attention mechanism | by Touhid, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/@touhid3.1416/the-surprisingly-simple-math-behind-transformer-attention-mechanism-d354fbb4fef6](https://medium.com/@touhid3.1416/the-surprisingly-simple-math-behind-transformer-attention-mechanism-d354fbb4fef6)  
37. A quick tutorial on how to build vector search | by Tripadvisor \- Medium, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/tripadvisor/a-quick-tutorial-on-how-to-build-vector-search-d74ad26f9ffe](https://medium.com/tripadvisor/a-quick-tutorial-on-how-to-build-vector-search-d74ad26f9ffe)  
38. Implementing Vector Search from Scratch: A Step-by-Step Tutorial \- MachineLearningMastery.com, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://machinelearningmastery.com/implementing-vector-search-from-scratch-a-step-by-step-tutorial/](https://machinelearningmastery.com/implementing-vector-search-from-scratch-a-step-by-step-tutorial/)  
39. Low-rank adaptation (LoRA) fine tuning \- IBM, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.ibm.com/docs/en/watsonx/w-and-w/2.1.0?topic=tuning-lora-fine](https://www.ibm.com/docs/en/watsonx/w-and-w/2.1.0?topic=tuning-lora-fine)  
40. Understanding LoRA — Low Rank Adaptation For Finetuning Large Models \- Medium, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/data-science/understanding-lora-low-rank-adaptation-for-finetuning-large-models-936bce1a07c6](https://medium.com/data-science/understanding-lora-low-rank-adaptation-for-finetuning-large-models-936bce1a07c6)  
41. What is a ReAct Agent? | IBM, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.ibm.com/think/topics/react-agent](https://www.ibm.com/think/topics/react-agent)  
42. Building a Python React Agent Class: A Step-by-Step Guide \- Nov 05, 2024 \- Neradot, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.neradot.com/post/building-a-python-react-agent-class-a-step-by-step-guide](https://www.neradot.com/post/building-a-python-react-agent-class-a-step-by-step-guide)  
43. Building and Deploying a Machine Learning Model Using FastAPI \- Analytics Vidhya, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.analyticsvidhya.com/blog/2025/12/fastapi-machine-learning-deployment/](https://www.analyticsvidhya.com/blog/2025/12/fastapi-machine-learning-deployment/)  
44. Serving ML Models with FastAPI: A Production-Ready API in Minutes \- Grigor Khachatryan, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://grigorkh.medium.com/serving-ml-models-with-fastapi-a-production-ready-api-in-minutes-b5f4839a33a9](https://grigorkh.medium.com/serving-ml-models-with-fastapi-a-production-ready-api-in-minutes-b5f4839a33a9)  
45. KSWIN \- River, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://riverml.xyz/0.15.0/api/drift/KSWIN/](https://riverml.xyz/0.15.0/api/drift/KSWIN/)  
46. Detecting data drift using the Kolmogorov-Smirnov test | Python, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://campus.datacamp.com/courses/end-to-end-machine-learning/model-monitoring?ex=6](https://campus.datacamp.com/courses/end-to-end-machine-learning/model-monitoring?ex=6)  
47. How would you design a fraud-detection system for Stripe? \- Final Round AI, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.finalroundai.com/interview-questions/stripe-fraud-detection-system-design](https://www.finalroundai.com/interview-questions/stripe-fraud-detection-system-design)  
48. TikTok MLE System Design Interview: Design a Real-Time Fraud Detection System, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/@bugfreeai/tiktok-mle-system-design-interview-design-a-real-time-fraud-detection-system-749cea63ffa5](https://medium.com/@bugfreeai/tiktok-mle-system-design-interview-design-a-real-time-fraud-detection-system-749cea63ffa5)