# **المرجع الشامل في هندسة خوارزميات التعلم الآلي: التحليل الرياضي، البناء البرمجي، والتطبيقات الواقعية**

## **مقدمة: فلسفة التعلم من البيانات والتحول النموذجي في الحوسبة**

يشهد العالم تحولاً جذرياً في كيفية معالجة المشكلات الحسابية، حيث ننتقل من البرمجة التقليدية القائمة على القواعد الصريحة (Rule-based Programming) إلى البرمجة الاحتمالية المعتمدة على البيانات (Data-driven Programming). في هذا السياق، لا يُعد التعلم الآلي (Machine Learning) مجرد مجموعة من الأدوات الإحصائية، بل هو حقل علمي يدمج بين الجبر الخطي، وحساب التفاضل والتكامل، ونظرية الاحتمالات، وعلوم الحاسوب لبناء أنظمة قادرة على التعميم (Generalization) من تجارب سابقة للتنبؤ بأحداث مستقبلية غير مرئية.1 يهدف هذا التقرير البحثي الموسع إلى تقديم تشريح دقيق وشامل لخوارزميات التعلم الآلي، متبنياً منهجية "الصندوق الأبيض" (White-box Approach)، حيث سنقوم بتفكيك كل خوارزمية لاستخراج جوهرها الرياضي، واشتقاق معادلاتها من المبادئ الأولى، ومن ثم إعادة بنائها برمجياً من الصفر (From Scratch) باستخدام لغة Python ومكتبة NumPy، وصولاً إلى ربط هذه المفاهيم المجردة بتطبيقات عملية في كبرى الشركات التقنية.

إن فهم التعلم الآلي بعمق يتطلب الغوص في ثلاثة أبعاد رئيسية: البعد التمثيلي (Representation) الذي يعنى بكيفية صياغة البيانات والنماذج، والبعد التقييمي (Evaluation) الذي يحدد معايير الأداء، والبعد التحسيني (Optimization) الذي يبحث عن أفضل المعلمات الممكنة. سنستعرض في هذا التقرير الطيف الكامل للخوارزميات، بدءاً من الانحدار الخطي البسيط وصولاً إلى الشبكات العصبية العميقة والتعلم المعزز، مع التركيز على التفاصيل الدقيقة التي غالباً ما يتم تجاوزها في الشروحات السطحية، مثل الاشتقاق الرياضي للتدرج (Gradient Derivation)، وتحليل مفاضلة التحيز والتباين (Bias-Variance Decomposition)، ودور التنظيم (Regularization) في التحكم في تعقيد النماذج.2

## ---

**الفصل الأول: الأسس الرياضية وهندسة البيانات**

قبل الخوض في تفاصيل الخوارزميات، يجب تأسيس القاعدة الرياضية التي ترتكز عليها جميع عمليات التعلم الآلي. إن إهمال هذه الأسس هو السبب الرئيسي لفشل الممارسين في تشخيص مشاكل النماذج أو تحسين أدائها.

### **1.1 الجبر الخطي: لغة البيانات**

تُمثل البيانات في التعلم الآلي عادةً كمتجهات (Vectors) ومصفوفات (Matrices). إذا كان لدينا مجموعة بيانات تحتوي على $m$ عينة و $n$ ميزة (Feature)، فإننا نمثلها كمصفوفة $X \\in \\mathbb{R}^{m \\times n}$. العمليات الأساسية التي سنعتمد عليها تشمل:

* **الضرب النقطي (Dot Product):** هو العملية الجوهرية لحساب التشابه بين المتجهات ولحساب المخرجات الخطية. للمعادلة $z \= w^T x \+ b$، يمثل الضرب النقطي $w^T x$ الإسقاط الهندسي للمدخلات على متجهات الأوزان.4  
* **القيم الذاتية والمتجهات الذاتية (Eigenvalues & Eigenvectors):** تلعب دوراً محورياً في خوارزميات تقليل الأبعاد مثل PCA، حيث تحدد الاتجاهات التي تحتوي على أكبر قدر من التباين في البيانات.5

### **1.2 حساب التفاضل: محرك التحسين**

تعتمد معظم خوارزميات التعلم الآلي على تحسين دالة هدف (Objective Function). ولتحقيق ذلك، نستخدم التفاضل لحساب التدرج (Gradient)، وهو متجه المشتقات الجزئية الذي يشير إلى اتجاه الزيادة القصوى للدالة.

$$\\nabla J(\\theta) \= \\left\[ \\frac{\\partial J}{\\partial \\theta\_1}, \\frac{\\partial J}{\\partial \\theta\_2}, \\dots, \\frac{\\partial J}{\\partial \\theta\_n} \\right\]^T$$

في خوارزمية الانحدار التدريجي (Gradient Descent)، نتحرك عكس اتجاه التدرج لتقليل الخطأ.6

### **1.3 نظرية الاحتمالات: التعامل مع عدم اليقين**

بما أن البيانات في العالم الحقيقي مشوشة وغير كاملة، يوفر الإطار الاحتمالي، وتحديداً نظرية بايز (Bayes' Theorem)، وسيلة لتحديث اعتقاداتنا حول النموذج بناءً على أدلة جديدة.

$$P(\\text{Model} | \\text{Data}) \= \\frac{P(\\text{Data} | \\text{Model}) \\cdot P(\\text{Model})}{P(\\text{Data})}$$

هذا المبدأ هو الأساس لخوارزميات مثل Naive Bayes وطرق التنظيم (Regularization) التي يمكن تفسيرها كفرض توزيعات مسبقة (Priors) على الأوزان.7

## ---

**الفصل الثاني: تحليل مفاضلة التحيز والتباين (Bias-Variance Tradeoff)**

يُعد تحليل الخطأ أحد أهم المفاهيم النظرية التي تحكم قدرة النموذج على التعميم. يمكن تفكيك الخطأ المتوقع لنموذج تعلم آلي إلى ثلاثة مكونات رئيسية، وفهم هذه المكونات هو المفتاح لبناء نماذج قوية.

### **2.1 الاشتقاق الرياضي لتحليل الخطأ**

لنفترض أن العلاقة الحقيقية بين المتغيرات هي $y \= f(x) \+ \\epsilon$، حيث $\\epsilon$ يمثل ضجيجاً عشوائياً بمتوسط صفري $E\[\\epsilon\] \= 0$ وتباين $\\sigma^2$. هدفنا هو تعلم دالة $\\hat{f}(x)$ تقرب $f(x)$ بأكبر قدر ممكن.  
مقياس الأداء هو متوسط مربع الخطأ (MSE) لعينة اختبار جديدة $x$:

$$\\text{MSE} \= \\mathbb{E} \\left\[ \\left( y \- \\hat{f}(x) \\right)^2 \\right\]$$  
بتعويض $y \= f(x) \+ \\epsilon$ وتطبيق خصائص التوقع والتباين، يمكننا اشتقاق المعادلة الشهيرة لتحليل الخطأ 8:

$$\\mathbb{E} \\left\[ (y \- \\hat{f}(x))^2 \\right\] \= \\underbrace{(\\mathbb{E}\[\\hat{f}(x)\] \- f(x))^2}\_{\\text{Bias}^2} \+ \\underbrace{\\mathbb{E}\[(\\hat{f}(x) \- \\mathbb{E}\[\\hat{f}(x)\])^2\]}\_{\\text{Variance}} \+ \\underbrace{\\sigma^2}\_{\\text{Irreducible Error}}$$

#### **الجدول 1: تفسير مكونات الخطأ**

| المكون | التفسير | التأثير على النموذج | الحل المقترح |
| ----: | ----: | ----: | ----: |
| **التحيز (Bias)** | الخطأ الناتج عن الافتراضات التبسيطية للخوارزمية (مثلاً، افتراض علاقة خطية لبيانات غير خطية). | نقص في التوافق (Underfitting). النموذج غير قادر على التقاط الأنماط الأساسية. | زيادة تعقيد النموذج، إضافة ميزات جديدة، تقليل التنظيم. |
| **التباين (Variance)** | حساسية النموذج للتغيرات الصغيرة في بيانات التدريب. | إفراط في التوافق (Overfitting). النموذج يحفظ الضجيج بدلاً من الإشارة. | زيادة بيانات التدريب، استخدام التنظيم (L1/L2)، تقليل تعقيد النموذج، استخدام طرق التجميع (Bagging). |
| **الخطأ غير القابل للاختزال** | الضجيج المتأصل في البيانات نفسها. | يمثل الحد الأدنى النظري للخطأ الذي لا يمكن تجاوزه. | تحسين جودة جمع البيانات أو تنظيفها. |

إن الهدف الأسمى لأي ممارس للتعلم الآلي هو إيجاد التوازن الدقيق بين التحيز والتباين لتقليل الخطأ الكلي.8

## ---

**الفصل الثالث: الانحدار الخطي (Linear Regression) \- التشريح الكامل**

الانحدار الخطي هو حجر الزاوية في النمذجة التنبؤية. ورغم بساطته الظاهرية، فإنه يحمل في طياته الكثير من العمق الرياضي والإحصائي.

### **3.1 الصياغة الرياضية والفرضيات**

يفترض النموذج الخطي وجود علاقة خطية بين المتغيرات المستقلة $X$ والمتغير التابع $y$.

$$h\_\\theta(x) \= \\theta\_0 \+ \\theta\_1 x\_1 \+ \\theta\_2 x\_2 \+ \\dots \+ \\theta\_n x\_n \= \\theta^T x$$

حيث $\\theta$ هو متجه الأوزان (بما في ذلك الحد الثابت $\\theta\_0$ المرتبط بـ $x\_0=1$).  
دالة التكلفة (Cost Function) المستخدمة عادة هي مجموع مربعات الخطأ (SSE) أو متوسطها (MSE)، وذلك لأن تقليل MSE يكافئ تعظيم الاحتمالية (Maximum Likelihood Estimation) بافتراض أن الأخطاء تتبع توزيعاً طبيعياً (Gaussian Distribution).

$$J(\\theta) \= \\frac{1}{2m} \\sum\_{i=1}^{m} (h\_\\theta(x^{(i)}) \- y^{(i)})^2$$

### **3.2 طرق الحل: المعادلة الطبيعية مقابل الانحدار التدريجي**

#### **أ. الحل التحليلي المباشر (Closed-Form Solution)**

يمكن إيجاد القيم المثلى للأوزان $\\theta$ مباشرة عن طريق مساواة مشتقة دالة التكلفة بالصفر، مما يقودنا إلى المعادلة الطبيعية (Normal Equation):

$$\\theta \= (X^T X)^{-1} X^T y$$

على الرغم من دقة هذا الحل، إلا أن تكلفته الحسابية عالية جداً $O(n^3)$ عند حساب معكوس المصفوفة، مما يجعله غير عملي للمجموعات البيانات الكبيرة ذات الميزات المتعددة.

#### **ب. خوارزمية الانحدار التدريجي (Gradient Descent)**

هذه هي الطريقة التكرارية الأكثر شيوعاً في التعلم الآلي والشبكات العصبية. تعتمد الفكرة على تحديث الأوزان تدريجياً في الاتجاه المعاكس للتدرج.6  
مشتقة دالة التكلفة بالنسبة للوزن $\\theta\_j$:

$$\\frac{\\partial J(\\theta)}{\\partial \\theta\_j} \= \\frac{1}{m} \\sum\_{i=1}^{m} (h\_\\theta(x^{(i)}) \- y^{(i)}) x\_j^{(i)}$$  
قاعدة التحديث في كل خطوة:

$$\\theta\_j := \\theta\_j \- \\alpha \\frac{\\partial J(\\theta)}{\\partial \\theta\_j}$$

حيث $\\alpha$ هو معدل التعلم (Learning Rate).

### **3.3 التنفيذ البرمجي من الصفر (Python Implementation)**

فيما يلي بناء كامل لفئة LinearRegression تحاكي مكتبة Scikit-Learn، باستخدام NumPy فقط، مع دعم للتنظيم (L1 و L2).10

Python

import numpy as np

class LinearRegressionScratch:  
    def \_\_init\_\_(self, learning\_rate=0.01, n\_iterations=1000, regularization=None, lambda\_reg=0.1):  
        self.learning\_rate \= learning\_rate  
        self.n\_iterations \= n\_iterations  
        self.regularization \= regularization  \# 'l1' or 'l2'  
        self.lambda\_reg \= lambda\_reg  
        self.weights \= None  
        self.bias \= None  
        self.cost\_history \=

    def fit(self, X, y):  
        """  
        تدريب النموذج باستخدام خوارزمية الانحدار التدريجي.  
        """  
        n\_samples, n\_features \= X.shape  
          
        \# تهيئة الأوزان والانحياز بالصفر  
        self.weights \= np.zeros(n\_features)  
        self.bias \= 0

        for i in range(self.n\_iterations):  
            \# 1\. التوقع الخطي (Forward Pass)  
            y\_predicted \= np.dot(X, self.weights) \+ self.bias  
              
            \# 2\. حساب التدرجات (Gradients)  
            error \= y\_predicted \- y  
            dw \= (1 / n\_samples) \* np.dot(X.T, error)  
            db \= (1 / n\_samples) \* np.sum(error)  
              
            \# 3\. إضافة حد التنظيم للتدرج (Regularization Term)  
            if self.regularization \== 'l2':  
                dw \+= (self.lambda\_reg / n\_samples) \* self.weights  
            elif self.regularization \== 'l1':  
                dw \+= (self.lambda\_reg / n\_samples) \* np.sign(self.weights)  
              
            \# 4\. تحديث المعلمات  
            self.weights \-= self.learning\_rate \* dw  
            self.bias \-= self.learning\_rate \* db  
              
            \# تسجيل التكلفة للمراقبة  
            cost \= np.mean(error\*\*2)  
            self.cost\_history.append(cost)

    def predict(self, X):  
        return np.dot(X, self.weights) \+ self.bias

مثال تطبيقي (حساب يدوي للخطوات):  
لنفترض البيانات التالية: $X \= $، $y \= $. نريد تعلم $\\theta$ (بافتراض الانحياز 0 للتبسيط).

1. **التهيئة:** $\\theta \= 0.5$، $\\alpha \= 0.1$.  
2. **التكرار الأول:**  
   * التوقعات: $h(1) \= 0.5 \\times 1 \= 0.5$، $h(2) \= 0.5 \\times 2 \= 1.0$.  
   * الأخطاء: $e\_1 \= 0.5 \- 2 \= \-1.5$، $e\_2 \= 1.0 \- 4 \= \-3.0$.  
   * التدرج: $\\frac{1}{2} ((-1.5 \\times 1\) \+ (-3.0 \\times 2)) \= \\frac{1}{2} (-1.5 \- 6\) \= \-3.75$.  
   * التحديث: $\\theta\_{new} \= 0.5 \- 0.1(-3.75) \= 0.5 \+ 0.375 \= 0.875$.  
     نلاحظ أن الوزن زاد واقترب من القيمة الحقيقية (التي هي 2)، وهذا يوضح آلية عمل الخوارزمية في تصحيح الخطأ ذاتياً.11

### **3.4 التنظيم (Regularization): Ridge vs Lasso**

لمعالجة مشكلة التباين العالي (Overfitting)، نضيف حداً جزائياً لدالة التكلفة.

* **Ridge (L2):** يضيف $\\lambda \\sum \\theta\_j^2$. يجبر الأوزان أن تكون صغيرة ولكنه نادراً ما يجعلها صفراً. المشتقة تتضمن $\\theta\_j$ مما يؤدي إلى تقليص نسبي (Weight Decay).  
* **Lasso (L1):** يضيف $\\lambda \\sum |\\theta\_j|$. يميل لجعل بعض الأوزان صفراً تماماً، مما يجعله أداة قوية لاختيار الميزات (Feature Selection). التدرج هنا ثابت ($\\pm \\lambda$) ولا يعتمد على قيمة الوزن، مما يدفع الأوزان الصغيرة نحو الصفر بسرعة.12

## ---

**الفصل الرابع: الانحدار اللوجستي (Logistic Regression) ومشكلة التصنيف**

على الرغم من اسمه، يُستخدم الانحدار اللوجستي للتصنيف (Classification). الفرق الجوهري عن الانحدار الخطي يكمن في دالة التفعيل ودالة التكلفة.

### **4.1 دالة السيجمويد (Sigmoid Function)**

لتحويل المخرجات الخطية إلى احتمالات محصورة بين 0 و 1، نستخدم دالة السيجمويد:

$$\\sigma(z) \= \\frac{1}{1 \+ e^{-z}}$$

حيث $z \= \\theta^T x$. هذه الدالة تمتاز بأن مشتقتها سهلة الحساب: $\\sigma'(z) \= \\sigma(z)(1 \- \\sigma(z))$، مما يسهل عملية الانتشار الخلفي لاحقاً في الشبكات العصبية.

### **4.2 دالة التكلفة: اللوغاريتم السالب (Log Loss / Cross-Entropy)**

لا يمكن استخدام MSE هنا لأن دالة الفرضية غير خطية، مما يجعل دالة التكلفة غير محدبة (Non-convex) ولها عدة قيعان محلية. بدلاً من ذلك، نستخدم دالة تعتمد على الاحتمالية القصوى (Maximum Likelihood):

$$J(\\theta) \= \- \\frac{1}{m} \\sum\_{i=1}^{m} \[y^{(i)} \\log(h\_\\theta(x^{(i)})) \+ (1 \- y^{(i)}) \\log(1 \- h\_\\theta(x^{(i)}))\]$$

هذه الدالة تعاقب النموذج بشدة إذا تنبأ باحتمالية عالية لفئة خاطئة (مثلاً، تنبأ بـ 0.99 للفئة 1 بينما الحقيقة 0).

### **4.3 التصنيف متعدد الفئات (Multiclass Classification)**

للتعامل مع أكثر من فئتين، نستخدم استراتيجيتين:

1. **واحد ضد الكل (One-vs-Rest):** تدريب $K$ نماذج ثنائية، كل نموذج يميز فئة واحدة عن البقية.  
2. دالة Softmax: تعميم للسيجمويد لعدة فئات، حيث تحسب احتمالية كل فئة بحيث يكون مجموع الاحتمالات 1\.

   $$P(y=k|x) \= \\frac{e^{z\_k}}{\\sum\_{j=1}^{K} e^{z\_j}}$$

## ---

**الفصل الخامس: خوارزمية الجيران الأقرب (K-Nearest Neighbors \- KNN)**

تُعتبر KNN خوارزمية فريدة لأنها "لا تتعلم" نموذجاً ثابتاً (Non-parametric & Lazy Learner). بدلاً من ذلك، تقوم بحفظ بيانات التدريب كاملة، وتؤجل الحسابات حتى لحظة التنبؤ.

### **5.1 المنطق الرياضي ومقاييس المسافة**

لتصنيف نقطة استعلام $q$، تبحث الخوارزمية عن أقرب $K$ جيران في فضاء الميزات.  
المسافة الأكثر شيوعاً هي المسافة الإقليدية (Euclidean Distance):

$$d(x, q) \= \\sqrt{\\sum\_{i=1}^{n} (x\_i \- q\_i)^2}$$

يمكن استخدام مقاييس أخرى مثل مسافة مانهاتن (مجموع الفروق المطلقة) وهي أفضل في الأبعاد العالية، أو مسافة منكوفسكي التي تعممهما.14

### **5.2 مشكلة لعنة الأبعاد (Curse of Dimensionality)**

تواجه KNN تحدياً كبيراً كلما زاد عدد الميزات (الأبعاد). في الأبعاد العالية، تصبح جميع النقاط متباعدة عن بعضها البعض بشكل متساوٍ تقريباً، مما يجعل مفهوم "الجار الأقرب" بلا معنى إحصائياً. الحل يكمن في تقليل الأبعاد (PCA) أو اختيار الميزات بعناية.

### **5.3 التنفيذ البرمجي الكامل**

Python

from collections import Counter

class KNNScratch:  
    def \_\_init\_\_(self, k=3):  
        self.k \= k

    def fit(self, X, y):  
        \# KNN هو متعلم كسول، لذا فإن عملية "التدريب" هي مجرد حفظ البيانات  
        self.X\_train \= X  
        self.y\_train \= y

    def euclidean\_distance(self, x1, x2):  
        return np.sqrt(np.sum((x1 \- x2)\*\*2))

    def predict(self, X):  
        predictions \= \[self.\_predict(x) for x in X\]  
        return np.array(predictions)

    def \_predict(self, x):  
        \# 1\. حساب المسافة بين النقطة الجديدة وكل نقاط التدريب  
        distances \= \[self.euclidean\_distance(x, x\_train) for x\_train in self.X\_train\]  
          
        \# 2\. استخراج مؤشرات أقرب k جيران  
        \# np.argsort يعيد مؤشرات العناصر مرتبة تصاعدياً  
        k\_indices \= np.argsort(distances)\[:self.k\]  
          
        \# 3\. جلب فئات هؤلاء الجيران  
        k\_nearest\_labels \= \[self.y\_train\[i\] for i in k\_indices\]  
          
        \# 4\. التصويت بالأغلبية  
        most\_common \= Counter(k\_nearest\_labels).most\_common(1)  
        return most\_common

.15

## ---

**الفصل السادس: أشجار القرار (Decision Trees) \- البناء والتقييم**

تحاكي أشجار القرار طريقة التفكير البشري عبر سلسلة من القواعد الشرطية (If-Else). قوتها تكمن في قابليتها للتفسير، لكنها تعاني من الميل للإفراط في التوافق.

### **6.1 معايير التقسيم: النقاء والمعلومات**

الهدف في كل عقدة هو اختيار الميزة والقيمة التي تقسم البيانات بأفضل شكل لفصل الفئات. نستخدم مقاييس رياضية لقياس "نقاء" العقدة.

#### **أ. مؤشر جيني (Gini Impurity)**

يقيس احتمالية تصنيف عنصر عشوائي بشكل خاطئ.

$$G \= 1 \- \\sum\_{i=1}^{C} p\_i^2$$

حيث $p\_i$ نسبة العينات من الفئة $i$ في العقدة. القيمة 0 تعني نقاءً تاماً.

#### **ب. الإنتروبيا (Entropy) وكسب المعلومات (Information Gain)**

مستمد من نظرية المعلومات لشانون. يقيس مقدار العشوائية أو عدم اليقين.

$$H(S) \= \- \\sum\_{i=1}^{C} p\_i \\log\_2(p\_i)$$

كسب المعلومات (Information Gain): هو الفرق بين إنتروبيا العقدة الأم ومجموع إنتروبيا العقد الأبناء الموزونة. تختار الخوارزمية التقسيم الذي يعظم هذا الكسب.17

### **6.2 مثال عددي مفصل لحساب الانقسام الأمثل**

لنفترض مجموعة بيانات صغيرة: 5 كرات زرقاء و 5 كرات حمراء (المجموع 10).

* إنتروبيا العقدة الأم:  
  $H(Parent) \= \- (0.5 \\log\_2 0.5 \+ 0.5 \\log\_2 0.5) \= 1$.  
* **انقسام مقترح (A):** يقسمها إلى (4 زرقاء، 0 حمراء) و (1 زرقاء، 5 حمراء).  
  * العقدة اليسرى (4,0): $H \= 0$ (نقية تماماً).  
  * العقدة اليمنى (1,5): احتمالات (1/6, 5/6).  
    $H \= \- (\\frac{1}{6} \\log\_2 \\frac{1}{6} \+ \\frac{5}{6} \\log\_2 \\frac{5}{6}) \\approx 0.65$.  
  * المتوسط الموزون للإنتروبيا:  
    $\\frac{4}{10}(0) \+ \\frac{6}{10}(0.65) \= 0.39$.  
  * كسب المعلومات: $IG \= 1 \- 0.39 \= 0.61$.  
    تقوم الخوارزمية بحساب الـ IG لكل تقسيم ممكن لكل ميزة وتختار الأكبر.19

## ---

**الفصل السابع: خوارزميات التجميع (Ensemble Methods)**

نظراً لأن شجرة القرار الواحدة ضعيفة وعالية التباين، فإن جمع عدة أشجار يحسن الأداء بشكل كبير.

### **7.1 الغابات العشوائية (Random Forests) \- التجميع (Bagging)**

تعتمد على فكرة **Bootstrap Aggregating**:

1. **Bootstrap:** أخذ عينات عشوائية مع الإرجاع من البيانات الأصلية لإنشاء مجموعات تدريب مختلفة.  
2. **Random Feature Selection:** عند كل انقسام في الشجرة، لا ننظر لكل الميزات، بل لمجموعة فرعية عشوائية منها. هذا يقلل الارتباط (Correlation) بين الأشجار.  
3. **Aggregation:** التصويت بالأغلبية (للتصنيف) أو المتوسط (للانحدار).20

### **7.2 التعزيز (Boosting) \- Gradient Boosting**

على عكس الغابات العشوائية التي تبني أشجاراً مستقلة، يبني التعزيز الأشجار بشكل تسلسلي، حيث تحاول كل شجرة جديدة تصحيح أخطاء الأشجار السابقة.  
آلية العمل:

1. تدريب شجرة أولى $F\_0(x)$.  
2. حساب **البواقي (Residuals)**: $r\_i \= y\_i \- F\_0(x\_i)$.  
3. تدريب شجرة جديدة $h\_1(x)$ للتنبؤ بهذه البواقي بدلاً من $y$ الأصلي.  
4. تحديث النموذج: $F\_1(x) \= F\_0(x) \+ \\eta h\_1(x)$.  
   تستمر العملية، مما يؤدي لنموذج قوي جداً يقلل التحيز والتباين معاً.22

## ---

**الفصل الثامن: آلات المتجهات الداعمة (Support Vector Machines \- SVM)**

تعتبر SVM من أكثر الخوارزميات أناقة رياضياً، حيث تحول مشكلة التصنيف إلى مشكلة تحسين هندسي.

### **8.1 تعظيم الهامش (Margin Maximization)**

الهدف هو إيجاد المستوى الفائق (Hyperplane) الذي يفصل بين الفئات بأكبر هامش ممكن. الهامش هو المسافة بين المستوى الفائق وأقرب نقطة من كل فئة (تسمى هذه النقاط بالمتجهات الداعمة).  
رياضياً، نريد تعظيم $\\frac{2}{||w||}$، وهو ما يكافئ تقليل $||w||^2$.  
المشكلة (Primal Form):

$$\\min\_{w,b} \\frac{1}{2} ||w||^2 \\quad \\text{subject to} \\quad y\_i (w^T x\_i \+ b) \\ge 1$$

### **8.2 المشكلة الثنائية (Dual Problem) وخدعة النواة (Kernel Trick)**

باستخدام مضاعفات لاغرانج، يمكن تحويل المشكلة إلى صيغة تعتمد فقط على الضرب النقطي بين الأزواج $(x\_i \\cdot x\_j)$.24  
هذا يسمح لنا باستخدام خدعة النواة: استبدال الضرب النقطي بدالة نواة $K(x\_i, x\_j)$ تحسب التشابه في فضاء ذي أبعاد أعلى دون الحاجة لتحويل البيانات فعلياً. أشهر النواة هي RBF (Radial Basis Function):

$$K(x, x') \= \\exp(-\\gamma ||x \- x'||^2)$$

هذا يسمح لـ SVM برسم حدود قرار غير خطية ومعقدة للغاية بكفاءة حسابية عالية.

## ---

**الفصل التاسع: الشبكات العصبية (Neural Networks) من الصفر**

هنا نصل إلى جوهر التعلم العميق. سنبني شبكة عصبية كاملة (Multilayer Perceptron) ونشتق خوارزمية الانتشار الخلفي التي حيرت الكثيرين.

### **9.1 المعماریة والانتشار الأمامي (Forward Propagation)**

لنفترض شبكة بطبقة مخفية واحدة.

* $Z\_1 \= X W\_1 \+ b\_1$  
* $A\_1 \= \\sigma(Z\_1)$ (تفعيل الطبقة المخفية)  
* $Z\_2 \= A\_1 W\_2 \+ b\_2$  
* $A\_2 \= \\sigma(Z\_2)$ (المخرج النهائي)

### **9.2 الاشتقاق الدقيق للانتشار الخلفي (Backpropagation)**

الهدف هو حساب مشتقة دالة الخطأ $E$ بالنسبة للأوزان لتحديثها. لنأخذ الوزن $W\_2$ كمثال.26  
باستخدام قاعدة السلسلة (Chain Rule):

$$\\frac{\\partial E}{\\partial W\_2} \= \\frac{\\partial E}{\\partial A\_2} \\cdot \\frac{\\partial A\_2}{\\partial Z\_2} \\cdot \\frac{\\partial Z\_2}{\\partial W\_2}$$  
لنقم بفك كل جزء (بافتراض خطأ تربيعي ودالة تفعيل Sigmoid):

1. $\\frac{\\partial E}{\\partial A\_2} \= \-(y \- A\_2)$  
2. $\\frac{\\partial A\_2}{\\partial Z\_2} \= A\_2(1 \- A\_2)$  
3. $\\frac{\\partial Z\_2}{\\partial W\_2} \= A\_1$

إذن، التدرج هو:

$$\\delta\_2 \= (A\_2 \- y) \\odot A\_2(1 \- A\_2)$$

$$\\frac{\\partial E}{\\partial W\_2} \= A\_1^T \\delta\_2$$  
بالنسبة للطبقة المخفية $W\_1$، نمرر الخطأ $\\delta\_2$ للخلف:

$$\\delta\_1 \= (\\delta\_2 W\_2^T) \\odot A\_1(1 \- A\_1)$$

$$\\frac{\\partial E}{\\partial W\_1} \= X^T \\delta\_1$$

### **9.3 كود الشبكة العصبية الكامل (NumPy Only)**

28

Python

import numpy as np

class NeuralNetwork:  
    def \_\_init\_\_(self, input\_size, hidden\_size, output\_size):  
        \# تهيئة الأوزان بشكل عشوائي لتكسير التناظر (Symmetry Breaking)  
        self.W1 \= np.random.randn(input\_size, hidden\_size) \* 0.01  
        self.b1 \= np.zeros((1, hidden\_size))  
        self.W2 \= np.random.randn(hidden\_size, output\_size) \* 0.01  
        self.b2 \= np.zeros((1, output\_size))

    def sigmoid(self, z):  
        return 1 / (1 \+ np.exp(-z))

    def sigmoid\_derivative(self, a):  
        return a \* (1 \- a)

    def forward(self, X):  
        self.z1 \= np.dot(X, self.W1) \+ self.b1  
        self.a1 \= self.sigmoid(self.z1)  
        self.z2 \= np.dot(self.a1, self.W2) \+ self.b2  
        self.a2 \= self.sigmoid(self.z2)  
        return self.a2

    def backward(self, X, y, output, learning\_rate):  
        m \= X.shape  
          
        \# حساب تدرجات الطبقة الثانية  
        error\_output \= output \- y  
        d\_output \= error\_output \* self.sigmoid\_derivative(output)  
        dW2 \= np.dot(self.a1.T, d\_output) / m  
        db2 \= np.sum(d\_output, axis=0, keepdims=True) / m

        \# حساب تدرجات الطبقة الأولى (Backpropagation)  
        error\_hidden \= np.dot(d\_output, self.W2.T)  
        d\_hidden \= error\_hidden \* self.sigmoid\_derivative(self.a1)  
        dW1 \= np.dot(X.T, d\_hidden) / m  
        db1 \= np.sum(d\_hidden, axis=0, keepdims=True) / m

        \# تحديث الأوزان  
        self.W1 \-= learning\_rate \* dW1  
        self.b1 \-= learning\_rate \* db1  
        self.W2 \-= learning\_rate \* dW2  
        self.b2 \-= learning\_rate \* db2

    def train(self, X, y, epochs=10000, lr=0.1):  
        for i in range(epochs):  
            output \= self.forward(X)  
            self.backward(X, y, output, lr)  
            if i % 1000 \== 0:  
                loss \= np.mean(np.square(y \- output))  
                print(f"Epoch {i}, Loss: {loss:.4f}")

## ---

**الفصل العاشر: التعلم غير المشرف (Unsupervised Learning)**

هنا لا توجد مخرجات $y$ لنتعلم منها. الهدف هو استكشاف الهيكل الخفي في البيانات.

### **10.1 خوارزمية K-Means**

تسعى لتقسيم البيانات إلى $K$ مجموعات (Clusters) بحيث يتم تقليل مجموع مربعات المسافات داخل كل مجموعة (WCSS).  
آلية التنبؤ predict في K-Means بسيطة: نحسب المسافة بين النقطة الجديدة وكل المراكز ونختار الأقرب.30

### **10.2 تحليل المكونات الأساسية (PCA)**

تقنية لتقليل الأبعاد تعتمد على الجبر الخطي.

1. تمركز البيانات (Centering).  
2. حساب مصفوفة التغاير (Covariance Matrix): $\\Sigma \= \\frac{1}{m} X^T X$.  
3. حساب القيم الذاتية والمتجهات الذاتية لـ $\\Sigma$.  
4. المتجهات الذاتية ذات القيم الأكبر تمثل المحاور التي تحتفظ بأكبر قدر من المعلومات (التباين). إسقاط البيانات عليها يقلل حجمها مع الحفاظ على جوهرها.5

## ---

**الفصل الحادي عشر: التعلم المعزز (Reinforcement Learning)**

يختلف هذا المجال جذرياً حيث يتعلم الوكيل (Agent) من خلال التفاعل مع البيئة وتلقي المكافآت أو العقوبات.

### **11.1 عمليات اتخاذ القرار الماركوفية (MDP)**

الإطار الرياضي لـ RL يتكون من حالات $S$، إجراءات $A$، احتمالات انتقال $P(s'|s,a)$، ومكافآت $R$.

### **11.2 معادلة بيلمان (Bellman Equation)**

حجر الزاوية في RL. تنص على أن قيمة حالة معينة تساوي المكافأة الفورية زائد قيمة الحالة التالية مخصومة بمعامل $\\gamma$.

$$V(s) \= \\max\_a \\left( R(s,a) \+ \\gamma \\sum\_{s'} P(s'|s,a) V(s') \\right)$$

.33

### **11.3 خوارزمية Q-Learning**

تسمح للوكيل بتعلم السياسة المثلى دون معرفة نموذج البيئة (Model-Free). يتم تحديث جدول Q-Table تكرارياً:

$$Q(s,a) \\leftarrow Q(s,a) \+ \\alpha \[r \+ \\gamma \\max\_{a'} Q(s', a') \- Q(s,a)\]$$

الحد $\\max\_{a'} Q(s', a')$ يمثل التقدير الأفضل للمستقبل، والفرق بينه وبين القيمة الحالية يسمى خطأ التوقع الزمني (TD Error).35

## ---

**الفصل الثاني عشر: دراسات حالة من الواقع العملي (Real-World Case Studies)**

الربط بين النظرية والتطبيق هو ما يميز الخبير. فيما يلي تحليل تقني لكيفية استخدام هذه الخوارزميات في شركات عالمية.

### **12.1 Netflix: التوصية عبر تحليل المصفوفات (Matrix Factorization)**

في مسابقة Netflix Prize، كان التحدي هو التنبؤ بتقييمات المستخدمين للأفلام. البيانات عبارة عن مصفوفة ضخمة جداً وفارغة (Sparse Matrix).  
الحل التقني: استخدموا SVD لتحليل المصفوفة $R$ إلى مصفوفتين مدمجتين (Embeddings) للمستخدمين والأفلام ($P$ و $Q$). التقييم المتوقع هو $r\_{ui} \= p\_u \\cdot q\_i$. يتم تعلم هذه المتجهات باستخدام Gradient Descent لتقليل الخطأ على التقييمات الموجودة. اكتشف النموذج تلقائياً مفاهيم مجردة (مثل "أفلام درامية"، "أفلام تعجب النساء") كأبعاد في هذه المتجهات.36

### **12.2 Uber: التنبؤ بالطلب باستخدام LSTM**

تتعامل Uber مع سلاسل زمنية معقدة تتأثر بالطقس، الأحداث الرياضية، والأعياد. النماذج الخطية التقليدية (ARIMA) تفشل في التقاط هذه العلاقات غير الخطية المعقدة.  
الحل التقني: استخدمت Uber شبكات LSTM (Long Short-Term Memory) مع نوافذ منزلقة (Sliding Windows). يتم دمج البيانات التاريخية للطلب مع ميزات خارجية (Exogenous Features) وتمريرها عبر طبقات التشفير (Autoencoders) لاستخراج الميزات قبل التنبؤ. هذا النهج قلل الخطأ بشكل كبير خلال الأحداث القصوى (Extreme Events).37

### **12.3 Rossmann: قوة تضمين الكيانات (Entity Embeddings)**

في مسابقة Kaggle لتوقع المبيعات، تفوق حل يستخدم الشبكات العصبية بفضل تقنية مبتكرة للتعامل مع البيانات الفئوية (Categorical Data) مثل "رقم المتجر" أو "يوم الأسبوع".  
الابتكار: بدلاً من استخدام One-Hot Encoding الذي ينتج متجهات ضخمة ومبعثرة، تم تعلم "تضمين" (Embedding) لكل فئة. أي تم تحويل "يوم الجمعة" إلى متجه كثيف (Dense Vector) مثل \[0.2, \-0.5, 0.9\]. المثير للدهشة أن النموذج تعلم تمثيل المتاجر القريبة جغرافياً بمتجهات متشابهة رياضياً، دون أن يتم تزويده بالإحداثيات الجغرافية صراحة، بل استنتجها فقط من أنماط المبيعات المتشابهة.39

## ---

**الفصل الثالث عشر: تمارين واختبارات شاملة**

### **13.1 أسئلة نظرية (Theoretical Quiz)**

1. **اشتقاق:** انطلق من دالة التكلفة للانحدار اللوجستي واشتق قاعدة التحديث للتدرج. أثبت أنها تتطابق شكلياً مع قاعدة تحديث الانحدار الخطي.  
2. **SVM:** لماذا نستخدم المشكلة الثنائية (Dual Problem) بدلاً من الأولية (Primal) عند استخدام Kernel؟ (تلميح: انظر لأبعاد المتجهات $w$).  
3. **أشجار القرار:** لديك بيانات. احسب Gini Impurity والإنتروبيا. أي المقياسين أكثر حساسية للشوائب؟

### **13.2 تحديات برمجية (Coding Challenges)**

1. **KNN المخصص:** قم بتعديل فئة KNNScratch لتستخدم "أوزان المسافة" (Distance Weights)، بحيث يكون للجيران الأقرب تأثير أكبر في التصويت من الجيران الأبعد.  
   * *تلميح:* بدلاً من التصويت بـ \+1، صوت بـ $1/distance$.  
2. **الانحدار متعدد المتغيرات:** استخدم فئة LinearRegression لنمذجة بيانات بولي نوميال (Polynomial) عن طريق إنشاء ميزات جديدة ($x^2, x^3$) يدوياً قبل التدريب.  
3. **كشف الاحتيال:** صمم Autoencoder باستخدام Keras لبيانات بطاقات الائتمان. درب النموذج على العمليات "الطبيعية" فقط. عند الاختبار، استخدم "خطأ إعادة البناء" (Reconstruction Error) كدرجة للكشف عن الاحتيال (Anomaly Score).40

## ---

**الخاتمة**

لقد قمنا في هذا التقرير برحلة عميقة عبر تضاريس التعلم الآلي، بدءاً من البنية الذرية للخوارزميات المتمثلة في المعادلات الرياضية، مروراً ببنائها كبرمجيات حية، وصولاً إلى تطبيقاتها العملاقة التي تشكل عالمنا الرقمي اليوم. إن إتقان هذا المجال لا يتأتى بمجرد استدعاء دوال جاهزة من Scikit-Learn، بل يتطلب الفهم العميق لما يدور داخل "الصندوق الأسود". هذا الفهم هو ما يمنحك القدرة على تشخيص الأخطاء، وتحسين النماذج، وابتكار حلول جديدة لمشكلات لم يسبق حلها.

#### **المصادر التي تم الاقتباس منها**

1. Machine learning \- Wikipedia, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://en.wikipedia.org/wiki/Machine\_learning](https://en.wikipedia.org/wiki/Machine_learning)  
2. Maths for Machine Learning \- GeeksforGeeks, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.geeksforgeeks.org/machine-learning/machine-learning-mathematics/](https://www.geeksforgeeks.org/machine-learning/machine-learning-mathematics/)  
3. Complete Roadmap of Mathematics for Machine Learning | HowAIWorks.ai, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://howaiworks.ai/blog/roadmap-mathematics-machine-learning-2025](https://howaiworks.ai/blog/roadmap-mathematics-machine-learning-2025)  
4. How to Get Started with Machine Learning: A Practical Roadmap, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/@digitalconsumer777/how-to-get-started-with-machine-learning-a-practical-roadmap-40c15013f105](https://medium.com/@digitalconsumer777/how-to-get-started-with-machine-learning-a-practical-roadmap-40c15013f105)  
5. Principal component analysis \- Wikipedia, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://en.wikipedia.org/wiki/Principal\_component\_analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)  
6. Linear regression: Gradient descent | Machine Learning \- Google for Developers, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent)  
7. Roadmap of Mathematics for Machine Learning | by Hrisav Bhowmick | Analytics Vidhya, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/analytics-vidhya/roadmap-of-mathematics-for-machine-learning-48f23baa57](https://medium.com/analytics-vidhya/roadmap-of-mathematics-for-machine-learning-48f23baa57)  
8. Bias–variance tradeoff \- Wikipedia, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://en.wikipedia.org/wiki/Bias%E2%80%93variance\_tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)  
9. Gradient Descent Derivation \- Chris McCormick, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://mccormickml.com/2014/03/04/gradient-descent-derivation/](https://mccormickml.com/2014/03/04/gradient-descent-derivation/)  
10. Linear Regression Implementation From Scratch using Python ..., تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.geeksforgeeks.org/machine-learning/linear-regression-implementation-from-scratch-using-python/](https://www.geeksforgeeks.org/machine-learning/linear-regression-implementation-from-scratch-using-python/)  
11. Gradient Descent with Linear Regression, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://ethen8181.github.io/machine-learning/linear\_regression/linear\_regession.html](https://ethen8181.github.io/machine-learning/linear_regression/linear_regession.html)  
12. Fighting Overfitting With L1 or L2 Regularization: Which One Is Better? \- Neptune.ai, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://neptune.ai/blog/fighting-overfitting-with-l1-or-l2-regularization](https://neptune.ai/blog/fighting-overfitting-with-l1-or-l2-regularization)  
13. Understanding L1 and L2 regularization with analytical and probabilistic views | by Yuki Shizuya | Intuition, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.ccs.neu.edu/home/vip/teach/MLcourse/1.1\_LinearRegression/LectureNotes/L1\_and\_L2\_reg\_regression,pdf.pdf](https://www.ccs.neu.edu/home/vip/teach/MLcourse/1.1_LinearRegression/LectureNotes/L1_and_L2_reg_regression,pdf.pdf)  
14. Implementing K-means Clustering from Scratch \- in Python | Mustafa Murat ARAT, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://mmuratarat.github.io/2019-07-23/kmeans\_from\_scratch](https://mmuratarat.github.io/2019-07-23/kmeans_from_scratch)  
15. Implementing the k-Nearest Neighbors (KNN) Algorithm from Scratch in Python \- Medium, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/@avijit.bhattacharjee1996/implementing-the-k-nearest-neighbors-knn-algorithm-from-scratch-in-python-3b83a4fe8](https://medium.com/@avijit.bhattacharjee1996/implementing-the-k-nearest-neighbors-knn-algorithm-from-scratch-in-python-3b83a4fe8)  
16. K-Nearest Neighbors (KNN) Classifier: Step-by-Step Python Implementation from Scratch, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://kenwuyang.com/posts/2022\_11\_02\_k\_nearest\_neighbors\_knn\_classifier\_step\_by\_step\_python\_implementation\_from\_scratch/](https://kenwuyang.com/posts/2022_11_02_k_nearest_neighbors_knn_classifier_step_by_step_python_implementation_from_scratch/)  
17. Gini Index and Entropy | 2 Ways to Measure Impurity in Data, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://datasciencedojo.com/blog/gini-index-and-entropy/](https://datasciencedojo.com/blog/gini-index-and-entropy/)  
18. Decision Trees Explained \- Entropy, Information Gain, Gini Index, CCP Pruning.., تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c/](https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c/)  
19. A Simple Explanation of Gini Impurity \- victorzhou.com, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://victorzhou.com/blog/gini-impurity/](https://victorzhou.com/blog/gini-impurity/)  
20. What Is Random Forest? | IBM, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.ibm.com/think/topics/random-forest](https://www.ibm.com/think/topics/random-forest)  
21. Build a Random Forest in Python from Scratch \- Inside Learning Machines, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://insidelearningmachines.com/build-a-random-forest-in-python/](https://insidelearningmachines.com/build-a-random-forest-in-python/)  
22. Supervised Machine Learning with Gradient Boosting \- University of San Diego Online Degrees, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://onlinedegrees.sandiego.edu/supervised-machine-learning-with-gradient-boosting/](https://onlinedegrees.sandiego.edu/supervised-machine-learning-with-gradient-boosting/)  
23. All You Need to Know about Gradient Boosting Algorithm − Part 1\. Regression \- Medium, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/data-science/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502](https://medium.com/data-science/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502)  
24. Support Vector Machines \- Stanford Engineering Everywhere, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://see.stanford.edu/materials/aimlcs229/cs229-notes3.pdf](https://see.stanford.edu/materials/aimlcs229/cs229-notes3.pdf)  
25. Support Vector Machines \- Dual formulation and Kernel Trick, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.cs.cmu.edu/\~aarti/Class/10315\_Fall20/lecs/svm\_dual\_kernel.pdf](https://www.cs.cmu.edu/~aarti/Class/10315_Fall20/lecs/svm_dual_kernel.pdf)  
26. Derivation of Backpropagation, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.cs.swarthmore.edu/\~meeden/cs81/s10/BackPropDeriv.pdf](https://www.cs.swarthmore.edu/~meeden/cs81/s10/BackPropDeriv.pdf)  
27. Lecture 6: Backpropagation, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.cs.toronto.edu/\~rgrosse/courses/csc321\_2017/readings/L06%20Backpropagation.pdf](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L06%20Backpropagation.pdf)  
28. Deep Neural net with forward and back propagation from scratch \- Python \- GeeksforGeeks, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.geeksforgeeks.org/python/deep-neural-net-with-forward-and-back-propagation-from-scratch-python/](https://www.geeksforgeeks.org/python/deep-neural-net-with-forward-and-back-propagation-from-scratch-python/)  
29. Let's code a Neural Network in plain NumPy | by Piotr Skalski | TDS Archive \- Medium, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/data-science/lets-code-a-neural-network-in-plain-numpy-ae7e74410795](https://medium.com/data-science/lets-code-a-neural-network-in-plain-numpy-ae7e74410795)  
30. Building a KMeans Clustering Algorithm from Scratch in Python | by Savaliyajainish, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/@savaliyajainish095/building-a-kmeans-clustering-algorithm-from-scratch-in-python-aa277d54866b](https://medium.com/@savaliyajainish095/building-a-kmeans-clustering-algorithm-from-scratch-in-python-aa277d54866b)  
31. How To Build Your Own K-Means Algorithm Implementation in Python From Scratch With K-Means++ Initialization | Towards AI, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://towardsai.net/p/l/how-to-build-your-own-k-means-algorithm-implementation-in-python-from-scratch-with-k-means-initialization](https://towardsai.net/p/l/how-to-build-your-own-k-means-algorithm-implementation-in-python-from-scratch-with-k-means-initialization)  
32. Mathematical understanding of Principal Component Analysis | by Yuki Shizuya | Intuition, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://medium.com/intuition/mathematical-understanding-of-principal-component-analysis-6c761004c2f8](https://medium.com/intuition/mathematical-understanding-of-principal-component-analysis-6c761004c2f8)  
33. Understanding the Bellman Equation in Reinforcement Learning \- DataCamp, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.datacamp.com/tutorial/bellman-equation-reinforcement-learning](https://www.datacamp.com/tutorial/bellman-equation-reinforcement-learning)  
34. Deriving Bellman's Equation in Reinforcement Learning \- Stats StackExchange, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://stats.stackexchange.com/questions/243384/deriving-bellmans-equation-in-reinforcement-learning](https://stats.stackexchange.com/questions/243384/deriving-bellmans-equation-in-reinforcement-learning)  
35. 18.2 Q-Learning, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://kenndanielso.github.io/mlrefined/blog\_posts/18\_Reinforcement\_Learning\_Foundations/18\_2\_Q\_learning.html](https://kenndanielso.github.io/mlrefined/blog_posts/18_Reinforcement_Learning_Foundations/18_2_Q_learning.html)  
36. Netflix Recommendations: Beyond the 5 stars (Part 1\) | by Netflix ..., تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429](https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429)  
37. Engineering Extreme Event Forecasting at Uber with Recurrent ..., تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.uber.com/blog/neural-networks/](https://www.uber.com/blog/neural-networks/)  
38. Engineering Uncertainty Estimation in Neural Networks for Time Series Prediction at Uber, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.uber.com/blog/neural-networks-uncertainty-estimation/](https://www.uber.com/blog/neural-networks-uncertainty-estimation/)  
39. Chapter 9 \- Tabular Modelling Deep Dive | Niyazi Kemer, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://niyazikemer.com/fastbook/2021/09/18/chapter-09.html](https://niyazikemer.com/fastbook/2021/09/18/chapter-09.html)  
40. part of 9th place (denoising auto-encoder NN) \- Kaggle, تم الوصول بتاريخ ‎ديسمبر 30, 2025، [https://www.kaggle.com/competitions/petfinder-adoption-prediction/writeups/bestoverfitting-part-of-9th-place-denoising-auto-e](https://www.kaggle.com/competitions/petfinder-adoption-prediction/writeups/bestoverfitting-part-of-9th-place-denoising-auto-e)