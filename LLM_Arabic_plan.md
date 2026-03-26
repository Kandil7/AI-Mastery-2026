<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# عاوز اعمل LLM Arabic يكون بمستوي عالم خبير باللغه العربيه و علومها و شاعرا و محقق لغوي

أقصر طريق عملي إنك **متحتاجش** تبني موديل من الصفر، لكن تختار Base قوي وتعمل له طبقة تدريب متخصصة تخليه بمستوى “عالم لغة + شاعر + محقق لغوي”.[^1_1][^1_2]

## 1. اختيار الـ Base Model

أفضل اختيارات كبداية (موديل واحد منهم فقط في أول نسخة):

- ALLaM-2-7B-Instruct (أو ALLaM‑7B‑Instruct‑preview):
    - متدرَّب خصيصًا على العربية + الإنجليزية وبري‑تريننج ضخم 4T إنجليزي + 1.2T عربي/إنجليزي.[^1_3][^1_1]
    - 7B parameters → مناسب لـ QLoRA على GPU واحد قوي (24–48GB) أو سحابة.[^1_4][^1_1]
- Qwen2.5‑7B/8B‑Instruct:
    - Reasoning قوي ومتعدد اللغات، وأداء ممتاز في بنشماركات عامة، يسهل تخصيصه بالعربي.[^1_5][^1_6][^1_4]

جدول سريع:


| العنصر | ALLaM‑7B-Instruct | Qwen2.5‑7B/8B |
| :-- | :-- | :-- |
| تركيز العربية | عالٍ جدًّا[^1_1][^1_3] | جيد إلى ممتاز[^1_6] |
| الحجم | 7B | 7B–8B |
| الأنسب لك | عالم لغوي عربي[^1_4][^1_7] | استدلال عام قوي[^1_6] |

لو هدفك “عالم لغة + شاعر”، ابدأ بـ ALLaM‑7B أو Qwen2.5‑7B أيهما أسهل لك في التشغيل حاليًا.[^1_1][^1_4]

## 2. تقسيم شخصية الموديل (القدرات المطلوبة)

أنت عايزه يجمع بين 3 أدوار:

1. عالم في علوم العربية: نحو، صرف، بلاغة، معاجم، قراءات، تفسير لغوي، أساليب العرب.[^1_8][^1_9]
2. شاعر: إنتاج شعر موزون (بحور الخليل)، نقد شعري، شرح صور بلاغية.[^1_8]
3. محقق لغوي: تصحيح إعراب، كشف لحن، مقارنة نسخ نصوص، اقتراح ضبط وتحقيق.[^1_9][^1_8]

هذا معناه أن الداتا وـ Evaluation لازم تتقسم لـ 3 “Sub‑skills”، وكل واحد له نوع بيانات واختبارات مختلفة.[^1_10][^1_9][^1_8]

## 3. مكوّن البيانات (High‑Level Roadmap)

على مستوى 3–6 شهور كمطوّر فرد (بدون الدخول في السكربتات الآن):[^1_11][^1_9][^1_10][^1_8]

1. طبقة Pretraining/Corpus (قراءة عربية عميقة):
    - كوربس عربي عام + تراثي (شاملة كتب التراث، الشروح، المعاجم، الدواوين).[^1_12][^1_9]
    - تنظيمه وتنضيفه (تطبيع، إزالة التكرار، فلترة الجودة) باستخدام أدوات مثل datatrove أو سكربتاتك.[^1_9][^1_12]
2. طبقة Post‑training (Instruction + Chat):
    - نحو/صرف/بلاغة:
        - أسئلة إعراب، طلب شرح قاعدة، تحويل أمثلة إلى سؤال/جواب، توضيح الصور البيانية.[^1_8][^1_9]
    - شعر:
        - طلب نظم على بحر معيّن، طلب نقد بيت، شرح استعارات وتشبيهات، مقارنة أساليب الشعراء.[^1_8]
    - تحقيق لغوي:
        - إدخال نص به أخطاء نحوية/إملائية وطلب تصحيح وتعليل، تمييز قراءة فصيحة من دارجة، اختيار الوجه الأصح من صيغ متعددة مع تعليل.[^1_9][^1_8]
3. طبقة بيانات صناعية (Synthetic):
    - توليد آلاف الأمثلة النحوية/البلاغية/الشعرية باستخدام LLM كبير ثم مراجعتها (LLM‑auditor + فلترة يدوية محدودة).[^1_11][^1_9][^1_8]
4. Evaluation:
    - استخدام بنشماركات عربية + مجموعة اختبارات يدوية أنت تبنيها (أسئلة نحو وبلاغة وشعر وتحقيق).[^1_12][^1_10][^1_9][^1_8]

## 4. خطة تدريب مبسطة (بدون تفاصيل كود)

الخط العملي الممكن تنفيذه بـ QLoRA على 7B:

1. اختيار Base:
    - ALLaM‑7B‑Instruct‑preview أو ALLaM‑2‑7B‑Instruct أو Qwen2.5‑7B‑Instruct.[^1_6][^1_2][^1_4][^1_1]
2. تجهيز Dataset Instruction على هيئة JSONL:
    - حقول مثل: instruction, input, output, metadata (skill = nahw/balagha/poetry/taḥqīq, difficulty).[^1_9][^1_8]
3. QLoRA Fine‑tuning:
    - تدريب على 30K–100K مثال عربي عالي الجودة في علوم العربية والشعر.[^1_4][^1_11]
4. اختبار يدوي + Iteration:
    - كل دورة تدريب بعدها تختبر بنفس مجموعة أسئلة “معيارية” لتشوف تحسّن الإعراب والبلاغة وقوة الشعر.[^1_10][^1_4][^1_9]

## 5. خطوة قادمة مفيدة لك

أنت عندك أصلًا برومبتات Deep Research وخطط تفصيلية للداتا وQLoRA “جاهزة للتحويل إلى سكربتات”.[^1_13][^1_14][^1_4][^1_8]
الخطوة المنطقية الآن: نربط الكلام ده بوضعك العملي (الهاردوير/السحابة + اختيار Base واحد + تحديد أول تخصص: نحو/بلاغة/شعر).

ما هو الهاردوير المتاح عندك حاليًا لتدريب الموديل (VRAM وعدد الـ GPUs أو نوع خدمة السحابة اللي تقدر تستخدمها)؟
<span style="display:none">[^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21]</span>

<div align="center">⁂</div>

[^1_1]: https://ai.azure.com/catalog/models/ALLaM-2-7b-instruct

[^1_2]: https://noqta.tn/en/tutorials/getting-started-with-allam-7b-instruct-preview

[^1_3]: https://arxiv.org/html/2407.15390v1

[^1_4]: https://www.perplexity.ai/search/816af35a-818f-45b9-a4a1-fcdede4dcd22

[^1_5]: https://www.webelight.com/blog/exploring-alibaba-qwen-two-point-five-model-a-potential-deepseek-rival

[^1_6]: https://qwenlm.github.io/blog/qwen2.5/

[^1_7]: https://www.perplexity.ai/search/509abe50-c152-4d49-b90b-8738f800f8fe

[^1_8]: https://www.perplexity.ai/search/4285148c-5563-4d51-bec7-d85b510d4db1

[^1_9]: https://www.perplexity.ai/search/33b369a1-2f21-469c-bc14-5cc6bfae35b8

[^1_10]: https://arxiv.org/html/2506.01920v1

[^1_11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12900375/

[^1_12]: https://arxiv.org/pdf/2506.01920.pdf

[^1_13]: https://www.perplexity.ai/search/45c61514-02e3-41d6-9fb4-60a2ab6f6986

[^1_14]: https://www.perplexity.ai/search/8fe0c614-b910-4d6a-beaf-292da30c7b71

[^1_15]: https://www.youtube.com/watch?v=31fnLx1BqQQ

[^1_16]: https://www.youtube.com/watch?v=tiFrbA1XPRA

[^1_17]: https://faculty.uobasrah.edu.iq/uploads/teaching/1700213789.pdf

[^1_18]: https://dn790006.ca.archive.org/0/items/Www.almaktabah.net-dohaMag/054-201204.pdf

[^1_19]: http://pdf.alanba.com.kw/pdf/2017/06/19-06-2017/19-06-2017.pdf

[^1_20]: https://huggingface.co/TRoboto/masc_kenlm_3grams_lm/resolve/main/vocab-200000.txt?download=true

[^1_21]: https://www.alarab.co.uk/pdf/2014/01/09-01/p1000.pdf


---

# ايه الادوار اللي ممكن احتاجها من LLM Arabic

الـ LLM العربي ممكن تقسّمه لأدوار (Capabilities) واضحة، وكل دور تقدر تعمله Track مستقل في التدريب والتقييم.[^2_1][^2_2]

## 1. أدوار لغوية أساسية

- **مصحّح لغوي (Proofreader)**: تصحيح نحو، صرف، إملاء، تشكيل، مع تعليل الأخطاء واقتراح الصياغة الأدق.[^2_3][^2_1]
- معلّم لغة (Tutor): شرح قواعد، ضرب أمثلة، عمل تمارين وتصحيحها، تدريج الصعوبة.[^2_4][^2_2]
- محلّل نصي: استخراج إعراب، علاقات نحوية، بلاغة (تشبيه، استعارة، كناية)، وتحديد الأسلوب.[^2_5][^2_1]


## 2. أدوار فهم وإنتاج النص

- كاتب محتوى عربي: مقالات، تقارير، رسائل رسمية، محتوى تسويقي، محادثة خدمة عملاء.[^2_6][^2_7]
- ملخِّص النصوص: تلخيص كتب/مقالات/تقارير مع الحفاظ على النقاط الأساسية.[^2_3]
- محوّل أساليب: تحويل نص من فصيح إلى مبسّط، من أكاديمي إلى شعبي، أو العكس.[^2_1][^2_4]


## 3. أدوار معرفية وثقافية

- مجيب أسئلة (QA) عام: أسئلة عامة بالعربية حول التاريخ، الدين، العلوم… إلخ.[^2_8][^2_2]
- خبير تراث عربي: التعامل مع نصوص تراثية، شروح، دواوين، سياقات ثقافية عربية.[^2_2][^2_1]
- داعم لهجات: فهم/توليد لهجات (مصري، خليجي، مغربي…) في المحادثة وخدمة العملاء.[^2_7][^2_4][^2_6]


## 4. أدوار متقدمة (تخصّص “عالم لغة”)

- عالم نحو وبلاغة: تحليل أبيات ونصوص، شرح شواهد، تفكيك الأساليب، مقارنة مدارس النحو والبلاغة.[^2_5][^2_1]
- شاعر/ناقد أدبي: نظم شعر على بحور مختلفة، نقد القصائد، شرح الصور والرموز.[^2_1][^2_5]
- محقّق لغوي/نصي: مقارنة نسخ نصوص، اقتراح قراءات أرجح لغويًا، كشف الدخيل أو الركيك.[^2_9][^2_1]


## 5. أدوار تطبيقية (منتج/بيزنس)

- مساعد أعمال عربي: صياغة عقود، خطابات رسمية، مراسلات بريدية، سياسات داخلية.[^2_6][^2_7]
- محلّل رأي عام: تحليل سوشال ميديا عربية، استخلاص مشاعر واتجاهات.[^2_7][^2_3]
- مساعد تعليمي: شروحات مناهج لغة عربية، أسئلة امتحانات، تصحيح إجابات الطلبة.[^2_4][^2_2]

لو ركّزت في أول نسخة من الموديل على 3–4 أدوار بس، تحب الأولوية تبقى للدور التعليمي (معلّم نحو/بلاغة)، ولا للإنتاج الإبداعي (شعر وكتابة)، ولا للتحقيق اللغوي المتقدم؟
<span style="display:none">[^2_10][^2_11][^2_12][^2_13][^2_14][^2_15]</span>

<div align="center">⁂</div>

[^2_1]: https://arxiv.org/pdf/2505.08004.pdf

[^2_2]: https://arxiv.org/html/2510.13430v1

[^2_3]: https://aixplain.com/wp-content/uploads/2025/05/aiXplain-Arabic-Benchmark-Report-May-2025-v2.1.pdf

[^2_4]: https://cacm.acm.org/arab-world-regional-special-section/the-landscape-of-arabic-large-language-models/

[^2_5]: https://archive.org/download/ArabicRhetoricCourse-OpenUniversityOfNigeria/Arabic Rhetoric 1.pdf

[^2_6]: https://arabic.ai/arabic-ai-llm/

[^2_7]: https://widebot.ai/aql/llm

[^2_8]: https://aclanthology.org/2022.osact-1.19/

[^2_9]: https://arxiv.org/html/2506.01920v1

[^2_10]: https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Arabic

[^2_11]: https://falcon-lm.github.io/blog/falcon-arabic/

[^2_12]: https://arxiv.org/pdf/2506.01920.pdf

[^2_13]: https://huggingface.co/blog/leaderboard-arabic

[^2_14]: https://github.com/tiiuae/Arabic-LLM-Benchmarks

[^2_15]: https://arxiv.org/abs/2510.13430


---

# للدور التعليمي (معلّم نحو/بلاغة)

لدور **معلّم** نحو وبلاغة، أهم شيء إنك تعرّف للموديل “وظائف تعليمية” واضحة تبني عليها الداتا والتقييم.[^3_1][^3_2]

## 1. وظائف المعلّم (Teaching Functions)

- شرح القواعد: تعريف القاعدة، تقسيمها، أمثلة صحيحة/خاطئة، وبيان العلل.[^3_3][^3_4]
- توليد تمارين: أسئلة اختيار من متعدد، إعراب، تصحيح خطأ، إكمال جملة، استخراج صورة بلاغية.[^3_5][^3_6]
- تصحيح الإجابات مع تعليق: يحكم بصح/خطأ، يوضّح موضع الخطأ، ويقترح تصحيح مع تفسير مختصر.[^3_7][^3_8][^3_2]
- حوار سقراطي: بدل ما يجاوب مباشرة، يسأل المتعلّم أسئلة تقوده لاكتشاف القاعدة بنفسه.[^3_2]


## 2. مهارات النحو المطلوبة

- تحليل الجملة: تمييز نوع الجملة، الفاعل، المفعول، المبتدأ، الخبر، التوابع… إلخ.[^3_9][^3_10]
- الإعراب التفصيلي: إعراب كلمة كلمة مع بيان العامل والعلامة والتقدير إن وجد.[^3_4][^3_10]
- كشف الأخطاء النحوية: مقارنة بين صيغ، واختيار الأصح وتعليل الخطأ في غيرها (نمط GEC عربي).[^3_8][^3_7]
- إعادة صياغة صحيحة: تحويل جملة ملخبطة إلى صياغة فصيحة سليمة مع الحفاظ على المعنى.[^3_11][^3_10]


## 3. مهارات البلاغة المطلوبة

- التعرف على الصور البلاغية: تشبيه، استعارة، كناية، مجاز، طباق، جناس… إلخ.[^3_3][^3_4]
- شرح الأثر الجمالي: لماذا هذه الصورة موفّقة؟ ما علاقتها بالمعنى والسياق؟[^3_3]
- المقارنة بين أساليب: أسلوب خبري/إنشائي، تقرير/إنشاء، جزالة/ركاكة، اختيار الأبلغ من صياغتين.[^3_10][^3_3]
- إنشاء أمثلة تعليمية: تأليف أمثلة صغيرة توضّح كل نوع بلاغي، مع تدرّج في الصعوبة.[^3_5][^3_4]


## 4. سلوك المعلّم (Pedagogy)

- تدرّج في الشرح: يبدأ بمستوى الطالب (يُستدل من برومبت: “أنا مبتدئ/متوسط/متقدّم”).[^3_1][^3_2]
- feedback بنّاء: لا يكتفي بالتصحيح، بل يوضّح “لماذا” و”كيف تتجنّب الخطأ لاحقًا”.[^3_2]
- استخدام أسئلة تشخيصية: في أول الحوار يختبر مستواه ببعض الأسئلة السريعة.[^3_6][^3_2]


## 5. كيف تعكس هذا في الداتا

لكل مثال تعليمي، حاول تحط:

- instruction: دور المعلّم + نوع النشاط (شرح قاعدة، تمرين، تصحيح إجابة…).[^3_5][^3_1]
- input: جملة/سؤال الطالب/نص مطلوب تحليله.
- output: جواب منظّم: (1) الحكم، (2) الشرح، (3) أمثلة إضافية/نصيحة.[^3_7][^3_8][^3_2]

لو هنبدأ بتراكم داتا يدوي/نص تراثي، تحب أول batch يكون تمارين إعراب ولا تمارين بلاغة (تشبيه/استعارة مثلًا)؟
<span style="display:none">[^3_12][^3_13][^3_14][^3_15][^3_16][^3_17]</span>

<div align="center">⁂</div>

[^3_1]: https://cacm.acm.org/arab-world-regional-special-section/the-landscape-of-arabic-large-language-models/

[^3_2]: https://arxiv.org/html/2502.05467v1

[^3_3]: https://www.academia.edu/85516391/A_Primer_of_Arabic_Rhetoric

[^3_4]: https://archive.org/download/ArabicRhetoricCourse-OpenUniversityOfNigeria/Arabic Rhetoric 1.pdf

[^3_5]: https://aclanthology.org/2025.arabicnlp-main.36/

[^3_6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12404542/

[^3_7]: https://arxiv.org/html/2312.08400v1

[^3_8]: https://aclanthology.org/2023.arabicnlp-1.9/

[^3_9]: https://arxiv.org/html/2510.13430v1

[^3_10]: https://arxiv.org/pdf/2505.08004.pdf

[^3_11]: https://localazy.com/blog/ai-8-llm-arabic-models-tested-to-translate

[^3_12]: https://falcon-lm.github.io/blog/falcon-arabic/

[^3_13]: https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Arabic

[^3_14]: https://www.sciencedirect.com/science/article/pii/S2666920X2500089X

[^3_15]: https://aclanthology.org/2024.icnlsp-1.34.pdf

[^3_16]: https://www.linkedin.com/posts/mandar-marathe-uk_github-al-balaghaarabic-rhetoric-the-activity-7396355388436160512-WBU4

[^3_17]: https://github.com/tiiuae/Arabic-LLM-Benchmarks


---

# عايز مزيج بسيط بين نحو وبلاغة للبداية

لبداية فيها مزيج بسيط نحو + بلاغة، خلّيه يؤدي “أنشطة تعليمية” موحّدة يجتمع فيها الإعراب مع شرح الصورة البلاغية.[^4_1][^4_2]

## 1. نوع الأنشطة اللي تبدأ بها

- جملة واحدة، مهمتان:

1) إعراب أساسيات (مبتدأ، خبر، فاعل، مفعول…).
2) تحديد لو فيها تشبيه/استعارة/كناية وشرحها باختصار.[^4_3][^4_1]
- جمل فيها خطأ نحوي أو تركيب ركيك:
    - اطلب منه: (أ) يصحّح، (ب) يوضّح نوع الخطأ، (ج) يبيّن إن كان في البلاغة ضعف (حشو، تكرار، صورة ضعيفة).[^4_4][^4_5]

مثال (كنمط داتا/سلوك):

- instruction:
“أعرب الجملة الآتية، ثم اذكر إن كان فيها صورة بلاغية (تشبيه/استعارة/كناية) وشرحها باختصار.”[^4_1]
- input:
“العلمُ نورٌ يبدّدُ ظلماتِ الجهلِ.”
- output (متوقّع):

1) إعراب منظم سطر بسطر.
2) يشرح أن هناك استعارة في تشبيه العلم بالنور أو الظلمات بالجهل، مع توضيح الأثر.[^4_6][^4_3]


## 2. قالب بسيط موحّد للداتا

لكل مثال مزيج نحو+بلاغة، خليه بالشكل ده (JSONL conceptual):

- instruction: نوع النشاط (إعراب + بلاغة + مستوى الطالب).
- input: الجملة/الفقرة أو سؤال الطالب.
- output:
    - قسم 1: الإعراب (منظم).
    - قسم 2: البلاغة (نوع الصورة + الشرح).
    - قسم 3 (اختياري): نصيحة تعليمية صغيرة للطالب.[^4_2][^4_1]


## 3. مستوى الصعوبة المبدئي

- جمل قصيرة من كتب تعليمية حديثة أو أمثلة من شروحات مبسّطة.[^4_7][^4_8]
- نوع بلاغة محدود في الأول (تشبيه صريح/استعارة مكنية/كناية عن صفة)، بدون دخول في دقائق علم المعاني.[^4_6][^4_3]

تحب أساعدك دلوقتي أبني لك 10–20 Template جاهزين (برومبت + input + output) تستخدمهم كأساس لتجميع/توليد داتا أولية للنحو+البلاغة؟
<span style="display:none">[^4_10][^4_11][^4_12][^4_13][^4_14][^4_15][^4_16][^4_17][^4_9]</span>

<div align="center">⁂</div>

[^4_1]: https://github.com/ARBML/CIDAR

[^4_2]: https://aclanthology.org/2025.arabicnlp-main.36/

[^4_3]: https://archive.org/download/ArabicRhetoricCourse-OpenUniversityOfNigeria/Arabic Rhetoric 1.pdf

[^4_4]: https://huggingface.co/papers/2312.08400

[^4_5]: https://arxiv.org/html/2312.08400v1

[^4_6]: https://www.academia.edu/85516391/A_Primer_of_Arabic_Rhetoric

[^4_7]: https://www.youtube.com/watch?v=mPqchkfWgEI

[^4_8]: https://u-caninstitute.com/our-blog/exercises-arabic-grammar/

[^4_9]: https://www.youtube.com/watch?v=4AXP-C3317k

[^4_10]: https://aclanthology.org/2025.arabicnlp-main.36.pdf

[^4_11]: https://www.linkedin.com/posts/computerengineering-graduate-program-kuwait-university_thesis-examination-announcement-titleevaluating-activity-7346458982128975872-ibKG

[^4_12]: https://www.verbling.com/find-teachers/arabic/grammar-development

[^4_13]: https://preply.com/en/online/s/arabic-grammar-tutors

[^4_14]: https://zenodo.org/records/15547781

[^4_15]: https://arxiv.org/abs/2312.08400

[^4_16]: https://www.youtube.com/watch?v=j-bG5TZIzLg

[^4_17]: https://huggingface.co/datasets/lightonai/ArabicWeb24


---

# عاوز اعمل كل الادوار مع اساس متين الموديل هيكون fine tuning

فهمت قصدك: عايز موديل واحد أساسه متين، وبالـ fine‑tuning يلبس “كل الأدوار” اللي كلمنا عنها (معلّم، مصحّح، شاعر، محقّق…).[^5_1]

## 1. الأساس المتين = اختيار Base + داتا نظيفة

- اختار Base عربي قوي جاهز (ALLaM‑7B/Qwen2.5‑7B/مشابه) ثم ركّز على جودة الداتا أكثر من حجمها.[^5_2][^5_3][^5_4]
- استخدم كوربس تعليمي/لغوي عالي الجودة كنواة (ArabicWeb‑Edu، ArabicWeb24، محتوى نحوي وبلاغي موثوق) بدل ويب خام.[^5_5][^5_6]


## 2. تقسيم الأدوار إلى “حزم مهام” (Heads)

بدل ما تخلط كل شيء، عرّف 4–5 حزم، وكل حزمة فيها أمثلة Instruction مميزة بميتابيانات:

- حزمة التعليم: شرح قواعد، تمارين نحو/بلاغة، تصحيح إجابات الطلبة.[^5_7][^5_8]
- حزمة التصحيح والتحرير: GEC عربي، إعادة صياغة، Proofreading.[^5_9][^5_10]
- حزمة الإبداع (شعر/كتابة): نظم، قصص قصيرة، تحسين الأسلوب.[^5_11][^5_7]
- حزمة التحقيق اللغوي/التراث: أسئلة عن الإعراب والبلاغة في نصوص تراثية، مثل أمثلة CIDAR.[^5_12][^5_13]
- حزمة المساعد العام/المحادثة: أسئلة عامة، مهام يومية، تعليم تفاعلي.[^5_14][^5_9]

كل مثال عندك يكون فيه metadata مثل: `"role": "tutor" | "proofreader" | "poet" | "muhhaqiq"` لتقدر توازنها في التدريب.[^5_12][^5_1]

## 3. مبدأ “الأساس قبل التخصص”

- أول Fine‑tune: ركّز على مهام عامة + تعليمية (نحو/بلاغة + تصحيح بسيط) بدون دخول عميق في التحقيق والشعر المعقّد.[^5_14][^5_1]
- جولات لاحقة: تعمل Fine‑tune إضافي (أو LoRA layers منفصلة) لأدوار متقدمة: شاعر، محقّق، خبير تراث.[^5_15][^5_12]


## 4. الاستفادة من داتا جاهزة

- CIDAR: فيها تعليمات نحو/بلاغة جاهزة بإخراج مرتب، ممتازة كأساس لحزمة “التعليم/التحقيق”.[^5_13][^5_12]
- ArabicWeb‑Edu: كوربس تعليمي عالي الجودة تقدر تستخرج منه نصوص وتمارين تعليمية.[^5_5]
- بنشماركات مثل ABB / OALL / CamelEval تساعدك تختبر إن الموديل فعلاً تحسّن في المهام العامة والتعليمية.[^5_16][^5_9][^5_14]


## 5. شكل Fine‑tuning العملي (مختصر)

- طبقة واحدة QLoRA على 7B، بداتا موحّدة schema (instruction, input, output, role, difficulty).[^5_12][^5_5]
- Sampling متوازن: مثلاً 40٪ تعليم نحو/بلاغة، 25٪ تصحيح، 20٪ مساعد عام، 15٪ شعر/تراث في أول نسخة.[^5_1][^5_14]
- بعد كل دورة تدريب، تختبر على:
    - بنشمارك عام (OALL/ABB).
    - حزمة أسئلة خاصة بيك للنحو والبلاغة والشعر.[^5_17][^5_18][^5_14]

لو حنرتّب شغلك الجاي، تحب أساعدك في: تصميم Schema نهائي للـ JSONL لكل الأدوار، ولا في اقتراح Mix نسب محددة (كم مثال لكل دور في أول 50K عينة)؟
<span style="display:none">[^5_19]</span>

<div align="center">⁂</div>

[^5_1]: https://arxiv.org/html/2510.13430v2

[^5_2]: https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard

[^5_3]: https://www.perplexity.ai/search/816af35a-818f-45b9-a4a1-fcdede4dcd22

[^5_4]: https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Arabic

[^5_5]: https://aclanthology.org/2025.arabicnlp-main.36.pdf

[^5_6]: https://lighton.ai/lighton-blogs/arabicweb24

[^5_7]: https://github.com/ARBML/CIDAR

[^5_8]: https://aclanthology.org/2025.arabicnlp-main.36/

[^5_9]: https://arxiv.org/html/2409.12623v1

[^5_10]: https://arxiv.org/abs/2312.08400

[^5_11]: https://arxiv.org/pdf/2505.08004.pdf

[^5_12]: https://huggingface.co/datasets/arbml/CIDAR

[^5_13]: https://huggingface.co/datasets/arbml/CIDAR/blob/main/README.md

[^5_14]: https://silma.ai/arabic-llm-benchmark

[^5_15]: https://arxiv.org/pdf/2402.03177.pdf

[^5_16]: https://www.tii.ae/insights/introducing-open-arabic-llm-leaderboard-empowering-arabic-language-modeling-community

[^5_17]: https://github.com/tiiuae/Arabic-LLM-Benchmarks

[^5_18]: https://crfm.stanford.edu/helm/arabic/latest/

[^5_19]: https://www.linkedin.com/pulse/arabic-llms-2025-benchmark-gulf-business-applications-rabehi-phd-ml8kf


---

# عايز سكيمة JSONL نهائية لكل الأدوار

هديك سكيمة واحدة مرِنة تلمّ كل الأدوار (معلّم، مصحّح، شاعر، محقّق…) على طريقة Instruction‑tuning الكلاسيكية (قريبة من CIDAR/InstAr) مع إضافة حقول للـ role وskill وmetadata.[^6_1][^6_2]

## 1. السكيمة المقترحة (حقول JSONL)

كل سطر في ملف الـ JSONL = مثال واحد بهذا الشكل:

```json
{
  "id": "ar-lang-0000123",
  "instruction": "اشرح هذه الجملة نحوياً وبلاغياً لطالب في المرحلة الثانوية.",
  "input": "العلمُ نورٌ يبدّدُ ظلماتِ الجهلِ.",
  "output": "أولاً: الإعراب ... ثانياً: البلاغة ...",
  "role": "tutor",
  "skills": ["nahw", "balagha"],
  "level": "intermediate",
  "domain": "education",
  "style": "fusha_classical",
  "source": "synthetic_manual",
  "task_type": "analysis_and_explanation",
  "difficulty": 2,
  "tags": ["sentence_parsing", "metaphor", "training_example_001"]
}
```

شرح الحقول الأساسية:

- instruction: صيغة الطلب/المهمّة بالعربي (إلزامي)، نفس فكرة CIDAR.[^6_1]
- input: النص/السؤال/السياق (ممكن يكون "" لو المهمة مستقلة).[^6_1]
- output: الإجابة النموذجية اللي عايز الموديل يتعلّمها.[^6_3][^6_1]

حقول للتحكّم في الأدوار:

- role:
    - `"tutor"` (معلّم)، `"proofreader"`, `"poet"`, `"muhhaqiq"`, `"assistant_general"` … إلخ.
- skills: array:
    - `"nahw"`, `"sarf"`, `"balagha"`, `"orthography"`, `"qa"`, `"style_editing"`, `"poetry"`, `"heritage"`.
- level:
    - `"beginner"`, `"intermediate"`, `"advanced"`, أو مستويات دراسية لو حبيت.

حقول سياقية/تحليلية:

- domain: `"education"`, `"business"`, `"academic"`, `"islamic_studies"`, `"general"`.
- style:
    - `"fusha_classical"`, `"fusha_modern"`, `"dialect_egyptian"`, `"mixed"`.
- task_type:
    - `"explanation"`, `"qa"`, `"correction"`, `"rewrite"`, `"analysis_and_explanation"`, `"generation"`, `"summarization"`, إلخ.[^6_4][^6_5]
- difficulty: رقم صغير 1–3 أو 1–5 (سهولة، متوسط، صعب).[^6_4]
- source: `"cidar"`, `"arabicweb_edu"`, `"synthetic_llm"`, `"manual_expert"`, `"textbook_X"`.[^6_6][^6_1]
- tags: حرة، عشان الفلترة والتحليل لاحقًا.


## 2. مثال واحد لكل دور رئيسي

### أ) معلّم نحو/بلاغة (tutor)

```json
{
  "id": "ar-tutor-0001",
  "instruction": "أعرب الجملة التالية ثم وضّح الصورة البلاغية فيها باختصار لطالب مبتدئ.",
  "input": "الكتابُ صديقٌ لا يخونُ صاحِبَه.",
  "output": "أولاً: الإعراب: الكتابُ: مبتدأ مرفوع وعلامة رفعه الضمة الظاهرة. صديقٌ: خبر مرفوع... ثانياً: البلاغة: في تشبيه الكتاب بالصديق تشبيه بليغ يدل على الأنس...",
  "role": "tutor",
  "skills": ["nahw", "balagha"],
  "level": "beginner",
  "domain": "education",
  "style": "fusha_modern",
  "task_type": "analysis_and_explanation",
  "difficulty": 1,
  "source": "synthetic_manual",
  "tags": ["i3rab", "tashbih"]
}
```


### ب) مصحّح لغوي (proofreader)

```json
{
  "id": "ar-proof-0001",
  "instruction": "صحّح الأخطاء النحوية والإملائية في الفقرة الآتية، ثم اشرح أهم ثلاثة أخطاء لطالب في المرحلة الجامعية.",
  "input": "ان العلم هوه طريق النجاه، فمن أحب ان ينجو فعليه بالقراءه المستمره.",
  "output": "النص المصحَّح: \"إنَّ العلمَ هو طريقُ النجاةِ، فمن أحبَّ أن ينجو فعليه بالقراءةِ المستمرة.\" الشرح: 1) \"ان\" صوابها \"إنَّ\" لأنها حرف توكيد ونصب...",
  "role": "proofreader",
  "skills": ["nahw", "orthography", "style_editing"],
  "level": "intermediate",
  "domain": "education",
  "style": "fusha_modern",
  "task_type": "correction",
  "difficulty": 2,
  "source": "synthetic_manual",
  "tags": ["gec", "spelling", "style"]
}
```


### ج) شاعر / كاتب إبداعي (poet)

```json
{
  "id": "ar-poet-0001",
  "instruction": "انظم بيتين من الشعر العمودي على بحر الكامل عن فضل طلب العلم، بلغة فصحى سهلة تناسب طالب ثانوي.",
  "input": "",
  "output": "في طلبِ العلمِ انفسحتْ دروبٌ\nتُفضي إلى العليا وتدفعُ كلَّ غِيِّ\nفاسعَ إليهِ بقلبِ حرٍّ صادقٍ\nفالجهلُ ليلٌ والعُلومُ ضياءُ فَجْرِ",
  "role": "poet",
  "skills": ["poetry"],
  "level": "intermediate",
  "domain": "education",
  "style": "fusha_modern",
  "task_type": "generation",
  "difficulty": 2,
  "source": "synthetic_llm",
  "tags": ["bahar_kamil", "value_of_knowledge"]
}
```


### د) محقّق لغوي/تراث (muhhaqiq)

```json
{
  "id": "ar-muhhaqiq-0001",
  "instruction": "اقرأ الجملة التالية من نص تراثي، ثم: 1) أعرب الكلمات الرئيسة، 2) وضّح المعنى العام بلغة معاصرة، 3) علّق على أي موضع قد يلتبس على طالب اليوم.",
  "input": "وما نيلُ المطالبِ بالتمنّي ولكن تُؤخذُ الدنيا غِلابا.",
  "output": "أولاً: الإعراب: ما: أداة نفي... نيلُ: مبتدأ مرفوع... ثانياً: المعنى المعاصر: لا تُدرك الأهداف بمجرد الأماني... ثالثاً: موضع الالتباس: كلمة \"غِلابا\"...",
  "role": "muhhaqiq",
  "skills": ["nahw", "balagha", "heritage"],
  "level": "advanced",
  "domain": "heritage",
  "style": "fusha_classical",
  "task_type": "analysis_and_explanation",
  "difficulty": 3,
  "source": "classical_poetry",
  "tags": ["tarassul", "poetry_explanation"]
}
```


### هـ) مساعد عام بالعربية (assistant_general)

```json
{
  "id": "ar-assistant-0001",
  "instruction": "اشرح بإيجاز الفرق بين الجملة الاسمية والجملة الفعلية لطالب في المرحلة الإعدادية، مع مثال لكل نوع.",
  "input": "",
  "output": "الجملة الاسمية تبدأ باسم، مثل: \"العلمُ نافعٌ\"... الجملة الفعلية تبدأ بفعل، مثل: \"ينتفعُ الطالبُ بالعلم\"...",
  "role": "assistant_general",
  "skills": ["nahw"],
  "level": "beginner",
  "domain": "education",
  "style": "fusha_modern",
  "task_type": "explanation",
  "difficulty": 1,
  "source": "synthetic_manual",
  "tags": ["basic_grammar", "school"]
}
```


## 3. ملاحظات تنفيذية

- السكيمة دي متوافقة مع طريقة datasets زي CIDAR (instruction/input/output) لكن مضافة لها metadata قوية للتحكم والفلترة.[^6_2][^6_1]
- تقدر بسهولة تعمل sampling متوازن حسب role/skills/level قبل التدريب، أو تحرق أدوار معينة في مراحل مختلفة من الـ fine‑tune.[^6_7][^6_3]
- لو ناوي تجرّب Json‑style كامل (كل شيء داخل object واحد) زي JsonTuning، نقدر نطوّر السكيمة دي لمدخل/مخرج JSON صِرف بدل نص حر.[^6_8][^6_4]

تحب الخطوة الجاية تكون: أديك سكربت Python (HF datasets + merge) يحوّل مصادر مختلفة (CIDAR، داتا يدوية) إلى الملف JSONL بهذه السكيمة؟
<span style="display:none">[^6_10][^6_11][^6_12][^6_13][^6_14][^6_15][^6_16][^6_9]</span>

<div align="center">⁂</div>

[^6_1]: https://github.com/ARBML/CIDAR

[^6_2]: https://arxiv.org/html/2407.02147v2

[^6_3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12900375/

[^6_4]: https://aclanthology.org/2025.findings-acl.1232.pdf

[^6_5]: https://www.digitalocean.com/community/tutorials/json-for-finetuning-machine-learning-models

[^6_6]: https://aclanthology.org/2025.arabicnlp-main.36.pdf

[^6_7]: https://arxiv.org/html/2510.13430v2

[^6_8]: https://latitude.so/blog/how-json-schema-works-for-llm-data

[^6_9]: https://community.openai.com/t/how-to-create-a-correct-jsonl-for-training/693897

[^6_10]: https://arxiv.org/html/2502.16377v2

[^6_11]: https://www.reddit.com/r/MachineLearning/comments/1iwxtmb/r_training_llms_for_strict_json_schema_adherence/

[^6_12]: https://learn.microsoft.com/en-us/answers/questions/2281533/fine-tuning-ai-to-return-structured-json-business

[^6_13]: https://www.nature.com/articles/s41598-025-16647-5

[^6_14]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^6_15]: https://www.kaggle.com/code/lonnieqin/instruction-tuning-with-llm

[^6_16]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0341905


---

# هحتاج اعمله الكشف في المعجم و اصل الكلمات و الافعال و تصريفتها و مصادرها و arabic tokenizer ؟

هتفتكر في الصواب: عشان تحول الموديل لـ “عالم لغوي حقيقي” لازم تجمع بين 3 طبقات:

1) **LLM + fine‑tuning** (السلوك الذي تبنيه على الـ skema اللي شرحتها).
2) **أنظمة خارجية للمورفولوجيا والمعجم** (كشف الجذر، الصرف، المعاني).
3) **Tokenizer عربي مهيّأ أو مُعدّل** يتفهم الخرائط الجذرية/الصرفية إن أردت.

***

## 1. الكشف في المعجم / أصل الكلمة وجذرها

بدل ما تبني Dictionary من الصفر، استعمل أدوات مورفولوجية جاهزة وصغ محرك سؤال:

- أدوات ممتازة:
    - **MADAMIRA / CAMeL Tools**: تحلّل الكلمة → جذر + صيغة + POS + إعراب مختصر (مثالي لـ محقق/عالم لغة).[^7_1][^7_2][^7_3]
    - **Alma**: مورفولوجيا سريعة (جذر، Lemma، POS tag) ودقيقة ومجانية.[^7_4]
    - موارد جذرية مثل: **Murabaa** و**CALEM** (Roots + Patterns + Stems) لو حابب تبني خدمة موسوعية داخلية.[^7_5]

طريقة العمل المتوقعة:

- تدخل الكلمة ← تطلع الجذر، الباب (يفعل، ينفع… إلخ)، الصيغة (مصدر، اسم فعل، مبالغة…).[^7_6][^7_5]
- القاعدة: الموديل يجيب: “هذا جذرها: ك‑ت‑ب، بابها: فَعَل، مصدرها: كِتَاب، مصادر أخرى…” ثم تستخدم المورفولوجيا كـ “أداة داخلية” للتحقق.

***

## 2. مصادر الكلمة والتصريفات (العطف على الـ fine‑tuning)

لو حابب تضيف لنفس الموديل كفاءة “عالم لغوي” في الصرف:

- استخدم موارد جاهزة:
    - معاجم إلكترونية (مثل **Lugha** أو **AraMorph**‑based lexicons) تربط الكلمة بكل أشكالها.[^7_7][^7_5]
    - مصادر جذرية (مثل **Khoja’s stemmer** ونظام الجذر الثلاثي/الرباعي).[^7_8][^7_9]

طريقة التطبيق مع LLM:

- في الـ **pre‑processing / retrieval**:
    - ادخل الكلمة → احصل على جذرها + صيغها + معانيها من المعجم/Analyzer → أدخل هذه البيانات في البرومبت كـ “مصدر موثوق” قبل ما يجاوب.
- مثال (دور في الـ instruction):
    - “أجب على السؤال عن أصل الكلمة وتصريفاتها مستخدمًا النتائج المرفقة من برنامج مورفولوجي عربي.”

بهطريقة دي، الموديل يبقى **قوي في التفكير والشرح**، وبرامج الـ Analyzer تبقى **قوية في الكشف الحرفي والجذري**، وتوحد بينهما في الـ prompt/rag.

***

## 3. Arabic tokenizer… هل تحتاج “Tokenization خاصة”؟

willReturn تجيب سؤال مهم:
حتى لو tokenizer الموديل غير مبني على الجذور، فالـ LLM modern ممكن يتعلم “التمايز الجذري” ضمنياً، لكن الـ morphologically‑aware tokenization يساعد في التدريب والفهم.[^7_10][^7_7]

### أ) الخيارات:

- **Tokenizer معيّن عربي مجمع**:
    - مثلاً tokenization مثل ما يوفره **MADAMIRA** (tokenization + stem/clitic separation)؛ تستخدمه في الـ preprocessing قبل إدخال النص لـ LLM، وتحفظ مسارات الجذر/الصيغة في بياناتك.[^7_2][^7_3]
- **استخدام tokenizer معيّن موديلك مع تضمين metadata**:
    - تستخدم الـ tokenizer العادي لـ ALLaM/Qwen (BPE/…).
    - تضيف حقل `morphology_features` في الـ skema:
        - `"root": "ktb", "pattern": "fa33al", "pos": "noun_mass", "lemma": "kiṭāb", "derivations": ["kutub", "kātib", "maktab"]`
        - تستخدم أدوات مثل Alma أو MADAMIRA عشان ملّي هذا الحقل في الـ dataset قبل الـ fine‑tune.[^7_5][^7_4]


### ب) ماذا تُضيف في الـ Instruction‑tuning؟

- مثال (مزيج بين المورفولوجيا والـ LLM):

```json
{
  "instruction": "أوضح أصل هذه الكلمة وجذرها، وصيغها المشتقة، ودورها النحوي والإعراب في الجملة.",
  "input": "الكلمة: 'الكاتب'. الجملة: 'الكاتبُ قارئٌ مثاليٌ.'",
  "output": "الجذر: ك‑ت‑ب، الباب: فاعل، مصدرها: كِتَاب، وظيفتها: فاعل مرفوع…",
  "role": "muhhaqiq",
  "skills": ["nahw", "morphology"],
  "morphology_features": {
    "word": "كاتب",
    "root": "ktb",
    "pattern": "fa33al",
    "pos": "noun", "lemma": "kātib",
    "derivations": ["maktab", "maktaba", "kutub", "kitāb"]
  },
  "task_type": "analysis_and_explanation"
}
```

بهتكتب:

- تستخدم مورفولوجيا (MADAMIRA/Alma) تولّد `morphology_features` من النص، ثم تضمنها في الـ JSONL.
- الموديل يتعلم يربط الجذر والصيغة بالمحتوى الإعرابي والمعاني في الـ output.

***

## 4. الخلاصة: ماذا تبني عمليًا؟

- **طبقتان متوازيتان**:

1) LLM معيّن تدربّه على الـ skema السابقة (معلم، مصحّح، شاعر، محقّق…).
2) Pipeline مورفولوجي منفصل (MADAMIRA / Alma + Murabaa/CALEM) يعطيك:
        - جذر الكلمة،
        - الباب،
        - الصرف (مصدر، مبالغة، مفعول،…).
- **الـ Arabic tokenizer**:
    - استخدم tokenizer الموديل العادي،
    - أضف بيانات مورفولوجية كـ metadata في الـ dataset،
    - استغلها في الـ prompt أو في الـ fine‑tune كـ “features” تُعلّم الموديل على الربط الجذري والصيغي.[^7_10][^7_5]

لو تحب، أقدر أساعدك في:

- تصميم Pipeline مورفولوجي (MADAMIRA/Alma + Python) يخرّج JSON يطابق الـ skema السابقة،
- أو أكتب مثال متكامل: من جملة → Tokenization + Morphology → تغذية LLM كـ user/assistant في الـ dataset.

تحب نبدأ بـ: خطوات مفصلة لـ **Pipeline مورفولوجي (MADAMIRA/Alma + HF Dataset)**، ولا مثال كامل بـ جملة → جذر/صرف → مدخل Dataset عربي؟
<span style="display:none">[^7_11][^7_12][^7_13][^7_14][^7_15]</span>

<div align="center">⁂</div>

[^7_1]: https://nyuad.nyu.edu/en/research/faculty-labs-and-projects/computational-approaches-to-modeling-language-lab/research/morphological-analysis-of-arabic.html

[^7_2]: http://www.lrec-conf.org/proceedings/lrec2014/pdf/593_Paper.pdf

[^7_3]: https://www.academia.edu/92240702/Madamira_A_fast_comprehensive_tool_for_morphological_analysis_and_disambiguation_of_arabic

[^7_4]: https://www.sciencedirect.com/science/article/pii/S1877050924030138

[^7_5]: https://aclanthology.org/2026.abjadnlp-1.47.pdf

[^7_6]: https://aclanthology.org/L18-1181.pdf

[^7_7]: https://mail.ijettjournal.org/Volume-73/Issue-9/IJETT-V73I9P126.pdf

[^7_8]: https://arxiv.org/pdf/1203.3584.pdf

[^7_9]: https://thesai.org/Downloads/Volume14No5/Paper_43-New_Arabic_Root_Extraction_Algorithm.pdf

[^7_10]: https://arxiv.org/abs/2603.15773

[^7_11]: https://dl.acm.org/doi/10.1145/1390749.1390767

[^7_12]: https://www.sciencedirect.com/science/article/pii/S1110866519303469

[^7_13]: https://www.academia.edu/86795237/An_intelligent_use_of_stemmer_and_morphology_analysis_for_Arabic_information_retrieval

[^7_14]: https://aclanthology.org/L18-1415.pdf

[^7_15]: https://www.semanticscholar.org/paper/MADAMIRA:-A-Fast,-Comprehensive-Tool-for-Analysis-Pasha-Al-Badrashiny/f5886c433084d1baaa24152cd3e4c555c79bfe4f


---

# لو عاوزه يفهم الكلام او يفكر بالعربي

لـ تفكير حقيقي بالعربي надо تركز على **3 طبقات**:
اللغة (الـ tokenizer والـ embeddings)، الفهم (الـ fine‑tune)، والمهارة (الـ morfology + التراث الذي تبنيه له).[^8_1][^8_2]

## 1. كيف تجعله يفكر بالعربي (وليس مترجماً))

- **استخدم موديل عربي قوي كـ Base**:
    - مثل ALLaM‑7B/Qwen2.5‑7B، وتدربه على نص عربي خالص (من كتب، تراث، مراجع لغوية) بدل مزيج عربي–إنجليزي.[^8_3][^8_4]
- **اجعل الـ training data 100% عربي**:
    - الـ instruction/Input/Output كلها بالعربية الفصحى، بدون ترجمة حرفية من الإنجليزي، عشان العقل اللغوي يبني “شبكة مفاهيم” بالعربية.[^8_2][^8_5]

هذه الخطوة تجعل الموديل يتعامل مع اللغة العربية مثل “لغة أولى”، وتفكيره طبيعي في العربية.

***

## 2. بناء طبقة التفكير المنظّم (المنطق + التعليم)

- **اجعله لا يجيب فقط، بل يبرر**:
    - كل إجابة له فيها 3 أجزاء تقريبًا:

1) الحكم أو القرار،
2) السبب (شرح قواعد/منطق بلاغي)،
3) مثال أو توضيح إضافي.
    - مثل:
        - قاعدة نحوية → مثال صحيح + خطأ + توضيح،
        - صورة بلاغية → تحليل أثرها النفسي والجمالي.
- استخدم **التعليمات مثل سؤال محاضر**:
    - مثال:
        - “اشرح قاعدة الإعراب في الجملة الآتية كما لو كنت تشرح لطالب مبتدئ، ثم اعرض مثالًا بسيطًا آخر لتوضيحها.”

بهذه الطريقة، تدربه على التفكير التعليمي المنظّم، وتصير العربية هي لغة التفكير الرسمية له.

***

## 3. تطبيق مفهوم “التفكير بالعربي” في الـ skema

- أضف هذا المبدأ في **الـ instruction** و**الـ output**:
    - مثال إدخال:

```json
{
  "instruction": "فكّر باللغة العربية الفصحى، وافهم الجملة الآتية ثم علّق عليها نحوياً وبلاغياً.",
  "input": "العلمُ نافعٌ في كل أمرٍ، والجهلُ ضارٌّ في كل موضعٍ.",
  "output": "أولاً: من الناحية النحوية، الجملة بسيطة وتعتمد على البناء المناسب للجملة الاسمية... ثانياً، من الناحية البلاغية، نرى مفارقة بين العلم والجهل..."
}
```

- تأكد من:
    - جميع المتغيرات (الـ metadata مثل الجذر، الفعل، التصريف) تكون **موضحة بالعربية** في الـ output.
    - الـ mental model (التفكير الذهني) يكون مبني على مثال عربي حقيقي، وليس مثال مترجم.

***

## 4. كيف تجعله يفهم الكلام بالعربية

- **القراءة واسعة بالتراث واللغة**:
    - تدربه على نصوص تراثية وكتب لغوية وتعليمية (مثلاً: كتب النحو، البلاغة، معاجم، شروح الشعر).[^8_6][^8_2]
- **استخدام RAG أو knowledge retrieval**:
    - بناء مكتبة معرفية عربية (Vector DB + RAG) تحتوي مصادر موثوقة، ويُستخدم LLM للربط وشرح النتائج، وليس ترجمتها.
- **Dialogue-based training**:
    - محادثات تفاعلية بالعربية الكاملة، حيث يسأل المستخدم ثم يردّ الموديل بنفس الأسلوب، مع توضيح思路 (الخطوات والتفكير).

***

## 5. تطبيق عملي سريع

- كل سؤال موجه له يكون بالعربية، ويطلب منه:
    - **التحليل المنهجي**،
    - **الشرح الواضح**،
    - **الاستشهاد بالتراث العربي** إن أمكن.

بهذه الطريقة، يُمارس اللغة العربية كـ “لغة التفكير الأولى” وليس فقط أداة لـ input أو output.

تحب أصمم لك مثال متكامل: جملة → تحليل نحوي وبلاغي بالعربية → تدرج في مستوى الصعوبة؟

<div align="center">⁂</div>

[^8_1]: https://arxiv.org/abs/2603.15773

[^8_2]: https://arxiv.org/pdf/2505.08004.pdf

[^8_3]: https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Arabic

[^8_4]: https://ai.azure.com/catalog/models/ALLaM-2-7b-instruct

[^8_5]: https://arxiv.org/html/2510.13430v2

[^8_6]: https://archive.org/download/ArabicRhetoricCourse-OpenUniversityOfNigeria/Arabic Rhetoric 1.pdf


---

# ازاي اهندل موضوع الفصحي و اللهجات المختلفه

التعامل الأمثل مع **الفصحى واللهجات** في موديلك ياخد مسارين:

1) موديل موحد بالفصحى يُحسن الهجات،
2) دور معيّن يفهم/ينتقل بين الفصحى واللهجة (مثل مترجم/محفّظ/معلّم пациента).[^9_1][^9_2]

***

## 1. دور الفصحى: “اللغة القيمة”

- اجعل **اللغة الدنيا** للموديل هي **العربية الفصحى MSA**:
    - الـ **pretraining** و**finetuning** الأساسي يكون على نصوص فصيحة عالية الجودة (كتب لغة، تراث، مراجع دراسية، ArabicWeb‑Edu...).[^9_3][^9_4]
- في الـ **instruction**:
    - قل صريحًا:
        - “أجب دائمًا بالفصحى، إلا إذا طُلِب منك صراحة استخدام لهجة معينة.”
    - هذا يُبقي التفكير الداخلي والفسيولوجي بالفصحى، حتى لو كان الإدخال مهجّنًا.[^9_2][^9_1]

بهتراض: الموديل يفكر ويحلّل بالفصحى، لكن يُفهم الهجّة ويُترجمها إن لزم.

***

## 2. معالجة اللهجات: ثلاث استراتيجيات

### أ) تصنيف دور “اللهجة” في الموديل

- يمكنك تعريف **3 أدوار رئيسية**:

| role | مهمته | مثال |
| :-- | :-- | :-- |
| `tutor` | فهم اللهجات وتحويلها إلى فصحى مبسطة مع شرح. | “أكتب الجملة ده بالفصحى وبيّن الفرق.” [^9_5][^9_6] |
| `translator` | ترجمة فصحى ↔ لهجة معيّنة (مصري، خليجي، شامي…). | “ترجم هذا النص الفصيح إلى لهجة مصرية عامية. لا تُغيّر المعنى.” [^9_7][^9_8] |
| `assistant_general` | المحادثة اليومية مع مزيج فصحى-لهجة مقبولة. | يُتنبأ باللهجة من سياق الـ input ويُجيب على نفس المستوى، مع توضيح عند الحاجة. [^9_9][^9_6] |

### ب) استخدام dialect→MSA كـ “وظيفة” داخل الـ instruction

- يمكن تدريب الموديل على **Tasks of Dialect→MSA** (مثل بنشماركات MADAR, MDC, SADA…):
    - أمثلة: جملة مصريّة → نسخة فصيحة، مع الحفاظ على المعنى.[^9_5][^9_8][^9_9]
- في الـ skema:

```json
{
  "instruction": "حول الجملة الآتية من اللهجة المصرية إلى العربية الفصحى، مع الحفاظ على المعنى.",
  "input": "أنا متحمّش النهارده كنت تعبان بقى.",
  "output": "أنا لم أُحِمَّ اليوم لأنني كنت متعبًا.",
  "role": "translator",
  "skills": ["dialect_egyptian", "formal_transfer"],
  "domain": "daily_life",
  "task_type": "dialect_to_msa"
}
```

بهتكتب: الموديل يُفهم اللهجات لكن يُعيد إنتاجها في الفصحى (أو العكس) حسب الـ role.

### ج) دعم اللهجة في الـ fine‑tuning (إن رغبت)

- لو حابب موديل معيّن يكون قوي في لهجة معيّنة (مثل المصري أو الخليجي):
    - تستخدم موديل معيّن (مثل Nile‑Chat للهجّة المصريّة) كـ “branch” أو “LoRA” متخصص في الـ dialect + تربطه مع موديل الفصحى في الـ inference عبر routing (Prompt + routing logic).[^9_10][^9_6]
- في الـ JSONL:
    - حدد `dialect`: `"dialect_egyptian"`, `"dialect_gulf"`, `"dialect_levant"`…
    - مثلاً:

```json
"role": "assistant_general",
"skills": ["dialect_egyptian", "formal_transfer"],
"dialect": "dialect_egyptian",
"style": "fusha_when_explaining_only",
```


***

## 3. مبدأ مفيد: “اللغة للمعرفة، واللهجة للتفاعل”

- أسلوب معمّم مفيد في الـ instruction:
    - عند المعرفة:
        - “أجب بالفصحى، واشرح الأفكار باللغة العربية الفصحى.”
    - عند التفاعل اليومي:
        - “أجب بلغة مماثلة للهجة المستخدم، مع استخدام الفصحى عند توضيح القواعد أو المفاهيم.”[^9_6][^9_2]

بهطريقة دي، تضمن أن الموديل:

- يفهم أي لهجة تقريبًا،
- يُنتج معرفة موثوقة بالفصحى،
- ويتفاعل مع المستخدم في لغته الأقرب.

***

## 4. خطة عملية قريبة من مهارة العالم اللغوي

- أضف إلى الـ skema:
    - حقل `dialect` (أو `regional_variant`).
    - حقل `formality` (الفصحى، العامية المحترمة، العامية اليومية).[^9_11][^9_2]
- في الـ dataset:
    - مثال: جملة مصريّة → معلّم يشرح معانيها بالفصحى + يبيّن الفرق بينها وبين الفصحى.[^9_8][^9_5]

تحب الخطوة الجاية تكون:

- أسلوب دقيق لـ **Prompt design** يُجبره يجيب بالفصحى حتى لو السؤال باللهجة،
- أو مثال كامل لـ Dataset يدمج بين فصحى ولبنانية/مصريّة في نفس الـ role?
<span style="display:none">[^9_12][^9_13][^9_14][^9_15][^9_16][^9_17]</span>

<div align="center">⁂</div>

[^9_1]: https://dl.acm.org/doi/pdf/10.1145/3737453

[^9_2]: https://cacm.acm.org/arab-world-regional-special-section/the-landscape-of-arabic-large-language-models/

[^9_3]: https://aclanthology.org/2025.arabicnlp-main.36.pdf

[^9_4]: https://lighton.ai/lighton-blogs/arabicweb24

[^9_5]: https://aclanthology.org/2024.osact-1.14.pdf

[^9_6]: https://arxiv.org/html/2409.11404v1

[^9_7]: https://www.sciencedirect.com/science/article/pii/S1877050926001158

[^9_8]: https://arxiv.org/html/2507.20301v1

[^9_9]: https://aclanthology.org/2026.abjadnlp-1.41.pdf

[^9_10]: https://www.middleeastainews.com/p/mbzuai-nile-chat-egyptian-arabic

[^9_11]: https://communities.springernature.com/posts/unlocking-arabic-formality-transfer-with-large-language-models-a-deep-dive-into-dialect-to-msa-translation

[^9_12]: https://arxiv.org/html/2406.18120v2

[^9_13]: https://www.isca-archive.org/syndata4genai_2024/alharbi24_syndata4genai.pdf

[^9_14]: https://prezi.com/p/9xksswh4cbow/automatic-translation-of-arabic-dialects-into-modern-standard-arabic-msa-a-comparative-hybrid-data-science-approach/

[^9_15]: https://www.sciencedirect.com/science/article/pii/S1877050924029855

[^9_16]: https://www.linkedin.com/pulse/arabic-llms-2025-benchmark-gulf-business-applications-rabehi-phd-ml8kf

[^9_17]: https://arxiv.org/html/2501.13419v1


---

# ايه اللي محتاج افكر فيه تاني او احسنه

أنت تقريبًا غطّيت 80٪ من الفكرة، والباقي هو تحسينات “نضج المنتج” مش بس “تشغيل موديل”.[^10_1][^10_2]

## 1. تقييم منهجي مش إحساس فقط

لازم تخطط من الآن لـ evaluation واضح قبل ما تدوس Train:

- استخدم بنشماركات عربية جاهزة: OALL / Arabic‑LLM‑Benchmarks / ArabicWeb‑Edu models عشان تشوف وضع الـ base والموديل بعد الـ fine‑tune.[^10_2][^10_3][^10_1]
- إبنِ أنت “Mini‑Benchmark” خاص بك:
    - 50–100 سؤال نحو، 50 بلاغة، 30 شعر، 30 تحقيق، 50 لهجات→فصحى، وتحتفظ بنسخة ثابتة تقارن عليها كل نسخة موديل.[^10_1][^10_2]

الفكرة: أي تغيير في الـ data أو الـ hyperparameters تقيس أثره، مش تحسّه.

## 2. جودة الداتا قبل كميّتها

النقطة الحساسة في العربي الآن هي quality filtering، خصوصًا للتعليم وعلوم اللغة.[^10_4][^10_5]

- ادرس منهج ArabicWeb‑Edu:
    - ركّزوا على **تعليمي/تفسيري**، فلتروا الويب بشدة، والنتيجة إن الموديلات المدرّبة على داتا عالية الجودة كسبت بنشماركات بوضوح.[^10_4]
- اعمل طبقة “Data Auditor”:
    - إمّا LLM آخر أو سكربت تقييم بسيط لكل مثال:
        - هل العربي سليم؟
        - هل الشرح واضح؟
        - هل الأسلوب ثابت؟
    - مشابه للي عملته GemmAr‑7B (تنضيف شديد + standardization قبل تحويل النصوص لـ instruction pairs).[^10_6][^10_7]

لو في شيء تصرف عليه وقت حقيقي، فهو فلترة وتنظيم الداتا قبل ما تدخل GPU.

## 3. توازن الأدوار والـ capacities

مع تعدد الأدوار (معلّم، مصحّح، شاعر، محقّق، مفسّر لهجات)، سهل جدًا إن دور يطغى على الباقي.[^10_5][^10_8]

- فكّر في الـ mix من الآن:
    - كام٪ من الداتا تعليم نحو/بلاغة؟
    - كام٪ تصحيح لغوي؟
    - كام٪ شعر/تراث؟
    - كام٪ لهجات/فصحى؟
- اشتغل بأسلوب staged:
    - مرحلة 1: أساس تعليمي (نحو/بلاغة + تصحيح).
    - مرحلة 2: إضافة شعر + تحقيق.
    - مرحلة 3: لهجات/فصحى وتحوّلات الأسلوب.

كل مرحلة تقيس أثرها قبل الانتقال.

## 4. Alignment: الأسلوب، السلامة، والثقافة

عربي = ثقافة + دين + حساسيات اجتماعية؛ الدراسات الأخيرة على Arabic LLMs نبّهت إن كثير موديلات قوية لغويًا لكن ضعيفة في alignment الثقافي.[^10_5][^10_1]

- حدِّد “شخصية” الموديل:
    - عالم لغة مهذّب، لا يسخر، لا يستخدم ألفاظ سوقية إلا لو في سياق شرح مقبول، يحترم الحساسية الدينية والثقافية.
- ضف طبقة **Safety / Policy** بسيطة في الـ fine‑tune:
    - أمثلة على أسئلة حساسة + ردود منضبطة،
    - مع تعليمات واضحة في الـ instruction (ما الذي يتجنّبه، وكيف يعتذر/يحيل).

ده مهم لو ناوي تطلّعه كـ منتج مستخدمين حقيقيين مش بس R\&D.

## 5. استقرار النسخ وإدارة التجارب

نقطة كثير ناس بتغلط فيها: بيتوهوا وسط عشرات التجارب وما يعرفوش أي نسخة فعليًا أفضل.[^10_9][^10_2]

- فكّر من دلوقتي في:
    - تسمية واضحة لكل Run (model + data mix + hyperparams).
    - حفظ Config + Commit في Git لكل تجربة.
    - Log للنتائج على بنشماركات + mini‑benchmark الخاص بك.

حتى لو هتشتغل على GPU واحد و50K مثال، ال discipline ده هيخلي مشروعك قابل للتوسّع.

## 6. نقاط تحسين إضافية محتملة

- التفكير المتسلسل (chain‑of‑thought) بالعربي:
    - استخدم جزء من الداتا فيه “شرح خطوات التفكير” وليس فقط الإجابة النهائية، زي اللي عملته بعض مشاريع Arabic instruction‑tuning.[^10_8][^10_6]
- استغلال داتا موجودة بدل إعادة اختراع العجلة:
    - InstAr / CIDAR / ArabicWeb‑Edu / GemmAr instruction dataset كنقطة انطلاق، وتبني فوقها بطبقة علوم لغة متخصصة.[^10_7][^10_10][^10_4]
- Plan للاستخدام الفعلي:
    - هيتخاطب معه ازاي؟ API؟ Web app للطلاب؟ Chatbot في LMS؟ ده هيأثر على الـ prompt templates وواجهة الموديل.

لو اخترنا حاجة واحدة تركّز تحسّنها في الشهر الجاي، تحب تكون: جودة وتنضيف الداتا؟ ولا نظام تقييم (بنشمارك + mini‑benchmark)؟
<span style="display:none">[^10_11][^10_12][^10_13][^10_14][^10_15][^10_16][^10_17]</span>

<div align="center">⁂</div>

[^10_1]: https://arxiv.org/html/2510.13430v1

[^10_2]: https://github.com/tiiuae/Arabic-LLM-Benchmarks

[^10_3]: https://silma.ai/arabic-llm-benchmark

[^10_4]: https://aclanthology.org/2025.arabicnlp-main.36.pdf

[^10_5]: https://arxiv.org/html/2507.14688v1

[^10_6]: https://arxiv.org/abs/2407.02147

[^10_7]: https://arxiv.org/html/2407.02147v1

[^10_8]: https://deeplearn.org/arxiv/505592/gemmar:-enhancing-llms-through-arabic-instruction-tuning

[^10_9]: https://www.linkedin.com/pulse/arabic-llms-2025-benchmark-business-applications-part-rabehi-phd-d91ff

[^10_10]: https://github.com/ARBML/CIDAR

[^10_11]: https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Arabic

[^10_12]: https://localazy.com/blog/ai-8-llm-arabic-models-tested-to-translate

[^10_13]: https://arabicnlp2025.sigarab.org

[^10_14]: https://arxiv.org/abs/2510.13430

[^10_15]: https://aidatainsider.com/ai/llm-developers-building-for-language-diversity-in-2025/

[^10_16]: https://etasr.com/index.php/ETASR/article/view/10331

[^10_17]: https://www.emergentmind.com/papers/2407.02147


---

# أريد أركز أولًا على جودة وتنضيف الداتا

أهم حاجة دلوقتي إن “تنضيف الداتا” يبقى Pipeline واضح ومتكرر، مش شوية Regex عشوائية.[^11_1][^11_2]

## 1. 3 مستويات لتنضيف الداتا

1) مستوى حرفي/تقني (نفس فكرة ArabicWeb24 / refined‑arabic):

- إزالة HTML, scripts, emojis، الروابط، IDs.[^11_3][^11_1]
- التأكد من أن النص عربي فعلاً (language detection) + إزالة السطور اللي مش جمل (قوائم، منيو، spam).[^11_1][^11_3]

2) مستوى لغوي عربي:

- تطبيع الحروف (أ/إ/آ → ا، ى → ي، ة ↔ ه حسب ما تقرر)، توحيد علامات الوقف، إزالة التشكيل لو مش محتاجه.[^11_4][^11_5][^11_6]
- إزالة التكرار (نفس الجملة مكررة، أو spam بنفس الصيغة)، على مستوى الوثيقة والجملة (minhash / repetition removal).[^11_3][^11_1]

3) مستوى “جودة تعليمية”:

- حذف النصوص الركيكة، المكسّرة، أو المليانة سبّ/إعلانات/نصب (C4-style bad words + heuristics عربية).[^11_2][^11_1][^11_3]
- فلترة حسب الطول والمعنى: جمل قصيرة جدًا أو بلا سياق تتشال.[^11_7][^11_1]


## 2. Pipeline عملي بسيط (تقدر تطبقه خلال أسابيع)

بناءً على ArabicWeb24 + مراجعة Mind‑the‑Gap وArabicWeb‑Edu:[^11_8][^11_9][^11_10][^11_2][^11_1]

1) مرحلة RAW → Clean:

- قراءة الملفات الخام (كتب PDF محوّلة، Web crawl، scripts…).
- تشغيل:
    - HTML strip + Unicode normalize.
    - language detection → keep Arabic‑dominant فقط.
    - basic regex للتخلص من: الروابط، @handles، emojis، أرقام ID الطويلة.

2) مرحلة Heuristics عربية:

- Keep فقط السطور اللي:
    - فيها حروف عربية بنسبة معينة،
    - طولها ≥ 5 كلمات،
    - تنتهي بنقطة/علامة استفهام/تعجب/نقطتين (تشبه جملة).[^11_5][^11_3]
- Remove صفحات فيها أقل من 5 جمل، أو مليانة junk (قوائم، navbars…).[^11_1][^11_3]

3) مرحلة Dedup:

- MinHash أو أداة جاهزة (datatrove / refined‑arabic repetition_removal) لإزالة المستندات والجمل المكررة أو المتشابهة > 75٪.[^11_2][^11_3][^11_1]

4) مرحلة Safety / Toxic Filter:

- قائمة كلمات سيئة (عربي + إنجليزي) زي ما عملوا في ArabicWeb24، وتفلتر أي document فيه نسبة عالية منها.[^11_2][^11_1]

5) مرحلة “Education Filter” الخاصة بيك:

- سكربت بسيط يصنّف الجمل:
    - نحو/بلاغة/شرح/تعليم/شعر/تراث/سوشال ميديا…
- تحتفظ بالمجالات اللي تخدم مشروعك (تعليمي، تراث، لغوي) وتقلل من chatter منخفض الجودة.[^11_10][^11_8]


## 3. بالنسبة لـ داتا الـ instruction (مش بس نص خام)

لأن مشروعك تعليم نحو/بلاغة/عالم لغة، الداتا الأهم هي الـ instruction pairs، وهنا الجودة أهم من الحجم.[^11_11][^11_8][^11_10]

- اعمل “data checklist” لكل مثال:
    - صياغة instruction واضحة،
    - input نظيف ومفهوم،
    - output: لغة سليمة + شرح منظم + بدون حشو أو تكرار.
- استعمل LLM‑auditor:
    - موديل كبير (حتى إنجليزي/متعدّد) يقيم كل مثال (جودة اللغة، وضوح التعليم، طول مناسب) ويعطي score، وترمي الأمثلة الضعيفة.[^11_12][^11_8][^11_11]


## 4. مصادر تتعلم منها أسلوب التنضيف

- ArabicWeb24 blog + كودهم (datatrove filters + dedup + bad words).[^11_1][^11_2]
- refined‑arabic repo: تطبيق عملي لفكرة C4‑style cleaning مع تهيئة للعربي.[^11_3]
- ArabicWeb‑Edu: منهج “احنا عايزين داتا تعليمية فقط، نظيفة جدًا” وكيف قيّموا impact لكل خطوة.[^11_10]
- Mind the Gap: توصيات مباشرة لبناء Arabic post‑training datasets عالية الجودة (تنويع المهام، metadata، مراجعة).[^11_9][^11_8]


## 5. خطوة عملية تقدر تبدأ بيها الآن

- اختر نوع واحد من المصادر (مثلاً: شروحات نحو/بلاغة PDFs أو مقالات تعليمية).
- بنى لها Pipeline بسيط بثلاث خطوات: normalization + heuristics + dedup.
- جرّب تعمل نسخة v0 من dataset نظيفة، حتى لو بس 5K–10K جملة، وشوف بنفسك الفرق في شكل النص.

تحب الخطوة الجاية أساعدك فيها تكون: تصميم checklist تفصيلي (بنود) لجودة مثال الـ instruction نفسه (مش النص الخام)، ولا pseudo‑code/سكربت عالي المستوى للـ cleaning pipeline بالعربي؟
<span style="display:none">[^11_13][^11_14][^11_15][^11_16][^11_17]</span>

<div align="center">⁂</div>

[^11_1]: https://huggingface.co/blog/MayFarhat/arabicweb24

[^11_2]: https://lighton.ai/lighton-blogs/arabicweb24

[^11_3]: https://github.com/mohataher/refined-arabic

[^11_4]: https://github.com/h9-tect/Arabic_nlp_preprocessing

[^11_5]: https://stackoverflow.com/questions/71834104/arabic-dataset-cleaning-removing-everything-but-arabic-text

[^11_6]: https://www.kaggle.com/code/mohamedabdelmohsen/preprocessing-techniques-on-arabic-text-modeling

[^11_7]: https://hplt-project.org/HPLT_D3_1___Software_for_cleaning_data_sets.pdf

[^11_8]: https://arxiv.org/html/2507.14688v1

[^11_9]: https://arxiv.org/abs/2507.14688

[^11_10]: https://aclanthology.org/2025.arabicnlp-main.36.pdf

[^11_11]: https://arxiv.org/abs/2407.02147

[^11_12]: https://deeplearn.org/arxiv/505592/gemmar:-enhancing-llms-through-arabic-instruction-tuning

[^11_13]: https://arxiv.org/abs/2511.07080

[^11_14]: https://www.academia.edu/49252262/A_hybrid_Technique_for_Cleaning_Missing_and_Misspelling_Arabic_Data_in_Data_Warehouse

[^11_15]: https://aclanthology.org/2025.arabicnlp-main.26/

[^11_16]: https://hackernoon.com/how-i-built-a-python-pipeline-to-analyze-16695-arabic-tweets-on-x

[^11_17]: https://www.hnjournal.net/ar/4-7-11/


---

# افضل حل لهندلة التشكيل

أفضل حل إنك تفصل بين:

1) كيف تخزّن وتدرّب على التشكيل في الداتا.
2) كيف تتعامل معه في inference وحالات الاستخدام المختلفة.[^12_1][^12_2]

## 1. في الداتا والتدريب

- للـ LLM العام (فهم/شرح/نحو/بلاغة):
    - اشتغل أساسًا على **نصوص بدون تشكيل كامل**، مع الاحتفاظ بالتشكيل في أمثلة محددة (قرآن، شعر، قواعد) حيث يكون جزءًا من المعلومة.[^12_3][^12_1]
    - طهّر التشكيل العشوائي/الخاطئ قبل التدريب؛ كثير من الداتا الويب فيها tashkeel غلط يزيد الضوضاء.[^12_4][^12_2]
- لو عايزه يكون **جيد في فهم التشكيل**:
    - احتفظ بنسخة Diacritized نظيفة من بعض corpora (مثل نصوص تراث/قرآن/حديث/شعر) بعد تنظيفها وتوحيد نمط التشكيل، كما فعلت نماذج متخصصة مثل Sadeed.[^12_2][^12_5]

نتيجة: الموديل يتعلم العربية “العادية” بدون تشويش، مع exposure كافي يفهم التشكيل عندما يراه.

## 2. في الإخراج (inference) لحالات مختلفة

- للاستخدام النصي العادي (قراءة، تعليم نحو/بلاغة):
    - خليه يكتب **من غير تشكيل** افتراضيًا، إلا في الكلمات التي يحتاج يبيّن فيها الفرق (مثل أمثلة نحوية).
    - لما تحتاج تبيين حركة معيّنة، خليه يشير “بالشرح” بدل التشكيل الكامل (مثل: “مرفوع بالضمة على آخره”).
- للصلاة/قرآن/TTS أو تعليم التجويد:
    - اعتبر التشكيل Task منفصل:
        - إمّا تستخدم موديل Diacritization متخصص (sequence‑to‑sequence أو BERT‑based) مثل Sadeed أو النماذج التي تستعرضها المسوحات الحديثة.[^12_6][^12_7][^12_2]
        - أو تستخدم LLM كبير كـ diacritizer على سطور قصيرة فقط، مع تقييم CER/WER.[^12_8][^12_5][^12_4]

هنا الأفضل تبني **خدمة منفصلة للتشكيل** تتغذى بالنص من الـ LLM وليس تدمج المشكلة داخل نفس الـ fine‑tune.

## 3. سياسة عملية تقترحها الأبحاث الحديثة

من الدراسات الأخيرة على Diacritization وLLMs:[^12_5][^12_8][^12_4][^12_2]

- أثناء التنضيف:
    - أزل التشكيل من معظم النصوص، واحتفظ فقط بنصوص diacritized ذات جودة معروفة (مصاحف معتمدة، كتب حديث مشكولة، corpora متخصصة).
    - وحّد نمط التشكيل (مثل حذف سكون زائدة، توحيد تنوين… إلخ) قبل استخدامها.
- في التدريب:
    - اعتبر التشكيل **skill/role** منفصل:
        - `role: "diacritizer"` مع أمثلة خاصة،
        - وباقي الأدوار تظل بدون تشكيل في الـ output.
- في التطبيق:
    - إن كان المشروع تعليمي نحو/بلاغة عام، خليه:
        - يشرح الحركات بالكلام،
        - أو يشكّل كلمات معيّنة فقط في body الشرح (مش الجملة كلها).


## 4. الخلاصة العملية لمشروعك

1) نظّف كل الداتا من التشكيل العشوائي، وخلّ التشكيل موجود فقط في مصادر ذات جودة عالية وموسومة (قرآن، شعر، نصوص تعليمية مشكولة).[^12_1][^12_2]
2) درّب الموديل على نص بدون تشكيل بوصفه الافتراضي، مع أمثلة محددة فيها تشكيل مضبوط كجزء من الشرح أو التحليل.[^12_5][^12_1]
3) لو تحتاج إخراج مشكول بالكامل (قراءة/تجويد/TTS)، استخدم **diacritization model/خدمة خارجية**، أو LoRA صغيرة متخصصة، بدل ما تفرض التشكيل على كل مخرجات الموديل الأساسي.[^12_8][^12_4][^12_2]

لو حددت لي أهم استخدام عندك للتشكيل (تعليم نحو، TTS، قرآن، أو توضيح غموض المعنى)، أقدر أضبط لك سياسة أدق ومثال برومبت/بايبلاين لكل حالة.
<span style="display:none">[^12_10][^12_11][^12_12][^12_13][^12_14][^12_15][^12_16][^12_9]</span>

<div align="center">⁂</div>

[^12_1]: https://www.cambridge.org/core/journals/natural-language-engineering/article/survey-of-automatic-arabic-diacritization-techniques/8D869B09FE33D60F3C801A97AFD6C5F1

[^12_2]: https://arxiv.org/html/2504.21635v1

[^12_3]: https://arxiv.org/html/2507.14688v1

[^12_4]: https://aclanthology.org/2025.emnlp-main.846/

[^12_5]: https://arxiv.org/html/2504.21635v2

[^12_6]: https://www.sciencedirect.com/science/article/abs/pii/S0957417424002811

[^12_7]: https://aclanthology.org/N19-1248/

[^12_8]: https://arxiv.org/pdf/2506.11602.pdf

[^12_9]: https://www.isca-archive.org/interspeech_2013/harrat13_interspeech.pdf

[^12_10]: https://www.linkedin.com/posts/louay-alshoum_ai-arabicai-voiceai-activity-7437517398041235456-HKPJ

[^12_11]: https://huggingface.co/papers?q=Arabic+diacritics

[^12_12]: https://github.com/ehsanasgari/Tahdheeb

[^12_13]: https://www.catalyzex.com/s/Arabic Text Diacritization

[^12_14]: https://asprunner.com/forums/topic/31201-Custom-Arabic-Search-Normalization-Handling-Similar-Letters-Diacritics-and-Tatweel

[^12_15]: https://aclanthology.org/2024.lrec-main.128.pdf

[^12_16]: https://kadsa.org/index.php/kadsa/article/download/15077/11251


---

# عاوز مصادر داتا و اودوات مناسبه للمرحله الاولي

أفضل بداية للمرحلة الأولى إنك تخلّيها “مرحلة بيانات فقط”: مصادر جاهزة + أدوات تنظيف/تجهيز، بدون تدريب لسه.[^13_1][^13_2]

## 1. كوربسات نص عام عربي (للخلفية اللغوية)

- ArabicWeb24 (HF: `lightonai/ArabicWeb24`):
    - ويب عربي نظيف + فلترة قوية + dedup، مناسب كمرجع أو طبقة نص عام لو احتجتها.[^13_3][^13_4][^13_1]
- ArabicWeb16 / قوائم Metatext:
    - Web أقدم وأوسع، مفيد لو عايز توسّع لاحقًا أو تستخرج أنواع نصوص معينة.[^13_5][^13_6]

في المرحلة الأولى استخدمها كمصدر “قراءة/تجارب تنظيف”، مش لازم تدخلها كلها في الـ fine‑tune.

## 2. داتا Instruction جاهزة بالعربي (نقطة انطلاق لـ roles)

- CIDAR (HF: `arbml/CIDAR`):
    - 10K instruction–output عربية، فيها جزء نحوي وتعليمي، ومراجَعة بشريًّا.[^13_7][^13_8][^13_9]
    - ممتازة كنموذج لأسلوب الـ instruction + output + الثقافة العربية.
- FineWeb‑Edu‑Ar / ArabicWeb‑Edu:
    - داتا ويب تعليمية (مصنّفة حسب quality/educational score) تقدر منها تستخرج نصوص لتمارين وشرح.[^13_10][^13_11][^13_12]

ابدأ بـ CIDAR كنقطة مرجعية للـ schema والأسلوب، ثم ابني فوقها داتا النحو/البلاغة المتخصصة بتاعتك.

## 3. مصادر تعليمية للنحو والبلاغة

- جزء من CIDAR نفسه فيه أسئلة نحو من موقع “اسأل المعلّم”.[^13_9][^13_13]
- كتب/كورسات بلاغة ونحو PDF: تستخرج منها أمثلة وتحوّلها لـ instruction pairs يدوي/نصف أوتوماتيكي (مع pipeline التنضيف اللي اتكلمنا عنه).[^13_14][^13_15][^13_10]

في المرحلة الأولى ركّز على 2–3 مصادر عالية الجودة (مثلاً: كتاب نحو مدرسي + كتاب بلاغة مبسّط + CIDAR).

## 4. أدوات تنظيف وتجهيز عربية

- ArabicWeb24 pipeline (datatrove + filters):
    - راجع خطواتهم كـ blueprint: لغة، طول، إزالة junk، bad words، dedup.[^13_4][^13_1]
- مكتبات preprocessing عربية:
    - `refined-arabic` (GitHub): normalization، إزالة التكرار، فلترة عربية، مفيدة كلبنات جاهزة.[^13_16]
    - `Arabic_nlp_preprocessing` (GitHub): توحيد الحروف، إزالة التشكيل/الرموز، إلخ.[^13_17]
    - `Tahdheeb` (تهذيب): cleaning/normalization عالية الجودة للنصوص العربية.[^13_18]

استخدم واحدة منهم كأساس، وعدّل حسب احتياجك (مثلاً: لا تزيل كل التشكيل إن كنت ستحتفظ به في نصوص معينة).

## 5. كيف تستغل هذه المصادر في المرحلة الأولى

مقترح عملي للـ 2–3 أسابيع الجاية:

1) حمّل CIDAR + عينة صغيرة من ArabicWeb24 + بعض PDFs نحو/بلاغة.
2) طبّق pipeline تنضيف بسيط مستوحى من ArabicWeb24 + refined‑arabic على نصوص الكتب/المقالات.[^13_1][^13_16]
3) صمّم JSONL بالسكيمة اللي بنيناها (instruction, input, output, role, skills, level…)، وابدأ بـ:
    - 2K–3K مثال تعليم نحو/بلاغة (بعضها من CIDAR، وبعضها من صنعك).
4) احتفِظ بكل شيء كـ v0_dataset مع لوج واضح عن: المصادر، خطوات التنضيف، وأي فلتر طبّقتَه.[^13_2][^13_10]

لو حابب، أقدر في الرسالة الجاية أعمل لك جدول صغير “Roadmap بيانات للمرحلة الأولى” يحدد: لكل أسبوع تعمل إيه بالظبط (مصدر، عدد أمثلة، وأداة مستخدمة).
<span style="display:none">[^13_19][^13_20][^13_21]</span>

<div align="center">⁂</div>

[^13_1]: https://lighton.ai/lighton-blogs/arabicweb24

[^13_2]: https://arxiv.org/html/2507.14688v1

[^13_3]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^13_4]: https://huggingface.co/blog/MayFarhat/arabicweb24

[^13_5]: https://metatext.io/datasets-list/arabic-language

[^13_6]: https://metatext.io/datasets/arabicweb16

[^13_7]: https://pure.kfupm.edu.sa/en/publications/cidar-culturally-relevant-instruction-dataset-for-arabic/

[^13_8]: https://aclanthology.org/2024.findings-acl.764/

[^13_9]: https://github.com/ARBML/CIDAR

[^13_10]: https://aclanthology.org/2025.arabicnlp-main.36.pdf

[^13_11]: https://aclanthology.org/2025.arabicnlp-main.36/

[^13_12]: https://arxiv.org/html/2411.06402v1

[^13_13]: https://www.themoonlight.io/en/review/cidar-culturally-relevant-instruction-dataset-for-arabic

[^13_14]: https://archive.org/download/ArabicRhetoricCourse-OpenUniversityOfNigeria/Arabic Rhetoric 1.pdf

[^13_15]: https://www.academia.edu/85516391/A_Primer_of_Arabic_Rhetoric

[^13_16]: https://github.com/mohataher/refined-arabic

[^13_17]: https://github.com/h9-tect/Arabic_nlp_preprocessing

[^13_18]: https://github.com/ehsanasgari/Tahdheeb

[^13_19]: https://www.linkedin.com/posts/2a2i_arabicweb24-creating-a-high-quality-arabic-activity-7227327248515694592-xbcK

[^13_20]: https://irep.mbzuai.ac.ae/items/0003b435-a5fb-4fcc-b252-8e1945565a73

[^13_21]: https://www.linkedin.com/posts/lighton_arabicweb24-creating-a-high-quality-arabic-activity-7227260529235374080-omKg


---

# نعم

أقترح لك Roadmap بيانات لـ 3 أسابيع، مركّزة وقابلة للتنفيذ.

## أسبوع 1: فهم المصادر وبناء Pipeline تنظيف

- هدف الأسبوع: يكون عندك أول Pipeline تنظيف + أول عينة داتا نظيفة.
- المهام:
    - تحميل CIDAR + عينة (مثلاً 50K–100K سطر) من ArabicWeb24 + 1–2 كتاب نحو/بلاغة PDF.[^14_1][^14_2][^14_3][^14_4]
    - تطبيق Cleaning بسيط على الكتب/المقالات (مش CIDAR لأنه جاهز نسبيًا):
        - إزالة HTML/رموز غريبة/روابط/إيموجي.
        - فلترة العربية (سطر فيه ≥ نسبة معيّنة من حروف عربية، ≥ 5 كلمات).[^14_5][^14_6]
        - إزالة التشكيل العشوائي، مع الاحتفاظ بالتشكيل فقط في نصوص تريدها لاحقًا.
    - إعداد سكربت واحد (Python) ينفّذ هذه الخطوات ويحفظ الناتج في ملف `clean_text_v0.txt`.[^14_7][^14_8]

**ناتج الأسبوع:**

- ملفات نص نظيفة v0 (كتاب/مقالة نحو وبلاغة + sample من الويب).
- سكربت Cleaning يعاد استخدامه.


## أسبوع 2: تصميم السكيمة وبدء بناء Instruction Data

- هدف الأسبوع: ملف JSONL صغير لكن نظيف بالسكيمة الموحّدة.
- المهام:
    - تثبيت سكيمة JSONL اللي اتفقنا عليها (instruction, input, output, role, skills, level, …).[^14_3][^14_9]
    - استخراج 200–300 مثال تعليم نحو/بلاغة من:
        - CIDAR (جزء الأسئلة التعليمية).[^14_10][^14_3]
        - كتب النحو/البلاغة بعد التنظيف (تحويل شرح/تمرين إلى instruction–output يدوي/نصف أوتوماتيكي).
    - ضمان جودة كل مثال بـ checklist بسيطة (وضوح التعليمات، صحة اللغة، تنظيم الشرح).
    - حفظ النتيجة في `ar_lang_teacher_v0.jsonl` (مثلاً 500–1,000 مثال عالي الجودة).

**ناتج الأسبوع:**

- أول Dataset تعليم نحو/بلاغة نظيف ومصنّف بالـ roles/skills.
- إحساس عملي بحجم الشغل اليدوي/شبه الآلي المطلوب.


## أسبوع 3: توسيع الداتا + تنظيم النسخ

- هدف الأسبوع: زيادة الحجم قليلاً وتنظيم النسخ.
- المهام:
    - توسيع `ar_lang_teacher_v0.jsonl` إلى 3K–5K مثال عن طريق:
        - توليد بعض الأمثلة صناعيًا بـ LLM + مراجعة سريعة.
        - استخراج مزيد من تمارين/شروح من الكتب أو المصادر التعليمية.
    - إضافة أدوار إضافية بشكل خفيف:
        - 10–20 مثال proofreader (تصحيح جمل مع تعليل).
        - 10–20 مثال poet/heritage (تحليل بيت شعر نحوًا وبلاغة).
    - تنظيم الفولدرات:
        - `data/raw/` (المصادر الأصلية).
        - `data/clean/` (النصوص بعد التنظيف).
        - `data/instructions/` (ملفات JSONL بإصدارات v0, v1…).
        - `scripts/` (سكربتات التنظيف والتحويل).

**ناتج الأسبوع:**

- Dataset v1 بحوالي 3K–5K مثال متنوع لكن مركز على الدور التعليمي.
- هيكل مشروع واضح جاهز للتوسع والتجارب لاحقًا.

بعد ما تخلص المرحلة دي، نقدر نخطط لمرحلة تالية: QLoRA سريعة بـ 5K–10K مثال على Base زي ALLaM‑7B أو Qwen2.5‑7B عشان تشوف أول تحسن ملحوظ في “معلّم نحو/بلاغة” قبل ما تكبّر الداتا.

هل تحب الخطوة الجاية تكون: نكتب معًا هيكل فولدرات مفصّل + أسماء ملفات واضحة وحاجات تحفظ فيها الـ logs والتجارب من البداية؟

<div align="center">⁂</div>

[^14_1]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^14_2]: https://lighton.ai/lighton-blogs/arabicweb24

[^14_3]: https://github.com/ARBML/CIDAR

[^14_4]: https://aclanthology.org/2025.arabicnlp-main.36.pdf

[^14_5]: https://huggingface.co/blog/MayFarhat/arabicweb24

[^14_6]: https://github.com/mohataher/refined-arabic

[^14_7]: https://github.com/h9-tect/Arabic_nlp_preprocessing

[^14_8]: https://github.com/ehsanasgari/Tahdheeb

[^14_9]: https://arxiv.org/html/2407.02147v2

[^14_10]: https://aclanthology.org/2024.findings-acl.764/


---

# المكتبه الشامله و الكتب التراثيه

المكتبة الشاملة والكتب التراثية كنز ضخم، بس لازم تدخلهم مشروعك بحذر وبنية واضحة.[^15_1][^15_2]

## 1. أيه هي Shamela / OpenITI عمليًا؟

- Shamela corpus: استخراج بحثي من نصوص المكتبة الشاملة ≈ مليار كلمة من نصوص تراثية عبر قرون، مع تنظيف أساسي وتحليل مورفولوجي بـ MADAMIRA (lemmatization, POS, features).[^15_3][^15_4][^15_2]
- OpenITI: مشروع أكبر يبني كوربس “machine‑actionable” من نصوص التراث (كثير منها من الشاملة)، مع metadata غنية (تاريخ المؤلف، النوع، نحو/صرف/بلاغة/تفسير…).[^15_5][^15_6][^15_1]

الأفضل لك تبدأ من إصدارات Shamela/OpenITI الجاهزة بدل تحاول تكشط الشاملة من الصفر.

## 2. الفايدة لموديل “عالم لغة”

من OpenITI بالذات تقدر تستخرج:

- كتب نحو: حوالي 143 كتاب نحوي ≈ 19 مليون كلمة، مناسبة لفهم اصطلاحات النحاة وتحليلهم.[^15_1]
- كتب صرف ومعاجم غريب: نصوص عن التصريف، الجذور، معاني الألفاظ، ممتازة لدور “المحقق اللغوي”.[^15_1]
- كتب بلاغة وأدب: شروح، أمثلة، نقد، تخدم دور “الشاعر/البلاغي”.[^15_7][^15_1]

تستخدمها أساسًا كـ **corpus قراءة + مصدر شواهد**، ثم تختار منها مقاطع مناسب تحويلها لـ instruction (سؤال/جواب، شرح، تمرين).

## 3. التحذير المهم: الانحياز والمجال

دراسات KITAB نبّهت إلى انحياز قوي في Shamela/OpenITI:

- غالبية النصوص سنية تقليدية، مع حضور كثيف لكتب الحديث والفقه، وأقل في مجالات أخرى.[^15_8][^15_1]
- هذا معناه: ممتازة للغة والتراث، لكن لا تعتمدها وحدها إن هتعرّف الموديل على “العالم الإسلامي” أو التنوع المذهبي.[^15_8]

لمشروعك (عالم لغة) هذا الانحياز أقل خطورة، لكن خليه في بالك لو توسّعت في فقه/عقيدة.

## 4. كيف تستخدم الشاملة/التراث في المرحلة الأولى؟

أنصحك:

- تبدأ من OpenITI (ready corpus) بدل التعامل اليدوي مع الشاملة:
    - يوفر لك نصوص .txt مع metadata (نوع = نحو/صرف/لغة/بلاغة).[^15_5][^15_1]
- تختار subset صغير موجه:
    - 10–20 كتاب نحو، 5–10 كتب بلاغة، 5–10 معاجم لغة / غريب.[^15_1]
- تمررها عبر Pipeline التنضيف العام (تطبيع، إزالة حشو، تقسيم لجمل/فقرات).
- من النص النظيف:
    - تقطع شروحات/أمثلة وتحولها يدوي/شبه آلي إلى instruction–output (مثلاً: “اشرح هذه القاعدة كما فعل ابن هشام لكن بلغة مبسطة”).

هكذا يكون التراث عندك **طبقة عمق** للموديل، مش بس ويب عشوائي.

## 5. ماذا تحتاج تفكر فيه إضافيًا مع التراث؟

- لغة قديمة: ستحتاج أمثلة تربط بين “الأسلوب التراثي” وشرح حديث مبسط للطالب.[^15_6][^15_1]
- الطول والتكرار: بعض الكتب ضخمة ومكررة؛ اعتمد على dedup وتقطيع فقرات قصيرة. Shamela corpus نفسه حل جزء من التكرار والنصوص المعادة، فاستفد من شغله.[^15_4][^15_9][^15_3]

لو حابب، أقدر أساعدك تختار “قائمة أول 10 كتب نحو/بلاغة/لغة من OpenITI” مناسبة كبذرة لموديل عالم اللغة، مع اقتراح كيف تقطّع كل كتاب لقطع صالحة للـ instruction.
<span style="display:none">[^15_10][^15_11][^15_12][^15_13][^15_14][^15_15]</span>

<div align="center">⁂</div>

[^15_1]: https://maximromanov.github.io/OpenITI/

[^15_2]: https://aclanthology.org/W16-4007/

[^15_3]: https://ar5iv.labs.arxiv.org/html/1612.08989

[^15_4]: https://belinkov.com/assets/pdf/lt4dh2016.pdf

[^15_5]: https://openiti.org/projects/OpenITI Corpus.html

[^15_6]: https://www.jstor.org/stable/45286327

[^15_7]: https://www.academia.edu/85992180/Digitizing_the_Textual_Heritage_of_the_Premodern_Islamicate_World_Principles_and_Plans

[^15_8]: https://kitab-project.org/Bias-in-the-OpenITI-corpus/

[^15_9]: https://www.bibliomed.org/fulltextpdf.php?mno=199918

[^15_10]: https://aclanthology.org/anthology-files/anthology-files/pdf/W/W16/W16-4007.pdf

[^15_11]: https://www.sciencedirect.com/science/article/pii/S1877050923011419

[^15_12]: https://www.semanticscholar.org/paper/Shamela:-A-Large-Scale-Historical-Arabic-Corpus-Belinkov-Magidow/b1eb22273386e209b4861b1da467d3bd1ce9ebdd

[^15_13]: https://belinkov.com/assets/pdf/lt4dh2016-slides.pdf

[^15_14]: https://mith.umd.edu/research/openiti-aocp/

[^15_15]: https://shamela.ws/book/148091


---

# نعم

أفضل ما نعمله دلوقتي إننا نبدأ من “تصوّر” لقائمة كتب + طريقة تقطيع عامّة، من غير التزام بأسماء بعينها (عشان اختيار الكتاب نفسه هيحددّه أنت حسب تفضيلك المدرسي/التراثي).

## 1. تصور لقائمة 10 كتب (نحو + بلاغة + لغة)

قسّمها كالتالي (أنت تختار العناوين الفعلية من OpenITI/الشاملة):

- نحو (5 كتب تقريبًا):
    - متون/شروح تعليمية متوسطة: كتاب مدرسـي معاصر، وشرح مبسّط على ألفية/أجرومية.
    - مرجع نحوي تحليلي (على نمط “مغني اللبيب” أو “شرح ابن عقيل” أو نظائرها).
- بلاغة (3 كتب تقريبًا):
    - كتاب تمهيدي في البيان والبديع،
    - كتاب أمثلة تطبيقية على البلاغة في القرآن والشعر.
- لغة/معجم/صرف (2 كتاب تقريبًا):
    - معجم لغوي مختصر أو معجم غريب،
    - كتاب صرف/أبنية الأفعال.

فكرة الجدول:


| النوع | الهدف من الكتاب | كيف ستستخدمه |
| :-- | :-- | :-- |
| نحو تعليمي | قواعد + تمارين + شروح مبسّطة | تحويل التمارين + الشروح إلى instruction |
| نحو تحليلي | تعليل قواعد + شواهد شعرية/قرآنية | أمثلة “عالم لغة” مع تبرير وتعليل |
| بلاغة | تشبيه/استعارة/كناية + أمثلة | استخراج أمثلة وتحويلها لأسئلة/تحليل |
| معجم/صرف | جذر، معاني، تصريفات | تغذية دور “محقّق لغوي” |

أنت تختار من OpenITI/الشاملة ما يقابل هذه الفئات.

## 2. طريقة تقطيع الكتب إلى “وحدات”

فكّر في كل كتاب كأنه متكون من وحدات صغيرة:

- وحدة “قاعدة”: تعريف + أمثلة.
- وحدة “تمرين”: سؤال + جواب في آخر الكتاب أو ضمن الشرح.
- وحدة “شرح بيت/آية”: نص + تحليل نحوي/بلاغي.

خطوات عامة:

1) تقسيم الكتاب إلى فقرات (paragraphs) اعتمادًا على العناوين والفواصل.
2) تمييز الفقرات التي تبدأ بـ “باب/فصل/قاعدة/مثال/تطبيق/مسألة”.
3) لكل وحدة:
    - استخرج:
        - نص القاعدة أو الشرح،
        - أمثلة مرتبطة،
        - إن وجد، أسئلة/تمارين.

## 3. تحويل وحدة “قاعدة نحو” → Instruction

مثال تخيلي لوحدة من شرح نحوي:

- في الكتاب:
    - “باب المبتدأ والخبر: المبتدأ هو اسم مرفوع مجرد من العوامل اللفظية، والخبر هو…”
    - أمثلة: “العلمُ نافعٌ، الطالبُ مجتهدٌ…”

تحويلها لـ مثال JSON واحد:

```json
{
  "instruction": "اشرح لطالب في المرحلة الثانوية قاعدة المبتدأ والخبر بلغة مبسطة، مع مثالين صحيحين ومثال خاطئ وتصحيحه.",
  "input": "",
  "output": "المبتدأ هو الاسم الذي نبدأ به الجملة لنتحدث عنه، وغالبًا يكون مرفوعًا، مثل: \"العلمُ\" في جملة \"العلمُ نافعٌ\". الخبر هو الجزء الذي نخبر به عن المبتدأ... مثال خاطئ: \"العلمَ نافعٌ\" والصواب: \"العلمُ نافعٌ\" لأن المبتدأ يجب أن يكون مرفوعًا...",
  "role": "tutor",
  "skills": ["nahw"],
  "level": "beginner",
  "domain": "education",
  "task_type": "explanation"
}
```

من أي باب نحوي، تقدر تولّد 2–3 أمثلة زي دي (شرح + أمثلة).

## 4. تحويل وحدة “شرح بيت بلاغيًا” → Instruction

افترض إن عندك في كتاب بلاغة فقرة تشرح بيت شعر وفيها تحديد نوع التشبيه/الاستعارة:

- في الكتاب:
    - البيت: “وإنَّ من البيان لسحرًا”
    - الشرح: بيان أن في العبارة استعارة/مجاز، ووجه الشبه، والأثر…

تحويلها:

```json
{
  "instruction": "حلّل البيت الآتي بلاغيًا لطالب جامعي: حدد نوع الصورة البلاغية واشرح أثرها.",
  "input": "وإنَّ من البيانِ لسحرًا.",
  "output": "في هذا القول تشبيه ضمني للبيان الجيد بالسحر؛ إذ يجعل السامع متأثرًا كما يتأثر المسحور. لا يُذكر المشبّه به صراحة، ولهذا يميل بعض البلاغيين إلى اعتباره استعارة أو تشبيهًا بليغًا...",
  "role": "tutor",
  "skills": ["balagha"],
  "level": "intermediate",
  "domain": "education",
  "task_type": "analysis_and_explanation"
}
```

من كل مثال في كتاب بلاغة، تقدر تستخرج واحد أو أكثر مثل هذا.

## 5. تحويل مادة معجم/صرف → “محقّق لغوي”

من فقرة في معجم أو كتاب صرف يشرح أصل كلمة وتصريفها:

- في الكتاب:
    - “كتب: أصل يدل على جمع الشيء بعضه إلى بعض؛ ومنه الكتاب لضم الحروف…”
    - صيغ: كاتب، مكتوب، كتاب، مكتبة…

تعمل مثال:

```json
{
  "instruction": "بيّن أصل الكلمة وتصريفاتها ومعناها في الاستعمال المعاصر.",
  "input": "الكلمة: \"كاتب\".",
  "output": "الكلمة من الجذر الثلاثي ك‑ت‑ب الذي يدل على جمع الشيء وضم بعضه إلى بعض... من هذا الجذر تأتي صيغ مثل: كتاب، مكتوب، مكتبة، كاتب. في الاستعمال المعاصر، \"كاتب\" تطلق على من يمارس الكتابة مهنة أو هواية...",
  "role": "muhhaqiq",
  "skills": ["morphology", "lexicon"],
  "level": "intermediate",
  "domain": "heritage",
  "task_type": "analysis_and_explanation"
}
```

كل مادة معجمية يمكن أن تتحول إلى سؤال من هذا النوع.

***

لو حابب، نقدر في الخطوة الجاية نشتغل على كتاب محدد تسميه لي (مثلاً اسم كتاب نحو أو بلاغة تحبه من الشاملة) وأبني لك عليه 3 قوالب حقيقية خطوة بخطوة (كيف تقطّع ملفه النصي، وتطلع منه 3 أمثلة JSON جاهزين).

---

# اختار الكتب المناسبه و اكمل

أنسب شيء نبدأ به بـ “مجموعة تأسيس” صغيرة لكن قوية من كتب مشهورة ومتاحة في الشاملة/OpenITI، تخدم نحو + بلاغة + لغة، وبعدين أوريك مثال عملي لتقطيع واحد منها إلى تعليمات.[^17_1][^17_2]

## 1. حزمة نحو مقترحة (5 كتب)

مقصدي هنا أسماء معروفة؛ تاخد منها أجزاء مناسبة، مش لازم الكل:

1) قطر الندى وبلّ الصدى (ابن هشام) + شرح مبسّط معاصر
2) شرح ابن عقيل على ألفية ابن مالك (اختيار أبواب أساسية، مش الكتاب كله)
3) أوضح المسالك إلى ألفية ابن مالك (ابن هشام) لبعض الأبواب النحوية
4) النحو الواضح (علي الجارم ومصطفى أمين) – جزءين: الابتدائي/الثانوي
5) كتب نحو مدرسية معاصرة (مثلاً كتاب قواعد لغة عربية للثانوي من وزارة التعليم)

الهدف:

- (4) و(5): أسلوب تعليمي حديث سهل.
- (1–3): عمق تراثي وتحليل نحوي قوي.


## 2. حزمة بلاغة مقترحة (3 كتب)

1) البلاغة الواضحة (علي الجارم – معاصر، شائع في التعليم)
2) جواهر البلاغة في المعاني والبيان والبديع (الهاشمي) – تختار منه الأبواب الواضحة والأمثلة
3) كتاب تطبيقات بلاغية على القرآن والشعر (أي كتاب تعليمي تطبيقي تجده مناسبًا في الشاملة/OpenITI)

الهدف:

- (1): تعريفات مبسطة وتمارين.
- (2): عمق تراثي وأمثلة كلاسيكية.
- (3): ربط مباشر بالقرآن والشعر.


## 3. حزمة لغة/صرف/معجم (2 كتب)

1) شذور الذهب في معرفة كلام العرب (ابن هشام) – فيه نحو وصرف ولغة
2) معجم وجيز/حديث أو معجم غريب (مثل “لسان العرب” لكن تختار مقالات محددة، أو معجم وسيط/وجيز معاصر)

الهدف:

- تغذية دور “محقّق لغوي” (جذر، معاني، تصريفات، استعمالات).

> ملاحظة: كل هذه الكتب موجودة بصيغ نصية في الشاملة، وكثير منها ضمن OpenITI أو مشاريع مماثلة.[^17_3][^17_4][^17_1]

***

## 4. مثال عملي: من “النحو الواضح” → 3 تعليمات JSON

افترض فقرة في “النحو الواضح” تشرح الجملة الاسمية بأمثلة وتمارين.

### مثال 1 – شرح قاعدة

```json
{
  "instruction": "اشرح لطالب في المرحلة الإعدادية ما هي الجملة الاسمية وما هي أركانها، مع مثالين بسيطين.",
  "input": "",
  "output": "الجملة الاسمية هي جملة تبدأ باسم، وتتركب غالبًا من مبتدأ وخبر. المبتدأ هو الاسم الذي نتحدث عنه، والخبر هو الكلام الذي نخبر به عنه. مثال 1: \"العلمُ نافعٌ\"؛ العلمُ: مبتدأ مرفوع بالضمة، نافعٌ: خبر مرفوع بالضمة. مثال 2: \"الطالبُ مجتهدٌ\"؛ الطالبُ: مبتدأ، مجتهدٌ: خبر.",
  "role": "tutor",
  "skills": ["nahw"],
  "level": "beginner",
  "domain": "education",
  "task_type": "explanation"
}
```


### مثال 2 – تمرين إعراب مع تصحيح

```json
{
  "instruction": "أعرب الجملة الآتية، ثم صحّح أي خطأ فيها واذكر السبب.",
  "input": "العلمَ نورٌ.",
  "output": "الإعراب الصحيح: العلمُ: مبتدأ مرفوع وعلامة رفعه الضمة الظاهرة على آخره. نورٌ: خبر مرفوع وعلامة رفعه الضمة. الخطأ في الجملة الأصلية هو نصب كلمة \"العلمَ\" مع أنها مبتدأ، والصواب أن تكون مرفوعة لأنها مبتدأ في جملة اسمية.",
  "role": "tutor",
  "skills": ["nahw"],
  "level": "beginner",
  "domain": "education",
  "task_type": "correction_and_explanation"
}
```


### مثال 3 – ربط نحو ببلاغة (من البلاغة الواضحة)

```json
{
  "instruction": "حلّل الجملة الآتية نحوياً وبلاغياً: عيّن المبتدأ والخبر، ثم اذكر إن كان فيها تشبيه أو استعارة.",
  "input": "العلمُ نورٌ يبدّدُ ظلماتِ الجهلِ.",
  "output": "نحوياً: العلمُ: مبتدأ مرفوع بالضمة. نورٌ: خبر مرفوع بالضمة. يبدّدُ: فعل مضارع مرفوع، والفاعل ضمير مستتر يعود على \"نور\". ظلماتِ: مفعول به منصوب بالكسرة لأنه جمع مؤنث سالم. الجهلِ: مضاف إليه مخفوض بالكسرة. بلاغياً: في الجملة تشبيه ضمني للعلم بالنور، وتشبيه للجهل بالظلمات، مما يبرز أثر العلم في إزالة الغموض والضلال كما يزيل النور الظلام.",
  "role": "tutor",
  "skills": ["nahw", "balagha"],
  "level": "intermediate",
  "domain": "education",
  "task_type": "analysis_and_explanation"
}
```


***

## 5. كيف تطبّق ده على باقي الكتب؟

- من كتب النحو التراثية (شرح ابن عقيل، أوضح المسالك):
    - تختار أبوابًا محددة (المبتدأ والخبر، الفاعل، المفعول به، النواسخ، الحال، التمييز…).
    - لكل باب:
        - مثال واحد “شرح مبسط” كما فعلنا في مثال 1.
        - مثال “تمرين + تصحيح” كما في مثال 2.
- من كتب البلاغة (البلاغة الواضحة، جواهر البلاغة):
    - تستخرج أمثلة تشبيه/استعارة/كناية، وتحولها لصيغ مثل مثال 3.
- من كتب اللغة/المعجم:
    - كل مادة (مدخل معجمي) تتحول لسؤال/جواب عن الجذر والمعنى والاستعمال، كما وضحنا سابقًا.

بهذا الشكل، قائمة الكتب المقترحة تصبح “مصنع” دائم لإنتاج أمثلة JSONL عالية الجودة.

لو تحب، في الخطوة الجاية أساعدك أكتب pseudo‑code بسيط: من ملف نصي (كتاب من الشاملة) → تقسيم لأبواب/فقرات → اختيار فقرات نحو/بلاغة → تصديرها كـ draft JSONL تحتاج فقط مراجعة خفيفة منك.

<div align="center">⁂</div>

[^17_1]: https://maximromanov.github.io/OpenITI/

[^17_2]: https://www.jstor.org/stable/45286327

[^17_3]: https://openiti.org/projects/OpenITI Corpus.html

[^17_4]: https://aclanthology.org/W16-4007/


---

# لو عاوز اضيف علم المعاني

أفضل طريقة تضيف بها **علم المعاني** إنك تعامله كـ skill مستقل جوه نفس إطار “معلّم اللغة”، بمجموعات مهام واضحة (tasks) وmetadata مخصّصة له.[^18_1][^18_2]

## 1. ما الذي يمثّل “علم المعاني” في الموديل؟

ركّز على مهام أساسية:

- مطابقة الكلام لمقتضى الحال: أسئلة عن لماذا اختير خبرية/إنشائية، تقديم/تأخير، تعريف/تنكير، قصر… إلخ.[^18_2][^18_3]
- أنواع الجمل: خبرية/إنشائية، غرض الخبر (التقرير، الإنكار، التعجب…) وغرض الاستفهام، الأمر، النهي…[^18_2]
- الإيجاز/الإطناب/المساواة: متى يكون التعبير موجزًا أو مطنَبًا، وما أثر ذلك في السياق.[^18_2]
- أساليب القصر والتخصيص: إنما، النفي والاستثناء، التقديم، تعريف المسند إليه/المسند…[^18_1][^18_2]

دي تتحوّل في الموديل لمهام تحليل وشرح، وليس فقط تعريفات نظرية.

## 2. إضافة “علم المعاني” للـ schema

في الـ JSONL اللي عندك، تضيف:

- `skills`: تضيف `"maani"` بجانب `"balagha"` أو بمفرده.
- `task_type` خاص:
    - `"maani_analysis"`, `"sentence_type_classification"`, `"qasd_al_khobar"`, `"qasr_analysis"`…

مثال:

```json
{
  "instruction": "بيّن نوع الجملة في المثال الآتي (خبرية أم إنشائية)، واذكر الغرض البلاغي من الخبر.",
  "input": "قد أفلحَ المؤمنون.",
  "output": "الجملة خبرية في ظاهرها، لكنها هنا ليست لمجرد الإخبار؛ إذ إن السامع يعلم أصلًا أن المؤمنين مفلحون بوعد الله، فيكون الغرض البلاغي تقوية الرجاء وتثبيت قلوب المؤمنين.",
  "role": "tutor",
  "skills": ["balagha", "maani"],
  "level": "advanced",
  "domain": "education",
  "task_type": "maani_analysis"
}
```


## 3. منين تجيب داتا علم المعاني؟

استخدم نفس الكتب البلاغية المقترحة، مع تركيز خاص على أبواب المعاني:

- “البلاغة الواضحة” + “جواهر البلاغة”: فيها أبواب كاملة للخبر والإنشاء، القصر، الوصل والفصل، الإيجاز والإطناب.[^18_1][^18_2]
- كتب شروح بلاغية على القرآن/الشعر (من الشاملة/OpenITI)، خصوصًا الأبواب اللي تشرح “لماذا جاء التعبير بهذا الوجه؟”.[^18_3][^18_4]

لكل باب:

- استخرج تعريف مبسط + 2–3 أمثلة.
- حوّل كل مثال لسؤال/جواب:
    - سؤال عن نوع الجملة، غرض الخبر، سبب التقديم/التأخير، وجه القصر…


## 4. أمثلة مهام جاهزة تضيفها للداتا

- تمييز نوع الجملة وغرضها:

```json
{
  "instruction": "حدد نوع الجملة (خبرية/إنشائية) واذكر الغرض البلاغي من الخبر.",
  "input": "وَمَا رَبُّكَ بِظَلَّامٍ لِّلْعَبِيدِ.",
  "output": "الجملة خبرية، والغرض البلاغي منها التوكيد والنفي القاطع لظلم الله لعباده، مما يبعث الطمأنينة في نفوس المؤمنين.",
  "skills": ["maani"]
}
```

- تحليل قصر:

```json
{
  "instruction": "حلّل أسلوب القصر في الجملة الآتية وبيّن طريقته وأثره.",
  "input": "إنما المؤمنون إخوةٌ.",
  "output": "في الجملة أسلوب قصر بـ\"إنما\"، حيث قُصر حقيقة الأخوة الحقيقية على المؤمنين، فيفيد التوكيد وحصر الأخوة الكاملة في دائرة الإيمان.",
  "skills": ["maani", "balagha"]
}
```

- تقديم/تأخير:

```json
{
  "instruction": "فسّر سرّ تقديم الجار والمجرور في الجملة الآتية.",
  "input": "في البيتِ رجلٌ.",
  "output": "تقديم الجار والمجرور \"في البيتِ\" يفيد الاهتمام بمكان وجود الرجل، وربما يدل على تخصيص وجوده بالبيت، خلافًا لقولنا: \"رجلٌ في البيت\" التي يكون فيها التركيز على وجود الرجل نفسه.",
  "skills": ["maani"]
}
```


## 5. كيف تدمجه في المرحلة الأولى بدون تعقيد

- في v1 من الداتا:
    - أضف فقط 200–300 مثال “علم معاني” منسقين بهذا الشكل، مركّزين على:
        - خبر/إنشاء،
        - غرض الخبر،
        - بعض أساليب القصر،
        - تقديم/تأخير بسيط.
- لاحقًا، تعمل Layer أعمق علميًا (أبواب الوصل والفصل، الفصل بين الجمل، إلخ)، لكن مش لازم في أول نسخة.

لو قلت لي: تحب تركز الأول في علم المعاني على “الخبر والإنشاء والقصر” ولا على “التقديم والتأخير والإيجاز/الإطناب”؟ أقدر أطلع لك 10 قوالب جاهزة لكل مجموعة تبني عليها داتا المرحلة الأولى.

<div align="center">⁂</div>

[^18_1]: https://archive.org/download/ArabicRhetoricCourse-OpenUniversityOfNigeria/Arabic Rhetoric 1.pdf

[^18_2]: https://www.academia.edu/85516391/A_Primer_of_Arabic_Rhetoric

[^18_3]: https://www.jstor.org/stable/45286327

[^18_4]: https://maximromanov.github.io/OpenITI/


---

# المرحلة الأولى بالكامل بالتفصيل

المرحلة الأولى هدفها واحد: **تأسيس داتا نظيفة ومركّزة** تبني عليها كل شيء بعد كده (بدون تدريب لسه أو بـ تجارب خفيفة جدًا في الآخر). كل اللي تحت ده تقدر تنفّذه في 4–6 أسابيع لو شغال لوحدك وبتركيز.

***

## 1. أهداف المرحلة الأولى (وشروط النجاح)

في نهاية المرحلة الأولى يكون عندك:

1) Pipeline تنظيف عربي واضح ومكرر (سكربتات ثابتة، مش شغل يدوي كل مرة).[^19_1][^19_2]
2) Dataset v1 صغير لكنه نظيف (3K–5K مثال) يغطي:
    - نحو أساسي (جملة اسمية/فعلية، مبتدأ/خبر، فاعل/مفعول، نواسخ…).
    - بلاغة (تشبيه/استعارة/كناية + بداية علم المعاني: خبر/إنشاء، قصر).[^19_3][^19_4]
3) هيكل فولدرات منظم، وLog بالتجارب والنسخ.

لو خرجت من المرحلة دي بده، فأنت جاهز تدخل أي Phase بعدها (توسيع داتا، QLoRA، إلخ).

***

## 2. تقسيم المرحلة لأربع مراحل عمل

### المرحلة 1: تجهيز البيئة وهيكل المشروع (يوم–يومين)

- هيكل فولدرات (مثال):

```
project_ar_linguist_llm/
  data/
    raw/
      web/
      books_nahw/
      books_balagha/
      lexicons/
    clean/
      web/
      books_nahw/
      books_balagha/
    instructions/
      v0/
      v1/
  scripts/
    cleaning/
    parsing_books/
    build_instructions/
  logs/
    data_notes.md
    experiments.md
```

- ملفات أساسية:
    - `logs/data_notes.md`: تسجل فيه ما تم إدخاله من مصادر + الخطوات.
    - `logs/experiments.md`: لاحقًا للتجارب، لكن حضّره من الآن.

***

### المرحلة 2: تجميع النصوص الخام + Pipeline تنظيف

#### 2.1. تجميع المصادر

1) مصادر ويب عامة عالية الجودة:
    - ArabicWeb24 (عينة فقط في البداية، مثلاً 50K–100K أسطر).[^19_2][^19_5]
2) كتب تعليمية:
    - نحو: “النحو الواضح” (الابتدائي/الثانوي)، مع 1 كتاب مدرسـي حديث.
    - بلاغة: “البلاغة الواضحة”، + كتاب تطبيقي صغير.
3) تراث/عمق:
    - نحو تراثي: أجزاء من “شرح ابن عقيل”، “قطر الندى” (أبواب منتقاة).
    - بلاغة/معاني: أجزاء من “جواهر البلاغة”.
    - لغة/صرف: أجزاء من “شذور الذهب” + مقالات من معجم موثوق.
4) Instruction جاهز:
    - CIDAR: لتحصل على أمثلة جاهزة + شكل واضح للـ instruction Arabic.[^19_6][^19_3]

ضع كل شيء كما هو في `data/raw/...`.

#### 2.2. Pipeline التنضيف (نص عام + كتب)

تبني سكربت واحد (أو مجموعة بسيطة) يعمل:

1) Normalization + إزالة الـ junk:
    - Strip HTML، remove links/emails/emojis، توحيد المسافات.[^19_7][^19_8][^19_2]
    - إزالة التشكيل العشوائي (مع استثناء نصوص أنت قررت تحافظ على تشكيلها).
2) فلترة عربية:
    - احتفظ بالأسطر التي:
        - ≥ 70٪ حروف عربية،
        - ≥ 5 كلمات،
        - ليست كلها أرقام/رموز.[^19_9][^19_10]
3) تقسيم لفقرات/جمل:
    - استخدم `.` `؟` `!` `؛` لتقسيم الجمل، مع مراعاة ألّا تقطع الآيات/الأحاديث الطويلة قطع سيئة قدر الإمكان.
4) Dedup (مهم جدًا):
    - على مستوى الجمل/الفقرات: حذف التكرار التام + شبه التام (minhash أو أداة جاهزة مثل refined‑arabic/dedup).[^19_1][^19_9]

الناتج:

- `data/clean/books_nahw/<book_name>_clean.txt`
- `data/clean/books_balagha/<book_name>_clean.txt`
- وهكذا…

***

### المرحلة 3: بناء Dataset v1 (نحو + بلاغة + علم المعاني)

الهدف النهائي هنا: ملف JSONL بإصدار مثلاً: `data/instructions/v1/ar_lang_teacher_v1.jsonl` بحوالي 3K–5K مثال، بمسكيمة thống nhất.

#### 3.1. تثبيت السكيمة (JSONL)

لكل مثال:

- حقول أساسية:

```json
{
  "id": "ar-tutor-0001",
  "instruction": "...",
  "input": "...",
  "output": "...",
  "role": "tutor",
  "skills": ["nahw", "balagha"],
  "level": "beginner",
  "domain": "education",
  "task_type": "analysis_and_explanation",
  "difficulty": 1,
  "source": "manual_nahw_alwadeh",
  "tags": ["jumla_ismiya", "i3rab"]
}
```

- قسم skills اللي هتستخدمه في v1:
    - `"nahw"`, `"balagha"`, `"maani"`, `"orthography"`, `"morphology"`, `"heritage"`.
- roles الأساسية في v1:
    - `"tutor"`, `"proofreader"`, `"assistant_general"`, `"muhhaqiq"` (بجرعة صغيرة في الأخير).


#### 3.2. حزم المهام في v1 (targets)

أ) نحو أساسي (≈ 1.5K–2K مثال):

- الجملة الاسمية والفعلية، المبتدأ والخبر، الفاعل، المفعول به، كان وأخواتها، إن وأخواتها، المضاف والمضاف إليه، الحال، التمييز.
- أنواع المهام:
    - شرح قاعدة + أمثلة.
    - إعراب جملة.
    - تصحيح خطأ مع تعليل.

ب) بلاغة (بيان + بداية معاني) (≈ 1K–1.5K مثال):

- تشبيه، استعارة، كناية، مجاز.
- بداية علم المعاني:
    - خبر vs إنشاء، غرض الخبر، أساليب القصر (إنما، النفي والاستثناء)، تقديم/تأخير بسيط.[^19_11][^19_12][^19_13]
- مهام:
    - تحديد نوع الصورة البلاغية وشرحها.
    - تحديد نوع الجملة وغرضها.
    - تحليل أسلوب قصر بسيط.

ج) Proofreading عربي تعليمي (≈ 500–1K مثال):

- جمل بها أخطاء نحوية/إملائية/أسلوبية بسيطة مستمدة من نصوص طلابية أو مولدة صناعيًا.
- المهام:
    - تصحيح النص + شرح 2–3 أخطاء رئيسية.

د) جرعة صغيرة “محقّق لغوي/تراث” (≈ 200–300 مثال):

- شروح لأبيات أو جمل تراثية (من شرح ابن عقيل/جواهر البلاغة/شذور الذهب).
- مهام:
    - إعراب نقاط أساسية.
    - شرح المعنى بلغة معاصرة.
    - تعليق على لفظ/تركيب قديم.


#### 3.3. طريقة إنتاج الأمثلة عمليًا

1) يدوياً من الكتب التعليمية:

- لكل باب في “النحو الواضح” + “البلاغة الواضحة”:
    - مثال واحد “شرح قاعدة” (instruction بدون input، output = شرح + أمثلة).
    - مثال واحد “تمرين + تصحيح” (instruction + جملة في input، output = إعراب/تحليل/تصحيح).
- من 10–15 باب نحوي وبلاغي ممكن بسهولة تطلع 300–500 مثال يدويًا في أيام قليلة.

2) شبه آلي من التراث:

- تخصيص سكربت بسيط:
    - يقطع فقرات مثل:
        - “مثال: …”
        - “ومن ذلك قوله: … ثم قال…”
    - يحفظها لك في CSV/JSON draft.
- أنت بعد كده:
    - تضيف instruction مناسب،
    - تنظف الشرح قليلاً (تخفيف اللغة، إزالة حشو)،
    - وتحولها لمثال نهائي.

3) أمثلة صناعية بمساعدة LLM:

- تستخدم موديل جاهز (GPT/Perplexity) لتوليد:
    - جمل فيها أخطاء نحوية،
    - جمل تحتاج تحليل بلاغي بسيط،
    - ثم تراجعها يدويًا قبل إدخالها في v1.

***

### المرحلة 4: ضمان الجودة (Checklist) + إقفال v1

#### 4.1. Checklist لجودة المثال الواحد

لكل (instruction, input, output):

- instruction:
    - واضح، بلغة فصحى بسيطة، يحدد المستوى (مبتدئ/متوسط…).
    - يحدد المطلوب بدقة (اشرح/أعرب/صحّح/حلّل…).
- input:
    - جملة/نص قصير واضح (لا يزيد عادة عن 1–2 سطر في v1).
    - بدون أخطاء غير مقصودة (إلا لو هدف السؤال كشفها).
- output:
    - لغة سليمة، بدون أخطاء نحوية/إملائية.
    - منظم (مثلاً: “أولًا: …، ثانيًا: …”)، خاصة في الشرح والتحليل.
    - يجاوب بالضبط على المطلوب، بلا حشو زائد.
- metadata:
    - role/skills/level/domain/task_type مضبوطة.
    - source/tag تساعدك ترجع للمصدر الأصلي إن احتجت.

يمكنك تبني مبدأ بسيط:

- “لو المثال مش ممتاز لدرجة أكون فخور به كمحتوى تعليمي لوحده، ما يدخلش v1.”


#### 4.2. أداة مراجعة (اختيارية لكن قوية)

- تطوّر سكربت صغير يستخدم LLM كبير كمُراجع:
    - يقرأ كل مثال، ويعطي Score 1–5 لجودة اللغة/التعليم/الوضوح.
    - ترمي أو تصلح الأمثلة تحت مستوى معين.
- عادةً ما تذكره الأبحاث عن Arabic post‑training datasets إن طبقة الـ auditing دي تفرق جدًا في جودة الموديل النهائي.[^19_14][^19_15]

***

## 3. نهاية المرحلة الأولى: ماذا تفعل؟

عند هذه النقطة يكون عندك:

- `data/instructions/v1/ar_lang_teacher_v1.jsonl` (3K–5K مثال عالي الجودة).
- Scripts تنظيف + parsing للكتب + بناء JSONL.
- Log واضح بالخطوات والمصادر.

اختياراتك بعد كده:

- تعمل QLoRA صغيرة (epoch أو اثنين) على Base زي ALLaM‑7B أو Qwen2.5‑7B على هذه الداتا فقط لتشوف أول “نَفَس” للموديل كـ معلّم.[^19_16][^19_17]
- أو تكمل على نفس Pipeline لتوسّع الداتا لـ 20K–50K مثال قبل أي تدريب.

لو تحب، الخطوة الجاية أقدر أساعدك أكتب لك “قائمة مهام دقيقة للأسبوع الأول من المرحلة دي” على شكل To‑Do daily (يوم 1/2/3…) عشان تبدأ فورًا من النهارده.

<div align="center">⁂</div>

[^19_1]: https://lighton.ai/lighton-blogs/arabicweb24

[^19_2]: https://huggingface.co/blog/MayFarhat/arabicweb24

[^19_3]: https://github.com/ARBML/CIDAR

[^19_4]: https://aclanthology.org/2025.arabicnlp-main.36.pdf

[^19_5]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^19_6]: https://aclanthology.org/2024.findings-acl.764/

[^19_7]: https://github.com/h9-tect/Arabic_nlp_preprocessing

[^19_8]: https://github.com/ehsanasgari/Tahdheeb

[^19_9]: https://github.com/mohataher/refined-arabic

[^19_10]: https://stackoverflow.com/questions/71834104/arabic-dataset-cleaning-removing-everything-but-arabic-text

[^19_11]: https://archive.org/download/ArabicRhetoricCourse-OpenUniversityOfNigeria/Arabic Rhetoric 1.pdf

[^19_12]: https://www.academia.edu/85516391/A_Primer_of_Arabic_Rhetoric

[^19_13]: https://www.jstor.org/stable/45286327

[^19_14]: https://arxiv.org/html/2507.14688v1

[^19_15]: https://arxiv.org/abs/2407.02147

[^19_16]: https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Arabic

[^19_17]: https://ai.azure.com/catalog/models/ALLaM-2-7b-instruct


---

# لو هعمل المشروع بمساعده AI agents ايه agents او roles اللي هحتاجها بالتفصيل

أنسب طريقة هي تبني المشروع كـ “فريق Agents متخصصين” حواليك أنت كـ **Architect / Human-in-the-loop**، مش Agent واحد بيعمل كل حاجة.[^20_1][^20_2]

## 1. Agents أساسية للبيانات (Data \& Cleaning)

### 1) Data Discovery / Scraper Agent

دوره: البحث عن المصادر وجمعها بشكل منظم.

- مهامه:
    - إيجاد وتحميل كوربسات جاهزة (ArabicWeb24, CIDAR, ArabicWeb‑Edu…).[^20_3][^20_4][^20_5]
    - اقتراح كتب نحو/بلاغة/تراث مناسبة، وتوليد قوائم URL/روابط shamela/OpenITI.
- الأدوات:
    - Web search + simple scrapers + واجهات Hugging Face Datasets API.[^20_6][^20_7]


### 2) Cleaning \& Normalization Agent

دوره: تطبيق Pipeline التنضيف على كل مصدر نص.

- مهامه:
    - اقتراح قواعد normalization عربية (توحيد الحروف، إزالة الضوضاء، التعامل مع التشكيل).[^20_8][^20_9][^20_10]
    - توليد/تعديل سكربتات Python (Regex + langdetect + segmentation + dedup) وتشغيلها على دفعات صغيرة.
- الأدوات:
    - Python + مكتبات preprocessing (refined‑arabic, Arabic_nlp_preprocessing, Tahdheeb).[^20_9][^20_10][^20_11]


### 3) Dedup \& Quality Filter Agent

دوره: تقليل التكرار وفلترة النص الرديء.

- مهامه:
    - استخدام أدوات dedup (MinHash, datatrove-like, NeMo Curator).[^20_12][^20_6][^20_9]
    - اقتراح thresholds، وتحليل عينات قبل/بعد، ورفع تقارير بالقيم المثلى.

أنت تراقب: لا تترك agent يفلتر بدون review لعينة.

***

## 2. Agents لتصميم وتحويل الداتا التعليمية

### 4) Schema \& Data Modeling Agent

دوره: حارس الـ JSONL schema.

- مهامه:
    - تثبيت وتحديث سكيمة (instruction, input, output, role, skills, level, …).
    - توليد قوالب جاهزة لكل نوع مهمة (نحو، بلاغة، معاني، تحقيق، لهجات…) مع أمثلة.[^20_13][^20_14]
    - التحقق أن أي ملف JSONL جديد يطابق السكيمة (Validation + تقارير أخطاء).


### 5) Instruction Generation Agent (Tutor Designer)

دوره: تحويل النصوص (كتب + شروح) إلى أمثلة تعليمية.

- مهامه:
    - أخذ فقرة من “النحو الواضح”/“البلاغة الواضحة” وتحويلها لـ 2–3 أمثلة:
        - شرح قاعدة،
        - تمرين + تصحيح،
        - تحليل نحو+بلاغة.[^20_4][^20_5]
    - احترام الـ schema: تعبئة role, skills, level, task_type تلقائيًا قدر الإمكان.
- الأدوات:
    - LLM قوي مع system prompt واضح: “أنت مصمم داتا تعليمية عربية، التزم بالسكيمة التالية…”


### 6) Heritage \& Maʿānī Agent

دوره: التعامل مع نصوص التراث (OpenITI, Shamela) واستخراج أمثلة علمية.

- مهامه:
    - قراءة قطع من شروح نحوية/بلاغية/معجمية، تلخيصها بلغة معاصرة.
    - اقتراح أسئلة “علم معاني” (خبر/إنشاء، قصر، تقديم/تأخير) من أمثلة قرآنية/شعرية.[^20_15][^20_16]

أنت هنا مهم جدًا: تراجع العيّنات الأولى، وتضع قواعد واضحة لما هو مقبول أو لا.

***

## 3. Agents للـ QA والتدقيق (Quality \& Evaluation)

### 7) Data Auditor / Critic Agent

دوره: “المصحّح اللغوي والممتحِن” للداتا.

- مهامه:
    - تقييم كل مثال (instruction, input, output) على محاور: صحة اللغة، وضوح الشرح، عدم الحشو، مناسب للمستوى.[^20_17][^20_18]
    - وضع Score (1–5) وتعليقات قصيرة، ووضع علمات (flags) للأمثلة التي تحتاج تعديل/حذف.
- الأدوات:
    - LLM كبير (حتى لو غير عربي 100٪) مع Prompt واضح + Rules، وبعدين أنت تراجع الأمثلة ذات التقييم المتدني أو التعليقات المهمة.


### 8) Style \& Consistency Agent

دوره: توحيد الأسلوب.

- مهامه:
    - ضبط الأسلوب إلى فصحى مبسطة، تنسيق المخرجات (أولًا/ثانيًا، نقاط، إلخ).
    - إزالة التناقضات في طريقة الشرح بين أمثلة ناتجة من مصادر مختلفة.
- طريقة العمل:
    - يقرأ output، ويعيد صياغته مع الحفاظ على المضمون، وفق “دليل أسلوب” تكتبه أنت (Tone Guide).


### 9) Safety \& Policy Agent

لو ناوي تطلع الموديل للجمهور:

- مهامه:
    - فلترة تعليمات/محتوى يتعارض مع سياسة المشروع (خطاب كراهية، إهانات، محتوى حساس غير مناسب…).[^20_19]
    - اقتراح صياغات بديلة أكثر احترامًا عند الحاجة.

***

## 4. Agents خاصة بالـ Model \& Training

### 10) Model Selection \& Config Agent

دوره: يساعدك في اختيار الـ base model وconfig QLoRA.

- مهامه:
    - مقارنة ALLaM / Qwen / LLaMA على ضوء بنشماركات عربية (OALL, Arabic‑LLM‑Benchmarks).[^20_20][^20_21][^20_22]
    - اقتراح إعدادات QLoRA مناسبة لحجم الداتا والـ GPU (lr, batch, epochs, max_seq_len…).[^20_18][^20_23]


### 11) Training Orchestrator Agent

دوره: يكتب ويشغّل سكربتات التدريب والتقييم البسيطة.

- مهامه:
    - توليد سكربتات Python/Colab (HF Trainer, PEFT, QLoRA) بناءً على وصفك.
    - تشغيل تجارب صغيرة (dry‑run) مع عدد عينات محدود، وقراءة الـ logs، واستخلاص أرقام أساسية (loss curves…).[^20_2][^20_1]


### 12) Evaluation \& Benchmark Agent

دوره: اختبار الموديل على بنشماركات عامة + mini‑benchmark خاص بك.

- مهامه:
    - تشغيل الـ model على:
        - مجموعة أسئلتك الخاصة (نحو/بلاغة/معاني/تحقيق/لهجات)،
        - بنشماركات جاهزة عربية (OALL, CamelEval, ArabicWeb‑Edu tasks).[^20_21][^20_22][^20_24]
    - تلخيص النتائج في جداول وتقارير قصيرة (قبل/بعد fine‑tune).

***

## 5. Agents للمنتج والتفاعل

### 13) Prompt \& UX Designer Agent

دوره: تصميم الـ system prompts وواجهات التفاعل.

- مهامه:
    - تصميم Prompts للأدوار (معلّم نحو، شاعر، محقّق…) بحيث يسهل استخدامها في الـ app أو الـ API.
    - اقتراح سيناريوهات استخدام (Use Cases) لطلاب، معلمين، باحثين.


### 14) Docs \& Knowledge Base Agent

دوره: بناء توثيق للمشروع.

- مهامه:
    - توليد README منظم، وثائق Data Pipeline، شرح كيف تستخدم الـ datasets، كيف تشغّل التدريب والتقييم.
    - تحديث “دفتر يوميات المشروع” (Project Log) من خلال ملخصات يومية/أسبوعية تُبنى على ملاحظاتك.

***

## 6. كيف تربط كل ده عمليًا بدون تعقيد؟

لا تحتاج تطبق الـ 14 Agent مرة واحدة؛ ممكن تبدأ بـ “Team صغير” في البداية:

1) Data Cleaning Agent
2) Instruction Generation Agent
3) Data Auditor Agent
4) Model Config + Training Agent

ثم تضيف الباقي مع الوقت حسب الحاجة.

مهم جدًا:

- أنت تبقى الـ Orchestrator البشري، تحدد الـ goals والـ pipelines، والـ Agents تنفذ أجزاء محددة، وتراجع أنت النتائج (خاصة في النحو/البلاغة/المعاني).

لو تحب، أقدر في الخطوة الجاية أقترح لك “أول 4 Agents” بالتحديد + أدوات/تقنية لكل واحد (LangChain / Autogen / CrewAI / سكربتات بسيطة)، مع مثال لتدفق عمل (Workflow) من:
`كتاب نحو خام → Cleaning Agent → Instruction Agent → Auditor Agent → JSONL v1`.
<span style="display:none">[^20_25][^20_26][^20_27][^20_28][^20_29][^20_30][^20_31][^20_32][^20_33][^20_34][^20_35]</span>

<div align="center">⁂</div>

[^20_1]: https://blog.dataengineerthings.org/integrating-llms-and-ai-agents-into-data-engineering-workflows-f0fb05eedb09

[^20_2]: https://www.mn.uio.no/ifi/studier/masteroppgaver/dkm/data-pipelines-creation-using-multi-agent-llms.html

[^20_3]: https://huggingface.co/datasets/lightonai/ArabicWeb24

[^20_4]: https://github.com/ARBML/CIDAR

[^20_5]: https://aclanthology.org/2025.arabicnlp-main.36.pdf

[^20_6]: https://github.com/NVIDIA-NeMo/Curator

[^20_7]: https://metatext.io/datasets-list/arabic-language

[^20_8]: https://huggingface.co/blog/MayFarhat/arabicweb24

[^20_9]: https://github.com/mohataher/refined-arabic

[^20_10]: https://github.com/ehsanasgari/Tahdheeb

[^20_11]: https://github.com/h9-tect/Arabic_nlp_preprocessing

[^20_12]: https://lighton.ai/lighton-blogs/arabicweb24

[^20_13]: https://aclanthology.org/2025.findings-acl.1232.pdf

[^20_14]: https://www.digitalocean.com/community/tutorials/json-for-finetuning-machine-learning-models

[^20_15]: https://maximromanov.github.io/OpenITI/

[^20_16]: https://www.jstor.org/stable/45286327

[^20_17]: https://arxiv.org/html/2507.14688v1

[^20_18]: https://arxiv.org/abs/2407.02147

[^20_19]: https://addepto.com/blog/from-data-pipelines-to-ai-agents-how-enterprises-operationalize-llm-systems-without-losing-reliability/

[^20_20]: https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Arabic

[^20_21]: https://github.com/tiiuae/Arabic-LLM-Benchmarks

[^20_22]: https://silma.ai/arabic-llm-benchmark

[^20_23]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12900375/

[^20_24]: https://arxiv.org/html/2409.12623v1

[^20_25]: https://ioblend.com/ioblend-integrates-ai-agents-into-data-pipelines/

[^20_26]: https://www.youtube.com/watch?v=R43Q0nIXa1Q

[^20_27]: https://www.linkedin.com/pulse/part-3-ai-core-llms-data-pipelines-industrial-pieter-van-schalkwyk-1rdnc

[^20_28]: https://www.sciencedirect.com/science/article/abs/pii/S1566253525009273

[^20_29]: https://www.youtube.com/watch?v=T315bC-xOm0

[^20_30]: https://www.linkedin.com/posts/aditya-hicounselor_ai-agenticai-llm-activity-7380950007530598403-_XIo

[^20_31]: https://datavizandai.github.io/2024/09/28/AI_Agents_vs._AI_Pipelines-3A_a_Practical_Guide_to_Coding_Your_LLM_Application.html

[^20_32]: https://arxiv.org/html/2509.23988v2

[^20_33]: https://www.youtube.com/watch?v=xFocS1TuzO0

[^20_34]: https://aws.plainenglish.io/ai-agents-in-data-engineering-the-next-evolution-in-automated-pipelines-523268db69d8

[^20_35]: https://keymakr.com/blog/agent-training-data-a-guide-to-llm-based-agent-training/


---

# first phase  full code google colab

الملف جاهز للتنزيل وتشغيله مباشرة على Google Colab.

## محتويات الـ Notebook (15 Cell)

| Cell | الوظيفة |
| :-- | :-- |
| 1 | تثبيت المكتبات (datasets, camel-tools, datasketch, pyarabic…) |
| 2 | Mount Google Drive + بناء هيكل الفولدرات كاملاً |
| 3 | تحميل CIDAR من HuggingFace وحفظه raw |
| 4 | تحميل عينة 100K من ArabicWeb24 (streaming) |
| 5 | Pipeline التنظيف (normalize + remove noise + heuristic filter) |
| 6 | Deduplication (MinHash LSH – Near + Exact duplicates) |
| 7 | تشغيل Pipeline على جميع ملفات الكتب + عينة الويب |
| 8 | Schema Class مع validate() وsave_dataset() |
| 9 | 10 seed examples جاهزة (نحو + بلاغة + علم المعاني + تصحيح) |
| 10 | تحويل CIDAR → سكيمتنا الموحدة تلقائيًا |
| 11 | Book Parser: يقطّع كتاب نصي → draft JSONL جاهز للمراجعة |
| 12 | LLM Auditor Agent (يقيّم كل مثال ويفلتر الضعيف) |
| 13 | Merge \& Balance → v1 (mix نسب 40% نحو / 25% بلاغة / 10% معاني…) |
| 14 | Stats Report (roles, skills, levels, avg length) |
| 15 | كتابة Project Log + خطوات المرحلة الجاية |

## كيف تستخدمه

1. افتح Colab وارفع الملف أو استخدم File → Open.
2. شغّل Cell 1 و2 أولاً.
3. ضع ملفات كتب النحو/البلاغة `.txt` في `data/raw/books_nahw/` و`data/raw/books_balagha/`.
4. شغّل باقي الـ cells بالترتيب.
5. راجع ملفات `_draft.jsonl` يدويًا قبل إدخالها في v1 (Cell 11).
6. ضع API Key لتشغيل Auditor في Cell 12 (اختياري لكن مهم جدًا).

---

# ضع ملفات كتب النحو/البلاغة .txt في data/raw/books_nahw/ وdata/raw/books_balagha/.

في النقطة دي فيه خطوتين بس مطلوبين منك يدويًا؛ الكولاب نفسه جاهز:

## 1. تجهيز ملفات الكتب

- حمّل نصوص الكتب اللي اخترناها (مثلاً “النحو الواضح”، “البلاغة الواضحة”، “شرح ابن عقيل”…) بصيغة **نص عادي** `.txt` من الشاملة أو أي مصدر آخر.
- لو نزلت PDF:
    - حوّله لـ `.txt` بأي أداة (Pandoc, Calibre, أو موقع تحويل PDF→TXT)،
    - راقب سريعًا إن الترميز UTF‑8 والعربي ظاهر صح.

سمِّ الملفات بشكل واضح، مثل:

- `nahw_alwadeh.txt`
- `balagha_alwadeha.txt`
- `sharh_ibn_aqeel.txt`


## 2. وضع الملفات في المسار الصحيح

بعد ما تشغّل Cell 2 في الكولاب (اللي بيبني المسارات):

- في Colab من الشريط الأيسر:
    - افتح تبويب Files.
    - اذهب إلى:
        - `/content/drive/MyDrive/project_ar_linguist_llm/data/raw/books_nahw/`
        - `/content/drive/MyDrive/project_ar_linguist_llm/data/raw/books_balagha/`
- ارفع الملفات كما يلي:
    - كتب النحو → فولدر `books_nahw`
    - كتب البلاغة → فولدر `books_balagha`

بعد الرفع:

1) شغّل Cell 7 في النوتبوك: هينظّف الكتب ويكتب نسخ نظيفة في `data/clean/books_*`.
2) شغّل Cell 11: هيولّد ملفات `_draft.jsonl` من الكتب، جاهزة للمراجعة والتحويل لـ v1.

لو حابب أشرح لك خطوة تحويل PDF الشاملة إلى TXT بشكل نظيف (مع Regex بسيطة لإزالة أرقام الصفحات والحواشي)، قل لي نوع الكتب اللي ناوي تبدأ بها (اسم كتاب أو اثنين) وأنا أكتب لك خطوات/سكربت صغير مخصوص لهم.

---

# books.json

الملف `books.json` عندك ممتاز كبنية ميتاداتا لمكتبة تراثية كاملة، وفيه كل ما نحتاجه لمرحلة البيانات الأولى.[^23_1]

## 1. ما الذي يحتويه books.json؟

كل عنصر في `books` يمثل كتابًا واحدًا مع حقول مثل:[^23_1]

- `title`: عنوان الكتاب.
- `cat_name`: التصنيف (مثلاً: كتب اللغة، الأدب، العروض والقوافي…).
- `file`: اسم ملف النص المقابل (`...txt`) اللي يفترض يكون عندك في نفس الداتا.
- `size_mb`: حجم الملف.
- `authors`: قائمة مؤلفين (اسم، سنة الوفاة، الدور).

مثال من الملف:[^23_1]

- كتاب لغة:
    - `title`: الطريق المستقيم في نظم علامات الترقيم
    - `cat_name`: كتب اللغة
    - `file`: `4_الطريق_المستقيم_في_نظم_علامات_الترقيم.txt`
- كتاب عروض/قافية:
    - `title`: القوافي للأخفش الأوسط
    - `cat_name`: العروض والقوافي
    - `file`: `737_القوافي_للأخفش_الأوسط.txt`

هذا يتيح لك:

- تصفية كتب النحو/اللغة/البلاغة/الأدب بسهولة عبر `cat_name`.
- معرفة اسم ملف الـ `.txt` لكل كتاب لاستخدامه في Pipeline التنظيف (يذهب إلى `data/raw/books_nahw/` أو `books_balagha/` حسب نوعه).


## 2. كيف تستخدمه مع النوتبوك اللي بنيّاه؟

في Colab:

1) انسخ ملف `books.json` إلى مشروعك، مثلًا إلى:
`/content/drive/MyDrive/project_ar_linguist_llm/data/raw/books.json`
2) أضف سكربت صغير (Cell جديد) يقرأ `books.json` ويختار الكتب المناسبة:

- نحو/لغة وصرف → انسخ ملفاتها إلى `books_nahw`.
- بلاغة/أدب/عروض → انسخ ملفاتها إلى `books_balagha`.

شيء زي:

- كل كتاب `cat_name` فيه: “كتب اللغة، العروض والقوافي، الأدب” تقدر تصنفه كـ مصدر نحو/بلاغة/لغة حسب ما تقرر.
- تستخدم `file` لربط الميتاداتا بالملف الفعلي `.txt` عندك في مجلد آخر (مثلاً مجلد كامل للشاملة).

لو تحب، أكتب لك الآن Cell بايثون جاهز للـ Colab:

- يقرأ `books.json`.
- يختار كتب:
    - `cat_name` ∈ {كتب اللغة، العروض والقوافي، الأدب}.
- يطبع لك قائمة منظمة: `title` + `file` عشان تعرف أي ملفات تنقلها إلى `books_nahw` و`books_balagha`.

هل عندك ملفات النص `.txt` في مجلد واحد (مثلاً dump للشاملة) على Google Drive، ولا لسه هتنقلها؟ بناءً على إجابتك أقدر أكتب السكربت بالمسارات المناسبة مباشرة.

<div align="center">⁂</div>

[^23_1]: books.json


---

# اشرحلي استفيد من كل الكتب ازاي

تقدّر تتعامل مع الكتب في `books.json` كأنها “مناجم خام” مختلفة، وكل مجموعة تخدم دور معيّن في الموديل (نحو، بلاغة، معاني، شعر، ثقافة…).[^24_1]

## 1. فهم خريطة الكتب من `books.json`

كل كتاب عندك موصوف تقريبًا بـ:[^24_1]

- `title`: اسم الكتاب (مثلاً: القوافي، مختارات شعراء العرب، وصايا الملوك…).
- `cat_name`: نوعه (الأدب، العروض والقوافي، كتب اللغة، التاريخ، الحديث، التخريج…).
- `file`: اسم ملف الـ `.txt` اللي فيه النص الكامل.

من هنا تقسّم المكتبة إلى “مجموعات استخدام”:

- كتب اللغة/النحو/الصرف.
- كتب البلاغة والأدب والشعر (مختارات شعراء العرب، التذكرة السعدية…).
- كتب الحديث/التفسير/الفقه (للاستشهاد الديني/السياقات).
- كتب التاريخ والتراجم (للسياق الثقافي واللغوي القديم).


## 2. دور كل مجموعة كتاب في مشروعك

### أ) كتب اللغة والنحو والصرف

استخدامها:

- استخراج **قواعد صريحة** + أمثلة → تحويلها إلى تعليمات teacher (شرح + تمارين + إعراب).
- استخراج أمثلة إعراب جاهزة من الشروح → تحويلها إلى “حلول نموذجية” في الـ output.

الهدف: تقوية مهارة **معلم النحو** و**المحقق الصرفي** (جذر، وزن، تصريف).

### ب) كتب البلاغة والأدب والشعر

استخدامها:

- استخراج الأبيات والأمثلة البلاغية:
    - تشبيه، استعارة، كناية، مجاز، صور معاني (خبر/إنشاء، قصر، تقديم وتأخير…).
- تحويل كل مثال إلى سؤال/جواب:
    - “حدّد نوع الصورة وأثرها”، “حلّل البيت نحوياً وبلاغياً”، “ما الغرض من هذا الخبر؟”.

الهدف: تقوية دور **معلم البلاغة** و**الشاعر/الناقد الأدبي**.

### ج) المختارات الشعرية (مثل “مختارات شعراء العرب”، “التذكرة السعدية”)

استخدامها:

- بنك ضخم للأبيات الجاهزة للتحليل أو النظم على غرارها.
- تدريب الموديل على **فهم الأساليب الشعرية**، واستحضار شواهد عند الشرح.

الهدف: مهارة **فهم الشعر الكلاسيكي** + توليد شعر جديد يستلهم نفس الروح، مع القدرة على شرحه.

### د) كتب الحديث والسيرة والأخلاق (مثل “مرويات المزاح والدعابة…”, “الأحاديث الواردة في اللعب والرياضة”)

استخدامها (بحذر):

- أمثلة تطبيقية على اللغة في سياق نبوي/صحابي (ألفاظ، تراكيب، بلاغة في الحديث).
- مواد تصلح لدور “المعلّم” و“المربي” في التطبيق، وليس لفقه الإفتاء.

الهدف: تغذية الموديل بأمثلة عربية صحيحة عالية القيمة، ومعرفة سياقات دينية عند شرح مفردات أو أمثال.

### هـ) كتب التاريخ والتراجم (مثل “مستفاد من ذيل تاريخ بغداد”، “موسوعة مرآة الحرمين…”)

استخدامها:

- سياقات لغوية/أسلوبية (كيف تُكتب التراجم، كيف يُسرد التاريخ، أنماط الجمل الطويلة).
- أمثلة للـ **تحقيق اللغوي**:
    - تحليل جملة تاريخية قديمة، شرح مفرداتها، إعادة صياغتها بالعربية المعاصرة.

الهدف: دعم دور **المحقق النصي/اللغوي** الذي يحوّل نص تراثي إلى شرح واضح لطلاب اليوم.

## 3. كيف تبني Pipeline استفادة من كل الكتب؟

فكّر في 3 مستويات:

### المستوى 1: “قراءة خام” (Pretraining/Background)

- كل الكتب (تقريبًا) تعتبر **كوربس قراءة** للموديل (لو عملت pretraining/continued pretraining).
- هنا تهمك: التنضيف، التطبيع، إزالة التكرار، دون تحويل مباشر إلى Instruction.


### المستوى 2: “قطع قابلة للتحويل إلى تعليمات”

- تختار أنواع كتب معيّنة لكل دور:
    - نحو/صرف: كتب اللغة والنحو.
    - بلاغة/معاني: كتب البلاغة والأدب والتذوّق.
    - شعر: مختارات الشعر.
    - تحقيق: كتب الشروح، التراجم، التاريخ.
- لكل كتاب:
    - تقطّعه إلى وحدات (باب، فصل، مسألة، مثال…).
    - تصنّف كل وحدة: قاعدة / مثال / تمرين / شرح بيت / حكاية / ترجمة.
    - تحوّل كل وحدة مناسبة إلى سطر JSONL (instruction + input + output).


### المستوى 3: “مراجع استرجاعية” (RAG/Tools)

- تستعمل الكتب كمراجع في **نظام استرجاع** مع الموديل:
    - عند سؤال عن بيت أو كتاب:
        - يبحث Agent في هذه النصوص،
        - يستخرج الفقرة المناسبة،
        - يشرحها الموديل اعتمادًا على النص الأصلي.
- خصوصًا كتب معاجم أو شروح، تكون مثل “معجم داخلي” يراجع منه الموديل بدل الاعتماد على الذاكرة فقط.


## 4. مثال مبسّط: كتاب أدب + كتاب شعر + كتاب لغة

افترض عندك:

- كتاب أدب: “وصايا الملوك” → حِكم وأمثال، جمل قوية.
    - تستخرج وصية واحدة؛
    - تعمل منها:
        - شرح لغوي/بلاغي،
        - سؤال للطالب: “ما الفكرة الرئيسة؟ ما الصورة البلاغية؟”.
- كتاب شعر: “مختارات شعراء العرب” → بيت رائع عن العلم أو الشجاعة.
    - تعمل منه:
        - تمرين تحليل بلاغي،
        - تمرين إعراب البيت،
        - مهمة: “اكتب بيتًا على نفس الوزن والمعنى بلغة مبسطة.”
- كتاب لغة: كتاب علامات الترقيم أو نحو/صرف.
    - تستخدمه لتوليد تمارين تصحيح:
        - جملة بلا علامات ترقيم صحيحة،
        - الموديل يضعها ويشرح السبب.

بهكذا، نفس المكتبة التي قد تبدو عشوائية تتحوّل إلى أربع رافعات:

1) كوربس قراءة عميق.
2) منجم تعليم نحو/بلاغة.
3) بنك شواهد شعرية وأمثال.
4) قاعدة استرجاع (RAG) للمحقق اللغوي.

لو قلت لي تصنيفين تحب تبدأ بهم (مثلاً: “كتب اللغة + الأدب” أو “الأدب + العروض والقوافي”) أقدر أساعدك نكتب فلتر Python على `books.json` يختار لك 20–30 كتابًا مناسبة كبداية، مع اقتراح دور كل واحد في الفيز الأولى من المشروع.

<div align="center">⁂</div>

[^24_1]: books.json


---

# هحتاج ايه بالظبط عشان اجهزهم

عشان تجهّز الكتب (لغة + نحو + أدب + بلاغة) للمرحلة الأولى، ستحتاج 3 أشياء رئيسية: ملفات النص نفسها، تنظيمها في فولدرات، ثم تمريرها في Pipeline التنضيف والبناء اللي جهزناه.

## 1. المتطلبات الأساسية

- `books.json` كما عندك الآن (ميتاداتا الكتب).[^25_1]
- فولدر يحتوي كل ملفات الشاملة `.txt` أو على الأقل الكتب التي يذكرها `books.json` في حقل `file` (مثلاً:
`shamela_dump/737_القوافي_للأخفش_الأوسط.txt`، …).[^25_1]
- مشروع Colab/Drive بالهيكل:

`/content/drive/MyDrive/project_ar_linguist_llm/`
وفيه `data/raw`, `data/clean`, `data/instructions`… (تم إنشاؤه في النوتبوك السابق).

## 2. تجهيز الكتب نفسها (خطوات عملية)

1) التأكد من وجود ملفات النص:
    - كل سطر في `books.json` له `file` مثل:
`737_القوافي_للأخفش_الأوسط.txt`, `740_مختارات_شعراء_العرب_لابن_الشجري.txt`.[^25_1]
    - هذه الملفات يجب أن تكون موجودة فعليًا في فولدر الـ dump (مثلاً: `shamela_txt/`).
2) اختيار الكتب المستهدَفة للمرحلة الأولى:
    - كتب اللغة والنحو: `cat_name` يشمل “كتب اللغة”، “النحو”، وربما بعض “العروض والقوافي” التي فيها شرح لغوي.[^25_1]
    - كتب الأدب/الشعر/البلاغة: `cat_name` مثل “الأدب”، “العروض والقوافي”.[^25_1]
    - ستحتاج فلتر صغير في Colab يقرأ `books.json` ويخرج قائمة عناوين + أسماء ملفات لهذه التصنيفات.
3) نسخ/ربط الملفات إلى فولدرات المشروع:
    - من فولدر الشاملة الأصلي (dump) → إلى:
        - `data/raw/books_nahw/` لكتب النحو/اللغة.
        - `data/raw/books_balagha/` لكتب الأدب/البلاغة/الشعر.
    - يمكنك البداية بـ 10–20 كتاب فقط.

## 3. ما الذي سيقوم به النوتبوك بعد تجهيز الملفات؟

بمجرد أن تضع ملفات `.txt` في المسارات الصحيحة:

1) Cell 7 (Process Raw Files):
    - يقرأ كل كتاب من `data/raw/books_nahw/` و`books_balagha/`.
    - يطبّق `clean_arabic_text()` + dedup.
    - يكتب نسخة نظيفة إلى `data/clean/books_nahw/..._clean.txt` و`books_balagha/..._clean.txt`.
2) Cell 11 (Book-to-Instructions Parser):
    - يقطع الكتب النظيفة إلى “Blocks” حسب العناوين (باب/فصل/مسألة/مثال…).
    - لكل block ينشئ مثال JSONL Draft بعلمات `needs_review`.
    - يحفظها في `data/instructions/v0/*_draft.jsonl` لتراجعها أنت وتحوّل المناسب منها إلى أمثلة تعليمية نهائية.

## 4. قائمة مختصرة بما تحتاج تحضّره قبل التشغيل

- مكان على Google Drive فيه:
    - `books.json` (موجود).
    - فولدر نصوص الشاملة `.txt` (أو على الأقل subset للكتب المختارة).
- معرفة مسار الـ dump على Drive (مثلاً: `/content/drive/MyDrive/shamela_txt/`).
- في Colab:
    - تعديل سكربت بسيط ليقرأ `books.json`, يختار الكتب حسب `cat_name`, ويقوم بنسخ ملفاتها إلى `data/raw/books_nahw/` و`books_balagha/`.

لو أحببت، في الرسالة القادمة أكتب لك Cell جاهز لـ Colab:

- يقرأ `books.json`.
- يرشّح كتب اللغة والنحو + الأدب/الشعر (مثلاً 20 كتاب).
- يحاول نسخ ملفاتها من فولدر dump عام إلى `data/raw/books_nahw/` و`books_balagha/` تلقائيًا (مع طباعة تقرير بالكتب التي نجح في نسخها والتي لم يجد ملفها).

هل مسار فولدر الشاملة txt عندك معروف (مثلاً: `MyDrive/shamela_txt/`)، أم تحب نكتب الكود بشكل عام وتعدّل المسار بنفسك؟

<div align="center">⁂</div>

[^25_1]: books.json


---

# authors.json

`authors.json` هو جدول المؤلفين الكامل اللي تقدر تربطه مع `books.json` عشان تختار كتب معينة بدقة (مثلاً: كتب السيوطي في البلاغة، ابن هشام في النحو…).[^26_1][^26_2]

## ما يحتويه authors.json

- `total`: عدد المؤلفين (3146).[^26_1]
- `authors`: قائمة مؤلفين، كل عنصر فيه:[^26_1]
    - `id`: رقم المؤلف (يتطابق مع `author_str` أو `authors[].id` في `books.json`).
    - `guid`: معرف عالمي.
    - `short_id`: اختصار.
    - `name`: اسم المؤلف (مثل: ابن حزم، ابن كثير، السيوطي…).
    - `death`: سنة الوفاة (هجري، أو 99999 للمعاصرين).[^26_1]

بربطه مع `books.json` تقدر:

- ترشّح كتب فترة زمنية معينة (مثلاً: قبل 500هـ، أو بين 600–800هـ).
- تختار مؤلفين بعينهم لمشروعك (ابن هشام، ابن مالك، السيوطي، الجاحظ، ابن قتيبة… إذا كانوا موجودين في القائمة).
- تميّز الكتب المعاصرة عن التراثية (باستخدام `death = 99999` للمعاصرين).[^26_1]

لو تحب، أقدر أكتب لك Cell جاهز لـ Colab:

- يقرأ `books.json` + `authors.json`.
- يربط المؤلفين بالكتب.
- يطبع لك مثلًا: كل كتب “كتب اللغة + الأدب” لمؤلفين ما قبل 800هـ، مع أسماء الملفات `.txt` عشان تنقلها لـ `books_nahw` و`books_balagha`.

هل تفضّل نركّز التراث على فترة زمنية معينة (مثلاً حتى 800هـ) ولا تخلط تراث قديم مع مؤلفين معاصرين في نفس المرحلة الأولى؟
<span style="display:none">[^26_3][^26_4][^26_5][^26_6]</span>

<div align="center">⁂</div>

[^26_1]: authors.json

[^26_2]: books.json

[^26_3]: books.json

[^26_4]: guid_index.json

[^26_5]: README.md

[^26_6]: categories.json


---

# عاوز احقق اقصي استفاده ممكنه من كل الكتب

أقصى استفادة = تتعامل مع مكتبة الشاملة اللي عندك على 3 مستويات مترابطة: كوربس قراءة، مصنع تعليمات، ومَرجِع استرجاع (RAG)، مع استغلال `books.json` + `authors.json` + `categories.json` بالكامل.[^27_1][^27_2][^27_3]

## 1) تخطيط استراتيجي للمكتبة (قبل أي كود)

فكّر في كل كتاب كأصل يمكن أن يخدم واحدًا (أو أكثر) من هذه الأدوار:

- خلفية لغوية عامة (language modeling):
كل النص تقريبًا بعد التنظيف → يفيد في pre/continued‑training.
- مادة تعليمية (instruction):
أبواب القواعد، الشروح، الأمثلة، التمارين → تتحول لسؤال/جواب، شرح، تصحيح.
- شواهد واستشهادات:
أبيات، أمثال، حكم، نصوص تراثية → تُستخدم لإثراء الشرح، أو عبر RAG عند الإجابة.

الهدف: ولا كتاب يضيع؛ لو ما صلح لمستوى، غالبًا سينفع في مستوى آخر.

***

## 2) استخدام `books.json` + `categories.json` + `authors.json` للتقسيم الذكي

### تقسيم حسب التصنيف (cat_name / categories.json)

من `books.json` و`categories.json` تقدر تبني مجموعات:[^27_3][^27_1]

- **نحو/لغة/صرف**: كتب اللغة، النحو، ربما بعض كتب “غريب الحديث” والمعاجم.
    - استخدام: قواعد، إعراب، صرف، أمثلة تعليمية، تصحيح لغوي.
- **بلاغة/أدب/شعر**: الأدب، العروض والقوافي، البلاغة.
    - استخدام: تشبيه، استعارة، كناية، معاني، نقد أدبي، شعر.
- **تفسير/حديث/فقه**: علوم القرآن، الحديث، العقيدة، الفقه.
    - استخدام: أمثلة على الأسلوب العربي، RAG للمعاني الشرعية (بحذر).
- **تاريخ/تراجم**: التاريخ، التراجم والطبقات.
    - استخدام: نصوص للتحقيق اللغوي، إعادة الصياغة، فهم الأساليب القديمة.


### تقاطع مع المؤلفين (authors.json)

من `authors.json`:[^27_2]

- تقدر تميّز “عمالقة اللغة” (ابن هشام، السيوطي، الجاحظ، ابن قتيبة، ابن يعيش… إلخ).
- تحدد حقبة زمنية:
    - مثلاً: قبل 500هـ = لغة أقرب للعصر الذهبي،
    - 600–900هـ = نضج البلاغة والنحو،
    - مؤلفون معاصرون (`death = 99999`) = أسلوب حديث مفيد للتعليم.

هذا يسمح لك:

- أن تجعل **التراث العميق** مصدرًا لعلم المعاني والبلاغة العليا.
- وتجعل **الكتب المعاصرة** مصدرًا لشرح مبسط وتمارين تعليمية.

***

## 3) أقصى استفادة على مستوى “كوربس القراءة” (Pre/Continued Training)

هنا الغرض: إعطاء الموديل إحساس شامل بأساليب العربية عبر القرون.

- تنظّف كل الكتب (pipeline التنظيف + dedup) وتبني منها كوربس كبير:
    - تقسمه منطقيًا:
        - `corpus_nahw_lang.txt`
        - `corpus_balagha_adab.txt`
        - `corpus_tafsir_hadith.txt`
        - `corpus_tarajim_tareekh.txt`
- لو عملت **continued pretraining** (أو حتى استغلاله في RAG embeddings)، الموديل سيستفيد من التنوع الأسلوبي والمعجمي.

الفرق المهم: هذه الطبقة لا تحتاج labels؛ فقط نص نظيف ومتنوّع.

***

## 4) أقصى استفادة على مستوى “مصنع تعليمات” (Instruction Factory)

هنا تأتي الفائدة الحقيقية لمشروع “عالم لغة”:

### أ) النحو واللغة

من كتب النحو/اللغة:

- استخرج أبواب القواعد →
    - حول كل باب إلى:
        - مثال شرح مبسط.
        - مجموعة تمارين (صح/خطأ، إعراب، ملء فراغ، اختيار من متعدد).
- استخرج جداول التصريف والموازين →
    - حولها لمهام: “استخرج الجذر، حدّد الوزن، استنتج المصدر…”.


### ب) البلاغة وعلم المعاني

من كتب البلاغة والأدب:

- أمثلة التشبيه والاستعارة والكناية →
    - لكل مثال: سؤال عن النوع + الأثر + الغرض البلاغي.
- أمثلة علم المعاني: خبر/إنشاء، قصر، تقديم وتأخير، إيجاز/إطناب →
    - حوّل الشرح النظري + الأمثلة إلى أسئلة تحليلية.


### ج) الشعر والأدب

من المختارات الشعرية والكتب الأدبية:

- لكل بيت مختار:
    - تمرين إعراب،
    - تمرين تحليل بلاغي،
    - تمرين “اكتب بيتًا مشابهًا في المعنى بأسلوبك”.


### د) التفسير والحديث والتاريخ

من شروح القرآن/الحديث والتاريخ:

- اتخاذ نماذج جمل/فقرات صعبة →
    - تحويلها إلى:
        - “اشرح المعنى بلغة معاصرة”،
        - “حلّل البناء النحوي والبلاغي”،
        - “استخرج المفردات الغريبة وفسّرها”.

بهذا، كل كتاب يصبح مصدرًا لعدة أنواع Examples، وليس مجرد نص يُقرأ.

***

## 5) أقصى استفادة على مستوى “RAG وأدوات مساعدة”

حتى بعد الـ fine‑tuning، الموديل لن يحمل كل النصوص في ذاكرته؛ هنا يأتي دور RAG:

- تبني فهارس (indices) على مجموعات كتب:
    - Index للغة والنحو،
    - Index للبلاغة والشعر،
    - Index للتراث (تفسير، تاريخ…).
- عند سؤال يحتاج استشهاد:
    - Agent بحث يستخرج فقرة من كتبك (اعتمادًا على العنوان/المؤلف/الكلمات المفتاحية)،
    - ثم LLM يشرحها أو يستشهد بها في الإجابة.

هذا يضمن أن:

- تستخدم الـ corpus بالكامل كـ “ظهر علمي” للموديل،
- وتقلّل من اختراع الشواهد/النقول الخاطئة.

***

## 6) كيف لا تضيع في الكمّ الهائل؟

لكي لا تغرق في آلاف الكتب:

1) استخدم `categories.json` + `books.json` + `authors.json` لاختيار **Core Set** صغيرة (مثلاً 30–50 كتاب في المرحلة الأولى).[^27_1][^27_2][^27_3]
2) لكل كتاب في الـ Core Set، حدد دوره الأساسي (نحو/بلاغة/شعر/تحقيق/تفسير…).
3) ضع حدًا عمليًا:
    - من كل كتاب، في المرحلة الأولى، تخرج فقط 20–50 مثال Instruction عالي الجودة.
    - مع الوقت تضيف المزيد لو احتجت.

***

إذا حابب، الخطوة الجاية أقدر أصمم لك “خريطة اختيار Core Set” اعتمادًا على `categories.json` + `books.json`:

- فلتر أوتوماتيكي يقترح مثلاً 10 كتب نحو/لغة + 10 بلاغة/أدب + 10 شعر/مختارات،
- مع جدول يبيّن لكل كتاب: التصنيف، المؤلف، سنة الوفاة، ودوره المقترح في المشروع (نحو/بلاغة/معاني/شعر/تحقيق).

<div align="center">⁂</div>

[^27_1]: books.json

[^27_2]: authors.json

[^27_3]: categories.json


---

# كمل

أكمل لك على السيناريو الأخير: تبدأ بـ 10 كتب نحو وبلاغة فقط، وتخدم بها المرحلة الأولى بأقصى استفادة.

## 1) ما الذي تحتاجه من الـ 10 كتب؟

لكل كتاب نحو/بلاغة في الـ Core Set، حدد له دورًا واضحًا:

- كتب نحو مدرسية/معاصرة:
    - مصدر لقواعد مبسطة + تمارين جاهزة.
    - تنتج منها: شرح قواعد، تمارين إعراب، تصحيح أخطاء.
- كتب نحو تراثية (شرح ابن عقيل، قطر الندى…):
    - مصدر لتعليل القواعد + شواهد شعرية/قرآنية.
    - تنتج منها: أسئلة “لماذا؟” و”ما وجه الإعراب؟” و”ما وجه الخلاف بين النحاة؟”.
- كتب بلاغة مدرسية (البلاغة الواضحة…):
    - مصدر لتعريفات وتشبيهات واستعارات سهلة.
    - تنتج منها: أسئلة تحديد نوع الصورة + الأثر البلاغي.
- كتب بلاغة/معاني تراثية (جواهر البلاغة…):
    - مصدر لعلم المعاني (خبر/إنشاء، قصر، تقديم وتأخير…).
    - تنتج منها: أسئلة تحليل معاني وغرض بلاغي.

بهذا، كل كتاب يُستغل على 3 مستويات: نص خام، تعليمات، شواهد.

## 2) خطة استغلال عملية لكل كتاب (في المرحلة الأولى)

لكل كتاب من الـ 10:

1) تمرير نصه الخام عبر Pipeline التنظيف → يحفظ في `data/clean/books_*`.
2) استخدام Parser (Cell 11 في النوتبوك) لإنتاج Draft JSONL:
    - كل باب/مسألة/فصل يتحول لبلوك نص جاهز للمراجعة.
3) أنت (أو Agent تعليمي) تختار من كل كتاب:
    - 10–20 بلوك مناسب.
    - تحوّل كل بلوك إلى 2–3 أمثلة Instruction نظيفة.

الهدف الواقعي:

- من كل كتاب 20–50 مثال → من 10 كتب = 200–500 مثال يدوي/شبه أوتوماتيكي عالي الجودة، تُدمج مع CIDAR وباقي الداتا في v1.


## 3) ماذا يضيف كل نوع كتاب لدور “عالم اللغة”؟

- نحو معاصر → لغة واضحة وتعليم مباشر للطلاب.
- نحو تراثي → عمق وتحليل واستحضار شواهد.
- بلاغة معاصرة → أمثلة مبسطة لشرح التشبيه والاستعارة والكناية.
- بلاغة/معاني تراثية → قدرة على تحليل الآيات والأبيات من زاوية معاني وقصر وتقديم وتأخير.

اجتماع هذه الأنواع هو الذي يجعل الموديل يفهم العربية “كعلم حي”، لا كمجرد قواعد محفوظة.

لو أحببت، أقدر في الخطوة القادمة أقترح لك بالاسم (من الشاملة) 10 كتب نحو وبلاغة (5 نحو + 5 بلاغة) مناسبة جدًّا لتكون الـ Core Set الأول، مع توصية موجزة لدور كل كتاب (تعليمي/تراثي/معاني/شعر).

---

# نعم

هقترح لك 10 كتب (5 نحو/لغة + 5 بلاغة/أدب) تصلح كـ Core Set للمرحلة الأولى، مع دور واضح لكل كتاب.

## 1) كتب النحو/اللغة (5 كتب)

1) النحو الواضح (الجزء المدرسي المشهور)

- الدور: أساس التعليم المعاصر (شرح بسيط + تمارين).
- الاستخدام:
    - استخراج قواعد الجملة الاسمية/الفعلية، المبتدأ والخبر، كان وأخواتها، إن وأخواتها، الحال، التمييز…
    - تحويل التمارين والأسئلة الموجودة فعلاً إلى Instruction + تصحيح + تعليل.

2) قطر الندى وبلّ الصدى – ابن هشام

- الدور: نحو تراثي متوسط، لغة قوية وجمل عربية أصيلة.
- الاستخدام:
    - أبواب مختارة (المبتدأ والخبر، الفاعل، المفعول…) → تعليل قواعد + شواهد.
    - أسئلة من نوع “اشرح قول ابن هشام” مع تبسيط للطالب.

3) شرح ابن عقيل على ألفية ابن مالك

- الدور: عمق نحوي وتحليل شواهد شعرية وقرآنية.
- الاستخدام:
    - اختيار أبواب محددة فقط (مثلاً: باب المبتدأ والخبر، باب النواسخ).
    - بناء مهام: إعراب بيت + تحليل تعليل ابن عقيل + تبسيطه.

4) شذور الذهب في معرفة كلام العرب – ابن هشام

- الدور: نحو + صرف + لغة، نافع لدور “المحقق اللغوي”.
- الاستخدام:
    - مقاطع تشرح أصول نحوية وصرفية، مع أمثلة تراثية.
    - تحويلها إلى أسئلة “ما الجذر؟ ما الوزن؟ ما وظيفة هذه الصيغة؟”.

5) كتاب معاصر في القواعد (مثلاً: “قواعد اللغة العربية للمرحلة الثانوية” أو أي كتاب مدرسي رسمي)

- الدور: ربط الموديل بالمناهج الحديثة، أسلوب وأسئلة شبيهة بما يراه الطلاب الآن.
- الاستخدام:
    - تمارين الكتاب → نماذج جاهزة لأسئلة امتحانية (اختر الإجابة الصحيحة، صحّح الخطأ، أعرب ما تحته خط…).


## 2) كتب البلاغة/الأدب (5 كتب)

6) البلاغة الواضحة – الجارم وأمين

- الدور: أساس البلاغة التعليمية (بيان + بديع + معاني) بلغة مبسطة.
- الاستخدام:
    - تعريفات بسيطة للتشبيه، الاستعارة، الكناية، المجاز، الخبر والإنشاء، القصر…
    - تحويل الأمثلة والتمارين الموجودة إلى مهام تحليل وشرح.

7) جواهر البلاغة – السيد أحمد الهاشمي

- الدور: عمق بلاغي تقليدي، خاصة في علم المعاني وأسرار التعبير.
- الاستخدام:
    - أمثلة معاني (خبر/إنشاء، قصر، تقديم وتأخير، إيجاز/إطناب) → أسئلة تحليل معاني وغرض بلاغي.
    - أمثلة تشبيه واستعارة وكناية أكثر ثراءً من الكتب المدرسية.

8) مختارات شعراء العرب لابن الشجري (أو أي مختارات شعرية مشابهة في مكتبتك، مثل “التذكرة السعدية”)

- الدور: بنك شواهد شعرية متنوعة.
- الاستخدام:
    - مهام: إعراب بيت، تحليل صورة بلاغية، شرح معنى بيت، استخراج نوع التشبيه/الاستعارة.
    - تمهيد لدور “الشاعر/الناقد”.

9) كتاب في الأدب والوصايا (مثل: وصايا الملوك، الزهرة)

- الدور: جمل نثرية بليغة، حكم وأمثال.
- الاستخدام:
    - مهام:
        - “اشرح هذه الحكمة بلغة مبسطة”،
        - “حلّل الصورة البلاغية في هذه الجملة”،
        - “حوّل النص من أسلوب تراثي إلى أسلوب معاصر مع الحفاظ على المعنى”.

10) كتاب في العروض والقوافي (مثل: القوافي للأخفش أو مختصر في العروض)

- الدور: دعم دور “الشاعر”، فهم البحور والقافية.
- الاستخدام:
    - مهام:
        - تحديد البحر من وزن بيت (بشكل مبسط)،
        - استنتاج أثر القافية والروي،
        - طلب نظم بيت على وزن بسيط/حر.


## 3) كيف تبدأ بهم عمليًا الآن؟

- من الشاملة أو من الـ dump الذي عندك:
    - استخرج النصوص لهذه الكتب الـ 10 (أو ما يقابلها عندك من أسماء قريبة).
    - ضعها في:
        - `data/raw/books_nahw/` للخمسة الأولى.
        - `data/raw/books_balagha/` للخمسة الأخيرة.
- شغّل Notebook المرحلة الأولى:
    - Cell 7 → تنظيف النصوص وكتابة `_clean.txt`.
    - Cell 11 → توليد `_draft.jsonl` لكل كتاب.
- من كل كتاب:
    - اختر يدويًا 20–50 draft جيد (باب واضح، مثال تعليمي، شرح مباشر).
    - عدّلها لتصبح Instruction نظيف (سؤال واضح + جواب نموذج + metadata) وفق السكيمة.

بهذا، خلال المرحلة الأولى، الـ Core Set = 10 كتب، وكل كتاب يعطيك عشرات الأمثلة عالية الجودة تغطي:

- نحو أساسي + تعليل تراثي.
- بلاغة (بيان + بديع + معاني).
- شعر وأدب وشواهد.

لو أحببت بعد كده، أكتب لك Cell في Colab يقرأ `books.json` + `authors.json` ويبحث عن هذه الأسماء (أو أقرب تطابق لها) ويطبع لك جدول: `title + cat_name + file + author + death` حتى تعرف الملفات الدقيقة التي ستستخدمها من الـ dump.

---

# ايه الادوادات اللي هحتاجها

هتحتاج “باقة” أدوات موزعة على 4 طبقات: إدارة ملفات، تنظيف ومعالجة عربية، بناء داتا Instruction، وتجارب تدريب/تقييم أولية.

## 1) بيئة العمل وإدارة الملفات

- Google Colab + Google Drive
    - لتشغيل النوتبوك، تخزين الـ dump، وتنظيم `project_ar_linguist_llm`.
- Python 3 مع مكتبات أساسية:
    - `os`, `glob`, `json`, `re`, `unicodedata`, `tqdm` (progress bars).


## 2) أدوات تحميل وتنظيف الكوربس

- Hugging Face Datasets
    - لتحميل CIDAR + ArabicWeb24 + أي داتا إضافية.
- مكتبات تنظيف عربية:
    - Regex + `langdetect` أو `fasttext` بسيط للغة.
    - سكربتات normalization (زي اللي في النوتبوك: توحيد الألف/الياء، إزالة التشكيل العشوائي، الروابط…).
    - ممكن تستفيد من أفكار `refined-arabic` و`Tahdheeb`، حتى لو ما استخدمت المكتبات حرفيًا.
- deduplication:
    - `datasketch` (MinHash + LSH) أو NeMo Curator لو حبيت شيء أثقل لاحقًا.


## 3) أدوات بناء داتا التعليم (Instruction)

- الـ Notebook اللي عملناه (Phase 1):
    - فيه:
        - دوال `clean_arabic_text`, `dedup_texts`.
        - `Example` dataclass + `save_dataset` + validator.
        - Book parser بسيط يولّد draft JSONL من الكتب.
- أي LLM مساعد (زي Perplexity أو GPT) تستخدمه كـ:
    - Instruction Generation Agent (لصياغة السؤال/الجواب من فقرة).
    - Data Auditor (لتقييم جودة المثال).


## 4) أدوات تقييم/تجريب أولي للموديل

- Hugging Face Transformers + PEFT (لما توصل لـ QLoRA):
    - `transformers`, `peft`, `accelerate` (تضيفها في نوتبوك جديد Phase 2).
- بنشماركات جاهزة (لاحقًا):
    - Arabic LLM Benchmarks (OALL، ArabicWeb‑Edu، CamelEval) لو حابب تجرب أداء الـ base والموديل بعد الـ fine‑tune.

باختصار:

- للمرحلة الأولى (اللي احنا فيها) يكفيك: Colab + Drive + Python + HF Datasets + سكربتات التنضيف والـ JSONL اللي جهزناها.
- لما تنتقل لمرحلة التدريب، تضيف أدوات Transformers/PEFT + كارت GPU محترم أو خدمة سحابة (RunPod/Colab Pro…).

تحب بعد كده أركّب لك Cell إضافي في النوتبوك يقرأ `books.json` + `authors.json` + `categories.json` ويطلع لك CSV منسَّق فيه الـ 10 كتب اللي اخترناهم بالظبط (عنوان، تصنيف، مؤلف، سنة وفاة، اسم ملف txt)؟

---

# camel tool ?

CAMeL Tools بالظبط من الأدوات اللي تستاهل تدخلها في مشروعك؛ هي Toolkit عربية جاهزة للمورفولوجيا وغيرها، وتكمل النواقص اللي عند الـ LLM.[^31_1][^31_2]

## 1) CAMeL Tools بتعمل إيه؟

مجموعة أدوات بايثون لـ Arabic NLP فيها:[^31_2][^31_3]

- preprocessing: تطبيع، تقطيع، tokenization، تنقيح.
- morphological modeling:
    - تحليل صرفي (morphological analyzer): يطلع لك root, lemma, POS, features.
    - توليد صيغ (generator): تبني تصريفات من جذر/lemma + features.
    - reinflection: تحويل صيغة لأخرى (مفرد↔جمع، مذكر↔مؤنث…).[^31_4][^31_5][^31_6]
- dialect identification: تمييز اللهجات (مصري/خليجي/…)/MSA.[^31_3][^31_2]
- NER + sentiment: مش أساسيين لمشروعك الآن لكن ممكن تستخدمهم لاحقًا.


## 2) أين تستخدمها في مشروع “عالم لغة”؟

أ) كأداة **Morphology/معجم** بجانب الموديل:

- تأخذ كلمة من النص (أو من سؤال الطالب) → ترسلها لـ CAMeL analyzer → ترجع:[^31_5][^31_4]
    - الجذر (root).
    - lemma.
    - POS (اسم/فعل/حرف).
    - features (جنس، عدد، زمن…).
- تستخدم النتائج في:
    - بناء `morphology_features` داخل الـ JSONL (سكيمة التدريب).
    - دعم دور “المحقق اللغوي / عالم الصرف” بمعلومات دقيقة خارجية.

ب) لتوليد بيانات صرفية عالية الجودة:

- عبر الـ generator/reinflector:[^31_6][^31_4]
    - توليد قائمة تصريفات لفعل/اسم معين.
    - استخدام هذه التصريفات لبناء تمارين:
        - “حوّل الفعل من الماضي إلى المضارع”،
        - “استخرج جمع هذه الكلمة، صيغة المبالغة، اسم المفعول…”.

ج) لتحسين التنظيف والتحليل:

- tokenizer/tagger من CAMeL Tools يعطيك segmentation عربي أفضل من regex.[^31_2][^31_3]
- يمكنك لاحقًا استخدام POS/dialect ID لتحسين فلترة الداتا (تمييز نصوص لهجة عن فصحى، تمييز نصوص شعرية/قرآنية…).[^31_7][^31_2]


## 3) أدمجها في الـ Pipeline إزاي؟

في Phase 1 اللي عندنا:

- في nNotebook الحالي أنا استخدمت cleaning بسيط بدون CAMeL؛ تقدر تضيف خطوة جديدة (في نوتبوك ثانٍ أو نفس النوتبوك):
    - تثبيت `camel-tools` (`pip install camel-tools`).[^31_8][^31_1]
    - تنزيل الـ data (`camel_data -i light` أو `defaults`).[^31_1]
    - بناء دالة `analyze_word` تستخدم `camel_tools.morphology.analyzer` لإرجاع root/lemma/pos.
    - لما تبني examples للـ JSONL، تضيف حقل إضافي اختياري `morphology_features` (أو تستخدم التحليل داخل الـ prompt وقت الإجابة).

في مرحلة لاحقة، ممكن يكون عندك Agent “Morphology Assistant” داخل النظام، شغله الأساسي أنه يستدعي CAMeL Tools لكل كلمة/جملة تحتاج تحليل صرفي دقيق، ويغذي الموديل بالنتيجة.

لو حابب، أقدر أكتب لك Cell Colab صغير:

- يثبت CAMeL Tools.
- يحمّل analyzer/generator.
- يريك مثال عملي:
    - كلمة → analyses (root, lemma, POS, features).
    - lemma + features → قائمة تصريفات.
<span style="display:none">[^31_10][^31_11][^31_12][^31_13][^31_14][^31_15][^31_9]</span>

<div align="center">⁂</div>

[^31_1]: https://github.com/CAMeL-Lab/camel_tools

[^31_2]: https://aclanthology.org/2020.lrec-1.868.pdf

[^31_3]: https://aclanthology.org/anthology-files/anthology-files/pdf/lrec/2020.lrec-1.868.pdf

[^31_4]: https://camel-tools.readthedocs.io/en/v1.5.5/cli/camel_morphology.html

[^31_5]: https://camel-tools.readthedocs.io/en/master/reference/camel_morphology_features.html

[^31_6]: https://camel-tools.readthedocs.io/en/latest/api/morphology/generator.html

[^31_7]: https://www.academia.edu/68108807/CAMeL_Tools_An_Open_Source_Python_Toolkit_for_Arabic_Natural_Language_Processing

[^31_8]: https://pypi.org/project/camel-tools/0.2.dev0/

[^31_9]: https://www.youtube.com/watch?v=dDT2Wvx0rpc

[^31_10]: https://www.semanticscholar.org/paper/CAMeL-Tools:-An-Open-Source-Python-Toolkit-for-Obeid-Zalmout/995ec006ac98a697ea38bd4eea8c1f3170a8adb4

[^31_11]: https://www.youtube.com/watch?v=RK9Tf5YkxqU

[^31_12]: https://camel.apache.org/blog/2025/10/camel-docling/

[^31_13]: https://arxiv.org/html/2506.18399v1

[^31_14]: https://discuss.huggingface.co/t/arabic-nlp-tutorial/3739

[^31_15]: https://aclanthology.org/2025.emnlp-main.1525.pdf


---

# استخدام camel_morphology لتحليل كتب النحو

أقوى استعمال لـ camel_morphology مع كتب النحو إنك تخليه “محرك صرف” تحت الموديل: يحلل كل كلمة/جملة في الكتب، وتستفيد منه في 3 حاجات: تنظيف وتحليل، بناء داتا تعليمية، وتغذية الـ JSONL بمعلومات صرفية.

## 1) فكرة البايبلاين مع كتب النحو

على مستوى الكتاب الواحد:

1) النص الخام (من الشاملة) → تنظيف (التطبيع + إزالة الضوضاء).
2) تقسيم لجمل/فقرات تعليمية.
3) لكل جملة تعليمية، تشغّل camel_morphology analyzer:
    - ترجع root, lemma, POS, features لكل token.
4) تستخدم النتائج في:
    - بناء تمارين صرف/نحو تلقائية.
    - إضافة حقل morphology في الـ JSONL (يخلي الموديل يربط بين النحو والصرف).

## 2) سيناريو عملي: جملة من “النحو الواضح”

مثال جملة:

> “نجحَ الطالبُ المجتهدُ في الامتحانِ.”

تعمل الآتي:

1) تحلل الجملة بـ camel_morphology:
    - “نجحَ”: فعل ماضٍ، lemma = نجح، root = ن ج ح.
    - “الطالبُ”: اسم، lemma = طالب، root = ط ل ب، مفرد، مذكر.
    - “المجتهدُ”: اسم، lemma = مجتهد، root = ج ه د، صفة.
    - “الامتحانِ”: اسم، lemma = امتحان، root = م ح ن أو امتحن (حسب التحليل).
2) تبني مثال JSONL:
```json
{
  "instruction": "أعرب الجملة الآتية تحليلًا نحويًا وصرفيًا.",
  "input": "نجحَ الطالبُ المجتهدُ في الامتحانِ.",
  "output": "نجحَ: فعل ماضٍ ثلاثي، من الجذر (ن ج ح)، مبني على الفتح.\nالطالبُ: اسم فاعل في المعنى، مفرد مذكر، من الجذر (ط ل ب)، فاعل مرفوع.\nالمجتهدُ: صفة للطالب، على وزن مُفتعِل، من الجذر (ج ه د)...",
  "role": "tutor",
  "skills": ["nahw","morphology"],
  "morphology_features": {
    "tokens": [
      {"form": "نجحَ", "lemma": "نجح", "root": "نجح", "pos": "verb_past"},
      {"form": "الطالبُ", "lemma": "طالب", "root": "طلب", "pos": "noun"},
      {"form": "المجتهدُ", "lemma": "مجتهد", "root": "جهد", "pos": "noun"},
      {"form": "الامتحانِ", "lemma": "امتحان", "root": "امتحن", "pos": "noun"}
    ]
  }
}
```

بهذا الشكل، كل مثال نحوي يصبح أيضًا درس صرفي؛ والموديل يتعلم يربط root/lemma/pos بالإعراب والشرح.

## 3) أنواع مهام تقدر تولّدها من كتب النحو باستخدام camel_morphology

1) تمارين استخراج الجذر والصيغة:
    - استخرج الجذر من الكلمة، حدّد وزنها، صنّف نوعها (اسم فاعل، مفعول، صيغة مبالغة…).
    - توليد أمثلة تلقائيًا: تختار كلمات من الكتب، وتحللها، وتبني سؤال + جواب.
2) تمارين مطابقة الإعراب مع التحليل الصرفي:
    - “هل يمكن أن يكون هذا الاسم فاعلًا؟” → تمرين يربط POS/features مع الوظيفة النحوية.
3) تمارين تصريف:
    - من lemma + features → توليد أشكال مختلفة:
        - فعل ماضٍ، مضارع، أمر، اسم فاعل، اسم مفعول.
    - تستخدم الـ generator/reinflector لإنتاج الإجابات النموذجية.
4) تمارين كشف اللبس:
    - كلمات ذات أكثر من تحليل صرفي (ambiguity) → تسأل الطالب أو الموديل:
        - “أي تحليل هو الأصح في هذا السياق؟ ولماذا؟”
    - camel_morphology يعطي كل التحليلات، وأنت تبني السؤال حولها.

## 4) أين بالضبط تدمجه في Pipeline الكتب؟

في النوتبوك:

- بعد Cell 7 (اللي بينتج `_clean.txt` للكتب)، تضيف مرحلة جديدة:

1) تقرأ `*_clean.txt` سطرًا سطرًا.
2) لكل جملة/فقرة تعليمية (مثلاً فيها 5–20 كلمة):
        - تشغل camel_morphology analyzer عليها.
        - تخزّن النتيجة إما:
            - داخل draft JSONL كمجرد `morphology_features` للمراجعة،
            - أو في ملف منفصل (مثلاً `bookname_morph.jsonl`) تعود له عند بناء الـ instructions.
- عندما تعمل Parser (Cell 11) الذي يحول blocks إلى draft Examples:
    - يمكنك استدعاء دالة تأخذ block وتضيف له ملخص مورفولوجي (مثل قائمة بالأفعال + جذورها، الأسماء + جذورها)، لتسهّل عليك صنع أسئلة صرفية.


## 5) كيف تستفيد “لأقصى حد” من camel_morphology مع كتب النحو؟

- لا تحاول تشكّل كل النصوص أو تربط كل كلمة؛ الهدف هو:
    - التركيز على الجمل/الأمثلة التي ستستخدمها تعليمياً.
- لشريحة مختارة من أمثلة “النحو الواضح” و“قطر الندى”:
    - حلّل الجملة بالكامل؛
    - اختر 2–3 كلمات في كل مثال تُبنى عليها مهام صرفية؛
    - أضف هذه المعلومات في الـ JSONL كـ features → الموديل يتعلم عليها مع الوقت.

لو أردت، أستطيع في خطوة لاحقة أن أكتب لك Cell Colab نموذجي:

- يثبت camel_tools.
- يحلل 3–4 جمل من كتاب نحو (من `books_nahw_clean.txt`).
- ويرجع لك مثال JSONL كامل فيه `instruction` + `input` + `output` + `morphology_features` جاهز للنسخ إلى Dataset.

---

# ايه تاني اللي ممكن اضيفه او احسنه

أهم التحسينات/الإضافات اللي هتفرّق معك في المرحلة الجاية:

## 1) ضبط “شخصية” الموديل (Style + Policy)

- تكتب دليل أسلوب للموديل:
    - لغة فصحى واضحة، مستوى رسمي محترم، لا سخرية، لا تهكّم على الطلاب.
    - طريقة ثابتة للإجابة: (تعريف مختصر → مثال/مثالان → تحذير من الخطأ الشائع).
- تستخدمه في كل أمثلة الـ JSONL (وفي الـ system prompt) عشان الموديل يطلع “عالم مهذّب ومعلّم صبور” مش مجرد مفسِّر قواعد.


## 2) بناء mini‑benchmark من أول يوم

- حتى قبل التدريب:
    - جهّز ملف صغير (مثلاً 100–150 سؤال) مقسوم:
        - 40 نحو، 30 بلاغة، 20 معاني، 10 صرف، 10 لهجات→فصحى.
    - تكتب فيه فقط: السؤال + expected key points في الإجابة.
- كل ما تعمل نسخة جديدة من الموديل (بعد fine‑tune صغير)، تختبر عليها نفس الملف وتشوف الفرق.
هذا يمنعك من الاعتماد على الانطباع.


## 3) تحسين تنوّع الأمثلة

- في v1 حاول تتجنب إن كل الأمثلة تبقى من نوع “اشرح” فقط.
- أضف أنواع مهام أخرى:
    - تصنيف (نوع الجملة، نوع الصورة البلاغية).
    - إعادة صياغة (من تعبير ركيك إلى فصيح).
    - ملء فراغات (أداة نصب/رفع/جر، حرف عطف مناسب…).
- هذا يساعد الموديل يكون مرن في classroom حقيقي، مش فقط شارح نظري.


## 4) Layer صغيرة للـ “Error Mining”

- من كتاب نحوي أو من بيانات طلاب (أو حتى جمل تولّدها أنت أو LLM)،
    - اجمع نماذج أخطاء شائعة: نصب المبتدأ، خلط بين إن وأن، همزات القطع والوصل، الخلط بين التاء المربوطة والمفتوحة…
- اعمل dataset صغير مخصص لـ GEC عربي تعليمي، بحيث الموديل يكون قوي جدًا في “تصحيح الطالب وتعليمه” لا في الشرح فقط.


## 5) Agent بسيط لإدارة الكتب

- بدل ما تتعامل يدويًا مع `books.json` ومجلدات النصوص، تبني سكربت/Agent صغير:
    - يقرأ `books.json` + `authors.json` + `categories.json`.
    - يسمح لك بفلترة:
        - حسب التصنيف (اللغة/الأدب)،
        - حسب المؤلف،
        - حسب الحقبة الزمنية.
    - يطبع لك Report بالكتب اللي دخلت المرحلة الحالية، واللي لسه pending.
ده يخلي إدارة المكتبة نفسها منظمة على المدى الطويل.


## 6) ضبط التوازن بين الفصحى واللهجة من البداية

- تضيف قليل من أمثلة “لهجة مصرية → فصحى تعليمية” في v1، لكن بوعي:
    - 1–2٪ فقط من الداتا،
    - role واضح (`translator` أو `tutor_dialect`).
- هذا يعطي الموديل إحساسًا باللهجة بدون ما يفسد نقاء الفصحى.


## 7) Workflow واضح للـ versions

- من دلوقتي استخدم versioning بسيط:
    - `dataset_v1`, `dataset_v2`…
    - `model_v0_base`, `model_v1_ft_nahw`, `model_v2_ft_nahw_balagha`…
- مع ملف log (أنت بدأت فعلاً) تكتب فيه:
    - إيش اللي تغيّر في كل نسخة،
    - على أي داتا تدربت،
    - ونتيجة mini‑benchmark.

لو اخترنا نقطة واحدة تحس أنها أضعف شيء عندك الآن: جودة أمثلة الـ instruction نفسها، ولا التنوّع في أنواع المهام، ولا إدارة المكتبة (books/authors/categories)؟

---

# full implementation plan in details

الـ Notebook الكامل جاهز للتنزيل.

## خريطة الـ 4 مراحل

الـ Notebook يحتوي 20 Cell موزعة على 4 مراحل متكاملة:

### Phase 0 – Setup (Cell 0–1)

- تثبيت كل الأدوات (camel-tools, qdrant, peft, transformers…)
- Mount Drive + بناء هيكل المشروع الكامل تلقائيًا


### Phase 1 – Data Foundation (Cell 2–7)

- **Cell 2**: تحميل `books.db` من الشاملة (8,425 كتاب / 16.4 GB)[^35_1]
- **Cell 3**: فلترة الكتب حسب التصنيف من `categories.json` (41 تصنيفًا، منها: النحو والصرف id=31، البلاغة id=35، الأدب id=32…)[^35_2]
- **Cell 4**: Smart Book Picker (مؤلفون ذوو أولوية + حد حجم لكل كتاب)
- **Cell 5**: نسخ الكتب المختارة إلى فولدرات المشروع + تقرير بالناجح والمفقود
- **Cell 6–7**: Pipeline تنظيف كامل → 7 corpora نظيفة بـ dedup


### Phase 2 – Instruction Factory (Cell 8–15)

- **Cell 8**: CAMeL Tools: analyze_token + analyze_sentence + generate_forms
- **Cell 9–10**: Schema كامل + 5 generators (morphology drill, irab, balagha, GEC, fill-blank)
- **Cell 11**: Book → Draft JSONL لكل corpus
- **Cell 12**: LLM Auditor Agent (GPT-4o-mini يفلتر بـ min_avg=3.5)
- **Cell 13**: GEC Dataset (5 أمثلة seed + قابل للتوسع)
- **Cell 14**: Mini-Benchmark (150 سؤال: نحو/صرف/بلاغة/معاني/GEC/شعر)
- **Cell 15**: Merge \& Balance → v1 (10K → 50K, نسب: نحو 35٪ / بلاغة 20٪ / معاني 10٪...)


### Phase 3 – Fine-Tuning (Cell 16–17)

- **Cell 16**: QLoRA على Qwen2.5-7B-Instruct (r=16, α=32, 4-bit)
- **Cell 17**: Evaluator تلقائي على Mini-Benchmark → نتائج per-skill


### Phase 4 – RAG + Deploy (Cell 18–20)

- **Cell 18**: Qdrant Index (3 collections: nahw / balagha / adab)
- **Cell 19**: FastAPI API كاملة (RAG + LLM في endpoint واحد `/ask`)
- **Cell 20**: Version Tracker يسجل كل نسخة تدريب مع نتائجها

<div align="center">⁂</div>

[^35_1]: README.md

[^35_2]: categories.json

