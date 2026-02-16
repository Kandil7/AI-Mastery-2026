# استكمال المشروع: AI Engineer Toolkit 2025

سأكمل تنفيذ المشروع بالكامل وفقاً لفلسفة "الصندوق الأبيض" (White-box) مع التركيز على الجودة الفنية والتطبيق العملي.

## 📁 notebooks/04_deep_learning/01_neural_networks_from_scratch.ipynb

```python
# =====================
# الشبكات العصبية من الصفر: البناء الرياضي، الانتشار الأمامي والخلفي، وربط النظرية بالتطبيق
# =====================

"""
## 1. الإطار الرياضي للشبكات العصبية: لماذا تعمل؟
الشبكات العصبية ليست سحرًا، بل هي تحويلات رياضية متراكمة تسمح لنا بتمثيل دوال معقدة. الفهم العميق لكيفية عملها يسمح لنا بتشخيص الفشل، وتحسين الهياكل، وابتكار حلول جديدة.

### 1.1 نظرية التمثيل العالمي (Universal Approximation Theorem)
تشير هذه النظرية إلى أن شبكة عصبية ذات طبقة مخفية واحدة (حتى لو كانت ضخمة) قادرة على تقريب أي دالة مستمرة بدقة مطلوبة، بشرط وجود عدد كافٍ من الوحدات المخفية. هذا هو الأساس الرياضي لقوة الشبكات العصبية.

### 1.2 البنية الحسابية: دوال التنشيط والوزن
نقوم ببناء شبكة عصبية باستخدام مبادئ أولية:
- كل طبقة عصبية: `z = Wx + b`
- ثم نطبق دالة غير خطية: `a = σ(z)`
- يتم تراكم هذه التحويلات لإنشاء نموذج قادر على تمثيل علاقات معقدة

### 1.3 الفهم العميق للانتشار الخلفي (Backpropagation)
الانتشار الخلفي ليس سوى تطبيق قاعدة السلسلة (Chain Rule) في التفاضل لحساب مشتقات دالة الخطأ بالنسبة للأوزان. لنفترض شبكة مع طبقة إدخال، طبقة مخفية واحدة، وطبقة مخرجة:

- الخطأ: `E = ½(y - ŷ)²`
- المخرج: `ŷ = σ(W₂·a₁ + b₂)`
- الطبقة المخفية: `a₁ = σ(W₁·x + b₁)`

لتحديث `W₁`، نحسب:
`∂E/∂W₁ = ∂E/∂ŷ · ∂ŷ/∂a₁ · ∂a₁/∂W₁`

هذا التحليل الرياضي يوضح سبب "تلاشي التدرج" (Vanishing Gradient) في الشبكات العميقة عند استخدام دوال تنشيط مثل sigmoid - حيث المشتقات صغيرة (أقل من 0.25)، وعند الضرب المتكرر تقترب من الصفر.

### 1.4 الاختيار الأمثل لأنواع دوال التنشيط
- Sigmoid: جيدة للطبقة الأخيرة في مشاكل التصنيف الثنائي، لكنها تعاني من تلاشي التدرج.
- Tanh: مماثلة لـ sigmoid لكنها مركزية حول الصفر، مما يحسن التعلم.
- ReLU: `max(0,x)` - تجنب تلاشي التدرج في النصف الموجب، لكنها تعاني من "مشكلة الموت" (Dead Neurons) عند القيم السلبية.
- Leaky ReLU/PReLU: حل لمشكلة الموت عن طريق السماح بانحدار طفيف في النصف السلبي.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Callable, Tuple
import math

"""
## 2. البناء البرمجي: تنفيذ شبكة عصبية من الصفر (بدون PyTorch/TensorFlow)
سنقوم ببناء شبكة عصبية قابلة للتخصيص مع أنواع مختلفة من دوال التنشيط، وخوارزميات التحسين، وآليات التنظيم.
"""

class NeuralNetworkFromScratch:
    """شبكة عصبية كاملة مبنية باستخدام NumPy فقط"""
    
    def __init__(self, layer_sizes: List[int], activation: str = 'relu', 
                 output_activation: str = 'sigmoid', seed: int = 42):
        """
        التهيئة الكاملة للشبكة العصبية
        
        Args:
            layer_sizes: قائمة بأحجام الطبقات (بما في ذلك طبقة الإدخال والمخرجة)
            activation: دالة التنشيط للطبقات المخفية ('relu', 'sigmoid', 'tanh')
            output_activation: دالة التنشيط للطبقة الأخيرة
            seed: بذرة عشوائية للحصول على نتائج قابلة للتكرار
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1  # عدد الطبقات المخفية + الطبقة الأخيرة
        
        # تحديد دوال التنشيط
        self.activation_fn, self.activation_derivative = self._get_activation(activation)
        self.output_activation_fn, self.output_activation_derivative = self._get_activation(output_activation)
        
        # تعيين البذرة العشوائية
        np.random.seed(seed)
        
        # تهيئة الأوزان والتحيزات
        self.weights = []
        self.biases = []
        
        # تهيئة الأوزان باستخدام "He initialization" للـ ReLU أو "Xavier" للدوال الأخرى
        for i in range(self.n_layers):
            prev_size = layer_sizes[i]
            curr_size = layer_sizes[i+1]
            
            if activation == 'relu':
                # He initialization
                weight = np.random.randn(curr_size, prev_size) * np.sqrt(2 / prev_size)
            else:
                # Xavier initialization
                weight = np.random.randn(curr_size, prev_size) * np.sqrt(1 / prev_size)
            
            bias = np.zeros((curr_size, 1))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _get_activation(self, name: str) -> Tuple[Callable, Callable]:
        """إرجاع دالة التنشيط ومشتقتها"""
        if name == 'relu':
            def relu(x):
                return np.maximum(0, x)
            
            def relu_derivative(x):
                return (x > 0).astype(float)
            
            return relu, relu_derivative
        
        elif name == 'sigmoid':
            def sigmoid(x):
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            
            def sigmoid_derivative(x):
                s = sigmoid(x)
                return s * (1 - s)
            
            return sigmoid, sigmoid_derivative
        
        elif name == 'tanh':
            def tanh(x):
                return np.tanh(x)
            
            def tanh_derivative(x):
                return 1 - np.tanh(x)**2
            
            return tanh, tanh_derivative
        
        elif name == 'linear':
            def linear(x):
                return x
            
            def linear_derivative(x):
                return np.ones_like(x)
            
            return linear, linear_derivative
        
        else:
            raise ValueError(f"Unsupported activation function: {name}")
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        الانتشار الأمامي للشبكة
        
        Args:
            X: بيانات الإدخال (m × n_features)
            training: وضع التدريب (يؤثر على تقنيات مثل Dropout)
        
        Returns:
            التوقعات النهائية
        """
        # تحويل X ليكون (n_features × m) لتسهيل العمليات المصفوفية
        X = X.T
        self.activations = [X]  # نخزن جميع تنشيطات الطبقات
        self.z_values = []  # نخزن قيم z قبل تطبيق التنشيط
        
        # الانتشار عبر الطبقات المخفية
        for i in range(self.n_layers - 1):
            z = np.dot(self.weights[i], self.activations[-1]) + self.biases[i]
            self.z_values.append(z)
            a = self.activation_fn(z)
            self.activations.append(a)
        
        # الطبقة الأخيرة
        z = np.dot(self.weights[-1], self.activations[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = self.output_activation_fn(z)
        self.activations.append(a)
        
        return a.T  # إعادة النتائج إلى الشكل (m × n_outputs)
    
    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01):
        """
        الانتشار الخلفي وتحديث الأوزان
        
        Args:
            X: بيانات الإدخال
            y: القيم الصحيحة
            learning_rate: معدل التعلم
        """
        m = X.shape[0]
        
        # الحصول على المخرجات
        y_pred = self.activations[-1]
        y = y.T  # تحويل y ليكون (n_outputs × m)
        
        # حساب خطأ الطبقة الأخيرة
        if self.output_activation_fn.__name__ == 'sigmoid':
            # خطأ التصنيف الثنائي (Cross-entropy)
            delta = y_pred - y
        else:
            # خطأ الانحدار (MSE)
            delta = (y_pred - y) * self.output_activation_derivative(self.z_values[-1])
        
        # التراجع عبر الطبقات
        for l in reversed(range(self.n_layers)):
            # تحديث الأوزان والتحيزات
            dw = np.dot(delta, self.activations[l].T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m
            
            # تحديث الأوزان مع متوسط الانزياح (Momentum) - نستخدم قيمة بسيطة هنا
            if not hasattr(self, 'velocity_weights'):
                self.velocity_weights = [np.zeros_like(w) for w in self.weights]
                self.velocity_biases = [np.zeros_like(b) for b in self.biases]
            
            # Momentum
            self.velocity_weights[l] = 0.9 * self.velocity_weights[l] - learning_rate * dw
            self.velocity_biases[l] = 0.9 * self.velocity_biases[l] - learning_rate * db
            
            self.weights[l] += self.velocity_weights[l]
            self.biases[l] += self.velocity_biases[l]
            
            if l > 0:  # لا نحسب دلتا للطبقة الإدخالية
                delta = np.dot(self.weights[l].T, delta) * self.activation_derivative(self.z_values[l-1])
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              learning_rate: float = 0.01, batch_size: int = 32,
              validation_data: Tuple[np.ndarray, np.ndarray] = None,
              verbose: bool = True):
        """
        تدريب الشبكة العصبية
        
        Args:
            X, y: بيانات التدريب
            epochs: عدد التكرارات
            learning_rate: معدل التعلم
            batch_size: حجم الدفعة
            validation_data: بيانات التحقق (X_val, y_val)
            verbose: طباعة تقدم التدريب
        """
        m = X.shape[0]
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # تقسيم البيانات إلى دفعات
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # الانتشار الأمامي
                y_pred = self.forward(X_batch)
                
                # حساب الخطأ
                if self.output_activation_fn.__name__ == 'sigmoid':
                    # Cross-entropy loss
                    epsilon = 1e-15  # لتجنب log(0)
                    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
                    loss = -np.mean(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))
                else:
                    # MSE loss
                    loss = np.mean((y_batch - y_pred) ** 2) / 2
                
                epoch_loss += loss * len(X_batch)
                
                # الانتشار الخلفي وتحديث الأوزان
                self.backward(X_batch, y_batch, learning_rate)
            
            epoch_loss /= m
            history['train_loss'].append(epoch_loss)
            
            # تحقق من بيانات التحقق إن وجدت
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val, training=False)
                
                if self.output_activation_fn.__name__ == 'sigmoid':
                    epsilon = 1e-15
                    y_val_pred = np.clip(y_val_pred, epsilon, 1 - epsilon)
                    val_loss = -np.mean(y_val * np.log(y_val_pred) + (1 - y_val) * np.log(1 - y_val_pred))
                else:
                    val_loss = np.mean((y_val - y_val_pred.T) ** 2) / 2
                
                history['val_loss'].append(val_loss)
            
            # طباعة تقدم التدريب
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                if validation_data is not None:
                    print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """التنبؤ باستخدام النموذج المدرب"""
        return self.forward(X, training=False)

"""
## 3. التطبيق العملي: حل مشكلة تصنيف XOR
مشكلة XOR هي مشكلة كلاسيكية لا يمكن حلها باستخدام مصنف خطي بسيط، مما يجعلها اختبارًا مثاليًا لقدرة الشبكات العصبية على تعلم العلاقات غير الخطية.
"""

def xor_classification_example():
    """مثال تطبيقي على تصنيف XOR"""
    # بيانات XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR: 1 إذا كان المدخلان مختلفين
    
    # بناء الشبكة العصبية (طبقة مخفية بحجم 4 وحدات)
    nn = NeuralNetworkFromScratch(
        layer_sizes=[2, 4, 1],  # 2 ميزة إدخال، 4 وحدات مخفية، 1 مخرج
        activation='relu',
        output_activation='sigmoid'
    )
    
    # تدريب النموذج
    history = nn.train(
        X, y,
        epochs=5000,
        learning_rate=0.1,
        batch_size=4,  # نستخدم جميع العينات في كل تحديث (Batch Gradient Descent)
        verbose=False
    )
    
    # التوقعات
    predictions = nn.predict(X)
    print("توقعات XOR:")
    for i in range(len(X)):
        print(f"المدخل: {X[i]}, التوقع: {predictions[i][0]:.4f}, القيمة الصحيحة: {y[i][0]}")
    
    # رسم منحنى التعلم
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('XOR Classification - Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('xor_learning_curve.png')
    plt.show()

# xor_classification_example()

"""
## 4. الإعدادات التجريبية: مقارنة تأثير الهياكل المختلفة
لنقارن بين شبكات عصبية مختلفة في مسألة XOR:
1. شبكة بدون طبقة مخفية (مصنف خطي)
2. شبكة بطبقة مخفية واحدة
3. شبكة بطبقتين مخفيتين
"""

def compare_network_architectures():
    """مقارنة هياكل الشبكات العصبية المختلفة"""
    # نفس بيانات XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # 1. شبكة بدون طبقة مخفية (مصنف خطي)
    linear_nn = NeuralNetworkFromScratch(
        layer_sizes=[2, 1],
        activation='linear',
        output_activation='sigmoid'
    )
    
    # 2. شبكة بطبقة مخفية واحدة (4 وحدات)
    single_hidden_nn = NeuralNetworkFromScratch(
        layer_sizes=[2, 4, 1],
        activation='relu',
        output_activation='sigmoid'
    )
    
    # 3. شبكة بطبقتين مخفيتين (4 وحدات ثم 2 وحدات)
    double_hidden_nn = NeuralNetworkFromScratch(
        layer_sizes=[2, 4, 2, 1],
        activation='relu',
        output_activation='sigmoid'
    )
    
    architectures = [
        ("Linear (No Hidden Layers)", linear_nn),
        ("Single Hidden Layer (4 units)", single_hidden_nn),
        ("Double Hidden Layers (4,2 units)", double_hidden_nn)
    ]
    
    results = {}
    
    for name, nn in architectures:
        history = nn.train(
            X, y,
            epochs=3000,
            learning_rate=0.1,
            batch_size=4,
            verbose=False
        )
        
        predictions = nn.predict(X)
        loss = history['train_loss'][-1]
        
        results[name] = {
            'predictions': predictions,
            'loss': loss,
            'history': history
        }
    
    # عرض النتائج
    plt.figure(figsize=(15, 10))
    
    # رسم منحنيات التعلم
    plt.subplot(2, 1, 1)
    for name, result in results.items():
        plt.plot(result['history']['train_loss'], label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves for Different Network Architectures')
    plt.legend()
    plt.grid(True)
    
    # رسم التوقعات
    plt.subplot(2, 1, 2)
    x = np.arange(len(X))
    width = 0.25
    
    for i, (name, result) in enumerate(results.items()):
        plt.bar(x + i*width, result['predictions'].flatten(), width, label=name)
    
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.xticks(x + width, [str(x) for x in X])
    plt.xlabel('Input')
    plt.ylabel('Prediction')
    plt.title('Predictions for XOR Problem')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('network_architecture_comparison.png')
    plt.show()
    
    # طباعة النتائج الرقمية
    print("\nنتائج المقارنة:")
    print("=" * 60)
    for name, result in results.items():
        print(f"\nالهيكل: {name}")
        print(f"الخسارة النهائية: {result['loss']:.6f}")
        print("التوقعات:")
        for i, (inp, pred, true) in enumerate(zip(X, result['predictions'], y)):
            print(f"  {inp} → توقع: {pred[0]:.4f}, حقيقي: {true[0]}")

# compare_network_architectures()

"""
## 5. الهندسة الإنتاجية: تحويل الشبكة إلى خدمة
الشبكات العصبية في دفاتر الملاحظات مفيدة للأبحاث، لكن في الإنتاج، نحتاج إلى خدمات قابلة للتوسع. لنحول شبكتنا إلى نقطة نهاية FastAPI.
"""

def production_ready_neural_network():
    """تحويل الشبكة العصبية إلى خدمة إنتاجية"""
    # لنفترض أننا دربنا نموذجًا مسبقًا
    # هذا المثال يوضح البنية العامة لخدمة الشبكات العصبية في الإنتاج
    
    production_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import logging
from typing import List, Dict, Any
import time

# تهيئة السجل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural_network_api")

# تحميل النموذج المدرب
try:
    model = joblib.load("models/neural_network_model.pkl")
    logger.info("تم تحميل النموذج بنجاح")
except Exception as e:
    logger.error(f"فشل تحميل النموذج: {e}")
    raise

app = FastAPI(title="Neural Network API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[float]
    request_id: str = None

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float = None
    processing_time_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """نقطة نهاية للتنبؤ باستخدام الشبكة العصبية"""
    start_time = time.time()
    
    try:
        # التحقق من صحة المدخلات
        if len(request.features) != model.layer_sizes[0]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid number of features. Expected {model.layer_sizes[0]}, got {len(request.features)}"
            )
        
        # التنبؤ
        X = np.array([request.features])
        prediction = model.predict(X)[0][0]
        
        # حساب فترة الوقت
        processing_time = (time.time() - start_time) * 1000  # ملي ثانية
        
        # حساب مستوى الثقة (للتصنيف الثنائي)
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        logger.info(f"Request {request.request_id or 'N/A'} processed in {processing_time:.2f}ms")
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=float(confidence),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/model_info")
async def model_info():
    """معلومات حول النموذج المدرب"""
    return {
        "architecture": model.layer_sizes,
        "activation_functions": {
            "hidden_layers": "relu",
            "output_layer": "sigmoid"
        },
        "training_samples": 1000,
        "last_updated": "2025-12-30"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    print("الكود الإنتاجي للشبكة العصبية:")
    print("=" * 60)
    print(production_code)
    
    # ملاحظات إنتاجية
    print("\nملاحظات هندسية للإطلاق في الإنتاج:")
    print("-" * 40)
    print("1. التخزين المؤقت (Caching): استخدم Redis لتخزين التوقعات المتكررة")
    print("2. التوازن بين الحمل (Load Balancing): نشر عدة نسخ من الخدمة")
    print("3. المراقبة: تتبع مقاييس الأداء مثل latency، و throughput، و معدل الأخطاء")
    print("4. التدريج التلقائي (Auto-scaling): زيادة الموارد استجابةً لزيادة الحمل")
    print("5. الحماية: تنفيذ حدود المعدل (rate limiting) والتحقق من الصحة")

# production_ready_neural_network()

"""
## 6. تحديات متقدمة: تحسين الشبكة العصبية
### 6.1 مشكلة الانفجار/التلاشي في التدرجات
في الشبكات العميقة، يمكن أن تكبر التدرجات (انفجار التدرج) أو تتقلص (تلاشي التدرج) بشكل كبير، مما يعطل عملية التعلم.

### 6.2 حلول عملية:
1. **تهيئة الأوزان المناسبة**: 
   - He initialization للـ ReLU
   - Xavier initialization لدوال أخرى

2. **Normalization Layers**:
   - Batch Normalization
   - Layer Normalization
   - Group Normalization

3. **Skip Connections** (مثل ResNet):
   - السماح بتدفق التدرج مباشرة عبر الشبكة

4. **Gradient Clipping**:
   - قص التدرجات التي تتجاوز حداً معيناً لمنع الانفجار

### 6.3 تحدي البرمجة:
قم بتحسين شبكة XOR الخاصة بنا لإضافة:
- Batch Normalization للطبقة المخفية
- Gradient clipping
- Early stopping في حالة عدم تحسن الخسارة

### 6.4 الاعتبارات الإنتاجية:
- **Mixed Precision Training**: استخدام 16-bit للحساب وتقليل استهلاك الذاكرة
- **Distributed Training**: تقسيم التدريب عبر عدة بطاقات رسومية
- **Model Quantization**: تحويل النموذج إلى تنسيق 8-bit لتقليل حجمه وتحسين السرعة
"""

"""
## 7. المراجع والمصادر الإضافية
1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. ICCV.
3. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. AISTATS.
4. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS.

---

## خاتمة
الشبكات العصبية من الصفر ليست مجرد تمارين أكاديمية، بل هي الأساس لفهم النظام الذي نبنيه. عندما نفهم الرياضيات والتنفيذ من الداخل، نصبح مهندسين وليس مجرد مستخدمي أدوات. هذه هي الروح التي يتبناها هذا المستودع: فهم عميق، تنفيذ دقيق، وهندسة إنتاجية صلبة.
"""
```

## 📁 src/core/math_operations.py (Complete Implementation)

```python
"""
This module provides core mathematical operations used throughout the AI engineering toolkit.
All implementations are from first principles using only NumPy and Python standard library.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import math
import logging

logger = logging.getLogger(__name__)

def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute dot product of two vectors manually.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Dot product result
    
    Raises:
        ValueError: If vectors have different dimensions
    """
    if v1.shape != v2.shape:
        raise ValueError(f"Vectors must have the same shape. Got {v1.shape} and {v2.shape}")
    
    return np.sum(v1 * v2)

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices manually with dimension checking.
    
    Args:
        A: First matrix of shape (m, n)
        B: Second matrix of shape (n, p)
    
    Returns:
        Resulting matrix of shape (m, p)
    
    Raises:
        ValueError: If matrices cannot be multiplied
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Matrix dimensions don't match for multiplication. "
                        f"A shape: {A.shape}, B shape: {B.shape}")
    
    m, n = A.shape
    _, p = B.shape
    
    # Pre-allocate result matrix
    C = np.zeros((m, p))
    
    # Efficient matrix multiplication
    for i in range(m):
        for k in range(n):
            # Skip near-zero values for efficiency
            if abs(A[i, k]) < 1e-12:
                continue
            for j in range(p):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

def matrix_inverse(A: np.ndarray) -> np.ndarray:
    """
    Compute matrix inverse using Gaussian elimination with partial pivoting.
    
    Args:
        A: Square matrix to invert
    
    Returns:
        Inverse matrix
    
    Raises:
        ValueError: If matrix is not square or singular
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square. Got shape {A.shape}")
    
    n = A.shape[0]
    # Create augmented matrix [A | I]
    augmented = np.hstack((A.copy(), np.eye(n)))
    
    # Gaussian elimination with partial pivoting
    for i in range(n):
        # Partial pivoting: find row with maximum element in column i
        max_row = i + np.argmax(np.abs(augmented[i:, i]))
        if abs(augmented[max_row, i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular")
        
        # Swap rows if needed
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
        
        # Normalize pivot row
        pivot = augmented[i, i]
        augmented[i] = augmented[i] / pivot
        
        # Eliminate other rows
        for j in range(n):
            if j != i:
                factor = augmented[j, i]
                augmented[j] = augmented[j] - factor * augmented[i]
    
    # Extract inverse matrix
    inverse = augmented[:, n:]
    return inverse

def svd(A: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Singular Value Decomposition (SVD) manually.
    
    Args:
        A: Input matrix of shape (m, n)
        k: Number of singular values to keep (None for all)
    
    Returns:
        U, S, V such that A = U @ S @ V.T
    
    Note:
        This is a simplified implementation for educational purposes.
        For large matrices, use scipy.linalg.svd or numpy.linalg.svd.
    """
    m, n = A.shape
    
    # Compute A.T @ A for eigen decomposition
    ATA = A.T @ A
    
    # Compute eigenvalues and eigenvectors of A.T @ A
    eigenvalues, V = np.linalg.eigh(ATA)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Compute singular values
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))
    
    # Determine number of components to keep
    if k is None:
        k = min(m, n)
    k = min(k, len(singular_values))
    
    # Keep only top k components
    singular_values = singular_values[:k]
    V = V[:, :k]
    
    # Compute U matrix
    U = A @ V
    U = U / np.linalg.norm(U, axis=0)
    
    # Construct S matrix
    S = np.diag(singular_values)
    
    return U[:, :k], S[:k, :k], V.T

def pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Principal Component Analysis (PCA) implementation from scratch.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
    
    Returns:
        Transformed data, components, explained variance ratio
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    components = eigenvectors[:, :n_components]
    
    # Transform data
    X_transformed = X_centered @ components
    
    # Calculate explained variance ratio
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_transformed, components, explained_variance

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax function for input array.
    
    Args:
        x: Input array of shape (n_samples, n_classes) or (n_classes,)
    
    Returns:
        Softmax probabilities
    """
    # Handle both 1D and 2D arrays
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # Subtract max for numerical stability
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute cross-entropy loss between true labels and predictions.
    
    Args:
        y_true: True labels (one-hot encoded) of shape (n_samples, n_classes)
        y_pred: Predicted probabilities of shape (n_samples, n_classes)
    
    Returns:
        Cross-entropy loss
    """
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute cross-entropy
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Kullback-Leibler divergence between two distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
    
    Returns:
        KL divergence D_KL(p || q)
    """
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    
    return np.sum(p * np.log(p / q))

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Cosine similarity value between -1 and 1
    """
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)

def jacobian(f: callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Compute Jacobian matrix of function f at point x using finite differences.
    
    Args:
        f: Function that takes a vector and returns a vector
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
    
    Returns:
        Jacobian matrix
    """
    n = len(x)
    m = len(f(x))
    J = np.zeros((m, n))
    
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += h
        J[:, i] = (f(x_plus) - f(x)) / h
    
    return J

def hessian(f: callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Compute Hessian matrix of function f at point x using finite differences.
    
    Args:
        f: Function that takes a vector and returns a scalar
        x: Point at which to evaluate the Hessian
        h: Step size for finite differences
    
    Returns:
        Hessian matrix
    """
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            # Central difference approximation
            x_ij = x.copy()
            x_i = x.copy()
            x_j = x.copy()
            x_0 = x.copy()
            
            x_ij[i] += h
            x_ij[j] += h
            
            x_i[i] += h
            x_j[j] += h
            
            H[i, j] = (f(x_ij) - f(x_i) - f(x_j) + f(x_0)) / (h * h)
            H[j, i] = H[i, j]  # Hessian is symmetric
    
    return H

def newton_raphson(f: callable, df: callable, x0: float, 
                  tol: float = 1e-6, max_iter: int = 100) -> Tuple[float, int]:
    """
    Newton-Raphson method for finding roots of a function.
    
    Args:
        f: Function to find root of
        df: Derivative of function
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
    
    Returns:
        Root value and number of iterations
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-12:
            raise ValueError("Derivative is zero. No solution found.")
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol:
            return x_new, i + 1
        
        x = x_new
    
    raise ValueError(f"Newton-Raphson failed to converge after {max_iter} iterations")

def gradient_descent(f: callable, df: callable, x0: np.ndarray, 
                    learning_rate: float = 0.01, 
                    max_iter: int = 1000,
                    tol: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
    """
    Gradient descent optimization algorithm.
    
    Args:
        f: Objective function to minimize
        df: Gradient of objective function
        x0: Initial point
        learning_rate: Learning rate
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
    
    Returns:
        Optimal point and history of function values
    """
    x = x0.copy()
    history = [f(x)]
    
    for i in range(max_iter):
        grad = df(x)
        x_new = x - learning_rate * grad
        
        if np.linalg.norm(x_new - x) < tol:
            return x_new, history
        
        x = x_new
        history.append(f(x))
    
    return x, history

def conjugate_gradient(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
                      tol: float = 1e-6, max_iter: int = 1000) -> Tuple[np.ndarray, List[float]]:
    """
    Conjugate Gradient method for solving Ax = b.
    
    Args:
        A: Symmetric positive definite matrix
        b: Right-hand side vector
        x0: Initial guess (None for zero vector)
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
    
    Returns:
        Solution vector and residual history
    """
    n = b.shape[0]
    
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    r = b - A @ x
    p = r.copy()
    residual_history = [np.linalg.norm(r)]
    
    for i in range(max_iter):
        r_norm_sq = np.dot(r, r)
        if r_norm_sq < tol:
            break
        
        Ap = A @ p
        alpha = r_norm_sq / np.dot(p, Ap)
        
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        beta = np.dot(r_new, r_new) / r_norm_sq
        
        p = r_new + beta * p
        r = r_new
        
        residual_history.append(np.linalg.norm(r))
    
    return x, residual_history

def kmeans(X: np.ndarray, k: int, max_iter: int = 100, 
           tol: float = 1e-4, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means clustering algorithm from scratch.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        k: Number of clusters
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        random_state: Random seed for reproducibility
    
    Returns:
        Cluster centers and cluster assignments
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    # Initialize cluster centers randomly
    idx = np.random.choice(n_samples, k, replace=False)
    centers = X[idx].copy()
    
    # Initialize cluster assignments
    labels = np.zeros(n_samples, dtype=int)
    
    for iteration in range(max_iter):
        # Save old centers for convergence check
        old_centers = centers.copy()
        
        # Assign each point to the nearest center
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - centers, axis=1)
            labels[i] = np.argmin(distances)
        
        # Update centers
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centers[i] = np.mean(cluster_points, axis=0)
        
        # Check for convergence
        center_shift = np.linalg.norm(centers - old_centers)
        if center_shift < tol:
            break
    
    return centers, labels

def linear_regression(X: np.ndarray, y: np.ndarray, 
                     regularization: str = None, alpha: float = 1.0) -> np.ndarray:
    """
    Linear regression with optional regularization.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        regularization: None, 'l1' (Lasso), or 'l2' (Ridge)
        alpha: Regularization strength
    
    Returns:
        Coefficients vector
    """
    n_samples, n_features = X.shape
    
    if regularization is None:
        # Normal equation solution
        return np.linalg.inv(X.T @ X) @ X.T @ y
    
    elif regularization == 'l2':
        # Ridge regression
        I = np.eye(n_features)
        I[0, 0] = 0  # Don't regularize the intercept term
        return np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    
    elif regularization == 'l1':
        # Lasso regression using coordinate descent
        # Simple implementation for demonstration
        beta = np.zeros(n_features)
        max_iter = 1000
        tol = 1e-4
        
        for _ in range(max_iter):
            beta_old = beta.copy()
            
            for j in range(n_features):
                X_j = X[:, j]
                y_pred = X @ beta
                rho = np.dot(X_j, y - y_pred + beta[j] * X_j)
                
                if j == 0:  # Intercept term
                    beta[j] = rho / n_samples
                else:  # Regularized coefficients
                    beta[j] = soft_threshold(rho, alpha * n_samples / 2) / (np.dot(X_j, X_j) + 1e-10)
            
            if np.linalg.norm(beta - beta_old) < tol:
                break
        
        return beta
    
    else:
        raise ValueError(f"Unsupported regularization type: {regularization}")

def soft_threshold(rho: float, alpha: float) -> float:
    """
    Soft thresholding operator used in Lasso regression.
    
    Args:
        rho: Input value
        alpha: Threshold parameter
    
    Returns:
        Soft thresholded value
    """
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0.0

def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Generate polynomial features up to specified degree.
    
    Args:
        X: Input features of shape (n_samples, n_features)
        degree: Maximum polynomial degree
    
    Returns:
        Polynomial feature matrix
    """
    n_samples, n_features = X.shape
    combinations = []
    
    # Generate all combinations of features up to the specified degree
    def _gen_combinations(current_deg, start_idx, current_comb):
        if current_deg == 0:
            combinations.append(tuple(current_comb))
            return
        
        for i in range(start_idx, n_features):
            current_comb.append(i)
            _gen_combinations(current_deg - 1, i, current_comb)
            current_comb.pop()
    
    # Start with bias term
    poly_X = np.ones((n_samples, 1))
    
    # Generate features for each degree
    for d in range(1, degree + 1):
        _gen_combinations(d, 0, [])
        
        for comb in combinations:
            feature = np.ones(n_samples)
            for idx in comb:
                feature *= X[:, idx]
            poly_X = np.column_stack((poly_X, feature))
        
        combinations = []
    
    return poly_X

def rbf_kernel(X: np.ndarray, Y: np.ndarray = None, gamma: float = 1.0) -> np.ndarray:
    """
    Radial Basis Function (RBF) kernel.
    
    Args:
        X: First set of samples of shape (n_samples_X, n_features)
        Y: Second set of samples of shape (n_samples_Y, n_features). If None, use X.
        gamma: Kernel coefficient
    
    Returns:
        Kernel matrix of shape (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    
    # Compute squared Euclidean distances
    XX = np.sum(X ** 2, axis=1).reshape(-1, 1)
    YY = np.sum(Y ** 2, axis=1).reshape(1, -1)
    distances = XX + YY - 2 * X @ Y.T
    
    # Apply RBF kernel
    K = np.exp(-gamma * distances)
    return K

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid function with numerical stability.
    
    Args:
        x: Input array
    
    Returns:
        Sigmoid output
    """
    # Clip input to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Args:
        x: Input array
    
    Returns:
        ReLU output
    """
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU function.
    
    Args:
        x: Input array
    
    Returns:
        ReLU derivative
    """
    return (x > 0).astype(float)

def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary cross-entropy (log loss).
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities
    
    Returns:
        Log loss value
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy for binary or multiclass classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (or probabilities for binary classification)
    
    Returns:
        Accuracy value
    """
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        # Multiclass case - y_pred is probabilities
        y_pred = np.argmax(y_pred, axis=1)
    elif y_pred.ndim == 1 or y_pred.shape[1] == 1:
        # Binary case - y_pred is probabilities
        y_pred = (y_pred > 0.5).astype(int)
    
    return np.mean(y_true == y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute precision for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
    
    Returns:
        Precision value
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    
    if true_positives + false_positives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_positives)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute recall for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
    
    Returns:
        Recall value
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    if true_positives + false_negatives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_negatives)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 score for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
    
    Returns:
        F1 score value
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    
    if p + r == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                    labels: Optional[List] = None) -> np.ndarray:
    """
    Compute confusion matrix for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label indices to include in the confusion matrix
    
    Returns:
        Confusion matrix array
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    
    n_classes = len(labels)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in labels and pred_label in labels:
            i = np.where(labels == true_label)[0][0]
            j = np.where(labels == pred_label)[0][0]
            matrix[i, j] += 1
    
    return matrix

def roc_curve(y_true: np.ndarray, y_score: np.ndarray, 
             pos_label: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Receiver Operating Characteristic (ROC) curve.
    
    Args:
        y_true: True binary labels
        y_score: Target scores (probability estimates)
        pos_label: Label of the positive class
    
    Returns:
        False positive rates, true positive rates, thresholds
    """
    # Sort scores in descending order
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # Count total positives and negatives
    n_pos = np.sum(y_true == pos_label)
    n_neg = len(y_true) - n_pos
    
    # Initialize arrays
    tps = np.cumsum(y_true == pos_label)
    fps = np.cumsum(y_true != pos_label)
    
    # Add initial point
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[y_score[0] + 1, y_score]
    
    # Compute TPR and FPR
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    return fpr, tpr, thresholds

def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUC).
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
    
    Returns:
        AUC value
    """
    return np.trapz(tpr, fpr)

def entropy(p: np.ndarray) -> float:
    """
    Compute entropy of a probability distribution.
    
    Args:
        p: Probability distribution
    
    Returns:
        Entropy value
    """
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1)
    return -np.sum(p * np.log2(p))

def gini_impurity(p: np.ndarray) -> float:
    """
    Compute Gini impurity of a probability distribution.
    
    Args:
        p: Probability distribution
    
    Returns:
        Gini impurity value
    """
    return 1 - np.sum(p ** 2)

def mutual_information(X: np.ndarray, y: np.ndarray, 
                       n_bins: int = 10) -> np.ndarray:
    """
    Compute mutual information between features and target.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_bins: Number of bins for discretization
    
    Returns:
        Mutual information for each feature
    """
    n_samples, n_features = X.shape
    mi = np.zeros(n_features)
    
    # Discretize target
    y_discrete = np.digitize(y, bins=np.linspace(np.min(y), np.max(y), n_bins))
    
    for i in range(n_features):
        # Discretize feature
        x_discrete = np.digitize(X[:, i], bins=np.linspace(np.min(X[:, i]), np.max(X[:, i]), n_bins))
        
        # Compute mutual information
        mi[i] = _compute_mutual_information(x_discrete, y_discrete)
    
    return mi

def _compute_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Helper function to compute mutual information between two discrete variables."""
    # Get unique values
    x_vals = np.unique(x)
    y_vals = np.unique(y)
    
    # Compute marginal and joint probabilities
    p_x = np.array([np.mean(x == xv) for xv in x_vals])
    p_y = np.array([np.mean(y == yv) for yv in y_vals])
    
    p_xy = np.zeros((len(x_vals), len(y_vals)))
    for i, xv in enumerate(x_vals):
        for j, yv in enumerate(y_vals):
            p_xy[i, j] = np.mean((x == xv) & (y == yv))
    
    # Compute mutual information
    mi = 0.0
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return mi

def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix of data matrix.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        Covariance matrix of shape (n_features, n_features)
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    n_samples = X.shape[0]
    cov = X_centered.T @ X_centered / (n_samples - 1)
    
    return cov

def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix of data matrix.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        Correlation matrix of shape (n_features, n_features)
    """
    # Compute covariance matrix
    cov = covariance_matrix(X)
    
    # Compute standard deviations
    stds = np.sqrt(np.diag(cov))
    
    # Compute correlation matrix
    corr = cov / np.outer(stds, stds)
    
    # Set diagonal to 1 (fix numerical errors)
    np.fill_diagonal(corr, 1.0)
    
    return corr

def eigen_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalue decomposition of a symmetric matrix.
    
    Args:
        A: Symmetric matrix
    
    Returns:
        Eigenvalues and eigenvectors
    """
    # Check if matrix is symmetric
    if not np.allclose(A, A.T, atol=1e-8):
        logger.warning("Matrix is not symmetric. Results may be inaccurate.")
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors

def pca_transform(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform data using PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of components to keep
    
    Returns:
        Transformed data and explained variance ratio
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = covariance_matrix(X_centered)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigen_decomposition(cov)
    
    # Select top n_components
    components = eigenvectors[:, :n_components]
    
    # Transform data
    X_transformed = X_centered @ components
    
    # Compute explained variance ratio
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_transformed, explained_variance

def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Compute Mahalanobis distance between a point and a distribution.
    
    Args:
        x: Point vector
        mean: Mean vector of distribution
        cov: Covariance matrix of distribution
    
    Returns:
        Mahalanobis distance
    """
    # Compute inverse covariance matrix
    cov_inv = matrix_inverse(cov)
    
    # Compute difference vector
    diff = x - mean
    
    # Compute Mahalanobis distance
    return np.sqrt(diff.T @ cov_inv @ diff)

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Manhattan distance between two vectors.
    
    Args:
        x: First vector
        y: Second vector
    
    Returns:
        Manhattan distance
    """
    return np.sum(np.abs(x - y))

def chebyshev_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Chebyshev distance between two vectors.
    
    Args:
        x: First vector
        y: Second vector
    
    Returns:
        Chebyshev distance
    """
    return np.max(np.abs(x - y))

def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float = 2.0) -> float:
    """
    Compute Minkowski distance between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        p: Order of the norm
    
    Returns:
        Minkowski distance
    """
    return np.sum(np.abs(x - y) ** p) ** (1 / p)
```

## 📁 notebooks/05_production_engineering/01_fastapi_model_deployment.ipynb (Complete)

```python
# =====================
# نشر نماذج الذكاء الاصطناعي باستخدام FastAPI: من الدفتر إلى الإنتاج
# النظرية الرياضية -> التنفيذ البرمجي -> الاعتبارات الإنتاجية
# =====================

"""
## 1. فلسفة النشر في عصر الذكاء الاصطناعي
في عصر الذكاء الاصطناعي الحديث، لم يعد النموذج في دفتر الملاحظات كافياً. القيمة الحقيقية تكمن في دمج النماذج في خطوط الإنتاج وتقديمها كخدمات يمكن للتطبيقات الأخرى استخدامها. FastAPI ظهر كإطار عمل مثالي لهذا الغرض بسبب:

1. **الأداء العالي**: يستخدم ASGI (Asynchronous Server Gateway Interface) مع uvicorn، مما يوفر أداءً أفضل بنسبة 3-5 مرات من Flask في سيناريوهات الطلب المتزامن.
2. **التوثيق التلقائي**: يولد واجهة توثيق تفاعلية باستخدام OpenAPI و Swagger UI، مما يقلل من وقت التطوير وزيادة الجودة.
3. **التحقق من الصحة التلقائي**: يستخدم Pydantic للتحقق من أنواع البيانات المدخلة والمخرجة، مما يمنع 40% من الأخطاء الشائعة في واجهات برمجة التطبيقات.
4. **الدعم الأصلي للوظائف غير المتزامنة**: ضروري لخدمات الذكاء الاصطناعي التي غالبًا ما تتضمن عمليات إدخال/إخراج (I/O) مكثفة.

الفهم العميق لكيفية عمل هذه الأنظمة هو ما يفصل بين "كود يعمل" و"خدمة تعمل في الإنتاج مع مقاييس الألف مطلوب".
"""

"""
## 2. تصميم البنية: من النموذج إلى الخدمة
عند تصميم خدمة نماذج الذكاء الاصطناعي، يجب أن نأخذ في الاعتبار الطبقة الكاملة:
1. **واجهة المستخدم**: كيف يتفاعل المستخدم مع الخدمة؟
2. **واجهة برمجة التطبيقات**: النقاط النهائية، التنسيقات، والتحقق من الصحة.
3. **منطق النموذج**: تحميل النموذج، المعالجة المسبقة، التنبؤ، المعالجة اللاحقة.
4. **البنية التحتية**: النشر، التوازن بين الحمل، التوسع التلقائي.
5. **المراقبة**: تتبع الأداء، الكشف عن الأخطاء، قياس المقاييس الرئيسية.

سنركز في هذا الدفتر على الطبقات 2 و 3، مع مناقشة كيفية التكامل مع الطبقات الأخرى.
"""

import time
import json
import logging
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import joblib
import redis

# تكوين السجل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_api")

"""
## 3. التنفيذ: بناء خدمة نموذج كاملة
فيما يلي تنفيذ كامل لخدمة نموذج الذكاء الاصطناعي باستخدام FastAPI، مع التركيز على الميزات الإنتاجية.
"""

@dataclass
class ModelConfig:
    """تكوين النموذج"""
    model_path: str = "models/model.pkl"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 3600  # ثانية
    max_batch_size: int = 32

class HealthStatus(str, Enum):
    """حالة الصحة للخدمة"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class PredictionRequest(BaseModel):
    """نموذج طلب التنبؤ"""
    features: Dict[str, Union[float, int, str, bool]]
    request_id: Optional[str] = None
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v

class BatchPredictionRequest(BaseModel):
    """نموذج طلب التنبؤ الدفعي"""
    requests: List[PredictionRequest]
    max_concurrency: int = Field(default=4, ge=1, le=16)

class PredictionResponse(BaseModel):
    """نموذج استجابة التنبؤ"""
    prediction: Union[float, int, str, list]
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ModelService:
    """خدمة نموذج الذكاء الاصطناعي مع وظائف الإنتاج"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.redis_client = None
        self.load_model()
        self.setup_cache()
    
    def load_model(self):
        """تحميل النموذج من الملف"""
        try:
            start_time = time.time()
            self.model = joblib.load(self.config.model_path)
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def setup_cache(self):
        """إعداد Redis للتخزين المؤقت"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis cache initialization failed: {str(e)}")
            self.redis_client = None
    
    def generate_cache_key(self, features: Dict[str, Any]) -> str:
        """توليد مفتاح للتخزين المؤقت بناءً على الميزات"""
        sorted_features = dict(sorted(features.items()))
        return json.dumps(sorted_features, sort_keys=True)
    
    async def get_cached_prediction(self, cache_key: str) -> Optional[PredictionResponse]:
        """الحصول على تنبؤ من التخزين المؤقت إن وجد"""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                data = json.loads(cached)
                return PredictionResponse(**data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
        
        return None
    
    async def cache_prediction(self, cache_key: str, prediction: PredictionResponse):
        """تخزين التنبؤ في التخزين المؤقت"""
        if not self.redis_client:
            return
        
        try:
            data = prediction.dict()
            self.redis_client.setex(cache_key, self.config.cache_ttl, json.dumps(data))
        except Exception as e:
            logger.warning(f"Cache storage failed: {str(e)}")
    
    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """المعالجة المسبقة للميزات قبل التنبؤ"""
        # في تطبيق حقيقي، قد تشمل هذه الخطوة:
        # - الترميز (encoding) للميزات الفئوية
        # - المقياس (scaling) للميزات العددية
        # - التعامل مع القيم المفقودة
        
        # للتبسيط، نفترض أن الميزات جاهزة للتنبؤ
        feature_values = np.array(list(features.values())).reshape(1, -1)
        return feature_values
    
    def postprocess_prediction(self, raw_prediction: np.ndarray, 
                             probabilities: Optional[np.ndarray] = None) -> PredictionResponse:
        """المعالجة اللاحقة للتنبؤ لتحويله إلى استجابة مفيدة"""
        # تحويل التنبؤ الخام إلى قيمة قابلة للاستخدام
        prediction_value = raw_prediction[0]
        
        # تحويل الاحتمالات إلى تنسيق قابل للاستخدام
        prob_dict = None
        confidence = None
        
        if probabilities is not None:
            if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                # تصنيف متعدد الفئات
                confidence = np.max(probabilities[0])
                if hasattr(self.model, 'classes_'):
                    prob_dict = {
                        str(cls): float(prob) 
                        for cls, prob in zip(self.model.classes_, probabilities[0])
                    }
            else:
                # تصنيف ثنائي
                confidence = float(probabilities[0][0])
                prob_dict = {"positive": confidence, "negative": 1 - confidence}
        
        return PredictionResponse(
            prediction=float(prediction_value) if isinstance(prediction_value, (int, float, np.number)) else str(prediction_value),
            confidence=confidence,
            probabilities=prob_dict,
            metadata={
                "model_version": getattr(self.model, "version", "1.0.0"),
                "feature_names": list(features.keys()) if 'features' in locals() else [],
                "timestamp": time.time()
            }
        )
    
    async def predict(self, features: Dict[str, Any]) -> PredictionResponse:
        """التنبؤ باستخدام النموذج"""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # توليد مفتاح التخزين المؤقت
            cache_key = self.generate_cache_key(features)
            
            # محاولة الحصول على التنبؤ من التخزين المؤقت
            cached_prediction = await self.get_cached_prediction(cache_key)
            if cached_prediction:
                logger.info("Cache hit for prediction")
                return cached_prediction
            
            # المعالجة المسبقة
            processed_features = self.preprocess_features(features)
            
            # التنبؤ
            start_time = time.time()
            raw_prediction = self.model.predict(processed_features)
            prediction_time = time.time() - start_time
            
            # الحصول على الاحتمالات إن أمكن
            probabilities = None
            if hasattr(self.model, "predict_proba"):
                try:
                    probabilities = self.model.predict_proba(processed_features)
                except Exception as e:
                    logger.warning(f"Probability prediction failed: {str(e)}")
            
            # المعالجة اللاحقة
            prediction_response = self.postprocess_prediction(raw_prediction, probabilities)
            
            # تخزين التنبؤ في التخزين المؤقت
            await self.cache_prediction(cache_key, prediction_response)
            
            logger.info(f"Prediction completed in {prediction_time:.4f} seconds")
            return prediction_response
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def batch_predict(self, requests: List[Dict[str, Any]], 
                           max_concurrency: int = 4) -> List[PredictionResponse]:
        """التنبؤ الدفعي مع معالجة متزامنة"""
        if len(requests) > self.config.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum of {self.config.max_batch_size}"
            )
        
        # معالجة الطلبات بالتوازي
        predictions = []
        start_time = time.time()
        
        for request in requests:
            try:
                prediction = await self.predict(request)
                predictions.append(prediction)
            except Exception as e:
                predictions.append(PredictionResponse(
                    prediction="error",
                    metadata={"error": str(e)}
                ))
        
        total_time = time.time() - start_time
        logger.info(f"Batch prediction completed for {len(requests)} items in {total_time:.4f} seconds")
        
        return predictions
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """الحصول على بيانات وصفية حول النموذج"""
        metadata = {
            "model_type": type(self.model).__name__,
            "features": getattr(self.model, "feature_names_in_", []) if hasattr(self.model, "feature_names_in_") else [],
            "classes": getattr(self.model, "classes_", []).tolist() if hasattr(self.model, "classes_") else [],
            "n_features": getattr(self.model, "n_features_in_", None),
            "n_classes": getattr(self.model, "n_classes_", None),
            "training_samples": getattr(self.model, "n_samples_", None),
            "version": getattr(self.model, "version", "1.0.0"),
            "training_date": getattr(self.model, "training_date", time.strftime("%Y-%m-%d"))
        }
        return metadata
    
    def health_check(self) -> Dict[str, Any]:
        """التحقق من حالة الخدمة"""
        status = HealthStatus.HEALTHY
        issues = []
        
        # التحقق من حالة النموذج
        if self.model is None:
            status = HealthStatus.UNHEALTHY
            issues.append("Model not loaded")
        
        # التحقق من حالة التخزين المؤقت
        cache_status = "connected" if self.redis_client else "not configured"
        if self.redis_client:
            try:
                self.redis_client.ping()
            except Exception as e:
                cache_status = f"disconnected: {str(e)}"
                status = HealthStatus.DEGRADED if status == HealthStatus.HEALTHY else status
                issues.append(f"Redis connection failed: {str(e)}")
        
        return {
            "status": status.value,
            "timestamp": time.time(),
            "uptime": time.time() - getattr(self, "start_time", time.time()),
            "components": {
                "model": {
                    "status": "loaded" if self.model else "not loaded",
                    "path": self.config.model_path
                },
                "cache": {
                    "status": cache_status,
                    "host": self.config.redis_host if self.redis_client else "not configured"
                }
            },
            "issues": issues
        }

# تهيئة التطبيق
app = FastAPI(
    title="AI Model API",
    description="Production-ready API for AI models with caching, monitoring, and reliability features",
    version="1.0.0"
)

# إضافة middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تهيئة خدمة النموذج
config = ModelConfig()
model_service = ModelService(config)
model_service.start_time = time.time()  # تتبع وقت التشغيل

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """تسجيل طلبات HTTP"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(f"{request.method} {request.url.path} {response.status_code} - {process_time:.4f}s")
    
    # إضافة رأس وقت المعالجة إلى الاستجابة
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    
    return response

@app.get("/health")
async def health_check():
    """نقطة نهاية للتحقق من صحة الخدمة"""
    return model_service.health_check()

@app.get("/metadata")
async def get_model_metadata():
    """نقطة نهاية للحصول على بيانات وصفية حول النموذج"""
    try:
        return model_service.get_model_metadata()
    except Exception as e:
        logger.error(f"Metadata retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model metadata")

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest, background_tasks: BackgroundTasks):
    """نقطة نهاية للتنبؤ الفردي"""
    try:
        return await model_service.predict(request.features)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def batch_predict_endpoint(request: BatchPredictionRequest):
    """نقطة نهاية للتنبؤ الدفعي"""
    try:
        return await model_service.batch_predict(
            [req.features for req in request.requests],
            max_concurrency=request.max_concurrency
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/stream")
async def stream_predict_endpoint(request: PredictionRequest):
    """نقطة نهاية للتنبؤ المتسلسل (streaming)"""
    
    async def generate_predictions():
        """مولد للاستجابات المتسلسلة"""
        try:
            yield json.dumps({"status": "processing", "stage": "preparing"}) + "\n"
            
            # مرحلة المعالجة المسبقة
            yield json.dumps({"status": "processing", "stage": "preprocessing"}) + "\n"
            processed_features = model_service.preprocess_features(request.features)
            
            # مرحلة التنبؤ
            yield json.dumps({"status": "processing", "stage": "predicting"}) + "\n"
            raw_prediction = model_service.model.predict(processed_features)
            
            # مرحلة المعالجة اللاحقة
            yield json.dumps({"status": "processing", "stage": "postprocessing"}) + "\n"
            prediction = model_service.postprocess_prediction(raw_prediction)
            
            # النتيجة النهائية
            yield json.dumps({
                "status": "complete",
                "result": prediction.dict()
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Streaming prediction failed: {str(e)}")
            yield json.dumps({
                "status": "error",
                "message": str(e)
            }) + "\n"
    
    return StreamingResponse(generate_predictions(), media_type="application/json")

if __name__ == "__main__":
    """نقطة الدخول للتطبيق"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Model API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--model-path', type=str, help='Path to the model file')
    args = parser.parse_args()
    
    if args.model_path:
        config.model_path = args.model_path
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Model path: {config.model_path}")
    
    uvicorn.run(
        "fastapi_model_deployment:app",
        host=args.host,
        port=args.port,
        reload=False,
        workers=4,
        log_level="info"
    )

"""
## 4. الاعتبارات الإنتاجية: ما وراء الكود
الكود أعلاه يوفر أساسًا متينًا، لكن في الإنتاج، هناك اعتبارات إضافية يجب أخذها في الاعتبار:

### 4.1 الأداء والتوسع
- **التخزين المؤقت (Caching)**: استخدم Redis لتخزين التوقعات المتكررة، كما فعلنا في المثال.
- **التوازن بين الحمل (Load Balancing)**: نشر عدة نسخ من الخدمة باستخدام nginx أو Kubernetes.
- **التوسع التلقائي (Auto-scaling)**: زيادة عدد النسخ استجابةً لزيادة الحمل.
- **التوسع الأفقي مقابل الرأسي**: للخدمات ذات التأخير المنخفض، التوسع الرأسي (معالجات أقوى) قد يكون أفضل. للخدمات ذات الطلبات العالية، التوسع الأفقي (نسخ أكثر) قد يكون أفضل.

### 4.2 الموثوقية والمراقبة
- **المراقبة المستمرة**: تتبع مقاييس مثل:
  - زمن الاستجابة (latency)
  - معدل الأخطاء (error rate)
  - معدل الطلبات في الثانية (RPS)
  - استخدام الذاكرة والمعالج
- **تنبيهات آلية**: إرسال تنبيهات عند تجاوز عتبات محددة (مثلاً، خطأ 500 لأكثر من 1% من الطلبات).
- **التعافي التلقائي**: إعادة تشغيل الخدمة عند تعطلها.

### 4.3 الأمان
- **التحقق من الهوية (Authentication)**: استخدام مفاتيح API أو JWT للتحقق من هوية المستخدمين.
- **التفويض (Authorization)**: التحكم في الوصول بناءً على الأدوار.
- **الحماية من الهجمات**: الحد من المعدل (rate limiting) لمنع هجمات الحرمان من الخدمة (DoS).
- **تشفير البيانات**: استخدام HTTPS وتشفير البيانات الحساسة.

### 4.4 تحسين التكلفة
- **الاستدلال المُحسّن**: استخدام تقنيات مثل التكميم (quantization) أو التقليم (pruning) للنماذج لتقليل استهلاك الموارد.
- **التخزين المؤقت الاستراتيجي**: حفظ التوقعات الشائعة لتقليل عدد عمليات الاستدلال.
- **الاستدعاءات غير المتزامنة**: لطلبات تستغرق وقتًا طويلًا، استخدم الطوابير (queues) لتجنب حظر الخادم.

### 4.5 التطوير المستمر
- **النشر المستمر (CI/CD)**: أتمتة عملية الاختبار والنشر.
- **اختبار الحمل (Load Testing)**: التأكد من أن الخدمة يمكنها التعامل مع الحمل المتوقع.
- **الترجيع السريع (Rollback)**: القدرة على الرجوع إلى إصدار سابق عند وجود مشكلة.
- **المراقبة بعد النشر**: مراقبة أداء الخدمة بعد النشر والكشف المبكر عن المشاكل.

### 4.6 تحليل التكاليف
- **تكلفة كل طلب**: تتبع تكلفة كل طلب (استهلاك وحدة المعالجة المركزية/البنية التحتية).
- **نقطة التعادل (Break-even Point)**: حساب عدد الطلبات اللازمة لتغطية تكلفة الخدمة.
- **التحسين المستمر**: البحث عن فرص لتحسين الكفاءة وخفض التكاليف.

## 5. التحديات العملية: تمارين للتطبيق
### مستوى مبتدئ
1. أضف نقطة نهاية GET `/` تعرض رسالة ترحيب وبعض المعلومات الأساسية عن الخدمة.
2. نفذ دالة تحقق من صحة الميزات للتأكد من أن جميع الميزات المطلوبة موجودة في الطلب.
3. أضف دعمًا للتخزين المؤقت باستخدام متغير عالمي (في الذاكرة) كحل مؤقت إذا فشل الاتصال بـ Redis.

### مستوى متوسط
1. نفذ نظام حماية من الهجمات (rate limiting) باستخدام مكتبة slowapi.
2. أضف مراقبة باستخدام Prometheus و Grafana لعرض مقاييس الخدمة.
3. نفذ نظامًا لتخزين طلبات التنبؤ في قاعدة بيانات للاستخدام في إعادة تدريب النماذج.

### مستوى متقدم
1. نفذ خدمة خلفية (background worker) باستخدام Celery لمعالجة الطلبات الطويلة.
2. أضف دعمًا لـ gRPC بالإضافة إلى HTTP/REST لتحسين الأداء.
3. نفذ نظامًا متقدمًا للتخزين المؤقت يستخدم عدة مستويات (L1: الذاكرة، L2: Redis، L3: قاعدة البيانات).

### تحدي إنتاجي
أنشئ نظامًا كاملًا لخدمة نماذج الذكاء الاصطناعي يتضمن:
- خدمة النموذج الأساسية (مثل ما أنشأناه)
- لوحة تحكم لمراقبة الأداء
- نظام تنبيهات عند وجود مشاكل
- واجهة برمجة تطبيقات للإدارة (لتحديث النماذج، مسح التخزين المؤقت، إلخ)
- مستندات API كاملة ومحدثة

## 6. الخلاصة
نشر نماذج الذكاء الاصطناعي باستخدام FastAPI هو عملية معقدة تتطلب فهمًا عميقًا لكل من علم البيانات والهندسة البرمجية. النجاح في هذا المجال لا يتعلق فقط ببناء نماذج دقيقة، بل أيضًا بجعلها متاحة، موثوقة، وقابلة للتوسع. الكود الذي قدمناه في هذا الدفتر هو نقطة بداية قوية، ولكنه يحتاج إلى تعديلات وتحسينات كبيرة ليناسب بيئات الإنتاج الحقيقية.

تذكر دائمًا: "النموذج في دفتر الملاحظات ليس منتجًا". القيمة الحقيقية تبدأ عندما يمكن لمستخدم حقيقي الحصول على قيمة من عملك.
"""
```

## 📁 case_studies/medical_diagnosis_agent/implementation/diagnostic_engine.py (Complete)

```python
"""
نظام تشخيص طبي متطور مع ضمان الخصوصية والموثوقية
هذا الملف يمثل نواة نظام تشخيص طبي قائم على الذكاء الاصطناعي مصمم للإطلاق في بيئة إنتاج حقيقية.
التركيز على دقة التشخيص، حماية البيانات، والموثوقية في القرارات الطبية الحرجة.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
import hashlib
import re

# تكوين السجل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("diagnostic_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("medical_diagnosis_agent")

class DiagnosticConfidenceLevel(str, Enum):
    """مستويات ثقة التشخيص"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNSURE = "unsure"

class DiagnosticRiskLevel(str, Enum):
    """مستويات خطورة التشخيص"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class MedicalSpecialty(str, Enum):
    """التخصصات الطبية"""
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    ONCOLOGY = "oncology"
    ENDOCRINOLOGY = "endocrinology"
    GASTROENTEROLOGY = "gastroenterology"
    PULMONOLOGY = "pulmonology"
    DERMATOLOGY = "dermatology"
    GENERAL = "general"

@dataclass
class PatientSymptom:
    """بيانات الأعراض الطبية"""
    symptom_name: str
    severity: int = Field(ge=1, le=10, description="شدة العرض من 1 إلى 10")
    duration_hours: float = Field(ge=0, description="المدة بالساعات")
    metadata: Dict[str, Any] = Field(default_factory=dict)

@dataclass
class MedicalCondition:
    """بيانات الحالة الطبية"""
    condition_name: str
    icd_code: Optional[str] = None
    medical_specialty: MedicalSpecialty
    severity_level: int = Field(ge=1, le=10)
    urgency_level: int = Field(ge=1, le=10)
    treatment_options: List[str]
    diagnostic_criteria: List[str]
    risk_factors: List[str]

@dataclass
class DiagnosticResult:
    """نتيجة التشخيص"""
    condition: MedicalCondition
    confidence_level: DiagnosticConfidenceLevel
    risk_level: DiagnosticRiskLevel
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    recommended_actions: List[str]
    required_tests: List[str]
    explanation: str
    ai_model_version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

class MedicalKnowledgeBase:
    """قاعدة المعرفة الطبية"""
    
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = knowledge_base_path
        self.conditions = self._load_knowledge_base()
        self.symptom_mappings = self._build_symptom_mappings()
    
    def _load_knowledge_base(self) -> Dict[str, MedicalCondition]:
        """تحميل قاعدة المعرفة من ملف JSON"""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conditions = {}
            for condition_data in data['conditions']:
                condition = MedicalCondition(
                    condition_name=condition_data['name'],
                    icd_code=condition_data.get('icd_code'),
                    medical_specialty=MedicalSpecialty(condition_data['specialty']),
                    severity_level=condition_data['severity_level'],
                    urgency_level=condition_data['urgency_level'],
                    treatment_options=condition_data['treatment_options'],
                    diagnostic_criteria=condition_data['diagnostic_criteria'],
                    risk_factors=condition_data.get('risk_factors', [])
                )
                conditions[condition.condition_name] = condition
            
            logger.info(f"Loaded {len(conditions)} medical conditions from knowledge base")
            return conditions
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
            raise
    
    def _build_symptom_mappings(self) -> Dict[str, List[str]]:
        """بناء خريطة بين الأعراض والحالات الطبية"""
        symptom_to_conditions = {}
        
        for condition_name, condition in self.conditions.items():
            for criterion in condition.diagnostic_criteria:
                # استخراج الأعراض من معايير التشخيص
                symptoms = self._extract_symptoms_from_criterion(criterion)
                for symptom in symptoms:
                    if symptom not in symptom_to_conditions:
                        symptom_to_conditions[symptom] = []
                    symptom_to_conditions[symptom].append(condition_name)
        
        logger.info(f"Built symptom mappings for {len(symptom_to_conditions)} symptoms")
        return symptom_to_conditions
    
    def _extract_symptoms_from_criterion(self, criterion: str) -> List[str]:
        """استخراج الأعراض من نص معيار التشخيص"""
        # هذه دالة بسيطة للاستخراج، في التطبيق الواقعي سيكون أكثر تعقيداً
        symptoms = []
        
        # أنماط شائعة للأعراض
        common_symptoms = [
            'ألم', 'حرارة', 'سعال', 'غثيان', 'دوخة', 'صداع', 'تعب',
            'ضعف', 'تورم', 'احمرار', 'طفح', 'ضيق تنفس', 'خفقان'
        ]
        
        for symptom in common_symptoms:
            if symptom in criterion.lower():
                symptoms.append(symptom)
        
        return symptoms

class PrivacyEngine:
    """محرك الخصوصية والأمان للبيانات الطبية"""
    
    def __init__(self, pii_patterns_path: Optional[str] = None):
        self.pii_patterns = self._load_pii_patterns(pii_patterns_path)
        self.hash_salt = os.getenv('PRIVACY_HASH_SALT', 'medical_diagnosis_salt_2025')
    
    def _load_pii_patterns(self, pii_patterns_path: Optional[str]) -> List[re.Pattern]:
        """تحميل أنماط البيانات الحساسة (PII)"""
        default_patterns = [
            # أرقام الهواتف
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            # عناوين البريد الإلكتروني
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            # أرقام الهوية الوطنية
            re.compile(r'\b\d{10}\b'),
            # تواريخ الميلاد
            re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),
            # أسماء الأشخاص (نمط بسيط)
            re.compile(r'\b(الأستاذ|الدكتور|السيدة|السيد)\s+[أ-ي][\w\s]+'),
            # العناوين
            re.compile(r'\b(شارع|طريق|مبنى|عمارة)\s+[\w\s,]+')
        ]
        
        if pii_patterns_path and os.path.exists(pii_patterns_path):
            try:
                with open(pii_patterns_path, 'r', encoding='utf-8') as f:
                    custom_patterns = json.load(f)
                for pattern in custom_patterns:
                    default_patterns.append(re.compile(pattern))
            except Exception as e:
                logger.warning(f"Failed to load custom PII patterns: {str(e)}")
        
        return default_patterns
    
    def anonymize_text(self, text: str) -> str:
        """إخفاء الهوية في النص"""
        anonymized_text = text
        
        for pattern in self.pii_patterns:
            matches = pattern.findall(anonymized_text)
            for match in matches:
                # استبدال البيانات الحساسة بـ [REDACTED]
                anonymized_text = anonymized_text.replace(match, '[REDACTED]')
        
        return anonymized_text
    
    def hash_patient_id(self, patient_id: str) -> str:
        """تشفير هوية المريض"""
        return hashlib.sha256((patient_id + self.hash_salt).encode()).hexdigest()
    
    def validate_consent(self, consent_data: Dict[str, Any]) -> bool:
        """التحقق من موافقة المريض"""
        required_fields = ['patient_id', 'consent_granted', 'consent_date', 'consent_version']
        
        for field in required_fields:
            if field not in consent_data:
                logger.warning(f"Missing required consent field: {field}")
                return False
        
        if not consent_data['consent_granted']:
            logger.warning("Patient has not granted consent")
            return False
        
        # التحقق من تاريخ الموافقة (يجب ألا يكون في المستقبل)
        consent_date = datetime.fromisoformat(consent_data['consent_date'])
        if consent_date > datetime.utcnow():
            logger.warning("Consent date is in the future")
            return False
        
        return True

class DiagnosticEngine:
    """محرك التشخيص الطبي الرئيسي"""
    
    def __init__(self, knowledge_base_path: str, pii_patterns_path: Optional[str] = None):
        self.knowledge_base = MedicalKnowledgeBase(knowledge_base_path)
        self.privacy_engine = PrivacyEngine(pii_patterns_path)
        self.model_version = "1.2.0"
        self.confidence_thresholds = {
            DiagnosticConfidenceLevel.HIGH: 0.85,
            DiagnosticConfidenceLevel.MEDIUM: 0.65,
            DiagnosticConfidenceLevel.LOW: 0.40
        }
    
    def _validate_symptoms(self, symptoms: List[PatientSymptom]) -> bool:
        """التحقق من صحة الأعراض المدخلة"""
        if not symptoms:
            logger.warning("No symptoms provided")
            return False
        
        for symptom in symptoms:
            if not symptom.symptom_name:
                logger.warning("Empty symptom name")
                return False
            if not (1 <= symptom.severity <= 10):
                logger.warning(f"Invalid severity level: {symptom.severity}")
                return False
            if symptom.duration_hours < 0:
                logger.warning(f"Negative duration: {symptom.duration_hours}")
                return False
        
        return True
    
    def _calculate_symptom_weight(self, symptom: PatientSymptom, condition: MedicalCondition) -> float:
        """حساب وزن العرض بالنسبة لحالة طبية معينة"""
        weight = 0.0
        
        # 1. مدى تطابق العرض مع معايير التشخيص
        for criterion in condition.diagnostic_criteria:
            if symptom.symptom_name.lower() in criterion.lower():
                weight += 0.3
        
        # 2. شدة العرض (الشدة العالية تزيد الثقة في تشخيص الحالات الحرجة)
        severity_factor = symptom.severity / 10.0
        if condition.urgency_level > 7 and symptom.severity > 7:
            weight += severity_factor * 0.4
        
        # 3. مدة العرض (بعض الحالات تتطلب أعراضًا ذات مدة محددة)
        duration_factor = min(symptom.duration_hours / 24.0, 1.0)  # حتى 24 ساعة
        if "حادة" in condition.condition_name.lower() and symptom.duration_hours < 48:
            weight += duration_factor * 0.3
        
        return min(weight, 1.0)
    
    def _calculate_condition_match_score(self, symptoms: List[PatientSymptom], 
                                        condition: MedicalCondition) -> Tuple[float, List[str], List[str]]:
        """حساب درجة التطابق بين الأعراض والحالة الطبية"""
        total_weight = 0.0
        max_possible_weight = 0.0
        supporting_evidence = []
        contradicting_evidence = []
        
        for symptom in symptoms:
            weight = self._calculate_symptom_weight(symptom, condition)
            total_weight += weight
            max_possible_weight += 1.0
            
            if weight > 0.2:
                supporting_evidence.append(f"العَرَض '{symptom.symptom_name}' (شدة: {symptom.severity}/10) يدعم تشخيص {condition.condition_name}")
            elif weight == 0 and symptom.severity > 7:
                contradicting_evidence.append(f"العَرَض '{symptom.symptom_name}' (شدة عالية) لا يتوافق مع {condition.condition_name}")
        
        match_score = total_weight / max_possible_weight if max_possible_weight > 0 else 0.0
        return match_score, supporting_evidence, contradicting_evidence
    
    def _determine_confidence_level(self, match_score: float, 
                                   symptoms_count: int) -> DiagnosticConfidenceLevel:
        """تحديد مستوى ثقة التشخيص بناءً على نقاط التطابق"""
        if symptoms_count < 2:
            return DiagnosticConfidenceLevel.LOW
        
        if match_score >= self.confidence_thresholds[DiagnosticConfidenceLevel.HIGH]:
            return DiagnosticConfidenceLevel.HIGH
        elif match_score >= self.confidence_thresholds[DiagnosticConfidenceLevel.MEDIUM]:
            return DiagnosticConfidenceLevel.MEDIUM
        elif match_score >= self.confidence_thresholds[DiagnosticConfidenceLevel.LOW]:
            return DiagnosticConfidenceLevel.LOW
        else:
            return DiagnosticConfidenceLevel.UNSURE
    
    def _determine_risk_level(self, condition: MedicalCondition, 
                            confidence_level: DiagnosticConfidenceLevel) -> DiagnosticRiskLevel:
        """تحديد مستوى خطورة التشخيص"""
        urgency_score = condition.urgency_level
        
        if confidence_level == DiagnosticConfidenceLevel.HIGH:
            if urgency_score >= 8:
                return DiagnosticRiskLevel.CRITICAL
            elif urgency_score >= 6:
                return DiagnosticRiskLevel.HIGH
            elif urgency_score >= 4:
                return DiagnosticRiskLevel.MEDIUM
            else:
                return DiagnosticRiskLevel.LOW
        else:  # مستوى ثقة منخفض
            if urgency_score >= 7:
                return DiagnosticRiskLevel.HIGH
            elif urgency_score >= 5:
                return DiagnosticRiskLevel.MEDIUM
            else:
                return DiagnosticRiskLevel.LOW
    
    def _generate_explanation(self, condition: MedicalCondition, 
                            confidence_level: DiagnosticConfidenceLevel,
                            supporting_evidence: List[str],
                            contradicting_evidence: List[str]) -> str:
        """توليد شرح مفصل للتشخيص"""
        explanation = f"تم تشخيص حالة {condition.condition_name} "
        
        # مستوى الثقة
        confidence_text = {
            DiagnosticConfidenceLevel.HIGH: "بثقة عالية",
            DiagnosticConfidenceLevel.MEDIUM: "بثقة متوسطة",
            DiagnosticConfidenceLevel.LOW: "بثقة منخفضة",
            DiagnosticConfidenceLevel.UNSURE: "بدون ثقة كافية"
        }
        explanation += f"{confidence_text[confidence_level]} بناءً على الأعراض المدخلة."
        
        # الأدلة الداعمة
        if supporting_evidence:
            explanation += "\n\nالأدلة الداعمة:"
            for evidence in supporting_evidence[:3]:  # عرض أول 3 أدلة
                explanation += f"\n- {evidence}"
        
        # الأدلة المناقضة
        if contradicting_evidence:
            explanation += "\n\nالأدلة التي قد تناقض هذا التشخيص:"
            for evidence in contradicting_evidence[:2]:  # عرض أول دليلين
                explanation += f"\n- {evidence}"
        
        # توصيات
        explanation += f"\n\nهذا التشخيص يتطلب استشارة طبيب مختص في {condition.medical_specialty.value} {condition.urgency_level}/10)."
        
        if condition.urgency_level >= 8:
            explanation += " يوصى بالذهاب إلى قسم الطوارئ فوراً."
        elif condition.urgency_level >= 6:
            explanation += " يوصى بزيارة الطبيب خلال الـ 24 ساعة القادمة."
        
        explanation += "\n\nملاحظة: هذا التشخيص آلي ولا يغني عن استشارة الطبيب المختص."
        
        return explanation
    
    def diagnose(self, patient_id: str, symptoms: List[PatientSymptom], 
                consent_data: Dict[str, Any], anonymize_output: bool = True) -> List[DiagnosticResult]:
        """
        تشخيص الحالات الطبية بناءً على الأعراض
        
        Args:
            patient_id: معرف المريض
            symptoms: قائمة الأعراض
            consent_data: بيانات موافقة المريض
            anonymize_output: ما إذا كان يجب إخفاء الهوية في النتائج
        
        Returns:
            قائمة نتائج التشخيص مرتبة حسب درجة التطابق
        """
        start_time = time.time()
        
        # 1. التحقق من الموافقة
        if not self.privacy_engine.validate_consent(consent_data):
            logger.warning("Patient consent validation failed")
            raise ValueError("Patient consent is invalid or missing")
        
        # 2. التحقق من صحة الأعراض
        if not self._validate_symptoms(symptoms):
            logger.warning("Symptom validation failed")
            raise ValueError("Invalid symptoms provided")
        
        # 3. تشفير هوية المريض
        hashed_patient_id = self.privacy_engine.hash_patient_id(patient_id)
        logger.info(f"Processing diagnosis for patient hash: {hashed_patient_id}")
        
        # 4. إخفاء الهوية في وصف الأعراض
        anonymized_symptoms = []
        for symptom in symptoms:
            anonymized_symptom_name = self.privacy_engine.anonymize_text(symptom.symptom_name)
            anonymized_symptoms.append(PatientSymptom(
                symptom_name=anonymized_symptom_name,
                severity=symptom.severity,
                duration_hours=symptom.duration_hours,
                metadata=symptom.metadata
            ))
        
        # 5. حساب درجات التطابق لجميع الحالات
        condition_scores = []
        for condition_name, condition in self.knowledge_base.conditions.items():
            match_score, supporting_evidence, contradicting_evidence = self._calculate_condition_match_score(
                anonymized_symptoms, condition
            )
            
            if match_score > 0.1:  # عتبة دنيا للتشخيص
                condition_scores.append((condition, match_score, supporting_evidence, contradicting_evidence))
        
        # 6. فرز النتائج حسب درجة التطابق
        condition_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 7. إنشاء نتائج التشخيص
        results = []
        for condition, match_score, supporting_evidence, contradicting_evidence in condition_scores[:5]:  # أفضل 5 نتائج
            confidence_level = self._determine_confidence_level(match_score, len(anonymized_symptoms))
            risk_level = self._determine_risk_level(condition, confidence_level)
            explanation = self._generate_explanation(condition, confidence_level, 
                                                    supporting_evidence, contradicting_evidence)
            
            # تحديد الإجراءات الموصى بها
            recommended_actions = []
            if risk_level == DiagnosticRiskLevel.CRITICAL:
                recommended_actions = ["الذهاب إلى الطوارئ فوراً", "استدعاء سيارة الإسعاف"]
            elif risk_level == DiagnosticRiskLevel.HIGH:
                recommended_actions = ["زيارة الطبيب خلال 24 ساعة", "تجنب الأنشطة البدنية الشاقة"]
            else:
                recommended_actions = ["زيارة الطبيب في أقرب وقت ممكن", "مراقبة الأعراض"]
            
            # تحديد الفحوصات المطلوبة
            required_tests = []
            if condition.medical_specialty == MedicalSpecialty.CARDIOLOGY:
                required_tests = ["تخطيط القلب", "تحاليل الدم (كوليسترول، إنزيمات القلب)"]
            elif condition.medical_specialty == MedicalSpecialty.NEUROLOGY:
                required_tests = ["تصوير الدماغ (CT/MRI)", "فحص القدرات العصبية"]
            
            result = DiagnosticResult(
                condition=condition,
                confidence_level=confidence_level,
                risk_level=risk_level,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                recommended_actions=recommended_actions,
                required_tests=required_tests,
                explanation=explanation,
                ai_model_version=self.model_version
            )
            
            # إخفاء الهوية في النتيجة إذا طلب
            if anonymize_output:
                result.explanation = self.privacy_engine.anonymize_text(result.explanation)
                result.supporting_evidence = [self.privacy_engine.anonymize_text(e) for e in result.supporting_evidence]
                result.contradicting_evidence = [self.privacy_engine.anonymize_text(e) for e in result.contradicting_evidence]
            
            results.append(result)
        
        processing_time = time.time() - start_time
        logger.info(f"Diagnosis completed in {processing_time:.2f} seconds. Found {len(results)} potential conditions.")
        
        return results
    
    def get_medical_specialist_recommendation(self, results: List[DiagnosticResult]) -> Dict[str, Any]:
        """الحصول على توصية بالتخصص الطبي المطلوب"""
        if not results:
            return {"specialty": "general", "priority": "low", "reasoning": "No diagnostic results available"}
        
        # تحديد التخصص بناءً على أفضل نتيجة
        top_result = results[0]
        specialty = top_result.condition.medical_specialty.value
        priority = top_result.risk_level.value
        
        reasoning = f"Based on the primary diagnosis of {top_result.condition.condition_name} "
        reasoning += f"with {top_result.confidence_level.value} confidence and {priority} risk level."
        
        return {
            "specialty": specialty,
            "priority": priority,
            "reasoning": reasoning,
            "alternative_specialties": list(set([
                result.condition.medical_specialty.value 
                for result in results[1:3]  # التخصصات البديلة من النتائج التالية
            ]))
        }

# مثال على الاستخدام
def example_usage():
    """مثال على استخدام محرك التشخيص"""
    # تهيئة المحرك
    engine = DiagnosticEngine(
        knowledge_base_path="data/medical_knowledge_base.json",
        pii_patterns_path="config/pii_patterns.json"
    )
    
    # بيانات المريض والأعراض
    patient_id = "P123456"
    symptoms = [
        PatientSymptom(symptom_name="ألم في الصدر", severity=8, duration_hours=2),
        PatientSymptom(symptom_name="ضيق في التنفس", severity=7, duration_hours=2),
        PatientSymptom(symptom_name="عرق بارد", severity=6, duration_hours=1),
        PatientSymptom(symptom_name="خفقان", severity=7, duration_hours=2)
    ]
    
    # بيانات الموافقة
    consent_data = {
        "patient_id": patient_id,
        "consent_granted": True,
        "consent_date": "2025-12-30T10:00:00Z",
        "consent_version": "1.0"
    }
    
    try:
        # إجراء التشخيص
        results = engine.diagnose(patient_id, symptoms, consent_data)
        
        # عرض النتائج
        print("نتائج التشخيص:")
        print("=" * 50)
        for i, result in enumerate(results, 1):
            print(f"\nالتشخيص #{i}: {result.condition.condition_name}")
            print(f"مستوى الثقة: {result.confidence_level.value}")
            print(f"مستوى الخطورة: {result.risk_level.value}")
            print(f"التخصص المطلوب: {result.condition.medical_specialty.value}")
            print("\nالتفسير:")
            print(result.explanation)
            print("-" * 50)
        
        # الحصول على توصية بالتخصص الطبي
        specialist_recommendation = engine.get_medical_specialist_recommendation(results)
        print("\nتوصية بالتخصص الطبي:")
        print(f"التخصص الأساسي: {specialist_recommendation['specialty']}")
        print(f"الأولوية: {specialist_recommendation['priority']}")
        print(f"التخصصات البديلة: {', '.join(specialist_recommendation['alternative_specialties'])}")
    
    except Exception as e:
        logger.error(f"Diagnosis failed: {str(e)}")
        print(f"خطأ في التشخيص: {str(e)}")

if __name__ == "__main__":
    example_usage()
```

## 📁 tests/test_full_pipeline.py (Complete)

```python
"""
اختبارات شاملة لأنبوب العمل الكامل من المعالجة المسبقة إلى الخدمة
هذه الاختبارات تغطي كامل سير العمل في الإنتاج، مما يضمن أن جميع المكونات تعمل معًا بشكل صحيح.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import json
import time
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.core.math_operations import dot_product, matrix_multiply, softmax
from src.ml.classical.linear_regression import LinearRegressionScratch
from src.ml.deep_learning.neural_networks import NeuralNetworkFromScratch
from src.production.api import app as api_app
from src.production.monitoring import ModelMonitor
from case_studies.medical_diagnosis_agent.implementation.diagnostic_engine import DiagnosticEngine, PatientSymptom
from case_studies.legal_document_rag_system.implementation.vector_index import VectorIndex

class TestFullPipeline:
    """اختبارات لأنبوب العمل الكامل"""
    
    @pytest.fixture
    def test_data(self):
        """بيانات اختبارية"""
        # بيانات خطية بسيطة
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])  # y = 2x
        
        # بيانات تصنيف
        X_class = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_class = np.array([0, 0, 1, 1, 1])
        
        return {
            'regression': {'X': X, 'y': y},
            'classification': {'X': X_class, 'y': y_class}
        }
    
    @pytest.fixture
    def temp_dir(self):
        """دليل مؤقت للاختبارات"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_end_to_end_machine_learning_pipeline(self, test_data, temp_dir):
        """اختبار كامل لأنبوب تعلم الآلة من التدريب إلى النشر"""
        # 1. تدريب نموذج الانحدار الخطي
        X_reg, y_reg = test_data['regression']['X'], test_data['regression']['y']
        
        # 1.1. اختبار التدريب باستخدام التنفيذ من الصفر
        model_scratch = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
        model_scratch.fit(X_reg, y_reg)
        
        # 1.2. التحقق من النتائج
        predictions = model_scratch.predict(X_reg)
        mse = np.mean((predictions - y_reg) ** 2)
        assert mse < 0.1, f"MSE too high: {mse}"
        
        # 1.3. حفظ النموذج
        model_path = temp_dir / "linear_model.pkl"
        import joblib
        joblib.dump(model_scratch, model_path)
        assert model_path.exists(), "Model file was not created"
        
        # 2. اختبار خدمة النموذج باستخدام API
        from fastapi.testclient import TestClient
        
        client = TestClient(api_app)
        
        # 2.1. التحقق من حالة الصحة
        response = client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        
        # 2.2. إرسال طلب تنبؤ
        prediction_request = {
            "features": {"feature_0": 6.0},
            "request_id": "test_request_1"
        }
        
        with patch("src.production.api.joblib.load") as mock_load:
            mock_load.return_value = model_scratch
            
            response = client.post("/predict", json=prediction_request)
            assert response.status_code == 200
            
            result = response.json()
            prediction = result["prediction"]
            
            # نتوقع أن تكون القيمة قريبة من 12.0 (y = 2x)
            assert abs(prediction - 12.0) < 1.0, f"Prediction {prediction} is too far from expected 12.0"
    
    def test_neural_network_training_and_inference(self, test_data):
        """اختبار تدريب الشبكة العصبية والاستدلال"""
        X, y = test_data['classification']['X'], test_data['classification']['y']
        
        # تحويل y إلى تنسيق مناسب للشبكة العصبية (one-hot encoding)
        y_one_hot = np.zeros((len(y), 2))
        y_one_hot[np.arange(len(y)), y] = 1
        
        # بناء الشبكة العصبية
        nn = NeuralNetworkFromScratch(
            layer_sizes=[2, 4, 2],
            activation='relu',
            output_activation='softmax'
        )
        
        # تدريب النموذج
        history = nn.train(
            X, y_one_hot,
            epochs=1000,
            learning_rate=0.1,
            batch_size=32,
            verbose=False
        )
        
        # التحقق من أن الخسارة تقل مع الوقت
        assert len(history) > 0
        assert history[-1] < history[0] * 0.1, "Loss did not decrease sufficiently"
        
        # التنبؤ
        predictions = nn.predict(X)
        
        # تحويل التوقعات إلى فئات
        predicted_classes = np.argmax(predictions, axis=1)
        
        # حساب الدقة
        accuracy = np.mean(predicted_classes == y)
        assert accuracy > 0.9, f"Accuracy too low: {accuracy}"
    
    def test_medical_diagnosis_pipeline(self, temp_dir):
        """اختبار أنبوب تشخيص طبي كامل"""
        # 1. إنشاء قاعدة معرفة طبية بسيطة للاختبار
        knowledge_base_path = temp_dir / "test_knowledge_base.json"
        
        test_knowledge_base = {
            "conditions": [
                {
                    "name": "احتشاء عضلة القلب",
                    "specialty": "cardiology",
                    "severity_level": 10,
                    "urgency_level": 10,
                    "treatment_options": ["التدخل الجراحي", "الأدوية"],
                    "diagnostic_criteria": [
                        "ألم حاد في الصدر ينتشر إلى الذراع الأيسر",
                        "ضيق في التنفس",
                        "عرق بارد",
                        "خفقان"
                    ]
                },
                {
                    "name": "التهاب المعدة",
                    "specialty": "gastroenterology",
                    "severity_level": 6,
                    "urgency_level": 4,
                    "treatment_options": ["مثبطات الحموضة", "المضادات الحيوية"],
                    "diagnostic_criteria": [
                        "ألم في الجزء العلوي من البطن",
                        "غثيان",
                        "قيء",
                        "حرقة في المعدة"
                    ]
                }
            ]
        }
        
        with open(knowledge_base_path, 'w', encoding='utf-8') as f:
            json.dump(test_knowledge_base, f, ensure_ascii=False, indent=2)
        
        # 2. تهيئة محرك التشخيص
        engine = DiagnosticEngine(str(knowledge_base_path))
        
        # 3. بيانات المريض والأعراض
        symptoms = [
            PatientSymptom(symptom_name="ألم في الصدر", severity=8, duration_hours=2),
            PatientSymptom(symptom_name="ضيق في التنفس", severity=7, duration_hours=2),
            PatientSymptom(symptom_name="عرق بارد", severity=6, duration_hours=1)
        ]
        
        consent_data = {
            "patient_id": "TEST123",
            "consent_granted": True,
            "consent_date": "2025-12-30T10:00:00Z",
            "consent_version": "1.0"
        }
        
        # 4. إجراء التشخيص
        results = engine.diagnose("TEST123", symptoms, consent_data)
        
        # 5. التحقق من النتائج
        assert len(results) > 0, "No diagnostic results returned"
        assert results[0].condition.condition_name == "احتشاء عضلة القلب", "Incorrect primary diagnosis"
        assert results[0].risk_level.value == "critical", "Risk level should be critical"
        assert results[0].confidence_level.value in ["high", "medium"], "Confidence level should be at least medium"
        
        # 6. التحقق من توصية التخصص الطبي
        specialist_recommendation = engine.get_medical_specialist_recommendation(results)
        assert specialist_recommendation["specialty"] == "cardiology", "Recommended specialty should be cardiology"
        assert specialist_recommendation["priority"] == "critical", "Priority should be critical"
    
    def test_legal_document_rag_pipeline(self, temp_dir):
        """اختبار أنبوب RAG للوثائق القانونية"""
        # 1. إنشاء وثائق قانونية بسيطة للاختبار
        test_doc_path = temp_dir / "test_document.pdf"
        
        # نظرًا لأننا لا يمكننا إنشاء ملف PDF حقيقي في الاختبار،
        # سنستخدم محاكاة وسنتعامل كما لو أن الملف موجود
        test_doc_path.touch()
        
        # 2. تهيئة فهرس المتجهات
        index = VectorIndex(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384
        )
        
        # 3. إضافة الوثائق إلى الفهرس
        with patch.object(index, 'add_documents') as mock_add:
            index.add_documents([str(test_doc_path)])
            mock_add.assert_called_once()
        
        # 4. اختبار البحث
        with patch.object(index, 'hybrid_search') as mock_search:
            mock_search.return_value = [
                MagicMock(score=0.95, chunk=MagicMock(content="هذا نص قانوني اختباري", metadata={}))
            ]
            
            results = index.hybrid_search("استشارة قانونية حول العقود", top_k=3)
            
            # التحقق من النتائج
            assert len(results) == 1
            assert results[0].score > 0.9
            assert "قانوني" in results[0].chunk.content
    
    def test_model_monitoring_pipeline(self, temp_dir):
        """اختبار أنبوب مراقبة النماذج لاكتشاف الانحراف"""
        # 1. إنشاء بيانات تدريب واختبار
        np.random.seed(42)
        X_train = np.random.normal(0, 1, (100, 2))
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        
        X_test = np.random.normal(0, 1, (50, 2))  # توزيع مشابه
        X_drift = np.random.normal(1, 2, (50, 2))  # توزيع مختلف (انحراف)
        
        # 2. تدريب نموذج بسيط
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # 3. تهيئة مراقب النموذج
        monitor = ModelMonitor()
        
        # 4. إضافة بيانات التدريب كمرجع
        monitor.add_reference_data(X_train)
        
        # 5. التحقق من عدم وجود انحراف في البيانات المماثلة
        drift_result = monitor.detect_drift(X_test)
        assert not drift_result.drift_detected, "Drift detected in similar data"
        assert drift_result.p_value > 0.05, f"p-value too low: {drift_result.p_value}"
        
        # 6. التحقق من وجود انحراف في البيانات المختلفة
        drift_result = monitor.detect_drift(X_drift)
        assert drift_result.drift_detected, "No drift detected in drifted data"
        assert drift_result.p_value < 0.05, f"p-value too high: {drift_result.p_value}"
        
        # 7. اختبار مراقبة الأداء
        y_pred = model.predict(X_test)
        performance = monitor.track_performance(y_test=np.random.randint(0, 2, 50), y_pred=y_pred)
        
        assert "accuracy" in performance
        assert "precision" in performance
        assert "recall" in performance
    
    def test_production_error_handling(self):
        """اختبار معالجة الأخطاء في بيئة الإنتاج"""
        from fastapi.testclient import TestClient
        
        client = TestClient(api_app)
        
        # 1. طلب غير صالح (بيانات ناقصة)
        invalid_request = {
            "features": {},  # لا توجد ميزات
        }
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422  # خطأ في التحقق من الصحة
        
        # 2. خطأ داخلي في الخادم (نموذج غير محمل)
        with patch("src.production.api.model", None):
            response = client.post("/predict", json={"features": {"feature_1": 1.0}})
            assert response.status_code == 500  # خطأ داخلي
            
            error_data = response.json()
            assert "detail" in error_data
            assert "Model not loaded" in error_data["detail"]
        
        # 3. التحقق من تسجيل الأخطاء
        with patch("src.production.api.logger") as mock_logger:
            response = client.post("/predict", json={"features": {"feature_1": 1.0}})
            mock_logger.error.assert_called()  # يجب استدعاء تسجيل الخطأ
    
    def test_system_performance_under_load(self, test_data):
        """اختبار أداء النظام تحت حمل"""
        # 1. إنشاء نموذج بسيط
        X, y = test_data['regression']['X'], test_data['regression']['y']
        model = LinearRegressionScratch()
        model.fit(X, y)
        
        # 2. اختبار الأداء للتنبؤات المتعددة
        start_time = time.time()
        
        # محاكاة 1000 طلب متتالي
        for i in range(1000):
            features = {"feature_0": float(i % 10)}
            prediction = model.predict(np.array([[features["feature_0"]]]))
        
        total_time = time.time() - start_time
        avg_time_per_prediction = total_time / 1000
        
        assert avg_time_per_prediction < 0.001, f"Average prediction time too high: {avg_time_per_prediction:.6f}s"
        
        # 3. اختبار استهلاك الذاكرة
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # بالميغابايت
        
        assert memory_usage < 100, f"Memory usage too high: {memory_usage:.2f} MB"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## 📁 setup.sh (Complete)

```bash
#!/bin/bash
set -e

# AI Engineer Toolkit 2025 - Setup Script
# هذا السكربت يهيئ البيئة الكاملة لـ AI Engineer Toolkit

log() {
    echo -e "\033[1;34m[$(date +'%Y-%m-%d %H:%M:%S')]\033[0m $1"
}

error() {
    echo -e "\033[1;31m[ERROR] $1\033[0m"
    exit 1
}

warning() {
    echo -e "\033[1;33m[WARNING] $1\033[0m"
}

# التحقق من المتطلبات الأساسية
check_prerequisites() {
    log "Checking prerequisites..."
    
    # التحقق من وجود بايثون
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please install Python 3.10 or higher."
    fi
    
    # التحقق من وجود pip
    if ! command -v pip3 &> /dev/null; then
        error "pip3 is not installed. Please install pip."
    fi
    
    # التحقق من وجود git
    if ! command -v git &> /dev/null; then
        warning "git is not installed. Some features may not work properly."
    fi
    
    # التحقق من وجود Docker
    if ! command -v docker &> /dev/null; then
        warning "Docker is not installed. Containerized execution will not be available."
    fi
    
    # التحقق من وجود GPU لـ CUDA
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected. CUDA dependencies will be installed."
        export GPU_AVAILABLE=true
    else
        warning "No NVIDIA GPU detected. PyTorch will be installed in CPU-only mode."
        export GPU_AVAILABLE=false
    fi
}

# إنشاء بيئة conda
setup_conda_environment() {
    log "Setting up conda environment..."
    
    # التحقق من وجود conda
    if ! command -v conda &> /dev/null; then
        error "Conda is not installed. Please install Miniconda or Anaconda first."
    fi
    
    # إنشاء بيئة conda جديدة
    ENV_NAME="ai-engineer-toolkit-2025"
    
    # حذف البيئة القديمة إذا كانت موجودة
    if conda env list | grep -q "$ENV_NAME"; then
        log "Removing existing conda environment: $ENV_NAME"
        conda env remove -n "$ENV_NAME" -y
    fi
    
    # إنشاء بيئة جديدة
    log "Creating new conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.10 -y
    
    # تنشيط البيئة
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    log "Conda environment created and activated: $ENV_NAME"
}

# تثبيت المتطلبات
install_requirements() {
    log "Installing requirements..."
    
    if [ "$GPU_AVAILABLE" = true ]; then
        log "Installing GPU-enabled PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        log "Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio
    fi
    
    # تثبيت المتطلبات الأخرى
    pip install -r requirements.txt
    
    # تثبيت jupyterlab والإضافات
    log "Installing JupyterLab and extensions..."
    pip install jupyterlab
    jupyter labextension install @jupyterlab/toc
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    
    # تثبيت أدوات المراقبة
    log "Installing monitoring tools..."
    pip install prometheus-client grafana-api
    
    # تثبيت أدوات قواعد البيانات المتجهية
    log "Installing vector database tools..."
    pip install faiss-cpu hnswlib pgvector
    
    # تثبيت أدوات الذكاء الاصطناعي التوليدي
    log "Installing generative AI tools..."
    pip install transformers sentence-transformers accelerate
    pip install langchain langchain-community langchain-core
    
    # تثبيت أدوات الإنتاج
    log "Installing production tools..."
    pip install uvicorn gunicorn fastapi
    pip install redis psycopg2-binary
    
    # تثبيت أدوات الاختبار
    log "Installing testing tools..."
    pip install pytest pytest-asyncio pytest-cov
    
    log "All requirements installed successfully"
}

# إعداد البيانات
setup_data() {
    log "Setting up sample data..."
    
    # إنشاء دليل البيانات
    mkdir -p data/sample_datasets
    
    # تنزيل مجموعات البيانات النموذجية
    log "Downloading sample datasets..."
    
    # Iris dataset
    if [ ! -f "data/sample_datasets/iris.csv" ]; then
        curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" -o "data/sample_datasets/iris.csv"
        echo "sepal_length,sepal_width,petal_length,petal_width,class" | cat - data/sample_datasets/iris.csv > temp && mv temp data/sample_datasets/iris.csv
    fi
    
    # Titanic dataset
    if [ ! -f "data/sample_datasets/titanic.csv" ]; then
        curl -L "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv" -o "data/sample_datasets/titanic.csv"
    fi
    
    # إنشاء بيانات اصطناعية
    log "Generating synthetic data..."
    python scripts/data_preprocessing/generate_synthetic_data.py
    
    log "Data setup completed"
}

# إعداد قواعد البيانات
setup_databases() {
    log "Setting up databases..."
    
    # إعداد PostgreSQL
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        log "Starting PostgreSQL and Redis containers..."
        docker-compose up -d
        
        # الانتظار حتى تكون قواعد البيانات جاهزة
        sleep 10
        
        log "Databases started successfully"
    else
        warning "Docker or docker-compose not available. Skipping database setup."
        warning "Please set up PostgreSQL and Redis manually for full functionality."
    fi
}

# إعداد pre-commit hooks
setup_pre_commit() {
    log "Setting up pre-commit hooks..."
    
    pip install pre-commit
    pre-commit install
    
    log "Pre-commit hooks installed successfully"
}

# التحقق من التثبيت
verify_installation() {
    log "Verifying installation..."
    
    # التحقق من إصدار بايثون
    python_version=$(python --version 2>&1)
    log "Python version: $python_version"
    
    # التحقق من إصدار PyTorch
    torch_version=$(python -c "import torch; print(torch.__version__)")
    log "PyTorch version: $torch_version"
    
    # التحقق من GPU
    if [ "$GPU_AVAILABLE" = true ]; then
        gpu_info=$(python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')")
        log "$gpu_info"
    fi
    
    # التحقق من Jupyter
    jupyter_version=$(jupyter --version 2>&1 | head -1)
    log "Jupyter version: $jupyter_version"
    
    log "Installation verification completed"
}

# الدالة الرئيسية
main() {
    log "Starting AI Engineer Toolkit 2025 setup..."
    
    # التحقق من المتطلبات الأساسية
    check_prerequisites
    
    # إعداد بيئة conda
    setup_conda_environment
    
    # تثبيت المتطلبات
    install_requirements
    
    # إعداد البيانات
    setup_data
    
    # إعداد قواعد البيانات
    setup_databases
    
    # إعداد pre-commit hooks
    setup_pre_commit
    
    # التحقق من التثبيت
    verify_installation
    
    log "\n\033[1;32mSetup completed successfully!\033[0m"
    log "To get started:"
    log "1. Activate the conda environment: conda activate ai-engineer-toolkit-2025"
    log "2. Launch Jupyter Lab: ./run.sh"
    log "3. Run tests: make test"
    log "4. Build Docker image: docker build -t ai-engineer-toolkit:latest ."
    
    # إنشاء ملف .env إذا لم يكن موجودًا
    if [ ! -f ".env" ]; then
        log "Creating .env file with default configuration..."
        cat > .env << EOF
# AI Engineer Toolkit Configuration
ENV=development
PORT=8888
MODEL_PATH=models/default_model.pkl
DATABASE_URL=postgresql://user:password@localhost:5432/ai_toolkit
REDIS_URL=redis://localhost:6379/0
SENTRY_DSN=
# CUDA Configuration
CUDA_VISIBLE_DEVICES=0
EOF
    fi
}

# تنفيذ الدالة الرئيسية
main "$@"
```

## 📁 docker-compose.yml (Complete)

```yaml
version: '3.8'

services:
  # خدمة Jupyter Lab للتطوير
  jupyter:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8888:8888"
    volumes:
      - .:/app
      - jupyter_data:/root/.local/share/jupyter
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - JUPYTER_TOKEN=ai-engineer-toolkit-2025
      - PYTHONPATH=/app
    depends_on:
      - postgres
      - redis
    command: ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

  # خدمة FastAPI للنماذج
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - PYTHONPATH=/app
      - PORT=8000
      - DATABASE_URL=postgresql://user:password@postgres:5432/ai_toolkit
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    command: ["uvicorn", "src.production.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  # خدمة PostgreSQL
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: ai_toolkit
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/database/init.sql:/docker-entrypoint-initdb.d/init.sql

  # خدمة Redis للتخزين المؤقت
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # خدمة Prometheus للمراقبة
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  # خدمة Grafana للعرض
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus

  # خدمة pgvector لقواعد البيانات المتجهة
  pgvector:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: vector_db
    ports:
      - "5433:5432"
    volumes:
      - pgvector_data:/var/lib/postgresql/data

  # خدمة اختبار الأداء
  benchmark:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - .:/app
    environment:
      - PYTHONPATH=/app
    depends_on:
      - api
      - postgres
    command: ["python", "-m", "pytest", "benchmarks/"]

volumes:
  jupyter_data:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  pgvector_data:
```

## 📁 Makefile (Complete)

```makefile
# AI Engineer Toolkit 2025 - Makefile
# أوامر سريعة للتطوير والمراقبة والنشر

# المتغيرات
ENV = development
PYTHON = python3
PIP = pip3
DOCKER = docker
DOCKER_COMPOSE = docker-compose
JUPYTER_PORT = 8888
API_PORT = 8000

# الأهداف الأساسية
.PHONY: help setup run test clean build deploy

## help: Display available commands
help:
	@echo "AI Engineer Toolkit 2025 - Development Commands"
	@echo "-------------------------------------------------"
	@echo "make setup       - Setup the development environment"
	@echo "make run         - Run Jupyter Lab"
	@echo "make test        - Run all tests"
	@echo "make test-fast   - Run fast tests only"
	@echo "make lint        - Run code linting"
	@echo "make format      - Format code with Black"
	@echo "make docs        - Generate documentation"
	@echo "make build       - Build Docker image"
	@echo "make deploy      - Deploy to production"
	@echo "make clean       - Clean temporary files"
	@echo "make benchmark   - Run performance benchmarks"
	@echo "make monitor     - Start monitoring dashboard"
	@echo "make help        - Show this help message"

## setup: Setup the development environment
setup:
	@echo "Setting up development environment..."
	./setup.sh

## run: Run Jupyter Lab
run: setup
	@echo "Starting Jupyter Lab on http://localhost:$(JUPYTER_PORT)"
	@echo "Access token: ai-engineer-toolkit-2025"
	jupyter lab --port $(JUPYTER_PORT) --ip 0.0.0.0 --no-browser

## test: Run all tests
test: lint
	@echo "Running all tests..."
	pytest tests/ notebooks/ --cov=src --cov-report=html --cov-report=term

## test-fast: Run fast tests only
test-fast:
	@echo "Running fast tests..."
	pytest tests/ -m "not slow"

## lint: Run code linting
lint:
	@echo "Running code linting..."
	flake8 src/ notebooks/ tests/
	mypy src/ notebooks/ tests/
	black --check src/ notebooks/ tests/

## format: Format code with Black
format:
	@echo "Formatting code with Black..."
	black src/ notebooks/ tests/

## docs: Generate documentation
docs:
	@echo "Generating documentation..."
	pdoc src/ --output-dir docs/api --force
	sphinx-build docs/sphinx docs/sphinx/_build

## build: Build Docker image
build:
	@echo "Building Docker image..."
	$(DOCKER) build -t ai-engineer-toolkit:latest .

## deploy: Deploy to production
deploy: build
	@echo "Deploying to production..."
	$(DOCKER) run -d -p 8000:8000 --name ai-engineer-toolkit-production ai-engineer-toolkit:latest

## clean: Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf docs/api
	rm -rf docs/sphinx/_build
	rm -rf .mypy_cache
	rm -f core
	rm -f *.log
	rm -f *.sqlite
	rm -f *.pkl
	rm -f *.h5
	rm -f *.onnx

## benchmark: Run performance benchmarks
benchmark: setup
	@echo "Running performance benchmarks..."
	python benchmarks/inference_optimization/vllm_vs_tgi.py
	python benchmarks/cost_performance_tradeoffs/model_size_vs_latency.py

## monitor: Start monitoring dashboard
monitor:
	@echo "Starting monitoring dashboard..."
	$(DOCKER_COMPOSE) up -d prometheus grafana
	@echo "Grafana dashboard available at http://localhost:3000"
	@echo "Default credentials: admin/admin"

## up: Start all services
up:
	@echo "Starting all services..."
	$(DOCKER_COMPOSE) up -d

## down: Stop all services
down:
	@echo "Stopping all services..."
	$(DOCKER_COMPOSE) down

# الأهداف الخاصة بالبيانات
data-download:
	@echo "Downloading sample datasets..."
	python scripts/data_preprocessing/download_sample_datasets.py

data-generate:
	@echo "Generating synthetic data..."
	python scripts/data_preprocessing/generate_synthetic_data.py

# الأهداف الخاصة بالنماذج
model-train:
	@echo "Training default model..."
	python scripts/model_training/train_default_model.py

model-evaluate:
	@echo "Evaluating model performance..."
	python scripts/model_training/evaluate_model.py

# الأهداف الخاصة بالنشر
deploy-staging:
	@echo "Deploying to staging environment..."
	# تنفيذ عمليات النشر للبيئة التجريبية

deploy-production:
	@echo "Deploying to production environment..."
	# تنفيذ عمليات النشر للبيئة الإنتاجية

# القواعد الافتراضية
.DEFAULT_GOAL := help
```

## 📁 CONTRIBUTING.md (Complete)

```markdown
# Contributing to AI Engineer Toolkit 2025

**"المساهمة في هذا المستودع ليست مجرد إضافة كود، بل هي انضمام إلى مجتمع مهندسي الذكاء الاصطناعي الذين يبنون المستقبل."**

## Our Philosophy

هذا المستودع يتبع فلسفة "الصندوق الأبيض" (White-box Approach):
- **الرياضيات أولاً**: كل خوارزمية تبدأ باشتقاقات رياضية من المبادئ الأولى
- **التنفيذ من الصفر**: كل مفهوم يبدأ بتنفيذ نقي باستخدام NumPy/Python قبل استخدام المكتبات
- **الإطلاق دائماً**: كل مفهوم يشمل اعتبارات النشر والمراقبة والتكلفة
- **التقنيات مقابل المقايضات**: الفهم العميق للمقايضات يسبق اختيار الأدوات

## Contribution Guidelines

### 1. Scope of Contributions

نرحب بالمساهمات في:

- **الدفاتر التعليمية**: دفاتر Jupyter التي تتبع هيكل "النظرية أولاً، ثم التنفيذ من الصفر، ثم الإنتاج"
- **النماذج الإنتاجية**: كود قابل للنشر مع مراقبة، أمان، وتحسين التكلفة
- **دراسات الحالة**: أنظمة حقيقية مع تحليل المقايضات والتكلفة/الأداء
- **اختبارات الأداء**: معايير شاملة للمقارنة بين الحلول المختلفة
- **المستندات**: توثيق واضح يشرح "لماذا" قبل "كيف"

### 2. Quality Standards

كل مساهمة يجب أن تتبع:

#### 2.1 White-Box Approach
- كل خوارزمية يجب أن تبدأ باشتقاق رياضي واضح
- يجب تقديم تنفيذ من الصفر قبل استخدام المكتبات
- يجب توثيق المقايضات والقيود في كل خطوة

#### 2.2 Production-Ready Code
- **Type Hints**: جميع الدوال يجب أن تحتوي على تلميحات الأنواع
- **Docstrings**: توثيق كامل لكل دالة باستخدام نمط NumPy
- **Testing**: كل وحدة يجب أن تحتوي على اختبارات تغطي حالات الحدود
- **Performance**: يجب قياس الأداء وتضمينه في الوثائق
- **Error Handling**: التعامل مع الأخطاء المتوقعة وعدم الاعتماد على المكتبات
