# Password Hashing Fundamentals

## Learning Objectives

After completing this lesson, you will understand:
- [ ] Why passwords must never be stored in plain text
- [ ] The difference between hashing and encryption
- [ ] How salt, pepper, and time-based attacks work
- [ ] Why Argon2 is the current industry standard
- [ ] How to implement secure password hashing in Python

## Prerequisites

- **Knowledge**: Basic Python, understanding of cryptographic fundamentals
- **Tools**: Python 3.11+, argon2-cffi library
- **Time**: 30-45 minutes

## Concepts Deep Dive

### What is Password Hashing?

**Definition**: Password hashing is a one-way cryptographic function that transforms a password into a fixed-size string (hash) that cannot be reversed.

**Why it matters**: 
- **Security**: If a database is breached, attackers cannot recover original passwords
- **Compliance**: Regulations like GDPR, PCI-DSS require secure password storage
- **Trust**: Users expect their passwords to be protected

**How it works**:
```
Plaintext Password  →  Hash Function  →  Stored Hash
"MyPassword123!"    →  SHA256/Argon2   →  "$argon2id$v=19$..."
```

Key properties:
1. **Deterministic**: Same input always produces same hash (with same salt)
2. **One-way**: Cannot reverse hash to get password
3. **Avalanche effect**: Small password change produces completely different hash
4. **Slow**: Intentionally computationally expensive to prevent brute force

---

### Hashing vs Encryption

| Aspect | Hashing | Encryption |
|---------|-----------|-------------|
| **Purpose** | One-way verification | Two-way secrecy |
| **Reversible?** | ❌ No | ✅ Yes (with key) |
| **Key Required** | ❌ No | ✅ Yes |
| **Use Case** | Passwords, file integrity | Data at rest, communication |
| **Algorithms** | Argon2, bcrypt, PBKDF2 | AES, ChaCha20 |

**Why not encryption for passwords?**
- If encryption key is stolen (along with database), all passwords are exposed
- Hashing doesn't need a key - the "key" is the password itself
- Hashing is designed to be slow (prevents brute force)

---

### The Hashing Arms Race

#### Attackers' Tools:
1. **Dictionary Attacks**: Try common passwords (password, 123456, qwerty)
2. **Brute Force**: Try every possible combination
3. **Rainbow Tables**: Pre-computed hashes for common passwords
4. **GPU/ASIC Attacks**: Parallel hardware to compute billions of hashes per second

#### Defenders' Tools:
1. **Salt**: Random data added before hashing to prevent rainbow tables
2. **Key Stretching**: Making computation intentionally slow
3. **Memory-Hard Functions**: Require lots of RAM (expensive for GPUs/ASICs)
4. **Pepper**: Secret added to all hashes (stored separately from database)

---

### Salt: The Rainbow Table Killer

**What is Salt?**
Random data (e.g., 16 bytes) added to password before hashing:
```
hash = argon2id(password + salt)
```

**Why it matters**:
- **Defeats Rainbow Tables**: Each salt creates unique hash space
- **Prevents Pre-computation**: Attacker must recompute for each salt
- **Same Password ≠ Same Hash**: "password" + salt1 ≠ "password" + salt2

**Implementation Detail**:
- Store salt WITH the hash (in the encoded string)
- Never reuse salt across passwords
- Use cryptographically secure random generator

---

### Pepper: Extra Security Layer

**What is Pepper?**
Secret value added to ALL passwords, stored separately from database:
```
hash = argon2id(password + salt + pepper)
pepper = get_pepper_from_vault_or_hsm()
```

**Why it matters**:
- **Double Protection**: Even if DB is stolen, passwords remain protected
- **Key Separation**: Pepper must be stolen separately from DB
- **Reversible**: If pepper is compromised, you can change it and rehash

**Best Practices**:
- Store pepper in HSM (Hardware Security Module) or secret manager
- Rotate pepper periodically (requires rehashing all passwords)
- Never store pepper in application code or config files

---

### Memory-Hard Functions: GPU Resistance

**The Problem**:
- Modern GPUs can compute billions of SHA256 hashes per second
- Traditional hashing functions (SHA256, MD5) are TOO FAST for passwords

**The Solution: Memory-Hard Functions**:
- **Argon2**: Winner of Password Hashing Competition 2015
- **scrypt**: Older but still secure
- **bcrypt**: Widely used but not memory-hard

**How Memory-Hard Functions Work**:
```
# Simplified example
def argon2(password, salt):
    # Require 64MB RAM
    memory_blocks = allocate_memory(64 * 1024 * 1024)
    
    # Fill memory with random data based on password
    for i in range(3):  # time_cost
        fill_blocks(memory_blocks, password, salt)
    
    # Extract hash from memory
    return extract_hash(memory_blocks)
```

**GPU Impact**:
- GPUs have limited VRAM (4-8GB typically)
- Computing one hash requires 64MB VRAM
- Parallelizing 1000 hashes requires 64GB VRAM (impossible)
- **Result**: GPUs can't attack Argon2 effectively

---

### Argon2: The Current Standard

**Why Argon2 Won**:
- Proven security design (2015 competition)
- Tunable memory cost (future-proof against faster hardware)
- Resistance to side-channel attacks
- Independent implementation (no patents)

**Argon2 Variants**:
| Variant | Use Case |
|----------|-----------|
| **Argon2id** | Password hashing (recommended) |
| **Argon2d** | Key derivation (resistant to side-channel) |

**Configuration Parameters**:
```python
# Production-safe defaults
time_cost = 3        # Iterations (higher = slower)
memory_cost = 65536    # Memory in KiB (64MB)
parallelism = 4        # Threads (typically 1-8)
hash_len = 32          # Output length (256 bits)
salt_len = 16          # Salt length (128 bits)
```

**Parameter Selection Guide**:
- **time_cost**: Increase for more security, slower verification (3-4 recommended)
- **memory_cost**: Set based on available RAM (64MB minimum, 256MB recommended)
- **parallelism**: Match CPU cores (4-8 for servers, 1-2 for mobile)
- **Benchmark**: Test on your hardware, aim for 100-500ms per hash

---

### Time-Based Attacks: The Next Threat

**The Attack**:
1. Attacker registers account with known password
2. Attacker times hash computation: 100ms vs 200ms
3. **Timing leak reveals**: When user's password starts with "A" vs "B"
4. Attacker narrows down character by character

**The Defense**:
- **Constant-Time Comparison**: Always take same time regardless of result
- **Hash Comparison**: Use `hmac.compare_digest()` or Argon2's built-in verify
- **No Early Returns**: Never "return False immediately" on first mismatch

**Implementation**:
```python
# ❌ Vulnerable (timing attack)
def verify_vuln(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return False  # Early return leaks position
    return True

# ✅ Secure (constant-time)
def verify_secure(hash1, hash2):
    return hmac.compare_digest(hash1, hash2)  # Always full time
```

---

## Code Walkthrough

### Our Implementation

```python
from argon2 import PasswordHasher

class Argon2PasswordHasher:
    def __init__(self, time_cost=3, memory_cost=65536, parallelism=4):
        # Initialize with production-safe parameters
        self._hasher = PasswordHasher(
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=32,     # 256-bit hash
            salt_len=16,     # 128-bit salt
            type=argon2.Type.ID,  # Argon2id for passwords
        )
    
    def hash(self, password: str) -> str:
        """Hash password with random salt."""
        return self._hasher.hash(password)
    
    def verify(self, password: str, hashed: str) -> bool:
        """Constant-time password verification."""
        try:
            return self._hasher.verify(hashed, password)
        except argon2.exceptions.VerifyMismatchError:
            return False
```

### Key Decisions

**Decision 1: Argon2id vs Argon2d**
- **Chosen**: Argon2id
- **Reason**: Optimized for password hashing, faster verification
- **Trade-off**: Slightly less side-channel resistant (not critical for passwords)

**Decision 2: Singleton Pattern**
- **Chosen**: Global hasher instance
- **Reason**: Reuse parameters, avoid repeated initialization
- **Trade-off**: Harder to change parameters at runtime (acceptable)

**Decision 3: Parameter Hardcoding**
- **Chosen**: Defaults in code, configurable via environment
- **Reason**: Security-by-default, prevent misconfiguration
- **Trade-off**: Less flexibility (acceptable for passwords)

---

## Practical Exercise

### Exercise 1: Hash and Verify

**Task**: Write a script that:
1. Hashes a password using our hasher
2. Verifies the correct password
3. Verifies an incorrect password
4. Prints results

**Solution**:
```python
from src.adapters.security.password_hasher import hash_password, verify_password

# Hash a password
password = "MySecurePassword123!"
hashed = hash_password(password)
print(f"Hash: {hashed}")

# Verify correct password
print(f"Correct password: {verify_password(password, hashed)}")

# Verify incorrect password
print(f"Wrong password: {verify_password('WrongPassword!', hashed)}")
```

**Expected Output**:
```
Hash: $argon2id$v=19$m=65536,t=3,p=4$...
Correct password: True
Wrong password: False
```

### Exercise 2: Parameter Benchmarking

**Task**: Benchmark different memory_cost settings and measure performance.

**Solution**:
```python
import time
from src.adapters.security.password_hasher import Argon2PasswordHasher

for memory_cost in [16384, 65536, 262144]:  # 16MB, 64MB, 256MB
    hasher = Argon2PasswordHasher(memory_cost=memory_cost)
    
    start = time.time()
    hasher.hash("TestPassword123!")
    duration = (time.time() - start) * 1000  # Convert to ms
    
    print(f"Memory: {memory_cost//1024}MB, Time: {duration:.1f}ms")
```

---

## Testing & Validation

### Unit Tests

We've implemented comprehensive tests in `tests/unit/test_password_hasher.py`:

- ✅ Hash generation with valid passwords
- ✅ Rejection of empty/short passwords
- ✅ Correct password verification
- ✅ Incorrect password verification
- ✅ Salt randomness (different hashes for same password)
- ✅ Parameter detection (needs_rehash)

### Security Tests

Run security validation:

```bash
# Test password strength
python -c "
from src.adapters.security.password_hasher import hash_password
try:
    hash_password('short')
except ValueError as e:
    print(f'✓ Caught weak password: {e}')
"

# Test timing attack resistance
python -c "
import time
from src.adapters.security.password_hasher import verify_password
hashed = hash_password('CorrectPassword123!')

# Measure multiple verifications
times = []
for _ in range(10):
    start = time.time()
    verify_password('WrongPassword', hashed)
    times.append((time.time() - start) * 1000000)  # microseconds

print(f'Min: {min(times):.1f}μs, Max: {max(times):.1f}μs')
print(f'Variance: {max(times) - min(times):.1f}μs (should be small)')
"
```

---

## Production Considerations

### Performance Impact

| Parameter | Time per Hash | Security Level |
|-----------|---------------|---------------|
| time_cost=2, mem=16MB | ~100ms | Minimum |
| time_cost=3, mem=64MB | ~250ms | Recommended |
| time_cost=4, mem=256MB | ~1000ms | Maximum |

**Recommendation**: Aim for 200-500ms per hash on production hardware.

### Security Considerations

1. **Never log passwords**: Hashed or plain
2. **Never expose hash errors**: Generic "invalid credentials" only
3. **Rate limit login attempts**: Prevent credential stuffing
4. **Monitor failed attempts**: Detect brute force attacks
5. **Plan for rehashing**: If parameters change, migrate user passwords

### Operational Concerns

1. **Hash verification is expensive**: Consider caching recently verified hashes
2. **Password reset flow**: Use tokens, not hash comparison
3. **User migration**: If migrating from bcrypt, verify first then rehash
4. **Parameter updates**: Requires full password rehash campaign

### Monitoring Recommendations

Track metrics:
```python
# Log slow hashing (potential attack)
if hash_time > 1000:  # 1 second
    alert("Suspiciously slow hash verification")

# Log verification failures
if verification_failed:
    increment_metric("login_failed", tags=["user", user_id])

# Monitor hash parameter usage
if needs_rehash(detected_hash):
    alert("Password parameters outdated, plan rehash campaign")
```

---

## Further Reading

### Resources

- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [Password Hashing Competition (PHC)](https://www.password-hashing.net/)
- [Argon2 RFC Documentation](https://datatracker.ietf.org/doc/html/rfc9106)
- [Why You Should Hash, Not Encrypt](https://auth0.com/blog/2015/03/11/adding-hashing-to-your-toolbox/)
- [Book: "Password Books"](https://password-books.com/) - History of password security

### Papers

- "Argon2: The Memory-Hard Function for Password Hashing" (2015)
- "On the Security of Password Hashing Schemes" (2009)
- "Cost-Effective Password Hardening" (2017)

---

## Quiz

1. **Why shouldn't passwords be stored in plain text?**
   - A) It takes too much storage
   - B) It's harder to implement
   - C) Database breaches would expose all passwords
   - D) Plain text is faster to process

2. **What is the primary purpose of salt in password hashing?**
   - A) To make passwords longer
   - B) To prevent rainbow table attacks
   - C) To encrypt the hash
   - D) To enable password recovery

3. **Why is Argon2 considered better than SHA256 for passwords?**
   - A) Argon2 is faster
   - B) Argon2 is memory-hard (resists GPU attacks)
   - C) SHA256 produces longer hashes
   - D) Argon2 is reversible

4. **What is a timing attack?**
   - A) Attack when hashing takes too long
   - B) Attack using response time to infer information
   - C) Attack that targets system time clocks
   - D) Attack that slows down the hash function

5. **How does constant-time comparison prevent timing attacks?**
   - A) It uses a fixed time for all comparisons
   - B) It adds random delays
   - C) It compares only hash length, not content
   - D) It uses faster hardware

---

## Answer Key

1. **C** - Database breaches would expose all passwords if stored in plain text
2. **B** - Salt prevents pre-computed rainbow table attacks
3. **B** - Argon2 is memory-hard, making GPU attacks impractical
4. **B** - Timing attacks use response time variations to infer information
5. **A** - Constant-time comparison always takes the same time regardless of result

---

## Next Steps

After completing this lesson:
- [ ] Implement JWT authentication (Lesson 2)
- [ ] Build user registration flow (Lesson 3)
- [ ] Add rate limiting (Lesson 6)
