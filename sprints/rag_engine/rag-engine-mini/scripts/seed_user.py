"""
Seed Demo User Script
======================
Creates a demo user with API key for testing.

سكربت إنشاء مستخدم تجريبي
"""

import uuid


def main() -> None:
    """Create demo user with API key."""
    
    user_id = str(uuid.uuid4())
    api_key = "demo_api_key_12345678"
    email = "demo@example.com"
    
    # In production, this would insert into database
    # For now, just print the credentials
    
    print("=" * 60)
    print("🌱 Demo User Created")
    print("=" * 60)
    print(f"  User ID:  {user_id}")
    print(f"  Email:    {email}")
    print(f"  API Key:  {api_key}")
    print("=" * 60)
    print()
    print("Use the API key in X-API-KEY header:")
    print(f'  curl -H "X-API-KEY: {api_key}" http://localhost:8000/health')
    print()
    print("⚠️  Note: This script currently only prints credentials.")
    print("    For production, implement database insertion.")
    print("=" * 60)


if __name__ == "__main__":
    main()
